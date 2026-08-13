#include "cct/scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

namespace cct {
namespace {

constexpr std::uintmax_t kMaximumCheckpointBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumModelDimension = 4096U;
constexpr std::size_t kMaximumParameterCount = 8'000'000U;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::size_t checked_file_size(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    require(!error && size <= kMaximumCheckpointBytes, "Stage 5 checkpoint exceeds byte budget");
    return static_cast<std::size_t>(size);
}

std::size_t argmax(const std::vector<double>& values) {
    require(!values.empty(), "cannot argmax an empty vector");
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<double> softmax(const std::vector<double>& logits) {
    require(!logits.empty(), "cannot softmax an empty vector");
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> result(logits.size(), 0.0);
    double denominator = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        result[index] = std::exp(std::clamp(logits[index] - maximum, -60.0, 60.0));
        denominator += result[index];
    }
    require(denominator > 0.0 && std::isfinite(denominator), "softmax denominator is invalid");
    for (auto& value : result) value /= denominator;
    return result;
}

BaselineKind baseline_kind(Stage5ModelKind kind) {
    if (kind == Stage5ModelKind::DenseCausalAttention) return BaselineKind::DenseCausalAttention;
    if (kind == Stage5ModelKind::GRU) return BaselineKind::GRU;
    return BaselineKind::DiagonalSSM;
}

}  // namespace

std::vector<std::size_t> Stage5Vocabulary::encode_bytes(const std::string& text, bool append_end) {
    std::vector<std::size_t> result;
    result.reserve(text.size() + (append_end ? 1U : 0U));
    for (const auto character : text) result.push_back(static_cast<unsigned char>(character));
    if (append_end) result.push_back(kEndOfTextToken);
    return result;
}

std::string Stage5Vocabulary::decode_bytes(const std::vector<std::size_t>& tokens) {
    std::string result;
    for (const auto token : tokens) {
        if (token < 256) result.push_back(static_cast<char>(token));
        else if (token == kUnknownToken) result.push_back('?');
        else if (token == kEndOfTextToken) break;
        else throw std::runtime_error("invalid byte vocabulary token");
    }
    return result;
}

std::vector<std::size_t> Stage5Vocabulary::compact_encode(const std::string& text, const std::string& alphabet,
                                                          std::size_t unknown_token) {
    require(!alphabet.empty() && unknown_token >= alphabet.size(), "invalid compact vocabulary");
    std::vector<std::size_t> result;
    result.reserve(text.size());
    for (const auto character : text) {
        const auto position = alphabet.find(character);
        result.push_back(position == std::string::npos ? unknown_token : position);
    }
    return result;
}

std::string Stage5Vocabulary::compact_decode(const std::vector<std::size_t>& tokens, const std::string& alphabet,
                                              std::size_t unknown_token) {
    require(!alphabet.empty() && unknown_token >= alphabet.size(), "invalid compact vocabulary");
    std::string result;
    for (const auto token : tokens) result.push_back(token < alphabet.size() ? alphabet[token] : '?');
    return result;
}

Stage5LanguageModel::Stage5LanguageModel(Stage5ModelConfig config) : config_(std::move(config)) {
    require(config_.input_dim > 0 && config_.hidden_dim > 0 && config_.output_dim > 0,
            "Stage 5 model dimensions must be positive");
    initialize();
}

void Stage5LanguageModel::initialize() {
    if (config_.kind == Stage5ModelKind::CCTNoMemory || config_.kind == Stage5ModelKind::CCTFrozenMemory) {
        SequenceConfig sequence_config;
        sequence_config.input_dim = config_.input_dim;
        sequence_config.hidden_dim = config_.hidden_dim;
        sequence_config.output_dim = config_.output_dim;
        sequence_config.seed = config_.seed;
        cct_ = std::make_unique<SelectiveSequenceCore>(sequence_config);
        baseline_.reset();
    } else {
        baseline_ = std::make_unique<MatchedBaseline>(baseline_kind(config_.kind),
                                                       BaselineConfig{config_.input_dim, config_.hidden_dim,
                                                                      config_.output_dim, config_.seed});
        cct_.reset();
    }
}

std::string Stage5LanguageModel::name() const {
    switch (config_.kind) {
        case Stage5ModelKind::DenseCausalAttention: return "dense_causal_attention";
        case Stage5ModelKind::GRU: return "gru";
        case Stage5ModelKind::DiagonalSSM: return "diagonal_ssm";
        case Stage5ModelKind::CCTNoMemory: return "cct_no_memory";
        case Stage5ModelKind::CCTFrozenMemory: return "cct_frozen_memory";
    }
    return "unknown";
}

std::vector<std::vector<double>> Stage5LanguageModel::forward(const std::vector<std::vector<double>>& inputs) const {
    require(!inputs.empty(), "Stage 5 forward input is empty");
    if (baseline_) return baseline_->forward(inputs);
    return cct_->forward(inputs).outputs;
}

Stage5Evaluation Stage5LanguageModel::evaluate(
    const std::vector<std::vector<std::vector<double>>>& input_batch,
    const std::vector<std::vector<std::vector<double>>>& target_batch,
    const std::vector<std::vector<std::uint8_t>>& masks) const {
    require(input_batch.size() == target_batch.size(), "Stage 5 input/target batch size mismatch");
    require(masks.empty() || masks.size() == input_batch.size(), "Stage 5 mask batch size mismatch");
    Stage5Evaluation result;
    double squared_error = 0.0;
    double cross_entropy = 0.0;
    for (std::size_t batch = 0; batch < input_batch.size(); ++batch) {
        const auto outputs = forward(input_batch[batch]);
        require(outputs.size() == target_batch[batch].size(), "Stage 5 output/target sequence mismatch");
        for (std::size_t time = 0; time < outputs.size(); ++time) {
            const auto active = masks.empty() || masks[batch].empty() || masks[batch][time] != 0;
            if (!active) continue;
            require(outputs[time].size() == target_batch[batch][time].size(), "Stage 5 output/target width mismatch");
            const auto probabilities = softmax(outputs[time]);
            const auto target_class = argmax(target_batch[batch][time]);
            require(target_class < probabilities.size(), "Stage 5 target class is out of range");
            for (std::size_t feature = 0; feature < outputs[time].size(); ++feature) {
                const auto difference = outputs[time][feature] - target_batch[batch][time][feature];
                squared_error += difference * difference;
            }
            cross_entropy -= std::log(std::max(probabilities[target_class], 1e-12));
            if (argmax(outputs[time]) == target_class) ++result.token_accuracy;
            ++result.token_count;
        }
    }
    require(result.token_count > 0, "Stage 5 evaluation has no active tokens");
    result.mean_squared_loss = squared_error / static_cast<double>(result.token_count);
    result.cross_entropy = cross_entropy / static_cast<double>(result.token_count);
    result.token_accuracy /= static_cast<double>(result.token_count);
    return result;
}

void Stage5LanguageModel::train(
    const std::vector<std::vector<std::vector<double>>>& input_batch,
    const std::vector<std::vector<std::vector<double>>>& target_batch,
    const std::vector<std::vector<std::uint8_t>>& masks,
    const Stage5TrainConfig& config) {
    require(input_batch.size() == target_batch.size() && !input_batch.empty(), "invalid Stage 5 training batch");
    require(config.epochs > 0 && config.learning_rate > 0.0 && config.clip_norm > 0.0,
            "invalid Stage 5 training configuration");
    if (baseline_) {
        baseline_->train_finite_difference(input_batch, target_batch, masks, config.epochs, config.learning_rate,
                                           config.clip_norm);
        optimizer_step_ += static_cast<std::uint64_t>(config.epochs * input_batch.size());
    } else {
        for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
            for (std::size_t batch = 0; batch < input_batch.size(); ++batch) {
                const auto gradients = cct_->loss_and_gradients(input_batch[batch], target_batch[batch], masks[batch]);
                cct_->apply_sgd(gradients, config.learning_rate, config.clip_norm);
                ++optimizer_step_;
            }
        }
    }
    data_cursor_ = config.data_cursor + static_cast<std::uint64_t>(config.epochs * input_batch.size());
    manifest_fingerprint_ = config.manifest_fingerprint;
}

void Stage5LanguageModel::train_reference_finite_difference(
    const std::vector<std::vector<std::vector<double>>>& input_batch,
    const std::vector<std::vector<std::vector<double>>>& target_batch,
    const std::vector<std::vector<std::uint8_t>>& masks,
    const Stage5TrainConfig& config,
    const double finite_difference_epsilon) {
    require(input_batch.size() == target_batch.size() && !input_batch.empty(), "invalid Stage 5 reference training batch");
    require(config.epochs > 0U && config.learning_rate > 0.0 && config.clip_norm > 0.0 &&
                finite_difference_epsilon > 0.0 && std::isfinite(finite_difference_epsilon),
            "invalid Stage 5 reference training configuration");
    const auto batch_masks = masks.empty() ? std::vector<std::vector<std::uint8_t>>(input_batch.size()) : masks;
    require(batch_masks.size() == input_batch.size(), "Stage 5 reference mask batch size mismatch");
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        for (std::size_t batch = 0; batch < input_batch.size(); ++batch) {
            const auto original = parameter_vector();
            std::vector<double> gradients(original.size(), 0.0);
            for (std::size_t index = 0; index < original.size(); ++index) {
                auto plus = original;
                auto minus = original;
                plus[index] += finite_difference_epsilon;
                minus[index] -= finite_difference_epsilon;
                set_parameter_vector(plus);
                const auto plus_loss = evaluate({input_batch[batch]}, {target_batch[batch]}, {batch_masks[batch]}).mean_squared_loss;
                set_parameter_vector(minus);
                const auto minus_loss = evaluate({input_batch[batch]}, {target_batch[batch]}, {batch_masks[batch]}).mean_squared_loss;
                gradients[index] = (plus_loss - minus_loss) / (2.0 * finite_difference_epsilon);
            }
            set_parameter_vector(original);
            double norm_squared = 0.0;
            for (const auto gradient : gradients) norm_squared += gradient * gradient;
            const auto scale = std::min(1.0, config.clip_norm / std::max(std::sqrt(norm_squared), 1e-12));
            auto updated = original;
            for (std::size_t index = 0; index < updated.size(); ++index) updated[index] -= config.learning_rate * scale * gradients[index];
            set_parameter_vector(updated);
            ++optimizer_step_;
        }
    }
    data_cursor_ = config.data_cursor + static_cast<std::uint64_t>(config.epochs * input_batch.size());
    manifest_fingerprint_ = config.manifest_fingerprint;
}

std::size_t Stage5LanguageModel::parameter_count() const noexcept {
    return baseline_ ? baseline_->parameter_count() : cct_->parameter_count();
}

std::size_t Stage5LanguageModel::state_memory_bytes() const noexcept {
    return baseline_ ? baseline_->state_memory_bytes(1) : cct_->config().hidden_dim * sizeof(double);
}

std::vector<double> Stage5LanguageModel::parameter_vector() const {
    return baseline_ ? baseline_->parameter_vector() : cct_->parameter_vector();
}

void Stage5LanguageModel::set_parameter_vector(const std::vector<double>& values) {
    if (baseline_) baseline_->set_parameter_vector(values);
    else cct_->set_parameter_vector(values);
}

void Stage5LanguageModel::save_checkpoint(const std::string& path) const {
    std::ofstream stream(path);
    require(static_cast<bool>(stream), "could not write Stage 5 checkpoint");
    const auto parameters = parameter_vector();
    stream << "CCT_STAGE5_CHECKPOINT_V1\n" << static_cast<unsigned int>(config_.kind) << ' ' << config_.input_dim << ' '
           << config_.hidden_dim << ' ' << config_.output_dim << ' ' << config_.seed << ' ' << optimizer_step_ << ' '
           << data_cursor_ << ' ' << manifest_fingerprint_ << ' ' << parameters.size() << '\n' << std::setprecision(17);
    for (const auto value : parameters) stream << value << ' ';
    stream << '\n';
}

Stage5LanguageModel Stage5LanguageModel::load_checkpoint(const std::string& path) {
    static_cast<void>(checked_file_size(path));
    std::ifstream stream(path);
    require(static_cast<bool>(stream), "could not read Stage 5 checkpoint");
    std::string header;
    std::getline(stream, header);
    require(header == "CCT_STAGE5_CHECKPOINT_V1", "invalid Stage 5 checkpoint header");
    unsigned int kind = 0;
    Stage5ModelConfig config;
    std::size_t parameter_count = 0;
    Stage5LanguageModel* unused = nullptr;
    (void)unused;
    stream >> kind >> config.input_dim >> config.hidden_dim >> config.output_dim >> config.seed;
    require(config.input_dim > 0U && config.input_dim <= kMaximumModelDimension && config.hidden_dim > 0U &&
                config.hidden_dim <= kMaximumModelDimension && config.output_dim > 0U && config.output_dim <= kMaximumModelDimension,
            "Stage 5 checkpoint dimensions exceed budget");
    config.kind = static_cast<Stage5ModelKind>(kind);
    Stage5LanguageModel model(config);
    stream >> model.optimizer_step_ >> model.data_cursor_ >> model.manifest_fingerprint_ >> parameter_count;
    require(parameter_count > 0U && parameter_count <= kMaximumParameterCount, "Stage 5 checkpoint parameter count exceeds budget");
    std::vector<double> parameters(parameter_count, 0.0);
    for (auto& value : parameters) stream >> value;
    require(static_cast<bool>(stream), "truncated Stage 5 checkpoint");
    model.set_parameter_vector(parameters);
    return model;
}

Stage5MemoryEvaluation evaluate_stage5_memory_augmentation(PersistentMemory& memory) {
    MemoryRecord record;
    record.memory_id = 5001;
    record.content = "stage5 canary: language evidence anchor";
    record.embedding = {1.0, 0.0, 0.0, 0.0};
    record.created_at = 1;
    record.valid_from = 1;
    record.source = {"stage5_canary", 0, record.content.size()};
    record.confidence = 0.99;
    (void)memory.write(record, "stage5_frozen_memory_fixture");
    MemoryQuery query;
    query.embedding = {1.0, 0.0, 0.0, 0.0};
    query.source_id = "stage5_canary";
    query.budget = 1;
    const auto started = std::chrono::steady_clock::now();
    const auto hits = memory.retrieve(query);
    const auto finished = std::chrono::steady_clock::now();
    const auto latency = std::chrono::duration<double, std::milli>(finished - started).count();
    return {0, hits.size(), latency, !hits.empty() && hits.front().memory_id == 5001};
}

}  // namespace cct
