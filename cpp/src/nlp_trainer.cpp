#include "cct/nlp_trainer.hpp"

#include "cct/corpus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <unistd.h>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw NlpTrainingError(message);
}

std::string training_contract_digest(const NlpModelConfig& model, const NlpOptimizerConfig& optimizer,
                                    const std::string& tokenizer_hash, const std::string& dataset_hash) {
    std::ostringstream contract;
    contract << "cct-training-contract-v1|model_kind=" << static_cast<unsigned int>(model.kind) << "|vocabulary=" << model.vocabulary_size
             << "|embedding=" << model.embedding_dim << "|hidden=" << model.hidden_dim << "|context=" << model.context_length
             << "|seed=" << model.seed << "|objective=next-token-cross-entropy|mask=target-only-v1|optimizer=" << std::setprecision(17)
             << optimizer.learning_rate << ',' << optimizer.beta1 << ',' << optimizer.beta2 << ',' << optimizer.epsilon << ','
             << optimizer.weight_decay << ',' << optimizer.clip_norm << ',' << optimizer.warmup_steps << ',' << optimizer.batch_size << ',' << optimizer.total_steps << ','
             << optimizer.validation_interval_steps << "," << optimizer.worker_count << "|compact_vocabulary=" << (model.compact_vocabulary ? 1 : 0)
             << "|token_id_limit=" << model.token_id_limit << "|tokenizer=" << tokenizer_hash << "|dataset=" << dataset_hash << "|code=native-c++20-nlp-v4";
    return GovernedCorpus::content_sha256(contract.str());
}

void atomic_publish_checkpoint(const std::string& path, const std::string& content) {
    const std::filesystem::path target(path);
    const auto parent = target.parent_path().empty() ? std::filesystem::path(".") : target.parent_path();
    std::error_code directory_error;
    std::filesystem::create_directories(parent, directory_error);
    require(!directory_error, "could not create NLP checkpoint parent directory");
    const auto template_path = (parent / (target.filename().string() + ".tmp.XXXXXX")).string();
    std::vector<char> temporary_template(template_path.begin(), template_path.end());
    temporary_template.push_back('\0');
    const auto descriptor = ::mkstemp(temporary_template.data());
    require(descriptor >= 0, "could not create NLP checkpoint temporary file");
    const auto temporary_path = std::string(temporary_template.data());
    std::size_t written = 0U;
    while (written < content.size()) {
        const auto count = ::write(descriptor, content.data() + written, content.size() - written);
        if (count <= 0) {
            ::close(descriptor);
            static_cast<void>(::unlink(temporary_path.c_str()));
            throw NlpTrainingError("could not write NLP checkpoint temporary file");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
        static_cast<void>(::unlink(temporary_path.c_str()));
        throw NlpTrainingError("could not durably flush NLP checkpoint temporary file");
    }
    if (::rename(temporary_path.c_str(), target.c_str()) != 0) {
        static_cast<void>(::unlink(temporary_path.c_str()));
        throw NlpTrainingError("could not atomically publish NLP checkpoint");
    }
    const auto directory_descriptor = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(directory_descriptor >= 0, "could not open NLP checkpoint parent directory");
    const auto directory_sync = ::fsync(directory_descriptor);
    const auto directory_close = ::close(directory_descriptor);
    require(directory_sync == 0 && directory_close == 0, "could not durably publish NLP checkpoint directory entry");
}

double sigmoid(const double value) {
    if (value >= 0.0) {
        const auto z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const auto z = std::exp(value);
    return z / (1.0 + z);
}

double squared_norm(const std::vector<double>& values) {
    return std::inner_product(values.begin(), values.end(), values.begin(), 0.0,
                              std::plus<double>(), [](const double left, const double right) { return left * right; });
}

double vector_norm(const std::vector<double>& values) {
    return std::sqrt(std::max(0.0, squared_norm(values)));
}

std::string hex_encode(const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string hex_decode(const std::string& value) {
    if (value.size() % 2U != 0U) throw NlpTrainingError("hex field has odd length");
    const auto nibble = [](const char character) -> unsigned char {
        if (character >= '0' && character <= '9') return static_cast<unsigned char>(character - '0');
        if (character >= 'a' && character <= 'f') return static_cast<unsigned char>(character - 'a' + 10);
        if (character >= 'A' && character <= 'F') return static_cast<unsigned char>(character - 'A' + 10);
        throw NlpTrainingError("hex field contains invalid character");
    };
    std::string result;
    result.reserve(value.size() / 2U);
    for (std::size_t index = 0; index < value.size(); index += 2U) {
        result.push_back(static_cast<char>((nibble(value[index]) << 4U) | nibble(value[index + 1U])));
    }
    return result;
}

std::size_t parse_size(const std::string& value, const std::string& field) {
    if (value.empty()) throw NlpTrainingError("empty numeric field: " + field);
    std::size_t consumed = 0;
    std::size_t result = 0;
    try {
        result = std::stoull(value, &consumed, 10);
    } catch (const std::exception&) {
        throw NlpTrainingError("invalid numeric field: " + field);
    }
    if (consumed != value.size()) throw NlpTrainingError("invalid numeric suffix: " + field);
    return result;
}

std::vector<double> matvec(const std::vector<double>& parameters, const std::size_t offset,
                           const std::size_t rows, const std::size_t columns,
                           const std::vector<double>& input) {
    require(input.size() == columns, "NLP matrix input dimension mismatch");
    require(offset + rows * columns <= parameters.size(), "NLP matrix parameter range exceeds model");
    std::vector<double> result(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row] += parameters[offset + row * columns + column] * input[column];
        }
    }
    return result;
}

void matvec_into(const std::vector<double>& parameters, const std::size_t offset, const std::size_t rows, const std::size_t columns,
                 const std::vector<double>& input, std::vector<double>& output) {
    require(input.size() == columns, "NLP matrix input dimension mismatch");
    require(offset + rows * columns <= parameters.size(), "NLP matrix parameter range exceeds model");
    output.assign(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto row_offset = offset + row * columns;
        double total = 0.0;
        for (std::size_t column = 0; column < columns; ++column) total += parameters[row_offset + column] * input[column];
        output[row] = total;
    }
}

std::size_t token_slot(const NlpModelConfig& config, const TokenId token) {
    if (!config.compact_vocabulary) {
        require(static_cast<std::size_t>(token) < config.vocabulary_size, "NLP token ID exceeds model vocabulary");
        return static_cast<std::size_t>(token);
    }
    require(token == Tokenizer::kEosId || (token >= Tokenizer::kByteFirstId && token <= config.token_id_limit),
            "NLP compact vocabulary received an unavailable token ID");
    if (token == Tokenizer::kEosId) return 0U;
    return 1U + static_cast<std::size_t>(token - Tokenizer::kByteFirstId);
}

TokenId token_from_slot(const NlpModelConfig& config, const std::size_t slot) {
    require(slot < config.vocabulary_size, "NLP logit slot exceeds model vocabulary");
    if (!config.compact_vocabulary) return static_cast<TokenId>(slot);
    if (slot == 0U) return Tokenizer::kEosId;
    const auto token = static_cast<TokenId>(Tokenizer::kByteFirstId + static_cast<TokenId>(slot - 1U));
    require(token <= config.token_id_limit, "NLP compact vocabulary slot exceeds token ID limit");
    return token;
}

void softmax_into(const std::vector<double>& logits, std::vector<double>& probabilities) {
    require(!logits.empty(), "NLP softmax received empty logits");
    probabilities.resize(logits.size());
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    double denominator = 0.0;
    for (std::size_t index = 0U; index < logits.size(); ++index) {
        probabilities[index] = std::exp(std::clamp(logits[index] - maximum, -80.0, 80.0));
        denominator += probabilities[index];
    }
    require(std::isfinite(denominator) && denominator > 0.0, "NLP softmax denominator is non-finite");
    for (auto& probability : probabilities) probability /= denominator;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> probabilities;
    softmax_into(logits, probabilities);
    return probabilities;
}

std::uint64_t target_count(const NlpSequence& sequence) {
    return static_cast<std::uint64_t>(std::count(sequence.loss_mask.begin(), sequence.loss_mask.end(), static_cast<std::uint8_t>(1)));
}

void require_finite(const std::vector<double>& values, const std::string& message) {
    for (const auto value : values) require(std::isfinite(value), message);
}

std::string serialize_sequences(const std::vector<NlpSequence>& sequences) {
    std::ostringstream output;
    for (const auto& sequence : sequences) {
        output << sequence.sequence_id << '\t' << sequence.record_id << '\t' << sequence.input_ids.size() << '\n';
        for (const auto value : sequence.input_ids) output << value << ',';
        output << '\n';
        for (const auto value : sequence.target_ids) output << value << ',';
        output << '\n';
        for (const auto value : sequence.loss_mask) output << static_cast<unsigned int>(value) << ',';
        output << '\n';
    }
    return output.str();
}

std::size_t bounded_add(const std::size_t left, const std::size_t right) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return std::numeric_limits<std::size_t>::max();
    return left + right;
}

std::size_t bounded_mul(const std::size_t left, const std::size_t right) noexcept {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) return std::numeric_limits<std::size_t>::max();
    return left * right;
}

std::size_t model_kind_number(const NlpModelKind kind) {
    return static_cast<std::size_t>(kind);
}

NlpModelKind model_kind_from_number(const std::size_t value) {
    if (value > model_kind_number(NlpModelKind::DiagonalSSM)) throw NlpTrainingError("invalid NLP model kind");
    return static_cast<NlpModelKind>(value);
}

}  // namespace

std::string nlp_model_kind_name(const NlpModelKind kind) {
    switch (kind) {
        case NlpModelKind::Track1CctRecurrence: return "track1_cct_recurrence";
        case NlpModelKind::DenseCausalAttention: return "dense_causal_attention";
        case NlpModelKind::GRU: return "gru";
        case NlpModelKind::DiagonalSSM: return "diagonal_ssm";
    }
    throw NlpTrainingError("unknown NLP model kind");
}

NlpDataset NlpDataset::build(const std::vector<EncodedDocument>& train_documents,
                             const std::vector<EncodedDocument>& validation_documents,
                             const std::string& tokenizer_hash, const std::size_t context_length) {
    require(!train_documents.empty() && !validation_documents.empty(), "NLP dataset requires train and validation documents");
    require(!tokenizer_hash.empty(), "NLP dataset tokenizer hash is empty");
    require(context_length >= 2U && context_length <= 1'000'000U, "NLP context length is outside the supported range");
    std::unordered_set<std::string> record_ids;
    NlpDataset dataset;
    dataset.tokenizer_hash = tokenizer_hash;
    dataset.context_length = context_length;
    const auto append = [&](const std::vector<EncodedDocument>& documents, std::vector<NlpSequence>& destination,
                            std::size_t& token_count, const char* split) {
        const bool training_split = std::string(split) == "train";
        for (const auto& document : documents) {
            require(!document.record_id.empty() && document.tokens.size() >= 2U && record_ids.insert(document.record_id).second,
                    "NLP document is too short or record identity is duplicated");
            require(document.tokenizer_version == "cct-ase-tokenizer-v1", "NLP document tokenizer version is not Stage 10 V1");
            require(!document.evaluator_only && (training_split ? document.training_allowed : document.evaluation_allowed),
                    "NLP document eligibility does not match its requested split");
            std::size_t start = 0;
            std::size_t chunk = 0;
            while (start + 1U < document.tokens.size()) {
                const auto end = std::min(document.tokens.size(), start + context_length);
                if (end - start < 2U) break;
                NlpSequence sequence;
                sequence.sequence_id = std::string(split) + ":" + document.record_id + ":" + std::to_string(chunk);
                sequence.record_id = document.record_id;
                sequence.input_ids.reserve(end - start);
                sequence.target_ids.reserve(end - start);
                sequence.loss_mask.assign(end - start, 0U);
                for (std::size_t index = start; index < end; ++index) {
                    sequence.input_ids.push_back(document.tokens[index].id);
                    const bool target_available = index + 1U < end;
                    sequence.target_ids.push_back(target_available ? document.tokens[index + 1U].id : Tokenizer::kPadId);
                    sequence.loss_mask[index - start] = target_available ? 1U : 0U;
                }
                token_count += static_cast<std::size_t>(target_count(sequence));
                destination.push_back(std::move(sequence));
                start = end;
                ++chunk;
            }
        }
    };
    append(train_documents, dataset.train, dataset.train_tokens, "train");
    append(validation_documents, dataset.validation, dataset.validation_tokens, "validation");
    require(!dataset.train.empty() && !dataset.validation.empty() && dataset.train_tokens > 0U && dataset.validation_tokens > 0U,
            "NLP dataset produced no trainable tokens");
    dataset.dataset_hash = GovernedCorpus::content_sha256(
        tokenizer_hash + "\ncontext=" + std::to_string(context_length) + "\ntrain\n" + serialize_sequences(dataset.train) +
        "validation\n" + serialize_sequences(dataset.validation));
    return dataset;
}

NextTokenModel::NextTokenModel(NlpModelConfig config) : config_(std::move(config)) {
    validate_config();
    initialize();
}

void NextTokenModel::validate_config() const {
    require(model_kind_number(config_.kind) <= model_kind_number(NlpModelKind::DiagonalSSM), "NLP model kind is unsupported");
    if (config_.compact_vocabulary) {
        require(config_.token_id_limit >= Tokenizer::kByteFirstId && config_.token_id_limit <= std::numeric_limits<TokenId>::max(),
                "NLP compact vocabulary token ID limit is invalid");
        const auto expected_slots = 2U + static_cast<std::size_t>(config_.token_id_limit - Tokenizer::kByteFirstId);
        require(config_.vocabulary_size == expected_slots, "NLP compact vocabulary slot count is inconsistent with token ID limit");
    } else {
        require(config_.vocabulary_size >= Tokenizer::kByteFirstId + 256U && config_.vocabulary_size <= 1'000'000U,
                "NLP vocabulary is outside the supported range");
    }
    require(config_.embedding_dim > 0U && config_.embedding_dim <= 4096U && config_.hidden_dim > 0U &&
                config_.hidden_dim <= 4096U && config_.context_length >= 2U && config_.context_length <= 1'000'000U,
            "NLP model dimensions are outside the supported range");
    require(config_.vocabulary_size <= std::numeric_limits<TokenId>::max() && expected_parameter_count() <= 16'000'000U,
            "NLP model parameter budget or token ID range is invalid");
}

std::size_t NextTokenModel::embedding_offset() const noexcept { return 0U; }
std::size_t NextTokenModel::cct_offset() const noexcept { return config_.vocabulary_size * config_.embedding_dim; }
std::size_t NextTokenModel::gru_offset() const noexcept { return cct_offset(); }
std::size_t NextTokenModel::ssm_offset() const noexcept { return cct_offset(); }
std::size_t NextTokenModel::dense_offset() const noexcept { return cct_offset(); }

std::size_t NextTokenModel::head_offset() const noexcept {
    const auto hidden = config_.hidden_dim;
    const auto input = config_.embedding_dim;
    if (config_.kind == NlpModelKind::Track1CctRecurrence) return cct_offset() + 4U * hidden * input + 3U * hidden;
    if (config_.kind == NlpModelKind::GRU) return gru_offset() + 3U * (hidden * input + hidden * hidden + hidden);
    if (config_.kind == NlpModelKind::DiagonalSSM) return ssm_offset() + hidden + hidden * input;
    return dense_offset() + 3U * hidden * input;
}

std::size_t NextTokenModel::skip_offset() const noexcept { return head_offset() + config_.vocabulary_size * config_.hidden_dim; }
std::size_t NextTokenModel::bias_offset() const noexcept { return skip_offset(); }

std::size_t NextTokenModel::expected_parameter_count() const noexcept {
    const auto input = config_.embedding_dim;
    const auto hidden = config_.hidden_dim;
    const auto vocabulary = config_.vocabulary_size;
    const auto embedding = bounded_mul(vocabulary, input);
    const auto head = bounded_add(bounded_mul(vocabulary, hidden), vocabulary);
    if (config_.kind == NlpModelKind::Track1CctRecurrence) {
        return bounded_add(bounded_add(embedding, bounded_mul(4U, bounded_mul(hidden, input))), bounded_add(bounded_mul(3U, hidden), head));
    }
    if (config_.kind == NlpModelKind::GRU) {
        const auto gate = bounded_add(bounded_add(bounded_mul(hidden, input), bounded_mul(hidden, hidden)), hidden);
        return bounded_add(embedding, bounded_add(bounded_mul(3U, gate), head));
    }
    if (config_.kind == NlpModelKind::DiagonalSSM) return bounded_add(embedding, bounded_add(hidden, bounded_add(bounded_mul(hidden, input), head)));
    return bounded_add(embedding, bounded_add(bounded_mul(3U, bounded_mul(hidden, input)), head));
}

void NextTokenModel::initialize() {
    parameters_.assign(expected_parameter_count(), 0.0);
    std::mt19937_64 generator(config_.seed);
    const auto scale = 1.0 / std::sqrt(static_cast<double>(config_.embedding_dim + config_.hidden_dim));
    std::normal_distribution<double> distribution(0.0, scale);
    for (auto& parameter : parameters_) parameter = distribution(generator);
    if (config_.kind == NlpModelKind::Track1CctRecurrence) {
        const auto retain_bias_offset = cct_offset() + 4U * config_.hidden_dim * config_.embedding_dim + config_.hidden_dim;
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            parameters_[retain_bias_offset + index] = 2.0;
        }
    }
}

std::string NextTokenModel::name() const { return nlp_model_kind_name(config_.kind); }

std::vector<double> NextTokenModel::embedding(const TokenId id) const {
    const auto slot = token_slot(config_, id);
    return std::vector<double>(parameters_.begin() + static_cast<std::ptrdiff_t>(embedding_offset() + slot * config_.embedding_dim),
                               parameters_.begin() + static_cast<std::ptrdiff_t>(embedding_offset() + (slot + 1U) * config_.embedding_dim));
}

void NextTokenModel::validate_sequence(const NlpSequence& sequence) const {
    require(!sequence.sequence_id.empty() && !sequence.record_id.empty() && !sequence.input_ids.empty() &&
                sequence.input_ids.size() == sequence.target_ids.size() && sequence.input_ids.size() == sequence.loss_mask.size() &&
                sequence.input_ids.size() <= config_.context_length,
            "NLP sequence identity, shape, or context length is invalid");
    require(target_count(sequence) > 0U, "NLP sequence has no active loss positions");
    for (const auto id : sequence.input_ids) static_cast<void>(token_slot(config_, id));
    for (std::size_t index = 0; index < sequence.target_ids.size(); ++index) {
        require(sequence.loss_mask[index] == 0U || sequence.loss_mask[index] == 1U, "NLP loss mask is not binary");
        if (sequence.target_ids[index] != Tokenizer::kPadId) static_cast<void>(token_slot(config_, sequence.target_ids[index]));
    }
}

void project_logits_into(const std::vector<double>& parameters, std::size_t head_offset, std::size_t bias_offset,
                         std::size_t vocabulary, std::size_t hidden, const std::vector<double>& hidden_state,
                         std::vector<double>& logits);

std::vector<std::vector<double>> forward_track1_cct_recurrence(const std::vector<double>& parameters, const NlpModelConfig& config,
                                             const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto recurrent_offset = vocabulary * input;
    const auto head_offset = recurrent_offset + 4U * hidden * input + 3U * hidden;
    const auto bias_offset = head_offset + vocabulary * hidden;
    std::vector<double> hidden_state(hidden, 0.0);
    std::vector<double> previous_input(input, 0.0);
    std::vector<double> x(input, 0.0);
    std::vector<double> retain_raw(hidden, 0.0);
    std::vector<double> write_raw(hidden, 0.0);
    std::vector<double> candidate_raw(hidden, 0.0);
    std::vector<double> previous_effect(hidden, 0.0);
    std::vector<std::vector<double>> logits;
    logits.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        const auto offset = token_slot(config, id) * input;
        std::copy(parameters.begin() + static_cast<std::ptrdiff_t>(offset), parameters.begin() + static_cast<std::ptrdiff_t>(offset + input), x.begin());
        matvec_into(parameters, recurrent_offset + 2U * hidden * input, hidden, input, x, retain_raw);
        matvec_into(parameters, recurrent_offset + 3U * hidden * input, hidden, input, x, write_raw);
        matvec_into(parameters, recurrent_offset, hidden, input, x, candidate_raw);
        matvec_into(parameters, recurrent_offset + hidden * input, hidden, input, previous_input, previous_effect);
        for (std::size_t index = 0U; index < hidden; ++index) {
            const auto retain = sigmoid(retain_raw[index] + parameters[recurrent_offset + 4U * hidden * input + hidden + index]);
            const auto write = sigmoid(write_raw[index] + parameters[recurrent_offset + 4U * hidden * input + 2U * hidden + index]);
            const auto candidate = std::tanh(candidate_raw[index] + previous_effect[index] + parameters[recurrent_offset + 4U * hidden * input + index]);
            hidden_state[index] = retain * hidden_state[index] + write * candidate;
        }
        logits.emplace_back(vocabulary, 0.0);
        project_logits_into(parameters, head_offset, bias_offset, vocabulary, hidden, hidden_state, logits.back());
        previous_input = x;
    }
    return logits;
}

std::vector<std::vector<double>> forward_gru(const std::vector<double>& parameters, const NlpModelConfig& config,
                                             const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto recurrent_offset = vocabulary * input;
    const auto gate_size = hidden * input + hidden * hidden + hidden;
    const auto z_offset = recurrent_offset;
    const auto r_offset = z_offset + gate_size;
    const auto n_offset = r_offset + gate_size;
    const auto head_offset = n_offset + gate_size;
    const auto bias_offset = head_offset + vocabulary * hidden;
    std::vector<double> hidden_state(hidden, 0.0);
    std::vector<double> x(input, 0.0);
    std::vector<double> z(hidden, 0.0);
    std::vector<double> r(hidden, 0.0);
    std::vector<double> candidate(hidden, 0.0);
    std::vector<double> recurrent(hidden, 0.0);
    std::vector<std::vector<double>> logits;
    logits.reserve(sequence.input_ids.size());
    const auto gate = [&](const std::size_t offset, std::vector<double>& result) {
        matvec_into(parameters, offset, hidden, input, x, result);
        matvec_into(parameters, offset + hidden * input, hidden, hidden, hidden_state, recurrent);
        for (std::size_t index = 0U; index < hidden; ++index) result[index] = sigmoid(result[index] + recurrent[index] + parameters[offset + hidden * input + hidden * hidden + index]);
    };
    for (const auto id : sequence.input_ids) {
        const auto offset = token_slot(config, id) * input;
        std::copy(parameters.begin() + static_cast<std::ptrdiff_t>(offset), parameters.begin() + static_cast<std::ptrdiff_t>(offset + input), x.begin());
        gate(z_offset, z);
        gate(r_offset, r);
        matvec_into(parameters, n_offset, hidden, input, x, candidate);
        matvec_into(parameters, n_offset + hidden * input, hidden, hidden, hidden_state, recurrent);
        for (std::size_t index = 0U; index < hidden; ++index) candidate[index] = std::tanh(candidate[index] + r[index] * recurrent[index] + parameters[n_offset + hidden * input + hidden * hidden + index]);
        for (std::size_t index = 0U; index < hidden; ++index) hidden_state[index] = z[index] * hidden_state[index] + (1.0 - z[index]) * candidate[index];
        logits.emplace_back(vocabulary, 0.0);
        project_logits_into(parameters, head_offset, bias_offset, vocabulary, hidden, hidden_state, logits.back());
    }
    return logits;
}

std::vector<std::vector<double>> forward_ssm(const std::vector<double>& parameters, const NlpModelConfig& config,
                                             const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto recurrent_offset = vocabulary * input;
    const auto input_offset = recurrent_offset + hidden;
    const auto head_offset = input_offset + hidden * input;
    const auto bias_offset = head_offset + vocabulary * hidden;
    std::vector<double> hidden_state(hidden, 0.0);
    std::vector<double> x(input, 0.0);
    std::vector<double> effect(hidden, 0.0);
    std::vector<std::vector<double>> logits;
    logits.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        const auto offset = token_slot(config, id) * input;
        std::copy(parameters.begin() + static_cast<std::ptrdiff_t>(offset), parameters.begin() + static_cast<std::ptrdiff_t>(offset + input), x.begin());
        matvec_into(parameters, input_offset, hidden, input, x, effect);
        for (std::size_t index = 0U; index < hidden; ++index) hidden_state[index] = 0.999 * sigmoid(parameters[recurrent_offset + index]) * hidden_state[index] + effect[index];
        logits.emplace_back(vocabulary, 0.0);
        project_logits_into(parameters, head_offset, bias_offset, vocabulary, hidden, hidden_state, logits.back());
    }
    return logits;
}

std::vector<std::vector<double>> forward_dense(const std::vector<double>& parameters, const NlpModelConfig& config,
                                               const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto attention_offset = vocabulary * input;
    const auto q_offset = attention_offset;
    const auto k_offset = q_offset + hidden * input;
    const auto v_offset = k_offset + hidden * input;
    const auto head_offset = v_offset + hidden * input;
    const auto bias_offset = head_offset + vocabulary * hidden;
    const auto length = sequence.input_ids.size();
    std::vector<double> keys(length * hidden, 0.0);
    std::vector<double> values(length * hidden, 0.0);
    std::vector<double> x(input, 0.0);
    std::vector<double> query(hidden, 0.0);
    std::vector<double> key(hidden, 0.0);
    std::vector<double> value(hidden, 0.0);
    std::vector<double> scores(length, 0.0);
    std::vector<double> context(hidden, 0.0);
    std::vector<std::vector<double>> logits;
    logits.reserve(length);
    for (std::size_t time = 0U; time < length; ++time) {
        const auto offset = token_slot(config, sequence.input_ids[time]) * input;
        std::copy(parameters.begin() + static_cast<std::ptrdiff_t>(offset), parameters.begin() + static_cast<std::ptrdiff_t>(offset + input), x.begin());
        matvec_into(parameters, q_offset, hidden, input, x, query);
        matvec_into(parameters, k_offset, hidden, input, x, key);
        matvec_into(parameters, v_offset, hidden, input, x, value);
        std::copy(key.begin(), key.end(), keys.begin() + static_cast<std::ptrdiff_t>(time * hidden));
        std::copy(value.begin(), value.end(), values.begin() + static_cast<std::ptrdiff_t>(time * hidden));
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t position = 0U; position <= time; ++position) {
            const auto key_offset = position * hidden;
            double score = 0.0;
            for (std::size_t index = 0U; index < hidden; ++index) score += query[index] * keys[key_offset + index];
            scores[position] = score / std::sqrt(static_cast<double>(hidden));
            maximum = std::max(maximum, scores[position]);
        }
        double denominator = 0.0;
        for (std::size_t position = 0U; position <= time; ++position) {
            scores[position] = std::exp(std::clamp(scores[position] - maximum, -80.0, 80.0));
            denominator += scores[position];
        }
        require(std::isfinite(denominator) && denominator > 0.0, "dense attention denominator is non-finite");
        std::fill(context.begin(), context.end(), 0.0);
        for (std::size_t position = 0U; position <= time; ++position) {
            const auto value_offset = position * hidden;
            const auto weight = scores[position] / denominator;
            for (std::size_t index = 0U; index < hidden; ++index) context[index] += weight * values[value_offset + index];
        }
        logits.emplace_back(vocabulary, 0.0);
        project_logits_into(parameters, head_offset, bias_offset, vocabulary, hidden, context, logits.back());
    }
    return logits;
}

std::vector<std::vector<double>> model_forward(const std::vector<double>& parameters, const NlpModelConfig& config,
                                               const NlpSequence& sequence) {
    if (config.kind == NlpModelKind::Track1CctRecurrence) return forward_track1_cct_recurrence(parameters, config, sequence);
    if (config.kind == NlpModelKind::GRU) return forward_gru(parameters, config, sequence);
    if (config.kind == NlpModelKind::DiagonalSSM) return forward_ssm(parameters, config, sequence);
    return forward_dense(parameters, config, sequence);
}

void project_logits_into(const std::vector<double>& parameters, const std::size_t head_offset, const std::size_t bias_offset,
                         const std::size_t vocabulary, const std::size_t hidden, const std::vector<double>& hidden_state,
                         std::vector<double>& logits) {
    require(hidden_state.size() == hidden, "NLP projection hidden dimension mismatch");
    logits.assign(vocabulary, 0.0);
    for (std::size_t token = 0U; token < vocabulary; ++token) {
        const auto row_offset = head_offset + token * hidden;
        double value = parameters[bias_offset + token];
        for (std::size_t index = 0U; index < hidden; ++index) value += parameters[row_offset + index] * hidden_state[index];
        logits[token] = value;
    }
}

double cross_entropy_from_logits(const std::vector<std::vector<double>>& logits, const NlpSequence& sequence,
                                 const NlpModelConfig& config, std::size_t* token_count, double* accuracy) {
    require(logits.size() == sequence.target_ids.size(), "NLP logits/targets length mismatch");
    double loss = 0.0;
    std::size_t count = 0;
    std::size_t correct = 0;
    std::vector<double> probabilities;
    for (std::size_t time = 0; time < logits.size(); ++time) {
        if (sequence.loss_mask[time] == 0U) continue;
        softmax_into(logits[time], probabilities);
        const auto target = token_slot(config, sequence.target_ids[time]);
        require(target < probabilities.size(), "NLP target is outside logits");
        loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
        const auto prediction_slot = static_cast<std::size_t>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
        if (prediction_slot == target) ++correct;
        ++count;
    }
    require(count > 0U, "NLP loss has no active tokens");
    if (token_count != nullptr) *token_count = count;
    if (accuracy != nullptr) *accuracy = static_cast<double>(correct) / static_cast<double>(count);
    return loss / static_cast<double>(count);
}

NlpGradientResult track1_cct_gradients(const std::vector<double>& parameters, const NlpModelConfig& config,
                                const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto recurrent_offset = vocabulary * input;
    const auto head_offset = recurrent_offset + 4U * hidden * input + 3U * hidden;
    const auto bias_offset = head_offset + vocabulary * hidden;
    struct Cache {
        std::vector<double> x;
        std::vector<double> previous_x;
        std::vector<double> hidden_before;
        std::vector<double> hidden_after;
        std::vector<double> retain;
        std::vector<double> write;
        std::vector<double> candidate;
        std::vector<double> logits;
    };
    std::vector<Cache> cache;
    cache.reserve(sequence.input_ids.size());
    std::vector<double> hidden_state(hidden, 0.0);
    std::vector<double> previous_x(input, 0.0);
    for (const auto id : sequence.input_ids) {
        Cache item;
        item.x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(token_slot(config, id) * input),
                                     parameters.begin() + static_cast<std::ptrdiff_t>((token_slot(config, id) + 1U) * input));
        item.previous_x = previous_x;
        item.hidden_before = hidden_state;
        const auto retain_raw = matvec(parameters, recurrent_offset + 2U * hidden * input, hidden, input, item.x);
        const auto write_raw = matvec(parameters, recurrent_offset + 3U * hidden * input, hidden, input, item.x);
        const auto candidate_raw = matvec(parameters, recurrent_offset, hidden, input, item.x);
        const auto previous_effect = matvec(parameters, recurrent_offset + hidden * input, hidden, input, previous_x);
        item.retain.resize(hidden);
        item.write.resize(hidden);
        item.candidate.resize(hidden);
        item.hidden_after.resize(hidden);
        for (std::size_t index = 0; index < hidden; ++index) {
            item.retain[index] = sigmoid(retain_raw[index] + parameters[recurrent_offset + 4U * hidden * input + hidden + index]);
            item.write[index] = sigmoid(write_raw[index] + parameters[recurrent_offset + 4U * hidden * input + 2U * hidden + index]);
            item.candidate[index] = std::tanh(candidate_raw[index] + previous_effect[index] + parameters[recurrent_offset + 4U * hidden * input + index]);
            item.hidden_after[index] = item.retain[index] * hidden_state[index] + item.write[index] * item.candidate[index];
        }
        item.logits.assign(vocabulary, 0.0);
        for (std::size_t token = 0; token < vocabulary; ++token) {
            item.logits[token] = parameters[bias_offset + token];
            for (std::size_t index = 0; index < hidden; ++index) item.logits[token] += parameters[head_offset + token * hidden + index] * item.hidden_after[index];
        }
        cache.push_back(std::move(item));
        hidden_state = cache.back().hidden_after;
        previous_x = cache.back().x;
    }

    const auto active_tokens = static_cast<double>(target_count(sequence));
    std::vector<double> gradients(parameters.size(), 0.0);
    std::vector<double> d_hidden_next(hidden, 0.0);
    std::vector<double> d_input_carry(input, 0.0);
    std::vector<std::vector<double>> d_embedding(sequence.input_ids.size(), std::vector<double>(input, 0.0));
    double loss = 0.0;
    std::size_t correct = 0;
    for (std::size_t reverse = cache.size(); reverse-- > 0;) {
        auto& item = cache[reverse];
        std::vector<double> d_output(vocabulary, 0.0);
        if (sequence.loss_mask[reverse] != 0U) {
            const auto probabilities = softmax(item.logits);
            const auto target = token_slot(config, sequence.target_ids[reverse]);
            loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
            if (static_cast<std::size_t>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))) == target) ++correct;
            for (std::size_t token = 0; token < vocabulary; ++token) d_output[token] = probabilities[token] / active_tokens;
            d_output[target] -= 1.0 / active_tokens;
        }
        for (std::size_t token = 0; token < vocabulary; ++token) {
            gradients[bias_offset + token] += d_output[token];
            for (std::size_t index = 0; index < hidden; ++index) {
                gradients[head_offset + token * hidden + index] += d_output[token] * item.hidden_after[index];
                d_hidden_next[index] += d_output[token] * parameters[head_offset + token * hidden + index];
            }
        }
        std::vector<double> d_hidden_before(hidden, 0.0);
        std::vector<double> d_x(input, 0.0);
        std::vector<double> d_previous_x(input, 0.0);
        for (std::size_t index = 0; index < hidden; ++index) {
            const auto d_retain = d_hidden_next[index] * item.hidden_before[index];
            const auto d_write = d_hidden_next[index] * item.candidate[index];
            const auto d_candidate = d_hidden_next[index] * item.write[index];
            const auto d_retain_raw = d_retain * item.retain[index] * (1.0 - item.retain[index]);
            const auto d_write_raw = d_write * item.write[index] * (1.0 - item.write[index]);
            const auto d_candidate_raw = d_candidate * (1.0 - item.candidate[index] * item.candidate[index]);
            gradients[recurrent_offset + 4U * hidden * input + index] += d_candidate_raw;
            gradients[recurrent_offset + 4U * hidden * input + hidden + index] += d_retain_raw;
            gradients[recurrent_offset + 4U * hidden * input + 2U * hidden + index] += d_write_raw;
            for (std::size_t column = 0; column < input; ++column) {
                gradients[recurrent_offset + index * input + column] += d_candidate_raw * item.x[column];
                gradients[recurrent_offset + hidden * input + index * input + column] += d_candidate_raw * item.previous_x[column];
                gradients[recurrent_offset + 2U * hidden * input + index * input + column] += d_retain_raw * item.x[column];
                gradients[recurrent_offset + 3U * hidden * input + index * input + column] += d_write_raw * item.x[column];
                d_x[column] += parameters[recurrent_offset + index * input + column] * d_candidate_raw;
                d_previous_x[column] += parameters[recurrent_offset + hidden * input + index * input + column] * d_candidate_raw;
                d_x[column] += parameters[recurrent_offset + 2U * hidden * input + index * input + column] * d_retain_raw;
                d_x[column] += parameters[recurrent_offset + 3U * hidden * input + index * input + column] * d_write_raw;
            }
            d_hidden_before[index] += d_hidden_next[index] * item.retain[index];
        }
        for (std::size_t index = 0; index < input; ++index) {
            d_x[index] += d_input_carry[index];
            d_input_carry[index] = d_previous_x[index];
        }
        d_embedding[reverse] = std::move(d_x);
        d_hidden_next = std::move(d_hidden_before);
    }
    for (std::size_t time = 0; time < sequence.input_ids.size(); ++time) {
        const auto offset = token_slot(config, sequence.input_ids[time]) * input;
        for (std::size_t index = 0; index < input; ++index) gradients[offset + index] += d_embedding[time][index];
    }
    require_finite(gradients, "CCT analytic gradient became non-finite");
    NlpGradientResult result;
    result.cross_entropy = loss / active_tokens;
    result.token_count = static_cast<std::size_t>(active_tokens);
    result.gradient_norm = vector_norm(gradients);
    result.gradients = std::move(gradients);
    (void)correct;
    return result;
}

NlpGradientResult gru_gradients(const std::vector<double>& parameters, const NlpModelConfig& config, const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto recurrent_offset = vocabulary * input;
    const auto gate_size = hidden * input + hidden * hidden + hidden;
    const auto z_offset = recurrent_offset;
    const auto r_offset = z_offset + gate_size;
    const auto n_offset = r_offset + gate_size;
    const auto head_offset = n_offset + gate_size;
    const auto bias_offset = head_offset + vocabulary * hidden;
    struct Cache {
        std::vector<double> x;
        std::vector<double> hidden_before;
        std::vector<double> z;
        std::vector<double> r;
        std::vector<double> candidate;
        std::vector<double> candidate_recurrent;
        std::vector<double> hidden_after;
        std::vector<double> logits;
    };
    std::vector<Cache> cache;
    cache.reserve(sequence.input_ids.size());
    std::vector<double> hidden_state(hidden, 0.0);
    for (const auto id : sequence.input_ids) {
        Cache item;
        item.x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(token_slot(config, id) * input),
                                     parameters.begin() + static_cast<std::ptrdiff_t>((token_slot(config, id) + 1U) * input));
        item.hidden_before = hidden_state;
        const auto z_raw = matvec(parameters, z_offset, hidden, input, item.x);
        const auto r_raw = matvec(parameters, r_offset, hidden, input, item.x);
        const auto n_raw = matvec(parameters, n_offset, hidden, input, item.x);
        const auto z_recurrent = matvec(parameters, z_offset + hidden * input, hidden, hidden, hidden_state);
        const auto r_recurrent = matvec(parameters, r_offset + hidden * input, hidden, hidden, hidden_state);
        item.candidate_recurrent = matvec(parameters, n_offset + hidden * input, hidden, hidden, hidden_state);
        item.z.resize(hidden);
        item.r.resize(hidden);
        item.candidate.resize(hidden);
        item.hidden_after.resize(hidden);
        for (std::size_t index = 0U; index < hidden; ++index) {
            item.z[index] = sigmoid(z_raw[index] + z_recurrent[index] + parameters[z_offset + hidden * input + hidden * hidden + index]);
            item.r[index] = sigmoid(r_raw[index] + r_recurrent[index] + parameters[r_offset + hidden * input + hidden * hidden + index]);
            item.candidate[index] = std::tanh(n_raw[index] + item.r[index] * item.candidate_recurrent[index] +
                                               parameters[n_offset + hidden * input + hidden * hidden + index]);
            item.hidden_after[index] = item.z[index] * hidden_state[index] + (1.0 - item.z[index]) * item.candidate[index];
        }
        item.logits.assign(vocabulary, 0.0);
        for (std::size_t token = 0U; token < vocabulary; ++token) {
            item.logits[token] = parameters[bias_offset + token];
            for (std::size_t index = 0U; index < hidden; ++index) item.logits[token] += parameters[head_offset + token * hidden + index] * item.hidden_after[index];
        }
        cache.push_back(std::move(item));
        hidden_state = cache.back().hidden_after;
    }
    const auto active_tokens = static_cast<double>(target_count(sequence));
    std::vector<double> gradients(parameters.size(), 0.0);
    std::vector<double> d_hidden_next(hidden, 0.0);
    double loss = 0.0;
    std::size_t correct = 0U;
    for (std::size_t reverse = cache.size(); reverse-- > 0U;) {
        auto& item = cache[reverse];
        std::vector<double> d_output(vocabulary, 0.0);
        if (sequence.loss_mask[reverse] != 0U) {
            const auto probabilities = softmax(item.logits);
            const auto target = token_slot(config, sequence.target_ids[reverse]);
            loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
            if (static_cast<std::size_t>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))) == target) ++correct;
            for (std::size_t token = 0U; token < vocabulary; ++token) d_output[token] = probabilities[token] / active_tokens;
            d_output[target] -= 1.0 / active_tokens;
        }
        for (std::size_t token = 0U; token < vocabulary; ++token) {
            gradients[bias_offset + token] += d_output[token];
            for (std::size_t index = 0U; index < hidden; ++index) {
                gradients[head_offset + token * hidden + index] += d_output[token] * item.hidden_after[index];
                d_hidden_next[index] += d_output[token] * parameters[head_offset + token * hidden + index];
            }
        }
        std::vector<double> d_hidden_before(hidden, 0.0);
        std::vector<double> d_x(input, 0.0);
        std::vector<double> d_z(hidden, 0.0);
        std::vector<double> d_r(hidden, 0.0);
        std::vector<double> d_candidate(hidden, 0.0);
        for (std::size_t index = 0U; index < hidden; ++index) {
            d_z[index] = d_hidden_next[index] * (item.hidden_before[index] - item.candidate[index]);
            d_candidate[index] = d_hidden_next[index] * (1.0 - item.z[index]);
            d_hidden_before[index] += d_hidden_next[index] * item.z[index];
        }
        for (std::size_t index = 0U; index < hidden; ++index) {
            const auto d_candidate_raw = d_candidate[index] * (1.0 - item.candidate[index] * item.candidate[index]);
            const auto d_candidate_recurrent = d_candidate_raw * item.r[index];
            d_r[index] += d_candidate_raw * item.candidate_recurrent[index];
            gradients[n_offset + hidden * input + hidden * hidden + index] += d_candidate_raw;
            for (std::size_t column = 0U; column < input; ++column) {
                gradients[n_offset + index * input + column] += d_candidate_raw * item.x[column];
                d_x[column] += parameters[n_offset + index * input + column] * d_candidate_raw;
            }
            for (std::size_t column = 0U; column < hidden; ++column) {
                gradients[n_offset + hidden * input + index * hidden + column] += d_candidate_recurrent * item.hidden_before[column];
                d_hidden_before[column] += parameters[n_offset + hidden * input + index * hidden + column] * d_candidate_recurrent;
            }
        }
        for (std::size_t index = 0U; index < hidden; ++index) {
            const auto d_r_raw = d_r[index] * item.r[index] * (1.0 - item.r[index]);
            const auto d_z_raw = d_z[index] * item.z[index] * (1.0 - item.z[index]);
            gradients[r_offset + hidden * input + hidden * hidden + index] += d_r_raw;
            gradients[z_offset + hidden * input + hidden * hidden + index] += d_z_raw;
            for (std::size_t column = 0U; column < input; ++column) {
                gradients[r_offset + index * input + column] += d_r_raw * item.x[column];
                gradients[z_offset + index * input + column] += d_z_raw * item.x[column];
                d_x[column] += parameters[r_offset + index * input + column] * d_r_raw;
                d_x[column] += parameters[z_offset + index * input + column] * d_z_raw;
            }
            for (std::size_t column = 0U; column < hidden; ++column) {
                gradients[r_offset + hidden * input + index * hidden + column] += d_r_raw * item.hidden_before[column];
                gradients[z_offset + hidden * input + index * hidden + column] += d_z_raw * item.hidden_before[column];
                d_hidden_before[column] += parameters[r_offset + hidden * input + index * hidden + column] * d_r_raw;
                d_hidden_before[column] += parameters[z_offset + hidden * input + index * hidden + column] * d_z_raw;
            }
        }
        const auto embedding_offset = token_slot(config, sequence.input_ids[reverse]) * input;
        for (std::size_t column = 0U; column < input; ++column) gradients[embedding_offset + column] += d_x[column];
        d_hidden_next = std::move(d_hidden_before);
    }
    require_finite(gradients, "GRU analytic gradient became non-finite");
    return {loss / active_tokens, static_cast<std::size_t>(active_tokens), vector_norm(gradients), std::move(gradients)};
}

NlpGradientResult ssm_gradients(const std::vector<double>& parameters, const NlpModelConfig& config, const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto recurrent_offset = vocabulary * input;
    const auto input_offset = recurrent_offset + hidden;
    const auto head_offset = input_offset + hidden * input;
    const auto bias_offset = head_offset + vocabulary * hidden;
    struct Cache { std::vector<double> x; std::vector<double> hidden_before; std::vector<double> hidden_after; std::vector<double> retain; std::vector<double> logits; };
    std::vector<Cache> cache;
    cache.reserve(sequence.input_ids.size());
    std::vector<double> hidden_state(hidden, 0.0);
    for (const auto id : sequence.input_ids) {
        Cache item;
        item.x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(token_slot(config, id) * input),
                                     parameters.begin() + static_cast<std::ptrdiff_t>((token_slot(config, id) + 1U) * input));
        item.hidden_before = hidden_state;
        item.retain.resize(hidden);
        const auto effect = matvec(parameters, input_offset, hidden, input, item.x);
        item.hidden_after.resize(hidden);
        for (std::size_t index = 0U; index < hidden; ++index) {
            item.retain[index] = sigmoid(parameters[recurrent_offset + index]);
            item.hidden_after[index] = 0.999 * item.retain[index] * hidden_state[index] + effect[index];
        }
        item.logits.assign(vocabulary, 0.0);
        for (std::size_t token = 0U; token < vocabulary; ++token) {
            item.logits[token] = parameters[bias_offset + token];
            for (std::size_t index = 0U; index < hidden; ++index) item.logits[token] += parameters[head_offset + token * hidden + index] * item.hidden_after[index];
        }
        cache.push_back(std::move(item));
        hidden_state = cache.back().hidden_after;
    }
    const auto active_tokens = static_cast<double>(target_count(sequence));
    std::vector<double> gradients(parameters.size(), 0.0);
    std::vector<double> d_hidden_next(hidden, 0.0);
    double loss = 0.0;
    std::size_t correct = 0U;
    for (std::size_t reverse = cache.size(); reverse-- > 0U;) {
        auto& item = cache[reverse];
        std::vector<double> d_output(vocabulary, 0.0);
        if (sequence.loss_mask[reverse] != 0U) {
            const auto probabilities = softmax(item.logits);
            const auto target = token_slot(config, sequence.target_ids[reverse]);
            loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
            if (static_cast<std::size_t>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))) == target) ++correct;
            for (std::size_t token = 0U; token < vocabulary; ++token) d_output[token] = probabilities[token] / active_tokens;
            d_output[target] -= 1.0 / active_tokens;
        }
        for (std::size_t token = 0U; token < vocabulary; ++token) {
            gradients[bias_offset + token] += d_output[token];
            for (std::size_t index = 0U; index < hidden; ++index) {
                gradients[head_offset + token * hidden + index] += d_output[token] * item.hidden_after[index];
                d_hidden_next[index] += d_output[token] * parameters[head_offset + token * hidden + index];
            }
        }
        std::vector<double> d_x(input, 0.0);
        std::vector<double> d_hidden_before(hidden, 0.0);
        for (std::size_t index = 0U; index < hidden; ++index) {
            const auto d_effect = d_hidden_next[index];
            gradients[recurrent_offset + index] += d_effect * item.hidden_before[index] * 0.999 * item.retain[index] * (1.0 - item.retain[index]);
            d_hidden_before[index] += d_effect * 0.999 * item.retain[index];
            for (std::size_t column = 0U; column < input; ++column) {
                gradients[input_offset + index * input + column] += d_effect * item.x[column];
                d_x[column] += parameters[input_offset + index * input + column] * d_effect;
            }
        }
        const auto embedding_offset = token_slot(config, sequence.input_ids[reverse]) * input;
        for (std::size_t column = 0U; column < input; ++column) gradients[embedding_offset + column] += d_x[column];
        d_hidden_next = std::move(d_hidden_before);
    }
    require_finite(gradients, "diagonal SSM analytic gradient became non-finite");
    return {loss / active_tokens, static_cast<std::size_t>(active_tokens), vector_norm(gradients), std::move(gradients)};
}

NlpGradientResult dense_gradients(const std::vector<double>& parameters, const NlpModelConfig& config, const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto embedding_offset = 0U;
    const auto q_offset = vocabulary * input;
    const auto k_offset = q_offset + hidden * input;
    const auto v_offset = k_offset + hidden * input;
    const auto head_offset = v_offset + hidden * input;
    const auto bias_offset = head_offset + vocabulary * hidden;
    const auto scale = 1.0 / std::sqrt(static_cast<double>(hidden));
    struct Cache { std::vector<double> x; std::vector<double> q; std::vector<double> k; std::vector<double> v; std::vector<double> probabilities; std::vector<double> context; std::vector<double> logits; };
    std::vector<Cache> cache;
    cache.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        Cache item;
        item.x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(embedding_offset + token_slot(config, id) * input),
                                     parameters.begin() + static_cast<std::ptrdiff_t>(embedding_offset + (token_slot(config, id) + 1U) * input));
        item.q = matvec(parameters, q_offset, hidden, input, item.x);
        item.k = matvec(parameters, k_offset, hidden, input, item.x);
        item.v = matvec(parameters, v_offset, hidden, input, item.x);
        cache.push_back(std::move(item));
    }
    for (std::size_t time = 0U; time < cache.size(); ++time) {
        auto& item = cache[time];
        std::vector<double> scores(time + 1U, 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t position = 0U; position <= time; ++position) {
            scores[position] = std::inner_product(item.q.begin(), item.q.end(), cache[position].k.begin(), 0.0) * scale;
            maximum = std::max(maximum, scores[position]);
        }
        double denominator = 0.0;
        item.probabilities.resize(time + 1U, 0.0);
        for (std::size_t position = 0U; position <= time; ++position) {
            item.probabilities[position] = std::exp(std::clamp(scores[position] - maximum, -80.0, 80.0));
            denominator += item.probabilities[position];
        }
        require(std::isfinite(denominator) && denominator > 0.0, "dense attention gradient denominator is non-finite");
        for (auto& probability : item.probabilities) probability /= denominator;
        item.context.assign(hidden, 0.0);
        for (std::size_t position = 0U; position <= time; ++position)
            for (std::size_t index = 0U; index < hidden; ++index) item.context[index] += item.probabilities[position] * cache[position].v[index];
        item.logits.assign(vocabulary, 0.0);
        for (std::size_t token = 0U; token < vocabulary; ++token) {
            item.logits[token] = parameters[bias_offset + token];
            for (std::size_t index = 0U; index < hidden; ++index) item.logits[token] += parameters[head_offset + token * hidden + index] * item.context[index];
        }
    }
    const auto active_tokens = static_cast<double>(target_count(sequence));
    std::vector<double> gradients(parameters.size(), 0.0);
    std::vector<std::vector<double>> d_q(cache.size(), std::vector<double>(hidden, 0.0));
    std::vector<std::vector<double>> d_k(cache.size(), std::vector<double>(hidden, 0.0));
    std::vector<std::vector<double>> d_v(cache.size(), std::vector<double>(hidden, 0.0));
    std::vector<std::vector<double>> d_x(cache.size(), std::vector<double>(input, 0.0));
    double loss = 0.0;
    std::size_t correct = 0U;
    for (std::size_t reverse = cache.size(); reverse-- > 0U;) {
        auto& item = cache[reverse];
        std::vector<double> d_output(vocabulary, 0.0);
        if (sequence.loss_mask[reverse] != 0U) {
            const auto probabilities = softmax(item.logits);
            const auto target = token_slot(config, sequence.target_ids[reverse]);
            loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
            if (static_cast<std::size_t>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))) == target) ++correct;
            for (std::size_t token = 0U; token < vocabulary; ++token) d_output[token] = probabilities[token] / active_tokens;
            d_output[target] -= 1.0 / active_tokens;
        }
        std::vector<double> d_context(hidden, 0.0);
        for (std::size_t token = 0U; token < vocabulary; ++token) {
            gradients[bias_offset + token] += d_output[token];
            for (std::size_t index = 0U; index < hidden; ++index) {
                gradients[head_offset + token * hidden + index] += d_output[token] * item.context[index];
                d_context[index] += d_output[token] * parameters[head_offset + token * hidden + index];
            }
        }
        std::vector<double> d_probability(item.probabilities.size(), 0.0);
        for (std::size_t position = 0U; position < item.probabilities.size(); ++position) {
            d_probability[position] = std::inner_product(d_context.begin(), d_context.end(), cache[position].v.begin(), 0.0);
            for (std::size_t index = 0U; index < hidden; ++index) d_v[position][index] += item.probabilities[position] * d_context[index];
        }
        double probability_dot = 0.0;
        for (std::size_t position = 0U; position < item.probabilities.size(); ++position) probability_dot += item.probabilities[position] * d_probability[position];
        for (std::size_t position = 0U; position < item.probabilities.size(); ++position) {
            const auto d_score = item.probabilities[position] * (d_probability[position] - probability_dot);
            for (std::size_t index = 0U; index < hidden; ++index) {
                d_q[reverse][index] += d_score * cache[position].k[index] * scale;
                d_k[position][index] += d_score * item.q[index] * scale;
            }
        }
        for (std::size_t index = 0U; index < hidden; ++index) {
            for (std::size_t column = 0U; column < input; ++column) {
                gradients[q_offset + index * input + column] += d_q[reverse][index] * item.x[column];
                d_x[reverse][column] += parameters[q_offset + index * input + column] * d_q[reverse][index];
            }
        }
    }
    for (std::size_t position = 0U; position < cache.size(); ++position) {
        for (std::size_t index = 0U; index < hidden; ++index) {
            for (std::size_t column = 0U; column < input; ++column) {
                gradients[k_offset + index * input + column] += d_k[position][index] * cache[position].x[column];
                gradients[v_offset + index * input + column] += d_v[position][index] * cache[position].x[column];
                d_x[position][column] += parameters[k_offset + index * input + column] * d_k[position][index];
                d_x[position][column] += parameters[v_offset + index * input + column] * d_v[position][index];
            }
        }
        const auto embedding_offset_for_token = token_slot(config, sequence.input_ids[position]) * input;
        for (std::size_t column = 0U; column < input; ++column) gradients[embedding_offset_for_token + column] += d_x[position][column];
    }
    require_finite(gradients, "dense attention analytic gradient became non-finite");
    return {loss / active_tokens, static_cast<std::size_t>(active_tokens), vector_norm(gradients), std::move(gradients)};
}

NlpGradientResult NextTokenModel::loss_and_gradients(const NlpSequence& sequence) const {
    validate_sequence(sequence);
    if (config_.kind == NlpModelKind::Track1CctRecurrence) return track1_cct_gradients(parameters_, config_, sequence);
    if (config_.kind == NlpModelKind::GRU) return gru_gradients(parameters_, config_, sequence);
    if (config_.kind == NlpModelKind::DiagonalSSM) return ssm_gradients(parameters_, config_, sequence);
    return dense_gradients(parameters_, config_, sequence);
}

void NextTokenModel::validate_inference_state(const NlpInferenceState& state) const {
    require(state.kind == config_.kind && state.context_length == config_.context_length, "NLP inference state model contract mismatch");
    require(state.valid_length <= state.context_length && state.write_index < state.context_length, "NLP inference state cursor is invalid");
    require(state.hidden.size() == config_.hidden_dim && state.previous_input.size() == config_.embedding_dim &&
                state.input.size() == config_.embedding_dim && state.query.size() == config_.hidden_dim &&
                state.context.size() == config_.hidden_dim && state.logits.size() == config_.vocabulary_size &&
                state.scores.size() == config_.context_length && state.scratch1.size() == config_.hidden_dim && state.scratch2.size() == config_.hidden_dim &&
                state.scratch3.size() == config_.hidden_dim && state.scratch4.size() == config_.hidden_dim,
            "NLP inference state scratch shape is invalid");
    if (config_.kind == NlpModelKind::DenseCausalAttention) {
        require(state.keys.size() == config_.context_length * config_.hidden_dim && state.values.size() == config_.context_length * config_.hidden_dim,
                "NLP dense inference cache shape is invalid");
    }
}

NlpInferenceState NextTokenModel::create_inference_state() const {
    NlpInferenceState state;
    state.kind = config_.kind;
    state.context_length = config_.context_length;
    state.hidden.assign(config_.hidden_dim, 0.0);
    state.previous_input.assign(config_.embedding_dim, 0.0);
    state.input.assign(config_.embedding_dim, 0.0);
    state.query.assign(config_.hidden_dim, 0.0);
    state.context.assign(config_.hidden_dim, 0.0);
    state.logits.assign(config_.vocabulary_size, 0.0);
    state.scores.assign(config_.context_length, 0.0);
    state.scratch1.assign(config_.hidden_dim, 0.0);
    state.scratch2.assign(config_.hidden_dim, 0.0);
    state.scratch3.assign(config_.hidden_dim, 0.0);
    state.scratch4.assign(config_.hidden_dim, 0.0);
    if (config_.kind == NlpModelKind::DenseCausalAttention) {
        state.keys.assign(config_.context_length * config_.hidden_dim, 0.0);
        state.values.assign(config_.context_length * config_.hidden_dim, 0.0);
    }
    validate_inference_state(state);
    return state;
}

void NextTokenModel::next_logits_incremental_into(const TokenId token, NlpInferenceState& state, std::vector<double>& output) const {
    validate_inference_state(state);
    static_cast<void>(token_slot(config_, token));
    const auto input = config_.embedding_dim;
    const auto hidden = config_.hidden_dim;
    const auto vocabulary = config_.vocabulary_size;
    const auto embedding_offset_for_token = token_slot(config_, token) * input;
    std::copy(parameters_.begin() + static_cast<std::ptrdiff_t>(embedding_offset_for_token),
              parameters_.begin() + static_cast<std::ptrdiff_t>(embedding_offset_for_token + input), state.input.begin());
    const auto recurrent_offset = vocabulary * input;
    if (config_.kind == NlpModelKind::Track1CctRecurrence) {
        const auto head = recurrent_offset + 4U * hidden * input + 3U * hidden;
        const auto bias = head + vocabulary * hidden;
        matvec_into(parameters_, recurrent_offset + 2U * hidden * input, hidden, input, state.input, state.scratch1);
        matvec_into(parameters_, recurrent_offset + 3U * hidden * input, hidden, input, state.input, state.scratch2);
        matvec_into(parameters_, recurrent_offset, hidden, input, state.input, state.scratch3);
        matvec_into(parameters_, recurrent_offset + hidden * input, hidden, input, state.previous_input, state.scratch4);
        for (std::size_t index = 0U; index < hidden; ++index) {
            const auto retain = sigmoid(state.scratch1[index] + parameters_[recurrent_offset + 4U * hidden * input + hidden + index]);
            const auto write = sigmoid(state.scratch2[index] + parameters_[recurrent_offset + 4U * hidden * input + 2U * hidden + index]);
            const auto candidate = std::tanh(state.scratch3[index] + state.scratch4[index] + parameters_[recurrent_offset + 4U * hidden * input + index]);
            state.hidden[index] = retain * state.hidden[index] + write * candidate;
        }
        project_logits_into(parameters_, head, bias, vocabulary, hidden, state.hidden, output);
        state.previous_input = state.input;
    } else if (config_.kind == NlpModelKind::GRU) {
        const auto gate_size = hidden * input + hidden * hidden + hidden;
        const auto z_offset = recurrent_offset;
        const auto r_offset = z_offset + gate_size;
        const auto n_offset = r_offset + gate_size;
        const auto head = n_offset + gate_size;
        const auto bias = head + vocabulary * hidden;
        const auto gate = [&](const std::size_t offset, std::vector<double>& result) {
            matvec_into(parameters_, offset, hidden, input, state.input, result);
            matvec_into(parameters_, offset + hidden * input, hidden, hidden, state.hidden, state.scratch4);
            for (std::size_t index = 0U; index < hidden; ++index) result[index] = sigmoid(result[index] + state.scratch4[index] + parameters_[offset + hidden * input + hidden * hidden + index]);
        };
        gate(z_offset, state.scratch1);
        gate(r_offset, state.scratch2);
        matvec_into(parameters_, n_offset, hidden, input, state.input, state.scratch3);
        matvec_into(parameters_, n_offset + hidden * input, hidden, hidden, state.hidden, state.scratch4);
        for (std::size_t index = 0U; index < hidden; ++index) state.scratch3[index] = std::tanh(state.scratch3[index] + state.scratch2[index] * state.scratch4[index] + parameters_[n_offset + hidden * input + hidden * hidden + index]);
        for (std::size_t index = 0U; index < hidden; ++index) state.hidden[index] = state.scratch1[index] * state.hidden[index] + (1.0 - state.scratch1[index]) * state.scratch3[index];
        project_logits_into(parameters_, head, bias, vocabulary, hidden, state.hidden, output);
    } else if (config_.kind == NlpModelKind::DiagonalSSM) {
        const auto input_offset = recurrent_offset + hidden;
        const auto head = input_offset + hidden * input;
        const auto bias = head + vocabulary * hidden;
        matvec_into(parameters_, input_offset, hidden, input, state.input, state.scratch1);
        for (std::size_t index = 0U; index < hidden; ++index) state.hidden[index] = 0.999 * sigmoid(parameters_[recurrent_offset + index]) * state.hidden[index] + state.scratch1[index];
        project_logits_into(parameters_, head, bias, vocabulary, hidden, state.hidden, output);
    } else {
        const auto attention_offset = recurrent_offset;
        const auto q_offset = attention_offset;
        const auto k_offset = q_offset + hidden * input;
        const auto v_offset = k_offset + hidden * input;
        const auto head = v_offset + hidden * input;
        const auto bias = head + vocabulary * hidden;
        const auto physical = state.write_index;
        matvec_into(parameters_, q_offset, hidden, input, state.input, state.query);
        matvec_into(parameters_, k_offset, hidden, input, state.input, state.scratch1);
        matvec_into(parameters_, v_offset, hidden, input, state.input, state.scratch2);
        std::copy(state.scratch1.begin(), state.scratch1.end(), state.keys.begin() + static_cast<std::ptrdiff_t>(physical * hidden));
        std::copy(state.scratch2.begin(), state.scratch2.end(), state.values.begin() + static_cast<std::ptrdiff_t>(physical * hidden));
        const auto count = std::min(state.valid_length + 1U, state.context_length);
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t position = 0U; position < count; ++position) {
            const auto slot = state.valid_length < state.context_length ? position : (state.write_index + position) % state.context_length;
            const auto key_offset = slot * hidden;
            double score = 0.0;
            for (std::size_t index = 0U; index < hidden; ++index) score += state.query[index] * state.keys[key_offset + index];
            state.scores[position] = score / std::sqrt(static_cast<double>(hidden));
            maximum = std::max(maximum, state.scores[position]);
        }
        double denominator = 0.0;
        for (auto& score : state.scores) {
            score = std::exp(std::clamp(score - maximum, -80.0, 80.0));
            denominator += score;
        }
        require(std::isfinite(denominator) && denominator > 0.0, "incremental dense attention denominator is non-finite");
        std::fill(state.context.begin(), state.context.end(), 0.0);
        for (std::size_t position = 0U; position < count; ++position) {
            const auto slot = state.valid_length < state.context_length ? position : (state.write_index + position) % state.context_length;
            const auto value_offset = slot * hidden;
            const auto weight = state.scores[position] / denominator;
            for (std::size_t index = 0U; index < hidden; ++index) state.context[index] += weight * state.values[value_offset + index];
        }
        project_logits_into(parameters_, head, bias, vocabulary, hidden, state.context, output);
    }
    state.valid_length = std::min(state.valid_length + 1U, state.context_length);
    state.write_index = state.valid_length < state.context_length ? state.valid_length : (state.write_index + 1U) % state.context_length;
    require_finite(output, "NLP incremental inference logits became non-finite");
}

std::vector<double> NextTokenModel::next_logits(const std::vector<TokenId>& context) const {
    require(!context.empty() && context.size() <= config_.context_length, "NLP inference context length is invalid");
    auto state = create_inference_state();
    std::vector<double> output(config_.vocabulary_size, 0.0);
    for (const auto id : context) next_logits_incremental_into(id, state, output);
    return output;
}

std::vector<double> NextTokenModel::next_logits_incremental(const TokenId token, NlpInferenceState& state) const {
    std::vector<double> output(config_.vocabulary_size, 0.0);
    next_logits_incremental_into(token, state, output);
    return output;
}

TokenId NextTokenModel::token_id_from_logit_slot(const std::size_t slot) const { return token_from_slot(config_, slot); }

std::size_t NextTokenModel::logit_slot_for_token_id(const TokenId token) const { return token_slot(config_, token); }

double NextTokenModel::loss_only(const NlpSequence& sequence) const {
    validate_sequence(sequence);
    const auto logits = model_forward(parameters_, config_, sequence);
    return cross_entropy_from_logits(logits, sequence, config_, nullptr, nullptr);
}

NlpEvaluation NextTokenModel::evaluate(const std::vector<NlpSequence>& sequences) const {
    require(!sequences.empty(), "NLP evaluation has no sequences");
    const auto started = std::chrono::steady_clock::now();
    double loss_sum = 0.0;
    std::size_t token_count = 0;
    std::size_t correct = 0;
    for (const auto& sequence : sequences) {
        validate_sequence(sequence);
        const auto logits = model_forward(parameters_, config_, sequence);
        std::size_t count = 0;
        double accuracy = 0.0;
        const auto loss = cross_entropy_from_logits(logits, sequence, config_, &count, &accuracy);
        require(std::isfinite(loss), "NLP evaluation loss is non-finite");
        loss_sum += loss * static_cast<double>(count);
        token_count += count;
        correct += static_cast<std::size_t>(accuracy * static_cast<double>(count));
    }
    const auto finished = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(finished - started).count();
    const auto loss = loss_sum / static_cast<double>(token_count);
    NlpEvaluation result;
    result.cross_entropy = loss;
    result.perplexity = std::exp(std::min(loss, 50.0));
    result.token_accuracy = static_cast<double>(correct) / static_cast<double>(token_count);
    result.token_count = token_count;
    result.elapsed_seconds = elapsed;
    result.tokens_per_second = elapsed > 0.0 ? static_cast<double>(token_count) / elapsed : 0.0;
    result.finite = std::isfinite(result.cross_entropy) && std::isfinite(result.perplexity);
    require(result.finite, "NLP evaluation is non-finite");
    return result;
}

void NextTokenModel::copy_parameter_vector_to(std::vector<double>& values) const {
    values.resize(parameters_.size());
    std::copy(parameters_.begin(), parameters_.end(), values.begin());
}

void NextTokenModel::set_parameter_vector(const std::vector<double>& values) {
    require(values.size() == expected_parameter_count(), "NLP parameter vector size mismatch");
    require_finite(values, "NLP parameter vector contains non-finite value");
    parameters_ = values;
}

void NextTokenModel::set_parameter_vector(std::vector<double>&& values) {
    require(values.size() == expected_parameter_count(), "NLP parameter vector size mismatch");
    require_finite(values, "NLP parameter vector contains non-finite value");
    parameters_ = std::move(values);
}

std::size_t NextTokenModel::state_memory_bytes() const noexcept {
    if (config_.kind == NlpModelKind::DenseCausalAttention) return (2U * config_.context_length * config_.hidden_dim + config_.hidden_dim) * sizeof(double);
    if (config_.kind == NlpModelKind::Track1CctRecurrence) return (config_.hidden_dim + config_.embedding_dim) * sizeof(double);
    return config_.hidden_dim * sizeof(double);
}

void NextTokenModel::apply_gradient(const std::vector<double>& gradients, const NlpOptimizerConfig& optimizer,
                                    NlpTrainerState& state, double* applied_learning_rate) {
    require(gradients.size() == parameters_.size() && std::isfinite(optimizer.learning_rate) && optimizer.learning_rate > 0.0 &&
                std::isfinite(optimizer.clip_norm) && optimizer.clip_norm > 0.0 && state.optimizer_step < std::numeric_limits<std::size_t>::max(),
            "NLP simple optimizer settings or gradient shape is invalid");
    require_finite(gradients, "NLP simple optimizer gradient is non-finite");
    const auto norm = vector_norm(gradients);
    require(std::isfinite(norm), "NLP gradient norm is non-finite");
    const auto scale = std::min(1.0, optimizer.clip_norm / std::max(norm, 1e-12));
    const auto learning_rate = optimizer.learning_rate * scale;
    require(std::isfinite(learning_rate), "NLP simple optimizer learning rate is non-finite");
    auto candidate = parameters_;
    for (std::size_t index = 0; index < candidate.size(); ++index) candidate[index] -= learning_rate * gradients[index];
    require_finite(candidate, "NLP simple optimizer produced non-finite parameters");
    parameters_ = std::move(candidate);
    state.optimizer_step += 1U;
    if (applied_learning_rate != nullptr) *applied_learning_rate = learning_rate;
}

void NextTokenModel::save_model(std::ostream& stream) const {
    stream << (config_.compact_vocabulary ? "NLP_MODEL_V4\n" : "NLP_MODEL_V3\n") << model_kind_number(config_.kind) << ' ' << config_.vocabulary_size << ' '
           << config_.embedding_dim << ' ' << config_.hidden_dim << ' ' << config_.context_length << ' ' << config_.seed;
    if (config_.compact_vocabulary) stream << ' ' << 1U << ' ' << config_.token_id_limit;
    stream << '\n'
           << parameters_.size() << '\n' << std::setprecision(17);
    for (std::size_t index = 0U; index < parameters_.size(); ++index) {
        if (index > 0U) stream << ' ';
        stream << parameters_[index];
    }
    stream << '\n';
}

NextTokenModel NextTokenModel::load_model(std::istream& stream) {
    std::string header;
    require(static_cast<bool>(stream >> header) && (header == "NLP_MODEL_V2" || header == "NLP_MODEL_V3" || header == "NLP_MODEL_V4"),
            "invalid NLP model checkpoint header");
    std::size_t kind = 0;
    NlpModelConfig config;
    require(static_cast<bool>(stream >> kind >> config.vocabulary_size >> config.embedding_dim >> config.hidden_dim >>
                              config.context_length >> config.seed),
            "NLP model checkpoint configuration is incomplete");
    config.kind = model_kind_from_number(kind);
    if (header == "NLP_MODEL_V4") {
        unsigned int compact = 0U;
        require(static_cast<bool>(stream >> compact >> config.token_id_limit) && compact == 1U,
                "NLP compact model checkpoint metadata is invalid");
        config.compact_vocabulary = true;
    }
    if (!config.compact_vocabulary) {
        require(config.vocabulary_size >= Tokenizer::kByteFirstId + 256U && config.vocabulary_size <= 1'000'000U,
                "NLP model checkpoint vocabulary dimensions exceed budget");
    }
    require(config.embedding_dim > 0U && config.embedding_dim <= 4096U && config.hidden_dim > 0U && config.hidden_dim <= 4096U &&
                config.context_length >= 2U && config.context_length <= 1'000'000U,
            "NLP model checkpoint dimensions exceed budget");
    NextTokenModel model(config);
    std::size_t parameter_count = 0;
    require(static_cast<bool>(stream >> parameter_count) && parameter_count == model.parameter_count() && parameter_count <= 16'000'000U,
            "NLP model checkpoint parameter count is invalid or exceeds budget");
    std::vector<double> parameters(parameter_count, 0.0);
    for (auto& value : parameters) require(static_cast<bool>(stream >> value) && std::isfinite(value), "NLP model checkpoint parameter is invalid");
    model.set_parameter_vector(parameters);
    return model;
}

NlpTrainer::NlpTrainer(NlpModelConfig model_config, NlpOptimizerConfig optimizer_config,
                       std::string tokenizer_hash, std::string dataset_hash)
    : model_(std::move(model_config)), optimizer_config_(std::move(optimizer_config)),
      tokenizer_hash_(std::move(tokenizer_hash)), dataset_hash_(std::move(dataset_hash)) {
    require(!tokenizer_hash_.empty() && !dataset_hash_.empty(), "NLP trainer identity hashes cannot be empty");
    validate_optimizer();
    first_moment_.assign(model_.parameter_count(), 0.0);
    second_moment_.assign(model_.parameter_count(), 0.0);
    update_parameters_.resize(model_.parameter_count());
    update_first_moment_.resize(model_.parameter_count());
    update_second_moment_.resize(model_.parameter_count());
    state_.seed = model_.config().seed;
    state_.rng_state = "seed=" + std::to_string(state_.seed);
    training_contract_hash_ = training_contract_digest(model_.config(), optimizer_config_, tokenizer_hash_, dataset_hash_);
    checkpoint_info_.tokenizer_hash = tokenizer_hash_;
    checkpoint_info_.dataset_hash = dataset_hash_;
    checkpoint_info_.training_contract_hash = training_contract_hash_;
    checkpoint_info_.session_id = session_id_;
    checkpoint_info_.parent_checkpoint_hash = parent_checkpoint_hash_;
}

void NlpTrainer::validate_optimizer() const {
    require(std::isfinite(optimizer_config_.learning_rate) && optimizer_config_.learning_rate > 0.0 &&
                std::isfinite(optimizer_config_.beta1) && optimizer_config_.beta1 >= 0.0 && optimizer_config_.beta1 < 1.0 &&
                std::isfinite(optimizer_config_.beta2) && optimizer_config_.beta2 >= 0.0 && optimizer_config_.beta2 < 1.0 &&
                std::isfinite(optimizer_config_.epsilon) && optimizer_config_.epsilon > 0.0 &&
                std::isfinite(optimizer_config_.weight_decay) && optimizer_config_.weight_decay >= 0.0 &&
                std::isfinite(optimizer_config_.clip_norm) && optimizer_config_.clip_norm > 0.0 &&
                optimizer_config_.batch_size > 0U && optimizer_config_.total_steps > 0U &&
                optimizer_config_.validation_interval_steps > 0U && optimizer_config_.worker_count > 0U,
            "NLP optimizer configuration is invalid or non-finite");
}

double NlpTrainer::scheduled_learning_rate() const {
    validate_optimizer();
    const auto step = state_.optimizer_step + 1U;
    if (optimizer_config_.warmup_steps > 0U && step <= optimizer_config_.warmup_steps) {
        return optimizer_config_.learning_rate * static_cast<double>(step) / static_cast<double>(optimizer_config_.warmup_steps);
    }
    if (optimizer_config_.total_steps <= optimizer_config_.warmup_steps) return optimizer_config_.learning_rate;
    const auto remaining = optimizer_config_.total_steps - std::min(step, optimizer_config_.total_steps);
    const auto span = optimizer_config_.total_steps - optimizer_config_.warmup_steps;
    const auto scheduled = optimizer_config_.learning_rate * static_cast<double>(remaining) / static_cast<double>(span);
    require(std::isfinite(scheduled) && scheduled >= 0.0, "NLP scheduled learning rate is invalid");
    return scheduled;
}

NlpEvaluation NlpTrainer::evaluate(const std::vector<NlpSequence>& sequences) const { return model_.evaluate(sequences); }

NlpTrainingPoint NlpTrainer::train_step(const NlpDataset& dataset) {
    const auto training_started = std::chrono::steady_clock::now();
    require(dataset.dataset_hash == dataset_hash_ && dataset.tokenizer_hash == tokenizer_hash_, "NLP dataset identity mismatch");
    require(dataset.context_length == model_.config().context_length, "NLP dataset/model context length mismatch");
    require(!dataset.train.empty() && state_.optimizer_step < optimizer_config_.total_steps,
            "NLP training dataset is empty or optimizer budget is exhausted");
    require(optimizer_config_.batch_size <= std::numeric_limits<std::size_t>::max() - state_.data_cursor,
            "NLP data cursor would overflow");
    std::vector<double> aggregate_gradients(model_.parameter_count(), 0.0);
    double aggregate_loss = 0.0;
    std::size_t aggregate_tokens = 0U;
    std::vector<NlpGradientResult> batch_results(optimizer_config_.batch_size);
    const auto evaluate_batch_item = [&](const std::size_t batch_index) {
        const auto& sequence = dataset.train[(state_.data_cursor + batch_index) % dataset.train.size()];
        return model_.loss_and_gradients(sequence);
    };
    const auto worker_limit = std::min(optimizer_config_.worker_count, optimizer_config_.batch_size);
    for (std::size_t batch_start = 0U; batch_start < optimizer_config_.batch_size; batch_start += worker_limit) {
        const auto batch_end = std::min(batch_start + worker_limit, optimizer_config_.batch_size);
        std::vector<std::future<NlpGradientResult>> futures;
        if (worker_limit > 1U) futures.reserve(batch_end - batch_start);
        for (std::size_t batch_index = batch_start; batch_index < batch_end; ++batch_index) {
            if (worker_limit > 1U) {
                futures.emplace_back(std::async(std::launch::async, evaluate_batch_item, batch_index));
            } else {
                batch_results[batch_index] = evaluate_batch_item(batch_index);
            }
        }
        if (worker_limit > 1U) {
            for (std::size_t future_index = 0U; future_index < futures.size(); ++future_index) {
                batch_results[batch_start + future_index] = futures[future_index].get();
            }
        }
    }
    for (const auto& gradient_result : batch_results) {
        require(std::isfinite(gradient_result.cross_entropy) && std::isfinite(gradient_result.gradient_norm), "NLP training objective is non-finite");
        require(gradient_result.token_count > 0U && gradient_result.gradients.size() == model_.parameter_count(),
                "NLP training gradient shape is invalid");
        require(aggregate_tokens <= std::numeric_limits<std::size_t>::max() - gradient_result.token_count,
                "NLP training target-token count would overflow");
        aggregate_loss += gradient_result.cross_entropy * static_cast<double>(gradient_result.token_count);
        aggregate_tokens += gradient_result.token_count;
        for (std::size_t index = 0U; index < aggregate_gradients.size(); ++index) aggregate_gradients[index] += gradient_result.gradients[index] * static_cast<double>(gradient_result.token_count);
    }
    require(aggregate_tokens > 0U, "NLP training batch has no active target tokens");
    for (auto& gradient : aggregate_gradients) gradient /= static_cast<double>(aggregate_tokens);
    require_finite(aggregate_gradients, "NLP training gradient is non-finite");
    const auto aggregate_gradient_norm = vector_norm(aggregate_gradients);
    model_.copy_parameter_vector_to(update_parameters_);
    std::copy(first_moment_.begin(), first_moment_.end(), update_first_moment_.begin());
    std::copy(second_moment_.begin(), second_moment_.end(), update_second_moment_.begin());
    const auto gradient_norm = aggregate_gradient_norm;
    const auto scale = std::min(1.0, optimizer_config_.clip_norm / std::max(gradient_norm, 1e-12));
    const auto learning_rate = scheduled_learning_rate();
    const auto step = state_.optimizer_step + 1U;
    const auto first_correction = 1.0 - std::pow(optimizer_config_.beta1, static_cast<double>(step));
    const auto second_correction = 1.0 - std::pow(optimizer_config_.beta2, static_cast<double>(step));
    for (std::size_t index = 0; index < update_parameters_.size(); ++index) {
        const auto gradient = aggregate_gradients[index] * scale;
        update_first_moment_[index] = optimizer_config_.beta1 * update_first_moment_[index] + (1.0 - optimizer_config_.beta1) * gradient;
        update_second_moment_[index] = optimizer_config_.beta2 * update_second_moment_[index] + (1.0 - optimizer_config_.beta2) * gradient * gradient;
        const auto first_hat = update_first_moment_[index] / std::max(first_correction, 1e-12);
        const auto second_hat = update_second_moment_[index] / std::max(second_correction, 1e-12);
        update_parameters_[index] -= learning_rate * (first_hat / (std::sqrt(second_hat) + optimizer_config_.epsilon) + optimizer_config_.weight_decay * update_parameters_[index]);
    }
    require_finite(update_first_moment_, "NLP optimizer produced non-finite first moments");
    require_finite(update_second_moment_, "NLP optimizer produced non-finite second moments");
    require_finite(update_parameters_, "NLP optimizer produced non-finite parameters");
    model_.set_parameter_vector(update_parameters_);
    const auto training_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - training_started).count();
    NlpEvaluation validation;
    double validation_elapsed = 0.0;
    const bool validation_performed = step % optimizer_config_.validation_interval_steps == 0U;
    if (validation_performed) {
        const auto validation_started = std::chrono::steady_clock::now();
        validation = model_.evaluate(dataset.validation);
        validation_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - validation_started).count();
        require(std::isfinite(validation.cross_entropy) && std::isfinite(validation.perplexity), "NLP validation metrics are non-finite");
    }
    first_moment_.swap(update_first_moment_);
    second_moment_.swap(update_second_moment_);
    state_.optimizer_step = step;
    state_.data_cursor += optimizer_config_.batch_size;
    NlpTrainingPoint point;
    point.step = state_.optimizer_step;
    point.data_cursor = state_.data_cursor;
    point.learning_rate = learning_rate;
    point.train_loss = aggregate_loss / static_cast<double>(aggregate_tokens);
    point.token_count = aggregate_tokens;
    point.gradient_norm = gradient_norm;
    point.validation_loss = validation.cross_entropy;
    point.validation_perplexity = validation.perplexity;
    point.validation_performed = validation_performed;
    point.training_elapsed_seconds = training_elapsed;
    point.validation_elapsed_seconds = validation_elapsed;
    history_.push_back(point);
    return point;
}

NlpTrainingPoint NlpTrainer::train_preference_step(const NlpPreferencePair& pair, const double margin) {
    const auto training_started = std::chrono::steady_clock::now();
    require(std::isfinite(margin) && margin >= 0.0, "NLP preference margin is invalid");
    require(state_.optimizer_step < optimizer_config_.total_steps, "NLP preference optimizer budget is exhausted");
    const auto preferred = model_.loss_and_gradients(pair.preferred);
    const auto rejected = model_.loss_and_gradients(pair.rejected);
    require(preferred.gradients.size() == rejected.gradients.size() && preferred.token_count > 0U && rejected.token_count > 0U,
            "NLP preference gradient shape or token count is invalid");
    const auto raw_margin_loss = preferred.cross_entropy - rejected.cross_entropy + margin;
    require(std::isfinite(raw_margin_loss), "NLP preference loss is non-finite");
    std::vector<double> gradients(preferred.gradients.size(), 0.0);
    if (raw_margin_loss > 0.0) {
        for (std::size_t index = 0U; index < gradients.size(); ++index) gradients[index] = preferred.gradients[index] - rejected.gradients[index];
    }
    require_finite(gradients, "NLP preference gradient is non-finite");
    const auto gradient_norm = vector_norm(gradients);
    const auto scale = std::min(1.0, optimizer_config_.clip_norm / std::max(gradient_norm, 1e-12));
    const auto learning_rate = scheduled_learning_rate();
    const auto step = state_.optimizer_step + 1U;
    model_.copy_parameter_vector_to(update_parameters_);
    std::copy(first_moment_.begin(), first_moment_.end(), update_first_moment_.begin());
    std::copy(second_moment_.begin(), second_moment_.end(), update_second_moment_.begin());
    const auto first_correction = 1.0 - std::pow(optimizer_config_.beta1, static_cast<double>(step));
    const auto second_correction = 1.0 - std::pow(optimizer_config_.beta2, static_cast<double>(step));
    for (std::size_t index = 0U; index < update_parameters_.size(); ++index) {
        const auto gradient = gradients[index] * scale;
        update_first_moment_[index] = optimizer_config_.beta1 * update_first_moment_[index] + (1.0 - optimizer_config_.beta1) * gradient;
        update_second_moment_[index] = optimizer_config_.beta2 * update_second_moment_[index] + (1.0 - optimizer_config_.beta2) * gradient * gradient;
        const auto first_hat = update_first_moment_[index] / std::max(first_correction, 1e-12);
        const auto second_hat = update_second_moment_[index] / std::max(second_correction, 1e-12);
        update_parameters_[index] -= learning_rate * (first_hat / (std::sqrt(second_hat) + optimizer_config_.epsilon) + optimizer_config_.weight_decay * update_parameters_[index]);
    }
    require_finite(update_first_moment_, "NLP preference optimizer first moment is non-finite");
    require_finite(update_second_moment_, "NLP preference optimizer second moment is non-finite");
    require_finite(update_parameters_, "NLP preference optimizer parameters are non-finite");
    model_.set_parameter_vector(update_parameters_);
    first_moment_.swap(update_first_moment_);
    second_moment_.swap(update_second_moment_);
    state_.optimizer_step = step;
    state_.data_cursor += 1U;
    NlpTrainingPoint point;
    point.step = state_.optimizer_step;
    point.data_cursor = state_.data_cursor;
    point.learning_rate = learning_rate;
    point.train_loss = std::max(0.0, raw_margin_loss);
    point.gradient_norm = gradient_norm;
    point.token_count = preferred.token_count + rejected.token_count;
    point.training_elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - training_started).count();
    point.validation_performed = false;
    history_.push_back(point);
    return point;
}

std::vector<NlpTrainingPoint> NlpTrainer::train_steps(const NlpDataset& dataset, const std::size_t steps) {
    require(steps > 0U, "NLP training step count must be positive");
    std::vector<NlpTrainingPoint> points;
    points.reserve(steps);
    for (std::size_t index = 0; index < steps; ++index) points.push_back(train_step(dataset));
    return points;
}

std::vector<NlpTrainingPoint> NlpTrainer::train_preference_steps(const std::vector<NlpPreferencePair>& pairs, const std::size_t steps, const double margin) {
    require(!pairs.empty() && steps > 0U, "NLP preference training requires pairs and positive steps");
    std::vector<NlpTrainingPoint> points;
    points.reserve(steps);
    for (std::size_t index = 0U; index < steps; ++index) points.push_back(train_preference_step(pairs[state_.data_cursor % pairs.size()], margin));
    return points;
}

void NlpTrainer::validate_checkpoint_identity(const std::string& tokenizer_hash, const std::string& dataset_hash) const {
    require(tokenizer_hash == tokenizer_hash_ && dataset_hash == dataset_hash_, "NLP checkpoint identity mismatch");
}

void NlpTrainer::begin_continuation(const std::string& dataset_hash, const std::string& session_id,
                                    const std::string& parent_checkpoint_hash, const std::size_t session_steps) {
    require(!dataset_hash.empty() && !session_id.empty() && !parent_checkpoint_hash.empty(),
            "NLP continuation identity fields cannot be empty");
    require(session_steps > 0U, "NLP continuation session budget must be positive");
    if (!checkpoint_info_.checkpoint_hash.empty()) {
        require(parent_checkpoint_hash == checkpoint_info_.checkpoint_hash, "NLP continuation parent checkpoint hash mismatch");
    } else {
        require(parent_checkpoint_hash == "GENESIS", "NLP initial continuation must use GENESIS parent identity");
    }
    require(state_.optimizer_step <= std::numeric_limits<std::size_t>::max() - session_steps,
            "NLP continuation optimizer budget would overflow");
    dataset_hash_ = dataset_hash;
    optimizer_config_.total_steps = state_.optimizer_step + session_steps;
    state_.data_cursor = 0U;
    session_id_ = session_id;
    parent_checkpoint_hash_ = parent_checkpoint_hash;
    training_contract_hash_ = training_contract_digest(model_.config(), optimizer_config_, tokenizer_hash_, dataset_hash_);
    checkpoint_info_.dataset_hash = dataset_hash_;
    checkpoint_info_.training_contract_hash = training_contract_hash_;
    checkpoint_info_.session_id = session_id_;
    checkpoint_info_.parent_checkpoint_hash = parent_checkpoint_hash_;
}
void NlpTrainer::save_checkpoint(const std::string& path) const {
    std::ostringstream serialized;
    serialized << "CCT_NLP_CHECKPOINT_V3\n";
    serialized << "tokenizer_hash=" << hex_encode(tokenizer_hash_) << "\n";
    serialized << "dataset_hash=" << hex_encode(dataset_hash_) << "\n";
    serialized << "training_contract_hash=" << hex_encode(training_contract_hash_) << "\n";
    serialized << "session_id=" << hex_encode(session_id_) << "\n";
    serialized << "parent_checkpoint_hash=" << hex_encode(parent_checkpoint_hash_) << "\n";
    serialized << "optimizer_step=" << state_.optimizer_step << "\n";
    serialized << "data_cursor=" << state_.data_cursor << "\n";
    serialized << "seed=" << state_.seed << "\n";
    serialized << "rng_state=" << hex_encode(state_.rng_state) << "\n";
    serialized << "optimizer=" << std::setprecision(17) << optimizer_config_.learning_rate << ' ' << optimizer_config_.beta1 << ' '
               << optimizer_config_.beta2 << ' ' << optimizer_config_.epsilon << ' ' << optimizer_config_.weight_decay << ' '
               << optimizer_config_.clip_norm << ' ' << optimizer_config_.warmup_steps << ' ' << optimizer_config_.batch_size << ' '
               << optimizer_config_.total_steps << ' ' << optimizer_config_.validation_interval_steps << ' ' << optimizer_config_.worker_count << "\n";
    serialized << "history_count=" << history_.size() << "\n";
    for (const auto& point : history_) {
        serialized << "history=" << point.step << ' ' << point.data_cursor << ' ' << point.learning_rate << ' '
                   << point.train_loss << ' ' << point.validation_loss << ' ' << point.validation_perplexity << ' '
                   << point.gradient_norm << ' ' << point.token_count << ' ' << (point.validation_performed ? 1 : 0) << " 0 0\n";
    }
    serialized << "moments_count=" << first_moment_.size() << "\n";
    serialized << "first_moment=";
    for (std::size_t index = 0U; index < first_moment_.size(); ++index) {
        if (index > 0U) serialized << ' ';
        serialized << first_moment_[index];
    }
    serialized << "\nsecond_moment=";
    for (std::size_t index = 0U; index < second_moment_.size(); ++index) {
        if (index > 0U) serialized << ' ';
        serialized << second_moment_[index];
    }
    serialized << "\nmodel_begin\n";
    model_.save_model(serialized);
    serialized << "end=1\n";
    const auto content = serialized.str();
    atomic_publish_checkpoint(path, content);
    checkpoint_info_.checkpoint_hash = nlp_checkpoint_hash(content);
    checkpoint_info_.optimizer_step = state_.optimizer_step;
    checkpoint_info_.data_cursor = state_.data_cursor;
}

NlpTrainer NlpTrainer::load_checkpoint(const std::string& path, const std::string& expected_tokenizer_hash,
                                       const std::string& expected_dataset_hash) {
    constexpr std::uintmax_t maximum_checkpoint_bytes = 256U * 1024U * 1024U;
    constexpr std::size_t maximum_line_bytes = 64U * 1024U * 1024U;
    constexpr std::size_t maximum_history_count = 1'000'000U;
    constexpr std::size_t maximum_moment_count = 16'000'000U;
    std::error_code size_error;
    const auto file_bytes = std::filesystem::file_size(path, size_error);
    require(!size_error && file_bytes <= maximum_checkpoint_bytes, "NLP checkpoint exceeds byte budget");
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "could not read NLP checkpoint");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto content = buffer.str();
    std::istringstream lines(content);
    std::string line;
    require(static_cast<bool>(std::getline(lines, line)) && (line == "CCT_NLP_CHECKPOINT_V2" || line == "CCT_NLP_CHECKPOINT_V3"),
            "unsupported NLP checkpoint header");
    const bool lineage_format = line == "CCT_NLP_CHECKPOINT_V3";
    std::string tokenizer_hash;
    std::string dataset_hash;
    std::string contract_hash;
    std::string session_id;
    std::string parent_checkpoint_hash;
    std::size_t optimizer_step = 0;
    std::size_t data_cursor = 0;
    std::uint64_t seed = 0;
    std::string rng_state;
    NlpOptimizerConfig optimizer;
    std::size_t history_count = 0;
    std::vector<NlpTrainingPoint> history;
    std::size_t moments_count = 0;
    std::vector<double> first;
    std::vector<double> second;
    std::string model_text;
    bool model_started = false;
    bool saw_end = false;
    while (std::getline(lines, line)) {
        require(line.size() <= maximum_line_bytes, "NLP checkpoint line exceeds byte budget");
        if (line == "model_begin") {
            model_started = true;
            break;
        }
        if (line.rfind("tokenizer_hash=", 0) == 0) tokenizer_hash = hex_decode(line.substr(15));
        else if (line.rfind("dataset_hash=", 0) == 0) dataset_hash = hex_decode(line.substr(13));
        else         if (line.rfind("training_contract_hash=", 0) == 0) contract_hash = hex_decode(line.substr(23));
        else if (line.rfind("session_id=", 0) == 0) {
            require(lineage_format, "lineage field is not valid in V2 checkpoint");
            session_id = hex_decode(line.substr(11));
        } else if (line.rfind("parent_checkpoint_hash=", 0) == 0) {
            require(lineage_format, "lineage field is not valid in V2 checkpoint");
            parent_checkpoint_hash = hex_decode(line.substr(23));
        } else if (line.rfind("optimizer_step=", 0) == 0) optimizer_step = parse_size(line.substr(15), "optimizer_step");
        else if (line.rfind("data_cursor=", 0) == 0) data_cursor = parse_size(line.substr(12), "data_cursor");
        else if (line.rfind("seed=", 0) == 0) seed = static_cast<std::uint64_t>(parse_size(line.substr(5), "seed"));
        else if (line.rfind("rng_state=", 0) == 0) rng_state = hex_decode(line.substr(10));
        else if (line.rfind("optimizer=", 0) == 0) {
            std::istringstream values(line.substr(10));
            require(static_cast<bool>(values >> optimizer.learning_rate >> optimizer.beta1 >> optimizer.beta2 >> optimizer.epsilon >>
                                      optimizer.weight_decay >> optimizer.clip_norm >> optimizer.warmup_steps),
                    "checkpoint optimizer fields are incomplete");
            std::vector<std::size_t> optimizer_tail;
            std::size_t tail_value = 0U;
            while (values >> tail_value) optimizer_tail.push_back(tail_value);
            require(optimizer_tail.size() >= 1U && optimizer_tail.size() <= 4U, "checkpoint optimizer field count is invalid");
            if (optimizer_tail.size() == 1U) {
                optimizer.total_steps = optimizer_tail[0];
            } else if (optimizer_tail.size() == 2U) {
                optimizer.total_steps = optimizer_tail[0];
                optimizer.validation_interval_steps = optimizer_tail[1];
            } else {
                optimizer.batch_size = optimizer_tail[0];
                optimizer.total_steps = optimizer_tail[1];
                optimizer.validation_interval_steps = optimizer_tail[2];
                if (optimizer_tail.size() == 4U) optimizer.worker_count = optimizer_tail[3];
            }

        } else if (line.rfind("history_count=", 0) == 0) {
            history_count = parse_size(line.substr(14), "history_count");
            require(history_count <= maximum_history_count, "NLP checkpoint history exceeds budget");
            history.reserve(history_count);
        } else if (line.rfind("history=", 0) == 0) {
            std::istringstream values(line.substr(8));
            NlpTrainingPoint point;
            require(static_cast<bool>(values >> point.step >> point.data_cursor >> point.learning_rate >> point.train_loss >>
                                      point.validation_loss >> point.validation_perplexity >> point.gradient_norm >> point.token_count),
                    "checkpoint history row is incomplete");
            int validation_performed = 0;
            if (values >> validation_performed >> point.training_elapsed_seconds >> point.validation_elapsed_seconds) {
                require(validation_performed == 0 || validation_performed == 1, "checkpoint validation flag is invalid");
                point.validation_performed = validation_performed == 1;
                require(std::isfinite(point.training_elapsed_seconds) && std::isfinite(point.validation_elapsed_seconds),
                        "checkpoint training timing is invalid");
            }
            require(history.size() < maximum_history_count, "NLP checkpoint history exceeds budget");
            history.push_back(point);
        } else if (line.rfind("moments_count=", 0) == 0) {
            moments_count = parse_size(line.substr(14), "moments_count");
            require(moments_count > 0U && moments_count <= maximum_moment_count, "NLP checkpoint moments exceed budget");
            first.assign(moments_count, 0.0);
            second.assign(moments_count, 0.0);
        } else if (line.rfind("first_moment=", 0) == 0) {
            std::istringstream values(line.substr(13));
            for (auto& value : first) require(static_cast<bool>(values >> value) && std::isfinite(value), "checkpoint first moment is invalid");
        } else if (line.rfind("second_moment=", 0) == 0) {
            std::istringstream values(line.substr(14));
            for (auto& value : second) require(static_cast<bool>(values >> value) && std::isfinite(value), "checkpoint second moment is invalid");
        } else {
            throw NlpTrainingError("unknown NLP checkpoint field");
        }
    }
    require(model_started && !tokenizer_hash.empty() && !dataset_hash.empty() && history.size() == history_count && moments_count > 0U,
            "NLP checkpoint metadata is incomplete");
    while (std::getline(lines, line)) {
        require(line.size() <= maximum_line_bytes, "NLP checkpoint model line exceeds byte budget");
        if (line == "end=1") {
            saw_end = true;
            break;
        }
        model_text += line + "\n";
        require(model_text.size() <= maximum_checkpoint_bytes, "NLP checkpoint model payload exceeds byte budget");
    }
    require(saw_end && !model_text.empty(), "NLP checkpoint model payload or terminator is missing");
    std::istringstream model_stream(model_text);
    const auto model = NextTokenModel::load_model(model_stream);
    if (!expected_tokenizer_hash.empty()) require(tokenizer_hash == expected_tokenizer_hash, "checkpoint tokenizer hash mismatch");
    if (!expected_dataset_hash.empty()) require(dataset_hash == expected_dataset_hash, "checkpoint dataset hash mismatch");
    NlpTrainer trainer(model.config(), optimizer, tokenizer_hash, dataset_hash);
    trainer.model_.set_parameter_vector(model.parameter_vector());
    if (!contract_hash.empty()) require(contract_hash == trainer.training_contract_hash_, "checkpoint training contract hash mismatch");
    require(first.size() == trainer.first_moment_.size() && second.size() == trainer.second_moment_.size(), "checkpoint optimizer state size mismatch");
    trainer.first_moment_ = std::move(first);
    trainer.second_moment_ = std::move(second);
    trainer.state_.optimizer_step = optimizer_step;
    trainer.state_.data_cursor = data_cursor;
    trainer.state_.seed = seed;
    trainer.state_.rng_state = std::move(rng_state);
    trainer.history_ = std::move(history);
    trainer.session_id_ = std::move(session_id);
    trainer.parent_checkpoint_hash_ = std::move(parent_checkpoint_hash);
    trainer.checkpoint_info_.training_contract_hash = trainer.training_contract_hash_;
    trainer.checkpoint_info_.session_id = trainer.session_id_;
    trainer.checkpoint_info_.parent_checkpoint_hash = trainer.parent_checkpoint_hash_;
    trainer.checkpoint_info_.checkpoint_hash = nlp_checkpoint_hash(content);
    trainer.checkpoint_info_.optimizer_step = optimizer_step;
    trainer.checkpoint_info_.data_cursor = data_cursor;
    return trainer;
}

std::string nlp_checkpoint_hash(const std::string& serialized_checkpoint) {
    return GovernedCorpus::content_sha256(serialized_checkpoint);
}

}  // namespace cct
