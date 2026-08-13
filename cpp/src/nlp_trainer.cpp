#include "cct/nlp_trainer.hpp"

#include "cct/corpus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw NlpTrainingError(message);
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
    for (const unsigned char byte : value) {
        result.push_back(digits[byte >> 4U]);
        result.push_back(digits[byte & 0x0fU]);
    }
    return result;
}

std::string hex_decode(const std::string& value) {
    if (value.size() % 2U != 0U) throw NlpTrainingError("hex field has odd length");
    const auto nibble = [](const char value) -> unsigned char {
        if (value >= '0' && value <= '9') return static_cast<unsigned char>(value - '0');
        if (value >= 'a' && value <= 'f') return static_cast<unsigned char>(value - 'a' + 10);
        if (value >= 'A' && value <= 'F') return static_cast<unsigned char>(value - 'A' + 10);
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

std::vector<double> softmax(const std::vector<double>& logits) {
    require(!logits.empty(), "NLP softmax received empty logits");
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size(), 0.0);
    double denominator = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(std::clamp(logits[index] - maximum, -80.0, 80.0));
        denominator += probabilities[index];
    }
    require(std::isfinite(denominator) && denominator > 0.0, "NLP softmax denominator is non-finite");
    for (auto& probability : probabilities) probability /= denominator;
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
    require(context_length >= 2U, "NLP context length must be at least two");
    NlpDataset dataset;
    dataset.tokenizer_hash = tokenizer_hash;
    dataset.context_length = context_length;
    const auto append = [&](const std::vector<EncodedDocument>& documents, std::vector<NlpSequence>& destination,
                            std::size_t& token_count, const char* split) {
        const bool training_split = std::string(split) == "train";
        for (const auto& document : documents) {
            require(!document.record_id.empty() && document.tokens.size() >= 2U, "NLP document is too short");
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
    require(config_.vocabulary_size >= Tokenizer::kByteFirstId + 256U, "NLP vocabulary is smaller than the Stage 10 byte range");
    require(config_.embedding_dim > 0U && config_.hidden_dim > 0U && config_.context_length >= 2U,
            "NLP model dimensions are invalid");
    require(config_.vocabulary_size <= std::numeric_limits<TokenId>::max(), "NLP vocabulary exceeds token ID range");
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
    const auto embedding = vocabulary * input;
    const auto head = vocabulary * hidden + vocabulary;
    if (config_.kind == NlpModelKind::Track1CctRecurrence) return embedding + 4U * hidden * input + 3U * hidden + head;
    if (config_.kind == NlpModelKind::GRU) return embedding + 3U * (hidden * input + hidden * hidden + hidden) + head;
    if (config_.kind == NlpModelKind::DiagonalSSM) return embedding + hidden + hidden * input + head;
    return embedding + 3U * hidden * input + head;
}

void NextTokenModel::initialize() {
    parameters_.assign(expected_parameter_count(), 0.0);
    std::mt19937_64 generator(config_.seed);
    const auto scale = 1.0 / std::sqrt(static_cast<double>(config_.embedding_dim + config_.hidden_dim));
    std::normal_distribution<double> distribution(0.0, scale);
    for (auto& parameter : parameters_) parameter = distribution(generator);
    if (config_.kind == NlpModelKind::Track1CctRecurrence) {
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            parameters_[cct_offset() + 2U * config_.hidden_dim * config_.embedding_dim + index] = 2.0;
        }
    }
}

std::string NextTokenModel::name() const { return nlp_model_kind_name(config_.kind); }

std::vector<double> NextTokenModel::embedding(const TokenId id) const {
    require(id < config_.vocabulary_size, "NLP token ID exceeds model vocabulary");
    return std::vector<double>(parameters_.begin() + static_cast<std::ptrdiff_t>(embedding_offset() + id * config_.embedding_dim),
                               parameters_.begin() + static_cast<std::ptrdiff_t>(embedding_offset() + (id + 1U) * config_.embedding_dim));
}

void NextTokenModel::validate_sequence(const NlpSequence& sequence) const {
    require(!sequence.input_ids.empty() && sequence.input_ids.size() == sequence.target_ids.size() &&
                sequence.input_ids.size() == sequence.loss_mask.size() && sequence.input_ids.size() <= config_.context_length,
            "NLP sequence shape or context length is invalid");
    require(target_count(sequence) > 0U, "NLP sequence has no active loss positions");
    for (const auto id : sequence.input_ids) require(id < config_.vocabulary_size, "NLP input token ID is out of range");
    for (std::size_t index = 0; index < sequence.target_ids.size(); ++index) {
        if (sequence.loss_mask[index] != 0U) require(sequence.target_ids[index] < config_.vocabulary_size, "NLP target token ID is out of range");
    }
}

std::vector<std::vector<double>> forward_track1_cct_recurrence(const std::vector<double>& parameters, const NlpModelConfig& config,
                                             const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto embedding_offset = 0U;
    const auto recurrent_offset = vocabulary * input;
    const auto head_offset = recurrent_offset + 4U * hidden * input + 3U * hidden;
    const auto bias_offset = head_offset + vocabulary * hidden;
    std::vector<double> hidden_state(hidden, 0.0);
    std::vector<double> previous_input(input, 0.0);
    std::vector<std::vector<double>> logits;
    logits.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        const auto x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(embedding_offset + id * input),
                                           parameters.begin() + static_cast<std::ptrdiff_t>(embedding_offset + (id + 1U) * input));
        const auto retain_raw = matvec(parameters, recurrent_offset + 2U * hidden * input, hidden, input, x);
        const auto write_raw = matvec(parameters, recurrent_offset + 3U * hidden * input, hidden, input, x);
        const auto candidate_raw = matvec(parameters, recurrent_offset, hidden, input, x);
        const auto previous_effect = matvec(parameters, recurrent_offset + hidden * input, hidden, input, previous_input);
        for (std::size_t index = 0; index < hidden; ++index) {
            const auto retain = sigmoid(retain_raw[index] + parameters[recurrent_offset + 4U * hidden * input + hidden + index]);
            const auto write = sigmoid(write_raw[index] + parameters[recurrent_offset + 4U * hidden * input + 2U * hidden + index]);
            const auto candidate = std::tanh(candidate_raw[index] + previous_effect[index] + parameters[recurrent_offset + 4U * hidden * input + index]);
            hidden_state[index] = retain * hidden_state[index] + write * candidate;
        }
        std::vector<double> output(vocabulary, 0.0);
        for (std::size_t token = 0; token < vocabulary; ++token) {
            output[token] = parameters[ bias_offset + token];
            for (std::size_t index = 0; index < hidden; ++index) output[token] += parameters[head_offset + token * hidden + index] * hidden_state[index];
        }
        logits.push_back(std::move(output));
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
    std::vector<std::vector<double>> logits;
    logits.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        const auto x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(id * input),
                                           parameters.begin() + static_cast<std::ptrdiff_t>((id + 1U) * input));
        const auto gate = [&](const std::size_t offset) {
            auto result = matvec(parameters, offset, hidden, input, x);
            const auto recurrent = matvec(parameters, offset + hidden * input, hidden, hidden, hidden_state);
            for (std::size_t index = 0; index < hidden; ++index) result[index] = sigmoid(result[index] + recurrent[index] + parameters[offset + hidden * input + hidden * hidden + index]);
            return result;
        };
        const auto z = gate(z_offset);
        const auto r = gate(r_offset);
        auto candidate = matvec(parameters, n_offset, hidden, input, x);
        const auto recurrent = matvec(parameters, n_offset + hidden * input, hidden, hidden, hidden_state);
        for (std::size_t index = 0; index < hidden; ++index) candidate[index] = std::tanh(candidate[index] + r[index] * recurrent[index] + parameters[n_offset + hidden * input + hidden * hidden + index]);
        for (std::size_t index = 0; index < hidden; ++index) hidden_state[index] = z[index] * hidden_state[index] + (1.0 - z[index]) * candidate[index];
        std::vector<double> output(vocabulary, 0.0);
        for (std::size_t token = 0; token < vocabulary; ++token) {
            output[token] = parameters[bias_offset + token];
            for (std::size_t index = 0; index < hidden; ++index) output[token] += parameters[head_offset + token * hidden + index] * hidden_state[index];
        }
        logits.push_back(std::move(output));
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
    std::vector<std::vector<double>> logits;
    logits.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        const auto x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(id * input),
                                           parameters.begin() + static_cast<std::ptrdiff_t>((id + 1U) * input));
        const auto effect = matvec(parameters, input_offset, hidden, input, x);
        for (std::size_t index = 0; index < hidden; ++index) hidden_state[index] = 0.999 * sigmoid(parameters[recurrent_offset + index]) * hidden_state[index] + effect[index];
        std::vector<double> output(vocabulary, 0.0);
        for (std::size_t token = 0; token < vocabulary; ++token) {
            output[token] = parameters[bias_offset + token];
            for (std::size_t index = 0; index < hidden; ++index) output[token] += parameters[head_offset + token * hidden + index] * hidden_state[index];
        }
        logits.push_back(std::move(output));
    }
    return logits;
}

std::vector<std::vector<double>> forward_dense(const std::vector<double>& parameters, const NlpModelConfig& config,
                                               const NlpSequence& sequence) {
    const auto input = config.embedding_dim;
    const auto hidden = config.hidden_dim;
    const auto vocabulary = config.vocabulary_size;
    const auto embedding_offset = 0U;
    const auto attention_offset = vocabulary * input;
    const auto q_offset = attention_offset;
    const auto k_offset = q_offset + hidden * input;
    const auto v_offset = k_offset + hidden * input;
    const auto head_offset = v_offset + hidden * input;
    const auto bias_offset = head_offset + vocabulary * hidden;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> logits;
    inputs.reserve(sequence.input_ids.size());
    logits.reserve(sequence.input_ids.size());
    for (const auto id : sequence.input_ids) {
        inputs.emplace_back(parameters.begin() + static_cast<std::ptrdiff_t>(embedding_offset + id * input),
                            parameters.begin() + static_cast<std::ptrdiff_t>(embedding_offset + (id + 1U) * input));
        const auto time = inputs.size() - 1U;
        const auto query = matvec(parameters, q_offset, hidden, input, inputs[time]);
        std::vector<double> scores(time + 1U, 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t position = 0; position <= time; ++position) {
            const auto key = matvec(parameters, k_offset, hidden, input, inputs[position]);
            scores[position] = std::inner_product(query.begin(), query.end(), key.begin(), 0.0) /
                               std::sqrt(static_cast<double>(hidden));
            maximum = std::max(maximum, scores[position]);
        }
        double denominator = 0.0;
        for (auto& score : scores) {
            score = std::exp(std::clamp(score - maximum, -80.0, 80.0));
            denominator += score;
        }
        require(std::isfinite(denominator) && denominator > 0.0, "dense attention denominator is non-finite");
        std::vector<double> context(hidden, 0.0);
        for (std::size_t position = 0; position <= time; ++position) {
            const auto value = matvec(parameters, v_offset, hidden, input, inputs[position]);
            for (std::size_t index = 0; index < hidden; ++index) context[index] += scores[position] / denominator * value[index];
        }
        std::vector<double> output(vocabulary, 0.0);
        for (std::size_t token = 0; token < vocabulary; ++token) {
            output[token] = parameters[bias_offset + token];
            for (std::size_t index = 0; index < hidden; ++index) output[token] += parameters[head_offset + token * hidden + index] * context[index];
        }
        logits.push_back(std::move(output));
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

double cross_entropy_from_logits(const std::vector<std::vector<double>>& logits, const NlpSequence& sequence,
                                 std::size_t* token_count, double* accuracy) {
    require(logits.size() == sequence.target_ids.size(), "NLP logits/targets length mismatch");
    double loss = 0.0;
    std::size_t count = 0;
    std::size_t correct = 0;
    for (std::size_t time = 0; time < logits.size(); ++time) {
        if (sequence.loss_mask[time] == 0U) continue;
        const auto probabilities = softmax(logits[time]);
        const auto target = sequence.target_ids[time];
        require(target < probabilities.size(), "NLP target is outside logits");
        loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
        const auto prediction = static_cast<TokenId>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
        if (prediction == target) ++correct;
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
        item.x = std::vector<double>(parameters.begin() + static_cast<std::ptrdiff_t>(id * input),
                                     parameters.begin() + static_cast<std::ptrdiff_t>((id + 1U) * input));
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
            const auto target = sequence.target_ids[reverse];
            loss -= std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
            if (static_cast<TokenId>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))) == target) ++correct;
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
        const auto offset = static_cast<std::size_t>(sequence.input_ids[time]) * input;
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

NlpGradientResult NextTokenModel::loss_and_gradients(const NlpSequence& sequence) const {
    validate_sequence(sequence);
    if (config_.kind == NlpModelKind::Track1CctRecurrence) return track1_cct_gradients(parameters_, config_, sequence);
    const auto base_loss = loss_only(sequence);
    const auto original = parameters_;
    std::vector<double> gradients(original.size(), 0.0);
    constexpr double epsilon = 1e-5;
    for (std::size_t index = 0; index < original.size(); ++index) {
        auto plus = original;
        auto minus = original;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        NextTokenModel plus_model(config_);
        plus_model.set_parameter_vector(plus);
        NextTokenModel minus_model(config_);
        minus_model.set_parameter_vector(minus);
        gradients[index] = (plus_model.loss_only(sequence) - minus_model.loss_only(sequence)) / (2.0 * epsilon);
    }
    require_finite(gradients, "baseline finite-difference gradient became non-finite");
    return {base_loss, static_cast<std::size_t>(target_count(sequence)), vector_norm(gradients), std::move(gradients)};
}

double NextTokenModel::loss_only(const NlpSequence& sequence) const {
    validate_sequence(sequence);
    const auto logits = model_forward(parameters_, config_, sequence);
    return cross_entropy_from_logits(logits, sequence, nullptr, nullptr);
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
        const auto loss = cross_entropy_from_logits(logits, sequence, &count, &accuracy);
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

void NextTokenModel::set_parameter_vector(const std::vector<double>& values) {
    require(values.size() == expected_parameter_count(), "NLP parameter vector size mismatch");
    require_finite(values, "NLP parameter vector contains non-finite values");
    parameters_ = values;
}

std::size_t NextTokenModel::state_memory_bytes() const noexcept {
    if (config_.kind == NlpModelKind::DenseCausalAttention) return (2U * config_.context_length * config_.hidden_dim + config_.hidden_dim) * sizeof(double);
    return config_.hidden_dim * sizeof(double);
}

void NextTokenModel::apply_gradient(const std::vector<double>& gradients, const NlpOptimizerConfig& optimizer,
                                    NlpTrainerState& state, double* applied_learning_rate) {
    require(gradients.size() == parameters_.size(), "NLP gradient vector size mismatch");
    require(optimizer.learning_rate > 0.0 && optimizer.clip_norm > 0.0, "NLP optimizer settings are invalid");
    const auto norm = vector_norm(gradients);
    require(std::isfinite(norm), "NLP gradient norm is non-finite");
    const auto scale = std::min(1.0, optimizer.clip_norm / std::max(norm, 1e-12));
    const auto learning_rate = optimizer.learning_rate * scale;
    for (std::size_t index = 0; index < parameters_.size(); ++index) parameters_[index] -= learning_rate * gradients[index];
    state.optimizer_step += 1U;
    if (applied_learning_rate != nullptr) *applied_learning_rate = learning_rate;
}

void NextTokenModel::save_model(std::ostream& stream) const {
    stream << "NLP_MODEL_V3\n" << model_kind_number(config_.kind) << ' ' << config_.vocabulary_size << ' '
           << config_.embedding_dim << ' ' << config_.hidden_dim << ' ' << config_.context_length << ' ' << config_.seed << '\n'
           << parameters_.size() << '\n' << std::setprecision(17);
    for (const auto value : parameters_) stream << value << ' ';
    stream << '\n';
}

NextTokenModel NextTokenModel::load_model(std::istream& stream) {
    std::string header;
    require(static_cast<bool>(stream >> header) && (header == "NLP_MODEL_V2" || header == "NLP_MODEL_V3"),
            "invalid NLP model checkpoint header");
    std::size_t kind = 0;
    NlpModelConfig config;
    require(static_cast<bool>(stream >> kind >> config.vocabulary_size >> config.embedding_dim >> config.hidden_dim >>
                              config.context_length >> config.seed),
            "NLP model checkpoint configuration is incomplete");
    config.kind = model_kind_from_number(kind);
    NextTokenModel model(config);
    std::size_t parameter_count = 0;
    require(static_cast<bool>(stream >> parameter_count) && parameter_count == model.parameter_count(),
            "NLP model checkpoint parameter count is invalid");
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
    state_.seed = model_.config().seed;
    state_.rng_state = "seed=" + std::to_string(state_.seed);
    checkpoint_info_.tokenizer_hash = tokenizer_hash_;
    checkpoint_info_.dataset_hash = dataset_hash_;
}

void NlpTrainer::validate_optimizer() const {
    require(optimizer_config_.learning_rate > 0.0 && optimizer_config_.beta1 >= 0.0 && optimizer_config_.beta1 < 1.0 &&
                optimizer_config_.beta2 >= 0.0 && optimizer_config_.beta2 < 1.0 && optimizer_config_.epsilon > 0.0 &&
                optimizer_config_.clip_norm > 0.0 && optimizer_config_.total_steps > 0U,
            "NLP optimizer configuration is invalid");
}

double NlpTrainer::scheduled_learning_rate() const {
    const auto step = state_.optimizer_step + 1U;
    if (optimizer_config_.warmup_steps > 0U && step <= optimizer_config_.warmup_steps) {
        return optimizer_config_.learning_rate * static_cast<double>(step) / static_cast<double>(optimizer_config_.warmup_steps);
    }
    if (optimizer_config_.total_steps <= optimizer_config_.warmup_steps) return optimizer_config_.learning_rate;
    const auto remaining = optimizer_config_.total_steps - std::min(step, optimizer_config_.total_steps);
    const auto span = optimizer_config_.total_steps - optimizer_config_.warmup_steps;
    return optimizer_config_.learning_rate * static_cast<double>(remaining) / static_cast<double>(span);
}

NlpEvaluation NlpTrainer::evaluate(const std::vector<NlpSequence>& sequences) const { return model_.evaluate(sequences); }

NlpTrainingPoint NlpTrainer::train_step(const NlpDataset& dataset) {
    require(dataset.dataset_hash == dataset_hash_ && dataset.tokenizer_hash == tokenizer_hash_, "NLP dataset identity mismatch");
    require(!dataset.train.empty(), "NLP training dataset is empty");
    const auto& sequence = dataset.train[state_.data_cursor % dataset.train.size()];
    const auto gradient_result = model_.loss_and_gradients(sequence);
    require(std::isfinite(gradient_result.cross_entropy) && std::isfinite(gradient_result.gradient_norm), "NLP training objective is non-finite");
    require(gradient_result.token_count > 0U, "NLP training sequence has no active tokens");
    auto parameters = model_.parameter_vector();
    const auto gradient_norm = gradient_result.gradient_norm;
    const auto scale = std::min(1.0, optimizer_config_.clip_norm / std::max(gradient_norm, 1e-12));
    const auto learning_rate = scheduled_learning_rate();
    const auto step = state_.optimizer_step + 1U;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const auto gradient = gradient_result.gradients[index] * scale;
        first_moment_[index] = optimizer_config_.beta1 * first_moment_[index] + (1.0 - optimizer_config_.beta1) * gradient;
        second_moment_[index] = optimizer_config_.beta2 * second_moment_[index] + (1.0 - optimizer_config_.beta2) * gradient * gradient;
        const auto first_correction = 1.0 - std::pow(optimizer_config_.beta1, static_cast<double>(step));
        const auto second_correction = 1.0 - std::pow(optimizer_config_.beta2, static_cast<double>(step));
        const auto first_hat = first_moment_[index] / std::max(first_correction, 1e-12);
        const auto second_hat = second_moment_[index] / std::max(second_correction, 1e-12);
        parameters[index] -= learning_rate * (first_hat / (std::sqrt(second_hat) + optimizer_config_.epsilon) + optimizer_config_.weight_decay * parameters[index]);
    }
    require_finite(parameters, "NLP optimizer produced non-finite parameters");
    model_.set_parameter_vector(parameters);
    state_.optimizer_step = step;
    state_.data_cursor += 1U;
    NlpTrainingPoint point;
    point.step = state_.optimizer_step;
    point.data_cursor = state_.data_cursor;
    point.learning_rate = learning_rate;
    point.train_loss = gradient_result.cross_entropy;
    point.token_count = gradient_result.token_count;
    point.gradient_norm = gradient_norm;
    const auto validation = model_.evaluate(dataset.validation);
    point.validation_loss = validation.cross_entropy;
    point.validation_perplexity = validation.perplexity;
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

void NlpTrainer::validate_checkpoint_identity(const std::string& tokenizer_hash, const std::string& dataset_hash) const {
    require(tokenizer_hash == tokenizer_hash_ && dataset_hash == dataset_hash_, "NLP checkpoint identity mismatch");
}

void NlpTrainer::save_checkpoint(const std::string& path) const {
    std::ostringstream serialized;
    serialized << "CCT_NLP_CHECKPOINT_V2\n";
    serialized << "tokenizer_hash=" << hex_encode(tokenizer_hash_) << "\n";
    serialized << "dataset_hash=" << hex_encode(dataset_hash_) << "\n";
    serialized << "optimizer_step=" << state_.optimizer_step << "\n";
    serialized << "data_cursor=" << state_.data_cursor << "\n";
    serialized << "seed=" << state_.seed << "\n";
    serialized << "rng_state=" << hex_encode(state_.rng_state) << "\n";
    serialized << "optimizer=" << std::setprecision(17) << optimizer_config_.learning_rate << ' ' << optimizer_config_.beta1 << ' '
               << optimizer_config_.beta2 << ' ' << optimizer_config_.epsilon << ' ' << optimizer_config_.weight_decay << ' '
               << optimizer_config_.clip_norm << ' ' << optimizer_config_.warmup_steps << ' ' << optimizer_config_.total_steps << "\n";
    serialized << "history_count=" << history_.size() << "\n";
    for (const auto& point : history_) {
        serialized << "history=" << point.step << ' ' << point.data_cursor << ' ' << point.learning_rate << ' '
                   << point.train_loss << ' ' << point.validation_loss << ' ' << point.validation_perplexity << ' '
                   << point.gradient_norm << ' ' << point.token_count << "\n";
    }
    serialized << "moments_count=" << first_moment_.size() << "\n";
    serialized << "first_moment=";
    for (const auto value : first_moment_) serialized << value << ' ';
    serialized << "\nsecond_moment=";
    for (const auto value : second_moment_) serialized << value << ' ';
    serialized << "\nmodel_begin\n";
    model_.save_model(serialized);
    serialized << "end=1\n";
    const auto content = serialized.str();
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not write NLP checkpoint");
    stream << content;
    require(static_cast<bool>(stream), "could not finish NLP checkpoint");
    checkpoint_info_.checkpoint_hash = nlp_checkpoint_hash(content);
    checkpoint_info_.optimizer_step = state_.optimizer_step;
    checkpoint_info_.data_cursor = state_.data_cursor;
}

NlpTrainer NlpTrainer::load_checkpoint(const std::string& path, const std::string& expected_tokenizer_hash,
                                       const std::string& expected_dataset_hash) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "could not read NLP checkpoint");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto content = buffer.str();
    std::istringstream lines(content);
    std::string line;
    require(static_cast<bool>(std::getline(lines, line)) && line == "CCT_NLP_CHECKPOINT_V2", "unsupported NLP checkpoint header");
    std::string tokenizer_hash;
    std::string dataset_hash;
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
    while (std::getline(lines, line)) {
        if (line == "model_begin") {
            model_started = true;
            break;
        }
        if (line.rfind("tokenizer_hash=", 0) == 0) tokenizer_hash = hex_decode(line.substr(15));
        else if (line.rfind("dataset_hash=", 0) == 0) dataset_hash = hex_decode(line.substr(13));
        else if (line.rfind("optimizer_step=", 0) == 0) optimizer_step = parse_size(line.substr(15), "optimizer_step");
        else if (line.rfind("data_cursor=", 0) == 0) data_cursor = parse_size(line.substr(12), "data_cursor");
        else if (line.rfind("seed=", 0) == 0) seed = static_cast<std::uint64_t>(parse_size(line.substr(5), "seed"));
        else if (line.rfind("rng_state=", 0) == 0) rng_state = hex_decode(line.substr(10));
        else if (line.rfind("optimizer=", 0) == 0) {
            std::istringstream values(line.substr(10));
            require(static_cast<bool>(values >> optimizer.learning_rate >> optimizer.beta1 >> optimizer.beta2 >> optimizer.epsilon >>
                                      optimizer.weight_decay >> optimizer.clip_norm >> optimizer.warmup_steps >> optimizer.total_steps),
                    "checkpoint optimizer fields are incomplete");
        } else if (line.rfind("history_count=", 0) == 0) {
            history_count = parse_size(line.substr(14), "history_count");
            history.reserve(history_count);
        } else if (line.rfind("history=", 0) == 0) {
            std::istringstream values(line.substr(8));
            NlpTrainingPoint point;
            require(static_cast<bool>(values >> point.step >> point.data_cursor >> point.learning_rate >> point.train_loss >>
                                      point.validation_loss >> point.validation_perplexity >> point.gradient_norm >> point.token_count),
                    "checkpoint history row is incomplete");
            history.push_back(point);
        } else if (line.rfind("moments_count=", 0) == 0) {
            moments_count = parse_size(line.substr(14), "moments_count");
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
        if (line == "end=1") break;
        model_text += line + "\n";
    }
    require(!model_text.empty(), "NLP checkpoint model payload is missing");
    std::istringstream model_stream(model_text);
    const auto model = NextTokenModel::load_model(model_stream);
    if (!expected_tokenizer_hash.empty()) require(tokenizer_hash == expected_tokenizer_hash, "checkpoint tokenizer hash mismatch");
    if (!expected_dataset_hash.empty()) require(dataset_hash == expected_dataset_hash, "checkpoint dataset hash mismatch");
    NlpTrainer trainer(model.config(), optimizer, tokenizer_hash, dataset_hash);
    trainer.model_.set_parameter_vector(model.parameter_vector());
    require(first.size() == trainer.first_moment_.size() && second.size() == trainer.second_moment_.size(), "checkpoint optimizer state size mismatch");
    trainer.first_moment_ = std::move(first);
    trainer.second_moment_ = std::move(second);
    trainer.state_.optimizer_step = optimizer_step;
    trainer.state_.data_cursor = data_cursor;
    trainer.state_.seed = seed;
    trainer.state_.rng_state = std::move(rng_state);
    trainer.history_ = std::move(history);
    trainer.checkpoint_info_.checkpoint_hash = nlp_checkpoint_hash(content);
    trainer.checkpoint_info_.optimizer_step = optimizer_step;
    trainer.checkpoint_info_.data_cursor = data_cursor;
    return trainer;
}

std::string nlp_checkpoint_hash(const std::string& serialized_checkpoint) {
    return GovernedCorpus::content_sha256(serialized_checkpoint);
}

}  // namespace cct
