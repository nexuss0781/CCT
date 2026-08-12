#include "cct/sft.hpp"

#include "cct/corpus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw SftError(message);
}

std::string hex_encode(const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

std::string hex_decode(const std::string& value) {
    require(value.size() % 2U == 0U, "invalid hex field length");
    auto nibble = [](const char character) -> unsigned char {
        if (character >= '0' && character <= '9') return static_cast<unsigned char>(character - '0');
        if (character >= 'a' && character <= 'f') return static_cast<unsigned char>(character - 'a' + 10);
        if (character >= 'A' && character <= 'F') return static_cast<unsigned char>(character - 'A' + 10);
        throw SftError("invalid hex field character");
    };
    std::string output;
    output.reserve(value.size() / 2U);
    for (std::size_t index = 0; index < value.size(); index += 2U) {
        output.push_back(static_cast<char>((nibble(value[index]) << 4U) | nibble(value[index + 1U])));
    }
    return output;
}

std::vector<std::string> split(const std::string& value, const char delimiter) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        fields.push_back(value.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1U;
    }
    return fields;
}

std::string join_hex(const std::vector<std::string>& fields) {
    std::ostringstream output;
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index != 0U) output << '|';
        output << hex_encode(fields[index]);
    }
    return output.str();
}

std::string field(const std::vector<std::string>& fields, const std::size_t index) {
    require(index < fields.size(), "serialized SFT record is truncated");
    return hex_decode(fields[index]);
}

std::vector<double> softmax(const std::vector<double>& logits) {
    require(!logits.empty(), "cannot normalize empty SFT logits");
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size(), 0.0);
    double total = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        total += probabilities[index];
    }
    require(total > 0.0 && std::isfinite(total), "SFT softmax normalization failed");
    for (auto& probability : probabilities) probability /= total;
    return probabilities;
}

std::size_t positive_mod(const std::uint64_t value, const std::size_t modulus) {
    require(modulus > 0U, "modulus must be positive");
    return static_cast<std::size_t>(value % modulus);
}

std::string bool_text(const bool value) { return value ? "1" : "0"; }

bool parse_bool(const std::string& value) {
    require(value == "0" || value == "1", "invalid serialized boolean");
    return value == "1";
}

std::string prediction_label(const std::vector<double>& probabilities, const SftTaskSchema& schema) {
    const auto index = static_cast<std::size_t>(std::distance(probabilities.begin(),
        std::max_element(probabilities.begin(), probabilities.end())));
    require(index < schema.labels.size(), "schema label count does not match model");
    return schema.labels[index];
}

}  // namespace

std::string sft_task_kind_name(const SftTaskKind kind) {
    switch (kind) {
        case SftTaskKind::Classification: return "classification";
        case SftTaskKind::StructuredExtraction: return "structured_extraction";
        case SftTaskKind::GroundedQuestionAnswering: return "grounded_qa";
        case SftTaskKind::Summarization: return "summarization";
        case SftTaskKind::CodeUnderstanding: return "code_understanding";
        case SftTaskKind::WorkflowDrafting: return "workflow_drafting";
    }
    throw SftError("unknown SFT task kind");
}

std::string sft_output_kind_name(const SftOutputKind kind) {
    switch (kind) {
        case SftOutputKind::Label: return "label";
        case SftOutputKind::Json: return "json";
        case SftOutputKind::Grounded: return "grounded";
        case SftOutputKind::BoundedText: return "bounded_text";
        case SftOutputKind::Draft: return "draft";
    }
    throw SftError("unknown SFT output kind");
}

std::string sft_hash(const std::string& serialized) { return GovernedCorpus::content_sha256(serialized); }

SftManifest SftManifest::build(const std::vector<SftInstructionExample>& examples,
                               const std::vector<SftTaskSchema>& schemas) {
    require(!examples.empty(), "SFT manifest cannot be empty");
    require(!schemas.empty(), "SFT schema registry cannot be empty");
    std::map<std::string, SftTaskSchema> schema_map;
    for (const auto& schema : schemas) {
        require(!schema.task_id.empty() && !schema.schema_version.empty() && !schema.labels.empty(),
                "SFT schema has missing identity or labels");
        require(schema_map.emplace(schema.task_id, schema).second, "duplicate SFT task schema");
    }
    SftManifest manifest;
    manifest.examples = examples;
    for (auto& example : manifest.examples) {
        require(schema_map.contains(example.task_id), "SFT example references unknown task");
        const auto& schema = schema_map.at(example.task_id);
        require(example.schema_version == schema.schema_version && !example.example_id.empty() && !example.input.empty() &&
                    !example.target.empty() && !example.input_provenance.empty() && !example.target_provenance.empty() &&
                    !example.policy_class.empty() && !example.split.empty() && !example.evaluator_owner.empty(),
                "SFT example is missing required provenance or content");
        require(example.source_hash == sft_hash(example.input) && example.target_hash == sft_hash(example.target),
                "SFT example source/target hash mismatch");
        const auto canonical = example.example_id + "\n" + example.task_id + "\n" + example.schema_version + "\n" +
                               example.input + "\n" + example.target + "\n" + example.target_label + "\n" +
                               example.source_hash + "\n" + example.target_hash + "\n" + example.split;
        require(example.example_hash == sft_hash(canonical), "SFT example hash mismatch");
        require(!(example.evaluator_only && example.training_allowed), "evaluator-only example is training-eligible");
        require(!(example.evaluator_only && example.evaluation_allowed), "evaluator-only example is evaluation-eligible");
        require(example.training_allowed || example.evaluation_allowed, "SFT example has no declared eligibility");
    }
    std::ostringstream serialized;
    serialized << "CCT_SFT_MANIFEST_V1\n";
    for (const auto& example : manifest.examples) {
        serialized << "E|" << join_hex({example.example_id, example.task_id, example.schema_version, example.input, example.target,
                                         example.target_label, example.input_provenance, example.target_provenance, example.policy_class,
                                         example.split, example.evaluator_owner, example.source_hash, example.target_hash, example.example_hash,
                                         example.citation_id, std::to_string(example.source_span_start), std::to_string(example.source_span_end),
                                         bool_text(example.training_allowed), bool_text(example.evaluation_allowed), bool_text(example.evaluator_only)}) << "\n";
    }
    manifest.manifest_hash = sft_hash(serialized.str());
    return manifest;
}

std::vector<SftInstructionExample> SftManifest::training_examples() const {
    std::vector<SftInstructionExample> output;
    for (const auto& example : examples) if (example.training_allowed && !example.evaluator_only) output.push_back(example);
    return output;
}

std::vector<SftInstructionExample> SftManifest::evaluation_examples() const {
    std::vector<SftInstructionExample> output;
    for (const auto& example : examples) if (example.evaluation_allowed && !example.evaluator_only) output.push_back(example);
    return output;
}

bool SftManifest::contains_evaluator_training() const {
    for (const auto& example : examples) if (example.evaluator_only && example.training_allowed) return true;
    return false;
}

std::string SftManifest::serialize() const {
    std::ostringstream serialized;
    serialized << "CCT_SFT_MANIFEST_V1\n" << "H|" << hex_encode(manifest_hash) << "\n";
    for (const auto& example : examples) {
        serialized << "E|" << join_hex({example.example_id, example.task_id, example.schema_version, example.input, example.target,
                                         example.target_label, example.input_provenance, example.target_provenance, example.policy_class,
                                         example.split, example.evaluator_owner, example.source_hash, example.target_hash, example.example_hash,
                                         example.citation_id, std::to_string(example.source_span_start), std::to_string(example.source_span_end),
                                         bool_text(example.training_allowed), bool_text(example.evaluation_allowed), bool_text(example.evaluator_only)}) << "\n";
    }
    return serialized.str();
}

SftManifest SftManifest::deserialize(const std::string& serialized) {
    std::istringstream input(serialized);
    std::string line;
    require(std::getline(input, line) && line == "CCT_SFT_MANIFEST_V1", "unsupported SFT manifest version");
    SftManifest manifest;
    std::string declared_hash;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split(line, '|');
        require(!fields.empty(), "malformed SFT manifest line");
        if (fields[0] == "H") {
            require(fields.size() == 2U, "malformed SFT manifest header");
            declared_hash = hex_decode(fields[1]);
        } else if (fields[0] == "E") {
            require(fields.size() == 21U, "malformed SFT example field count");
            std::vector<std::string> values(fields.begin() + 1, fields.end());
            SftInstructionExample example;
            example.example_id = field(values, 0); example.task_id = field(values, 1); example.schema_version = field(values, 2);
            example.input = field(values, 3); example.target = field(values, 4); example.target_label = field(values, 5);
            example.input_provenance = field(values, 6); example.target_provenance = field(values, 7); example.policy_class = field(values, 8);
            example.split = field(values, 9); example.evaluator_owner = field(values, 10); example.source_hash = field(values, 11);
            example.target_hash = field(values, 12); example.example_hash = field(values, 13); example.citation_id = field(values, 14);
            example.source_span_start = static_cast<std::size_t>(std::stoull(field(values, 15)));
            example.source_span_end = static_cast<std::size_t>(std::stoull(field(values, 16)));
            example.training_allowed = parse_bool(field(values, 17)); example.evaluation_allowed = parse_bool(field(values, 18));
            example.evaluator_only = parse_bool(field(values, 19));
            manifest.examples.push_back(std::move(example));
        } else {
            throw SftError("unknown SFT manifest record");
        }
    }
    require(!manifest.examples.empty() && !declared_hash.empty(), "SFT manifest has no records or hash");
    manifest.manifest_hash = declared_hash;
    require(manifest.serialize() == serialized, "SFT manifest serialization is not canonical");
    return manifest;
}

FormattedInstruction SftFormatter::format(const SftInstructionExample& example, const SftTaskSchema& schema,
                                          const Tokenizer& tokenizer) {
    require(example.task_id == schema.task_id && example.schema_version == schema.schema_version,
            "SFT formatter schema identity mismatch");
    const std::string serialized = "<CCT_TASK_V1> task=" + example.task_id + " schema=" + schema.schema_version +
                                   " input=" + example.input + " <TARGET> " + example.target + " <END>";
    const auto target_marker = std::string("<TARGET> ");
    const auto target_start = serialized.find(target_marker) + target_marker.size();
    const auto target_end = target_start + example.target.size();
    const auto encoded = tokenizer.encode(serialized, example.example_id, false);
    FormattedInstruction formatted;
    formatted.example_id = example.example_id;
    formatted.serialized = serialized;
    formatted.target_token_start = encoded.tokens.size();
    formatted.target_token_end = 0;
    for (const auto& token : encoded.tokens) {
        formatted.token_ids.push_back(token.id);
        const bool active = token.kind != TokenKind::Control && token.source_start < target_end && token.source_end > target_start;
        formatted.loss_mask.push_back(active ? 1U : 0U);
        if (active) {
            formatted.target_token_start = std::min(formatted.target_token_start, formatted.token_ids.size() - 1U);
            formatted.target_token_end = formatted.token_ids.size();
        }
    }
    require(formatted.target_token_end > formatted.target_token_start, "SFT target produced no trainable tokens");
    require(formatted.serialized.substr(target_start, example.target.size()) == example.target, "SFT target span mismatch");
    return formatted;
}

std::string SftFormatter::mask_policy_name() { return "target-span-only-v1"; }

SftModel::SftModel(SftModelConfig config) : config_(std::move(config)) {
    validate();
    parameters_.assign(config_.label_count * config_.feature_dim + config_.label_count, 0.0);
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const auto mixed = (config_.seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(index + 1U));
        parameters_[index] = (static_cast<double>(positive_mod(mixed ^ (mixed >> 29U), 2001U)) / 1000.0 - 1.0) * 0.02;
    }
}

void SftModel::validate() const {
    require(!config_.base_checkpoint_hash.empty() && !config_.task_id.empty() && config_.feature_dim > 0U &&
                config_.label_count > 1U, "invalid SFT model configuration");
}

void SftModel::set_parameter_vector(const std::vector<double>& values) {
    require(values.size() == parameters_.size(), "SFT model parameter vector size mismatch");
    require(std::all_of(values.begin(), values.end(), [](const double value) { return std::isfinite(value); }),
            "SFT model parameter vector contains non-finite value");
    parameters_ = values;
}

std::string SftModel::parameter_checksum() const {
    std::ostringstream serialized;
    serialized << name() << '|' << config_.base_checkpoint_hash << '|' << config_.task_id << '|' << config_.feature_dim << '|'
               << config_.label_count << '|' << config_.seed << '|';
    serialized << std::setprecision(17);
    for (const auto value : parameters_) serialized << value << ',';
    return sft_hash(serialized.str());
}

std::string SftModel::name() const { return "sft-full-" + config_.task_id; }

std::vector<double> SftModel::features(const std::string& input) const {
    std::vector<double> result(config_.feature_dim, 0.0);
    result[0] = 1.0;
    if (config_.feature_dim > 1U) result[1] = std::min(4.0, static_cast<double>(input.size()) / 16.0);
    if (config_.feature_dim > 2U) {
        std::size_t letters = 0;
        for (const unsigned char byte : input) if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z')) ++letters;
        result[2] = input.empty() ? 0.0 : static_cast<double>(letters) / static_cast<double>(input.size());
    }
    if (config_.feature_dim > 3U) {
        std::size_t digits = 0;
        for (const unsigned char byte : input) if (byte >= '0' && byte <= '9') ++digits;
        result[3] = input.empty() ? 0.0 : static_cast<double>(digits) / static_cast<double>(input.size());
    }
    if (config_.feature_dim > 4U) result[4] = input.find("positive") != std::string::npos || input.find("invoice") != std::string::npos ? 1.0 : (input.find('{') != std::string::npos || input.find('[') != std::string::npos ? 0.5 : 0.0);
    if (config_.feature_dim > 5U) result[5] = input.find("negative") != std::string::npos ? 1.0 : (input.find('?') != std::string::npos ? 1.0 : 0.0);
    if (config_.feature_dim > 6U) result[6] = input.find("deny") != std::string::npos || input.find("secret") != std::string::npos ? 1.0 : 0.0;
    if (config_.feature_dim > 7U) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const unsigned char byte : input) { hash ^= byte; hash *= 1099511628211ULL; }
        result[7] = static_cast<double>(hash % 1000U) / 1000.0;
    }
    return result;
}

std::vector<double> SftModel::logits(const std::vector<double>& feature_values, const SftAdapter* adapter) const {
    require(feature_values.size() == config_.feature_dim, "SFT feature dimension mismatch");
    std::vector<double> output(config_.label_count, 0.0);
    for (std::size_t label = 0; label < config_.label_count; ++label) {
        for (std::size_t feature = 0; feature < config_.feature_dim; ++feature) {
            double weight = parameters_[label * config_.feature_dim + feature];
            if (adapter != nullptr) {
                const auto rank = adapter->spec().rank;
                for (std::size_t component = 0; component < rank; ++component) {
                    const auto adapter_offset = label * rank + component;
                    const auto feature_offset = config_.label_count * rank + component * config_.feature_dim + feature;
                    weight += adapter->parameter_vector()[adapter_offset] * adapter->parameter_vector()[feature_offset];
                }
            }
            output[label] += weight * feature_values[feature];
        }
        output[label] += parameters_[config_.label_count * config_.feature_dim + label];
    }
    return output;
}

std::size_t SftModel::label_index(const std::string& label, const SftTaskSchema& schema) {
    const auto found = std::find(schema.labels.begin(), schema.labels.end(), label);
    require(found != schema.labels.end(), "SFT target label is outside closed schema");
    return static_cast<std::size_t>(std::distance(schema.labels.begin(), found));
}

std::string SftModel::json_output(const SftInstructionExample& example, const SftTaskSchema& schema,
                                  const std::string& label, const double confidence) {
    std::ostringstream output;
    if (schema.output_kind == SftOutputKind::Json) {
        output << "{\"task_id\":\"" << schema.task_id << "\",\"label\":\"" << label << "\",\"confidence\":"
               << std::setprecision(8) << confidence << ",\"source_start\":" << example.source_span_start
               << ",\"source_end\":" << example.source_span_end << "}";
    } else if (schema.output_kind == SftOutputKind::Grounded) {
        output << "{\"answer\":\"" << label << "\",\"citation\":\"" << example.citation_id
               << "\",\"uncertainty\":\"bounded\"}";
    } else if (schema.output_kind == SftOutputKind::Draft) {
        output << "{\"draft\":\"" << label << "\",\"approval_required\":true}";
    } else {
        output << label;
    }
    return output.str();
}

SftPrediction SftModel::predict(const SftInstructionExample& example, const SftTaskSchema& schema,
                                const SftAdapter* adapter) const {
    require(example.task_id == config_.task_id && schema.task_id == config_.task_id, "SFT prediction task identity mismatch");
    const auto probabilities = softmax(logits(features(example.input), adapter));
    const auto label = prediction_label(probabilities, schema);
    const auto confidence = *std::max_element(probabilities.begin(), probabilities.end());
    SftPrediction prediction;
    prediction.task_id = schema.task_id;
    prediction.label = label;
    prediction.confidence = confidence;
    prediction.citation_id = example.citation_id;
    prediction.output = json_output(example, schema, label, confidence);
    prediction.schema_valid = schema.output_kind == SftOutputKind::Label ||
                              (prediction.output.size() <= schema.maximum_output_bytes && prediction.output.front() == '{' && prediction.output.back() == '}');
    prediction.citation_valid = !schema.requires_citations || (!example.citation_id.empty() && example.citation_id == prediction.citation_id);
    prediction.abstained = false;
    return StructuredDecoder::validate(prediction, example, schema);
}

SftEvaluation SftModel::evaluate(const std::vector<SftInstructionExample>& examples,
                                 const SftTaskSchema& schema, const SftAdapter* adapter) const {
    const auto started = std::chrono::steady_clock::now();
    SftEvaluation result;
    result.example_count = examples.size();
    double loss_sum = 0.0;
    for (const auto& example : examples) {
        const auto prediction = predict(example, schema, adapter);
        const auto target = label_index(example.target_label, schema);
        const auto probabilities = softmax(logits(features(example.input), adapter));
        loss_sum += -std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
        if (prediction.label == example.target_label) ++result.correct_count;
        if (prediction.schema_valid) ++result.valid_schema_count;
        if (prediction.citation_valid) ++result.valid_citation_count;
        if (prediction.abstained) ++result.abstention_count;
    }
    require(!examples.empty(), "cannot evaluate empty SFT example set");
    result.cross_entropy = loss_sum / static_cast<double>(examples.size());
    result.accuracy = static_cast<double>(result.correct_count) / static_cast<double>(examples.size());
    result.schema_validity = static_cast<double>(result.valid_schema_count) / static_cast<double>(examples.size());
    result.citation_integrity = static_cast<double>(result.valid_citation_count) / static_cast<double>(examples.size());
    result.abstention_rate = static_cast<double>(result.abstention_count) / static_cast<double>(examples.size());
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.finite = std::isfinite(result.cross_entropy) && std::isfinite(result.accuracy) && std::isfinite(result.schema_validity) &&
                    std::isfinite(result.citation_integrity);
    return result;
}

double SftModel::loss(const SftInstructionExample& example, const SftTaskSchema& schema, const SftAdapter* adapter) const {
    const auto probabilities = softmax(logits(features(example.input), adapter));
    const auto target = label_index(example.target_label, schema);
    return -std::log(std::max(probabilities[target], std::numeric_limits<double>::min()));
}

std::vector<double> SftModel::gradients(const SftInstructionExample& example, const SftTaskSchema& schema) const {
    const auto feature_values = features(example.input);
    const auto probabilities = softmax(logits(feature_values, nullptr));
    const auto target = label_index(example.target_label, schema);
    std::vector<double> gradient(parameters_.size(), 0.0);
    for (std::size_t label = 0; label < config_.label_count; ++label) {
        const auto error = probabilities[label] - (label == target ? 1.0 : 0.0);
        for (std::size_t feature = 0; feature < config_.feature_dim; ++feature) gradient[label * config_.feature_dim + feature] = error * feature_values[feature];
        gradient[config_.label_count * config_.feature_dim + label] = error;
    }
    return gradient;
}

void SftModel::apply_gradient(const std::vector<double>& gradient, const SftOptimizerConfig& optimizer) {
    require(gradient.size() == parameters_.size() && optimizer.learning_rate > 0.0, "invalid full SFT update");
    double norm = 0.0;
    for (const auto value : gradient) norm += value * value;
    norm = std::sqrt(norm);
    const auto scale = norm > optimizer.clip_norm ? optimizer.clip_norm / norm : 1.0;
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        parameters_[index] = parameters_[index] * (1.0 - optimizer.learning_rate * optimizer.weight_decay) - optimizer.learning_rate * gradient[index] * scale;
    }
    require(std::all_of(parameters_.begin(), parameters_.end(), [](const double value) { return std::isfinite(value); }),
            "full SFT update produced non-finite parameter");
}

void SftModel::save(std::ostream& stream) const {
    stream << "CCT_SFT_MODEL_V1\n" << std::quoted(config_.base_checkpoint_hash) << ' ' << std::quoted(config_.task_id) << ' '
           << config_.feature_dim << ' ' << config_.label_count << ' ' << config_.seed << '\n' << std::setprecision(17);
    for (const auto value : parameters_) stream << value << ' ';
    stream << '\n';
    require(static_cast<bool>(stream), "could not serialize SFT model");
}

SftModel SftModel::load(std::istream& stream) {
    std::string version;
    stream >> version;
    require(version == "CCT_SFT_MODEL_V1", "unsupported SFT model version");
    SftModelConfig config;
    stream >> std::quoted(config.base_checkpoint_hash) >> std::quoted(config.task_id) >> config.feature_dim >> config.label_count >> config.seed;
    SftModel model(config);
    for (auto& value : model.parameters_) stream >> value;
    require(static_cast<bool>(stream), "truncated SFT model");
    return model;
}

SftModel SftModel::merged(const SftAdapter& adapter) const {
    require(adapter.spec().base_checkpoint_hash == config_.base_checkpoint_hash, "adapter base checkpoint mismatch during merge");
    SftModel output = *this;
    const auto rank = adapter.spec().rank;
    const auto adapter_values = adapter.parameter_vector();
    for (std::size_t label = 0; label < config_.label_count; ++label) {
        for (std::size_t feature = 0; feature < config_.feature_dim; ++feature) {
            double delta = 0.0;
            for (std::size_t component = 0; component < rank; ++component) {
                delta += adapter_values[label * rank + component] * adapter_values[config_.label_count * rank + component * config_.feature_dim + feature];
            }
            output.parameters_[label * config_.feature_dim + feature] += delta;
        }
    }
    return output;
}

SftAdapter::SftAdapter(SftAdapterSpec spec, SftModelConfig base_config) : spec_(std::move(spec)), base_config_(std::move(base_config)) {
    require(!spec_.adapter_id.empty() && !spec_.task_id.empty() && spec_.rank > 0U && spec_.target_module == "output_projection",
            "invalid SFT adapter specification");
    require(spec_.base_checkpoint_hash == base_config_.base_checkpoint_hash && spec_.task_id == base_config_.task_id,
            "adapter does not match base model identity");
    parameters_.assign(base_config_.label_count * spec_.rank + spec_.rank * base_config_.feature_dim, 0.0);
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const auto mixed = spec_.rank + static_cast<std::uint64_t>(index + 1U) * 0x9e3779b97f4a7c15ULL;
        parameters_[index] = (static_cast<double>(positive_mod(mixed ^ (mixed >> 31U), 1001U)) / 1000.0 - 0.5) * 0.01;
    }
}

std::string SftAdapter::parameter_checksum() const {
    std::ostringstream serialized;
    serialized << spec_.adapter_id << '|' << spec_.task_id << '|' << spec_.base_checkpoint_hash << '|' << spec_.training_manifest_hash << '|'
               << spec_.rank << '|' << spec_.target_module << '|';
    serialized << std::setprecision(17);
    for (const auto value : parameters_) serialized << value << ',';
    return sft_hash(serialized.str());
}

void SftAdapter::set_parameter_vector(const std::vector<double>& values) {
    require(values.size() == parameters_.size(), "SFT adapter parameter vector size mismatch");
    require(std::all_of(values.begin(), values.end(), [](const double value) { return std::isfinite(value); }),
            "SFT adapter vector contains non-finite value");
    parameters_ = values;
}

std::vector<double> SftAdapter::gradients(const SftModel& base, const SftInstructionExample& example,
                                          const SftTaskSchema& schema) const {
    const auto merged_model = base.merged(*this);
    const auto full_gradient = merged_model.gradients(example, schema);
    std::vector<double> gradient(parameters_.size(), 0.0);
    const auto rank = spec_.rank;
    const auto base_values = parameters_;
    for (std::size_t label = 0; label < base_config_.label_count; ++label) {
        for (std::size_t component = 0; component < rank; ++component) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < base_config_.feature_dim; ++feature) {
                sum += full_gradient[label * base_config_.feature_dim + feature] * base_values[base_config_.label_count * rank + component * base_config_.feature_dim + feature];
            }
            gradient[label * rank + component] = sum;
        }
    }
    for (std::size_t component = 0; component < rank; ++component) {
        for (std::size_t feature = 0; feature < base_config_.feature_dim; ++feature) {
            double sum = 0.0;
            for (std::size_t label = 0; label < base_config_.label_count; ++label) sum += full_gradient[label * base_config_.feature_dim + feature] * base_values[label * rank + component];
            gradient[base_config_.label_count * rank + component * base_config_.feature_dim + feature] = sum;
        }
    }
    return gradient;
}

void SftAdapter::apply_gradient(const std::vector<double>& gradient, const SftOptimizerConfig& optimizer) {
    require(gradient.size() == parameters_.size() && optimizer.learning_rate > 0.0, "invalid adapter update");
    double norm = 0.0;
    for (const auto value : gradient) norm += value * value;
    norm = std::sqrt(norm);
    const auto scale = norm > optimizer.clip_norm ? optimizer.clip_norm / norm : 1.0;
    for (std::size_t index = 0; index < parameters_.size(); ++index) parameters_[index] -= optimizer.learning_rate * gradient[index] * scale;
}

void SftAdapter::save(std::ostream& stream) const {
    stream << "CCT_SFT_ADAPTER_V1\n" << std::quoted(spec_.adapter_id) << ' ' << std::quoted(spec_.task_id) << ' '
           << std::quoted(spec_.domain) << ' ' << std::quoted(spec_.version) << ' ' << spec_.rank << ' '
           << std::quoted(spec_.target_module) << ' ' << std::quoted(spec_.base_checkpoint_hash) << ' '
           << std::quoted(spec_.training_manifest_hash) << ' ' << spec_.permissions.size();
    for (const auto& permission : spec_.permissions) stream << ' ' << std::quoted(permission);
    stream << '\n' << std::setprecision(17);
    for (const auto value : parameters_) stream << value << ' ';
    stream << '\n';
}

SftAdapter SftAdapter::load(std::istream& stream) {
    std::string version;
    stream >> version;
    require(version == "CCT_SFT_ADAPTER_V1", "unsupported SFT adapter version");
    SftAdapterSpec spec;
    std::size_t permission_count = 0;
    stream >> std::quoted(spec.adapter_id) >> std::quoted(spec.task_id) >> std::quoted(spec.domain) >> std::quoted(spec.version)
           >> spec.rank >> std::quoted(spec.target_module) >> std::quoted(spec.base_checkpoint_hash) >> std::quoted(spec.training_manifest_hash) >> permission_count;
    for (std::size_t index = 0; index < permission_count; ++index) {
        std::string permission;
        stream >> std::quoted(permission);
        spec.permissions.push_back(std::move(permission));
    }
    SftModelConfig base_config{spec.base_checkpoint_hash, spec.task_id, 8U, 2U, 0U};
    SftAdapter adapter(spec, base_config);
    for (auto& value : adapter.parameters_) stream >> value;
    require(static_cast<bool>(stream), "truncated SFT adapter");
    return adapter;
}

void SftAdapterRegistry::register_adapter(const SftAdapter& adapter) {
    for (const auto& existing : adapters_) require(existing.spec().adapter_id != adapter.spec().adapter_id, "duplicate SFT adapter ID");
    adapters_.push_back(adapter);
}

bool SftAdapterRegistry::authorize(const std::string& adapter_id, const std::string& task_id, const std::string& base_hash,
                                   const std::string& permission) const {
    for (const auto& adapter : adapters_) {
        if (adapter.spec().adapter_id != adapter_id || adapter.spec().task_id != task_id || adapter.spec().base_checkpoint_hash != base_hash) continue;
        return std::find(adapter.spec().permissions.begin(), adapter.spec().permissions.end(), permission) != adapter.spec().permissions.end();
    }
    return false;
}

const SftAdapter& SftAdapterRegistry::load_authorized(const std::string& adapter_id, const std::string& task_id,
                                                       const std::string& base_hash, const std::string& permission) const {
    require(authorize(adapter_id, task_id, base_hash, permission), "SFT adapter authorization denied");
    for (const auto& adapter : adapters_) if (adapter.spec().adapter_id == adapter_id) return adapter;
    throw SftError("authorized SFT adapter disappeared");
}

std::string SftAdapterRegistry::serialize() const {
    std::ostringstream output;
    output << "CCT_SFT_REGISTRY_V1\n" << adapters_.size() << '\n';
    for (const auto& adapter : adapters_) {
        std::ostringstream encoded;
        adapter.save(encoded);
        output << hex_encode(encoded.str()) << '\n';
    }
    return output.str();
}

SftAdapterRegistry SftAdapterRegistry::deserialize(const std::string& serialized) {
    std::istringstream input(serialized);
    std::string version;
    std::size_t count = 0;
    input >> version >> count;
    require(version == "CCT_SFT_REGISTRY_V1", "unsupported SFT registry version");
    std::string line;
    std::getline(input, line);
    SftAdapterRegistry registry;
    for (std::size_t index = 0; index < count; ++index) {
        require(static_cast<bool>(std::getline(input, line)), "truncated SFT registry");
        std::istringstream adapter_stream(hex_decode(line));
        registry.register_adapter(SftAdapter::load(adapter_stream));
    }
    return registry;
}

SftPrediction StructuredDecoder::validate(const SftPrediction& original, const SftInstructionExample& example,
                                          const SftTaskSchema& schema) {
    auto prediction = original;
    if (prediction.output.size() > schema.maximum_output_bytes) {
        prediction.schema_valid = false;
        prediction.abstained = schema.allows_abstention;
    }
    if (schema.output_kind == SftOutputKind::Json || schema.output_kind == SftOutputKind::Grounded || schema.output_kind == SftOutputKind::Draft) {
        prediction.schema_valid = prediction.output.size() >= 2U && prediction.output.front() == '{' && prediction.output.back() == '}';
    }
    if (schema.output_kind == SftOutputKind::Grounded) {
        prediction.citation_valid = !schema.requires_citations || (!example.citation_id.empty() && prediction.output.find(example.citation_id) != std::string::npos);
        if (schema.requires_citations && !prediction.citation_valid) prediction.abstained = schema.allows_abstention;
    }
    if (schema.kind == SftTaskKind::WorkflowDrafting && prediction.output.find("approval_required\":true") == std::string::npos) {
        prediction.schema_valid = false;
    }
    return prediction;
}

}  // namespace cct
