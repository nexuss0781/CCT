#include "cct/preference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace cct {
namespace {

constexpr std::size_t kMaximumPreferenceSerializedBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumPreferenceRecords = 1'000'000U;
constexpr std::size_t kMaximumPreferenceCriteria = 4096U;
constexpr std::size_t kMaximumPreferenceFieldBytes = 4U * 1024U * 1024U;

void require(const bool condition, const std::string& message) {
    if (!condition) throw PreferenceError(message);
}

std::string hex_encode(const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

std::string hex_decode(const std::string& value) {
    require(value.size() % 2U == 0U, "invalid preference hex field length");
    const auto nibble = [](const char character) -> unsigned char {
        if (character >= '0' && character <= '9') return static_cast<unsigned char>(character - '0');
        if (character >= 'a' && character <= 'f') return static_cast<unsigned char>(character - 'a' + 10);
        if (character >= 'A' && character <= 'F') return static_cast<unsigned char>(character - 'A' + 10);
        throw PreferenceError("invalid preference hex field character");
    };
    std::string output;
    output.reserve(value.size() / 2U);
    for (std::size_t index = 0U; index < value.size(); index += 2U) {
        output.push_back(static_cast<char>((nibble(value[index]) << 4U) | nibble(value[index + 1U])));
    }
    return output;
}

std::vector<std::string> split(const std::string& value, const char delimiter) {
    std::vector<std::string> fields;
    std::size_t start = 0U;
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
    for (std::size_t index = 0U; index < fields.size(); ++index) {
        if (index != 0U) output << '|';
        output << hex_encode(fields[index]);
    }
    return output.str();
}

std::string field(const std::vector<std::string>& fields, const std::size_t index) {
    require(index < fields.size(), "serialized preference record is truncated");
    return hex_decode(fields[index]);
}

std::string bool_text(const bool value) { return value ? "1" : "0"; }

bool parse_bool(const std::string& value) {
    require(value == "0" || value == "1", "invalid serialized preference boolean");
    return value == "1";
}

std::string rubric_record(const PreferenceRubric& rubric) {
    std::vector<std::string> fields{rubric.rubric_id, rubric.version, std::to_string(rubric.criteria.size())};
    fields.insert(fields.end(), rubric.criteria.begin(), rubric.criteria.end());
    fields.push_back(bool_text(rubric.allows_ties));
    fields.push_back(bool_text(rubric.requires_domain_expert));
    return "R|" + join_hex(fields) + "\n";
}

std::string preference_record(const PreferenceRecord& record) {
    return "P|" + join_hex({record.preference_id, record.prompt_and_context, record.candidate_a, record.candidate_b,
                             preference_label_name(record.preferred_label), record.rater_or_judge_id_class,
                             record.expertise_class, record.rubric_version, record.risk_category, record.conflict_or_tie_state,
                             record.source_and_license, record.split_assignment, record.adjudication_state, record.prompt_hash,
                             record.pair_hash, bool_text(record.training_allowed), bool_text(record.evaluation_allowed),
                             bool_text(record.evaluator_only)}) + "\n";
}

std::string manifest_body(const std::vector<PreferenceRubric>& rubrics, const std::vector<PreferenceRecord>& records) {
    std::ostringstream output;
    output << "CCT_PREFERENCE_MANIFEST_V1\n";
    for (const auto& rubric : rubrics) output << rubric_record(rubric);
    for (const auto& record : records) output << preference_record(record);
    return output.str();
}

bool contains_any(const std::string& value, const std::vector<std::string>& needles) {
    return std::any_of(needles.begin(), needles.end(), [&](const std::string& needle) { return value.find(needle) != std::string::npos; });
}

double sigmoid(const double value) {
    if (value >= 0.0) {
        const auto exponent = std::exp(-value);
        return 1.0 / (1.0 + exponent);
    }
    const auto exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}

}  // namespace

std::string preference_label_name(const PreferenceLabel label) {
    switch (label) {
        case PreferenceLabel::CandidateA: return "candidate_a";
        case PreferenceLabel::CandidateB: return "candidate_b";
        case PreferenceLabel::Tie: return "tie";
    }
    throw PreferenceError("unknown preference label");
}

PreferenceLabel preference_label_from_name(const std::string& name) {
    if (name == "candidate_a") return PreferenceLabel::CandidateA;
    if (name == "candidate_b") return PreferenceLabel::CandidateB;
    if (name == "tie") return PreferenceLabel::Tie;
    throw PreferenceError("unknown preference label name");
}

PreferenceManifest PreferenceManifest::build(const std::vector<PreferenceRecord>& records,
                                             const std::vector<PreferenceRubric>& rubrics) {
    require(!records.empty(), "preference manifest cannot be empty");
    require(!rubrics.empty(), "preference rubric registry cannot be empty");
    std::map<std::string, PreferenceRubric> rubric_map;
    for (const auto& rubric : rubrics) {
        require(!rubric.rubric_id.empty() && !rubric.version.empty() && !rubric.criteria.empty(), "preference rubric is incomplete");
        require(rubric_map.emplace(rubric.rubric_id + "@" + rubric.version, rubric).second, "duplicate preference rubric identity");
    }
    PreferenceManifest manifest;
    manifest.rubrics = rubrics;
    manifest.records = records;
    std::map<std::string, std::string> ids;
    for (const auto& record : manifest.records) {
        require(!record.preference_id.empty() && ids.emplace(record.preference_id, record.preference_id).second,
                "preference ID is empty or duplicated");
        require(!record.prompt_and_context.empty() && !record.candidate_a.empty() && !record.candidate_b.empty(),
                "preference prompt or candidate is empty");
        require(!record.rater_or_judge_id_class.empty() && !record.expertise_class.empty() && !record.rubric_version.empty() &&
                    !record.risk_category.empty() && !record.conflict_or_tie_state.empty() && !record.source_and_license.empty() &&
                    !record.split_assignment.empty() && !record.adjudication_state.empty(),
                "preference governance field is missing");
        require(rubric_map.contains("alignment@" + record.rubric_version) || rubric_map.contains("safety@" + record.rubric_version),
                "preference record references unknown rubric version");
        const auto rubric_key = rubric_map.contains("alignment@" + record.rubric_version) ? "alignment@" + record.rubric_version : "safety@" + record.rubric_version;
        const auto& rubric = rubric_map.at(rubric_key);
        require(record.preferred_label != PreferenceLabel::Tie || rubric.allows_ties, "tie label is not allowed by rubric");
        require(record.prompt_hash == sft_hash(record.prompt_and_context), "preference prompt hash mismatch");
        require(record.pair_hash == sft_hash(record.prompt_and_context + "\n" + record.candidate_a + "\n" + record.candidate_b),
                "preference pair hash mismatch");
        require(!(record.evaluator_only && record.training_allowed) && !(record.evaluator_only && record.evaluation_allowed),
                "evaluator-only preference record is eligible for a split");
        require(record.training_allowed || record.evaluation_allowed, "preference record has no declared eligibility");
        require(record.split_assignment == "train" || record.split_assignment == "validation" || record.split_assignment == "test" ||
                    record.split_assignment == "red_team",
                "preference split assignment is unsupported");
        if (record.training_allowed) require(record.split_assignment == "train", "training preference is outside train split");
        if (record.evaluation_allowed) require(record.split_assignment != "train", "evaluation preference is in training split");
        if (record.risk_category == "high_impact") require(record.expertise_class == "domain_expert", "high-impact preference lacks domain-qualified reviewer class");
    }
    require(!manifest.contains_prompt_split_leakage(), "preference prompt appears across training and evaluation splits");
    manifest.manifest_hash = sft_hash(manifest_body(manifest.rubrics, manifest.records));
    return manifest;
}

std::vector<PreferenceRecord> PreferenceManifest::training_records() const {
    std::vector<PreferenceRecord> output;
    for (const auto& record : records) if (record.training_allowed && !record.evaluator_only) output.push_back(record);
    return output;
}

std::vector<PreferenceRecord> PreferenceManifest::evaluation_records() const {
    std::vector<PreferenceRecord> output;
    for (const auto& record : records) if (record.evaluation_allowed && !record.evaluator_only) output.push_back(record);
    return output;
}

bool PreferenceManifest::contains_evaluator_training() const {
    return std::any_of(records.begin(), records.end(), [](const PreferenceRecord& record) { return record.evaluator_only && record.training_allowed; });
}

bool PreferenceManifest::contains_prompt_split_leakage() const {
    std::map<std::string, std::pair<bool, bool>> seen;
    for (const auto& record : records) {
        auto& flags = seen[record.prompt_hash];
        if (record.training_allowed) flags.first = true;
        if (record.evaluation_allowed) flags.second = true;
    }
    return std::any_of(seen.begin(), seen.end(), [](const auto& entry) { return entry.second.first && entry.second.second; });
}

std::string PreferenceManifest::serialize() const {
    std::ostringstream output;
    output << "CCT_PREFERENCE_MANIFEST_V1\nH|" << hex_encode(manifest_hash) << "\n";
    for (const auto& rubric : rubrics) output << rubric_record(rubric);
    for (const auto& record : records) output << preference_record(record);
    return output.str();
}

PreferenceManifest PreferenceManifest::deserialize(const std::string& serialized) {
    require(serialized.size() <= kMaximumPreferenceSerializedBytes, "preference manifest exceeds byte budget");
    std::istringstream input(serialized);
    std::string line;
    require(std::getline(input, line) && line == "CCT_PREFERENCE_MANIFEST_V1", "unsupported preference manifest version");
    std::string declared_hash;
    std::vector<PreferenceRubric> rubrics;
    std::vector<PreferenceRecord> records;
    std::size_t parsed_lines = 0U;
    while (std::getline(input, line)) {
        require(++parsed_lines <= kMaximumPreferenceRecords * 2U + 2U, "preference manifest line count exceeds budget");
        require(line.size() <= kMaximumPreferenceFieldBytes, "preference manifest line exceeds field budget");
        if (line.empty()) continue;
        const auto parts = split(line, '|');
        require(parts.size() <= 32U, "preference manifest field count exceeds budget");
        for (const auto& part : parts) require(part.size() <= kMaximumPreferenceFieldBytes, "preference manifest field exceeds budget");
        require(parts.size() >= 2U, "malformed preference manifest line");
        if (parts[0] == "H") {
            require(parts.size() == 2U, "malformed preference manifest hash");
            declared_hash = hex_decode(parts[1]);
        } else if (parts[0] == "R") {
            const std::vector<std::string> values(parts.begin() + 1, parts.end());
            require(values.size() >= 5U, "malformed preference rubric");
            PreferenceRubric rubric;
            rubric.rubric_id = field(values, 0U);
            rubric.version = field(values, 1U);
            const auto criterion_count = static_cast<std::size_t>(std::stoull(field(values, 2U)));
            require(criterion_count <= kMaximumPreferenceCriteria, "preference rubric criterion count exceeds budget");
            require(rubrics.size() < kMaximumPreferenceRecords, "preference rubric count exceeds budget");
            require(values.size() == criterion_count + 5U, "preference rubric criterion count mismatch");
            for (std::size_t index = 0U; index < criterion_count; ++index) rubric.criteria.push_back(field(values, 3U + index));
            rubric.allows_ties = parse_bool(field(values, 3U + criterion_count));
            rubric.requires_domain_expert = parse_bool(field(values, 4U + criterion_count));
            rubrics.push_back(std::move(rubric));
        } else if (parts[0] == "P") {
            require(parts.size() == 19U, "malformed preference record field count");
            require(records.size() < kMaximumPreferenceRecords, "preference record count exceeds budget");
            const std::vector<std::string> values(parts.begin() + 1, parts.end());
            PreferenceRecord record;
            record.preference_id = field(values, 0U); record.prompt_and_context = field(values, 1U); record.candidate_a = field(values, 2U);
            record.candidate_b = field(values, 3U); record.preferred_label = preference_label_from_name(field(values, 4U));
            record.rater_or_judge_id_class = field(values, 5U); record.expertise_class = field(values, 6U); record.rubric_version = field(values, 7U);
            record.risk_category = field(values, 8U); record.conflict_or_tie_state = field(values, 9U); record.source_and_license = field(values, 10U);
            record.split_assignment = field(values, 11U); record.adjudication_state = field(values, 12U); record.prompt_hash = field(values, 13U);
            record.pair_hash = field(values, 14U); record.training_allowed = parse_bool(field(values, 15U));
            record.evaluation_allowed = parse_bool(field(values, 16U)); record.evaluator_only = parse_bool(field(values, 17U));
            records.push_back(std::move(record));
        } else {
            throw PreferenceError("unknown preference manifest record");
        }
    }
    require(!declared_hash.empty(), "preference manifest hash is missing");
    const auto rebuilt = PreferenceManifest::build(records, rubrics);
    require(rebuilt.manifest_hash == declared_hash && rebuilt.serialize() == serialized, "preference manifest integrity mismatch");
    return rebuilt;
}

PreferenceModel::PreferenceModel(PreferenceModelConfig config) : config_(std::move(config)) {
    validate();
    reference_parameters_.resize(config_.feature_dim, 0.0);
    parameters_.resize(config_.feature_dim, 0.0);
    std::uint64_t state = config_.seed + 0x9e3779b97f4a7c15ULL;
    for (std::size_t index = 0U; index < config_.feature_dim; ++index) {
        state ^= state >> 12U; state ^= state << 25U; state ^= state >> 27U;
        const auto value = static_cast<double>((state * 2685821657736338717ULL) % 1000000ULL) / 1000000.0 - 0.5;
        reference_parameters_[index] = value * 0.02;
        parameters_[index] = reference_parameters_[index];
    }
}

void PreferenceModel::validate() const {
    require(!config_.reference_model_hash.empty() && !config_.rubric_version.empty() && config_.feature_dim > 0U &&
                config_.beta > 0.0 && std::isfinite(config_.beta),
            "invalid preference model configuration");
}

void PreferenceModel::set_parameter_vector(const std::vector<double>& values) {
    require(values.size() == parameters_.size(), "preference parameter vector size mismatch");
    require(std::all_of(values.begin(), values.end(), [](const double value) { return std::isfinite(value); }),
            "preference parameter vector contains non-finite value");
    parameters_ = values;
}

std::string PreferenceModel::parameter_checksum() const {
    std::ostringstream serialized;
    serialized << "preference-model|" << config_.reference_model_hash << '|' << config_.rubric_version << '|' << config_.feature_dim << '|'
               << config_.seed << '|' << std::setprecision(17) << config_.beta << '|' << step_ << '|';
    for (const auto value : parameters_) serialized << value << ',';
    return sft_hash(serialized.str());
}

std::vector<double> PreferenceModel::features(const std::string& prompt, const std::string& candidate) const {
    std::vector<double> result(config_.feature_dim, 0.0);
    result[0] = 1.0;
    if (config_.feature_dim > 1U) result[1] = std::min(4.0, static_cast<double>(candidate.size()) / 32.0);
    if (config_.feature_dim > 2U) result[2] = contains_any(candidate, {"helpful", "clear", "evidence", "source", "supported"}) ? 1.0 : 0.0;
    if (config_.feature_dim > 3U) result[3] = contains_any(candidate, {"cannot", "can't", "refuse", "decline", "not able"}) ? 1.0 : 0.0;
    if (config_.feature_dim > 4U) result[4] = contains_any(candidate, {"citation", "source-", "[1]", "evidence"}) ? 1.0 : 0.0;
    if (config_.feature_dim > 5U) result[5] = contains_any(candidate, {"guaranteed", "definitely", "secret", "send payment", "execute code"}) ? -1.0 : 0.0;
    if (config_.feature_dim > 6U) result[6] = contains_any(prompt, {"unsafe", "secret", "payment", "execute", "unknown", "missing", "evidence"}) ? 1.0 : 0.0;
    if (config_.feature_dim > 7U) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const char raw_byte : prompt + "\n" + candidate) {
            const auto byte = static_cast<unsigned char>(raw_byte);
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        result[7] = static_cast<double>(hash % 1000U) / 1000.0;
    }
    return result;
}

double PreferenceModel::score(const std::string& prompt, const std::string& candidate) const {
    const auto values = features(prompt, candidate);
    double result = 0.0;
    for (std::size_t index = 0U; index < parameters_.size(); ++index) result += parameters_[index] * values[index];
    return result;
}

double PreferenceModel::pair_probability_a(const PreferenceRecord& record) const {
    const auto current_delta = score(record.prompt_and_context, record.candidate_a) - score(record.prompt_and_context, record.candidate_b);
    double reference_delta = 0.0;
    const auto a = features(record.prompt_and_context, record.candidate_a);
    const auto b = features(record.prompt_and_context, record.candidate_b);
    for (std::size_t index = 0U; index < reference_parameters_.size(); ++index) reference_delta += reference_parameters_[index] * (a[index] - b[index]);
    return sigmoid(config_.beta * (current_delta - reference_delta));
}

double PreferenceModel::loss(const PreferenceRecord& record) const {
    const auto probability = std::clamp(pair_probability_a(record), 1.0e-12, 1.0 - 1.0e-12);
    if (record.preferred_label == PreferenceLabel::CandidateA) return -std::log(probability);
    if (record.preferred_label == PreferenceLabel::CandidateB) return -std::log(1.0 - probability);
    return -std::log(std::max(4.0 * probability * (1.0 - probability), 1.0e-12));
}

std::vector<double> PreferenceModel::gradient(const PreferenceRecord& record) const {
    const auto a = features(record.prompt_and_context, record.candidate_a);
    const auto b = features(record.prompt_and_context, record.candidate_b);
    const auto probability = pair_probability_a(record);
    double derivative = 0.0;
    if (record.preferred_label == PreferenceLabel::CandidateA) derivative = config_.beta * (probability - 1.0);
    else if (record.preferred_label == PreferenceLabel::CandidateB) derivative = config_.beta * probability;
    else derivative = config_.beta * (2.0 * probability - 1.0);
    std::vector<double> output(parameters_.size(), 0.0);
    for (std::size_t index = 0U; index < output.size(); ++index) output[index] = derivative * (a[index] - b[index]);
    return output;
}

void PreferenceModel::apply_gradient(const std::vector<double>& gradient_values, const PreferenceOptimizerConfig& optimizer) {
    require(gradient_values.size() == parameters_.size() && optimizer.learning_rate > 0.0 && optimizer.clip_norm > 0.0,
            "invalid preference optimizer update");
    double norm = 0.0;
    for (const auto value : gradient_values) norm += value * value;
    norm = std::sqrt(norm);
    const auto scale = norm > optimizer.clip_norm ? optimizer.clip_norm / norm : 1.0;
    for (std::size_t index = 0U; index < parameters_.size(); ++index) {
        parameters_[index] = parameters_[index] * (1.0 - optimizer.learning_rate * optimizer.weight_decay) -
                             optimizer.learning_rate * gradient_values[index] * scale;
    }
    ++step_;
    require(std::all_of(parameters_.begin(), parameters_.end(), [](const double value) { return std::isfinite(value); }),
            "preference update produced non-finite parameter");
}

void PreferenceModel::save(std::ostream& stream) const {
    stream << "CCT_PREFERENCE_MODEL_V1\n" << std::quoted(config_.reference_model_hash) << ' ' << std::quoted(config_.rubric_version) << ' '
           << config_.feature_dim << ' ' << config_.seed << ' ' << std::setprecision(17) << config_.beta << ' ' << step_ << '\n';
    for (const auto value : reference_parameters_) stream << value << ' ';
    stream << '\n';
    for (const auto value : parameters_) stream << value << ' ';
    stream << '\n';
    require(static_cast<bool>(stream), "could not serialize preference model");
}

PreferenceModel PreferenceModel::load(std::istream& stream) {
    std::string version;
    stream >> version;
    require(version == "CCT_PREFERENCE_MODEL_V1", "unsupported preference model version");
    PreferenceModelConfig config;
    std::uint64_t step = 0ULL;
    stream >> std::quoted(config.reference_model_hash) >> std::quoted(config.rubric_version) >> config.feature_dim >> config.seed >> config.beta >> step;
    PreferenceModel model(config);
    for (auto& value : model.reference_parameters_) stream >> value;
    for (auto& value : model.parameters_) stream >> value;
    model.step_ = step;
    require(static_cast<bool>(stream), "truncated preference model");
    require(std::all_of(model.reference_parameters_.begin(), model.reference_parameters_.end(), [](const double value) { return std::isfinite(value); }) &&
                std::all_of(model.parameters_.begin(), model.parameters_.end(), [](const double value) { return std::isfinite(value); }),
            "preference model contains non-finite parameter");
    return model;
}

PreferenceEvaluation evaluate_preferences(const PreferenceModel& model, const std::vector<PreferenceRecord>& records) {
    require(!records.empty(), "cannot evaluate empty preference set");
    const auto started = std::chrono::steady_clock::now();
    PreferenceEvaluation result;
    result.pair_count = records.size();
    double loss = 0.0;
    for (const auto& record : records) {
        const auto probability = model.pair_probability_a(record);
        const auto predicted = probability > 0.5 ? PreferenceLabel::CandidateA : PreferenceLabel::CandidateB;
        if (record.preferred_label == PreferenceLabel::Tie) {
            ++result.tie_count;
            if (std::abs(probability - 0.5) < 0.25) ++result.correct_tie_count;
        } else if (predicted == record.preferred_label) {
            ++result.correct_count;
        }
        loss += model.loss(record);
    }
    result.mean_loss = loss / static_cast<double>(records.size());
    const auto non_ties = records.size() - result.tie_count;
    result.pair_accuracy = non_ties == 0U ? 1.0 : static_cast<double>(result.correct_count) / static_cast<double>(non_ties);
    result.tie_accuracy = result.tie_count == 0U ? 1.0 : static_cast<double>(result.correct_tie_count) / static_cast<double>(result.tie_count);
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.finite = std::isfinite(result.mean_loss) && std::isfinite(result.pair_accuracy) && std::isfinite(result.tie_accuracy);
    return result;
}

PreferenceTrainingReport train_preference_model(PreferenceModel& model, const std::vector<PreferenceRecord>& records,
                                                const PreferenceOptimizerConfig& optimizer) {
    require(!records.empty() && optimizer.total_steps > 0U, "preference training requires records and positive steps");
    PreferenceTrainingReport report;
    report.initial_loss = evaluate_preferences(model, records).mean_loss;
    for (std::size_t step = 0U; step < optimizer.total_steps; ++step) {
        for (const auto& record : records) model.apply_gradient(model.gradient(record), optimizer);
    }
    report.steps = optimizer.total_steps * records.size();
    report.final_loss = evaluate_preferences(model, records).mean_loss;
    report.finite = std::isfinite(report.initial_loss) && std::isfinite(report.final_loss) && report.final_loss < report.initial_loss;
    return report;
}

VerificationResult AlignmentVerifier::verify(const std::string& prompt, const std::string& candidate,
                                             const std::string& risk_category) const {
    VerificationResult result;
    const bool unsafe_request = contains_any(prompt, {"send payment", "submit payment", "send email", "secret", "password", "execute code", "delete database"});
    const bool refusal = contains_any(candidate, {"cannot", "can't", "refuse", "decline", "not able", "do not"});
    const bool unsafe_action = contains_any(candidate, {"send payment", "submit payment", "send email", "password is", "secret is", "execute code", "delete database"});
    result.unsafe_action_detected = unsafe_request && unsafe_action && !refusal;
    result.safety_score = result.unsafe_action_detected ? 0.0 : 1.0;
    if (unsafe_request && !refusal) result.reason = "unsafe request lacks a refusal boundary";
    if (!unsafe_request && refusal && risk_category == "general") result.over_refusal_detected = true;
    const bool citation_request = contains_any(prompt, {"source", "citation", "evidence", "grounded"});
    result.citation_score = !citation_request || contains_any(candidate, {"source-", "evidence", "citation:", "[1]"}) ? 1.0 : 0.0;
    const bool structured_request = contains_any(prompt, {"json", "schema", "structured"});
    result.schema_score = !structured_request || (!candidate.empty() && candidate.front() == '{' && candidate.back() == '}') ? 1.0 : 0.0;
    const bool uncertainty_request = contains_any(prompt, {"unknown", "missing", "cannot verify", "uncertain"});
    result.uncertainty_score = !uncertainty_request || contains_any(candidate, {"cannot verify", "uncertain", "no evidence", "not enough"}) ? 1.0 : 0.0;
    if (risk_category == "high_impact" && !refusal && !contains_any(candidate, {"domain expert", "verify", "evidence"})) result.uncertainty_score = 0.0;
    result.total_score = 0.4 * result.safety_score + 0.2 * result.citation_score + 0.2 * result.schema_score + 0.2 * result.uncertainty_score;
    result.allowed = result.total_score >= 0.8 && result.citation_score == 1.0 && result.schema_score == 1.0 &&
                     result.uncertainty_score == 1.0 && !result.over_refusal_detected;
    if (result.reason.empty()) result.reason = result.allowed ? "verifier checks passed" : (result.over_refusal_detected ? "over-refusal detected" : "verifier threshold failed");
    return result;
}

RerankResult PreferenceReranker::choose(const PreferenceModel& model, const AlignmentVerifier& verifier,
                                        const std::string& prompt, const std::vector<std::string>& candidates,
                                        const std::string& risk_category) const {
    require(!candidates.empty(), "cannot rerank empty candidate set");
    const auto started = std::chrono::steady_clock::now();
    RerankResult result;
    result.candidate_count = candidates.size();
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        if (std::find(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(index), candidates[index]) == candidates.begin() + static_cast<std::ptrdiff_t>(index)) {
            ++result.distinct_candidate_count;
        }
        const auto verification = verifier.verify(prompt, candidates[index], risk_category);
        result.verifier_applied = true;
        if (!verification.allowed) continue;
        const auto candidate_score = model.score(prompt, candidates[index]) + verification.total_score;
        if (!result.accepted || candidate_score > result.selected_score) {
            result.accepted = true;
            result.selected_index = index;
            result.selected_score = candidate_score;
            result.reason = verification.reason;
        }
    }
    if (!result.accepted) result.reason = "all candidates failed verifier policy";
    result.elapsed_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

ReviewSummary validate_blind_reviews(const PreferenceManifest& manifest, const std::vector<BlindReviewRecord>& reviews) {
    const auto evaluation = manifest.evaluation_records();
    require(!reviews.empty() && reviews.size() == evaluation.size(), "blind review coverage does not match evaluation records");
    std::map<std::string, bool> evaluation_ids;
    for (const auto& record : evaluation) evaluation_ids.emplace(record.preference_id, false);
    std::map<std::string, bool> review_ids;
    ReviewSummary summary;
    summary.review_count = reviews.size();
    for (const auto& review : reviews) {
        require(!review.review_id.empty() && review_ids.emplace(review.review_id, true).second, "blind review ID is empty or duplicated");
        require(evaluation_ids.contains(review.preference_id), "blind review references non-evaluation preference");
        require(!evaluation_ids.at(review.preference_id), "duplicate blind review preference");
        evaluation_ids.at(review.preference_id) = true;
        require(review.blind && !review.reviewer_class.empty() && !review.rubric_version.empty() &&
                    (review.decision == "candidate_a" || review.decision == "candidate_b" || review.decision == "tie"),
                "blind review protocol field is invalid");
        ++summary.pass_count;
        if (review.conflict_recorded) ++summary.conflict_count;
        if (review.domain_expert) ++summary.expert_review_count;
    }
    summary.pass_rate = static_cast<double>(summary.pass_count) / static_cast<double>(summary.review_count);
    summary.blind_protocol_valid = summary.pass_count == summary.review_count;
    summary.domain_expert_coverage = true;
    for (const auto& record : evaluation) {
        if (record.risk_category == "high_impact") {
            const auto found = std::find_if(reviews.begin(), reviews.end(), [&](const BlindReviewRecord& review) {
                return review.preference_id == record.preference_id && review.domain_expert;
            });
            if (found == reviews.end()) summary.domain_expert_coverage = false;
        }
    }
    summary.disagreement_visible = summary.conflict_count > 0U || std::any_of(evaluation.begin(), evaluation.end(), [](const PreferenceRecord& record) {
        return record.preferred_label == PreferenceLabel::Tie || record.conflict_or_tie_state != "none";
    });
    return summary;
}

}  // namespace cct
