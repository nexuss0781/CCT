#pragma once

#include "cct/sft.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class PreferenceLabel : std::uint8_t {
    CandidateA = 0,
    CandidateB = 1,
    Tie = 2
};

std::string preference_label_name(PreferenceLabel label);
PreferenceLabel preference_label_from_name(const std::string& name);

struct PreferenceRubric {
    std::string rubric_id;
    std::string version = "v1";
    std::vector<std::string> criteria;
    bool allows_ties = true;
    bool requires_domain_expert = false;
};

struct PreferenceRecord {
    std::string preference_id;
    std::string prompt_and_context;
    std::string candidate_a;
    std::string candidate_b;
    PreferenceLabel preferred_label = PreferenceLabel::Tie;
    std::string rater_or_judge_id_class;
    std::string expertise_class;
    std::string rubric_version;
    std::string risk_category;
    std::string conflict_or_tie_state;
    std::string source_and_license;
    std::string split_assignment;
    std::string adjudication_state;
    std::string prompt_hash;
    std::string pair_hash;
    bool training_allowed = false;
    bool evaluation_allowed = false;
    bool evaluator_only = false;
};

class PreferenceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct PreferenceManifest {
    std::string manifest_hash;
    std::vector<PreferenceRubric> rubrics;
    std::vector<PreferenceRecord> records;

    static PreferenceManifest build(const std::vector<PreferenceRecord>& records,
                                    const std::vector<PreferenceRubric>& rubrics);
    std::vector<PreferenceRecord> training_records() const;
    std::vector<PreferenceRecord> evaluation_records() const;
    bool contains_evaluator_training() const;
    bool contains_prompt_split_leakage() const;
    std::string serialize() const;
    static PreferenceManifest deserialize(const std::string& serialized);
};

struct PreferenceEvaluation {
    double mean_loss = 0.0;
    double pair_accuracy = 0.0;
    double tie_accuracy = 0.0;
    std::size_t pair_count = 0;
    std::size_t correct_count = 0;
    std::size_t tie_count = 0;
    std::size_t correct_tie_count = 0;
    double elapsed_seconds = 0.0;
    bool finite = false;
};

struct PreferenceModelConfig {
    std::string reference_model_hash;
    std::string rubric_version;
    std::size_t feature_dim = 8;
    std::uint64_t seed = 0;
    double beta = 0.1;
};

struct PreferenceOptimizerConfig {
    double learning_rate = 0.2;
    double clip_norm = 5.0;
    double weight_decay = 0.0;
    std::size_t total_steps = 80;
};

class PreferenceModel {
public:
    explicit PreferenceModel(PreferenceModelConfig config);

    const PreferenceModelConfig& config() const noexcept { return config_; }
    std::size_t parameter_count() const noexcept { return parameters_.size(); }
    std::uint64_t step() const noexcept { return step_; }
    std::vector<double> parameter_vector() const { return parameters_; }
    void set_parameter_vector(const std::vector<double>& values);
    std::string parameter_checksum() const;
    double score(const std::string& prompt, const std::string& candidate) const;
    double pair_probability_a(const PreferenceRecord& record) const;
    double loss(const PreferenceRecord& record) const;
    std::vector<double> gradient(const PreferenceRecord& record) const;
    void apply_gradient(const std::vector<double>& gradient, const PreferenceOptimizerConfig& optimizer);
    void save(std::ostream& stream) const;
    static PreferenceModel load(std::istream& stream);

private:
    PreferenceModelConfig config_;
    std::vector<double> reference_parameters_;
    std::vector<double> parameters_;
    std::uint64_t step_ = 0;

    void validate() const;
    std::vector<double> features(const std::string& prompt, const std::string& candidate) const;
};

PreferenceEvaluation evaluate_preferences(const PreferenceModel& model,
                                          const std::vector<PreferenceRecord>& records);

struct PreferenceTrainingReport {
    std::size_t steps = 0;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    bool finite = false;
};

PreferenceTrainingReport train_preference_model(PreferenceModel& model,
                                                const std::vector<PreferenceRecord>& records,
                                                const PreferenceOptimizerConfig& optimizer);

struct VerificationResult {
    double safety_score = 0.0;
    double citation_score = 0.0;
    double schema_score = 0.0;
    double uncertainty_score = 0.0;
    double total_score = 0.0;
    bool allowed = false;
    bool unsafe_action_detected = false;
    bool over_refusal_detected = false;
    std::string reason;
};

class AlignmentVerifier {
public:
    VerificationResult verify(const std::string& prompt, const std::string& candidate,
                              const std::string& risk_category) const;
};

struct RerankResult {
    std::size_t selected_index = 0;
    double selected_score = 0.0;
    double elapsed_milliseconds = 0.0;
    std::size_t candidate_count = 0;
    std::size_t distinct_candidate_count = 0;
    bool accepted = false;
    bool verifier_applied = false;
    std::string reason;
};

class PreferenceReranker {
public:
    RerankResult choose(const PreferenceModel& model, const AlignmentVerifier& verifier,
                        const std::string& prompt, const std::vector<std::string>& candidates,
                        const std::string& risk_category) const;
};

struct BlindReviewRecord {
    std::string review_id;
    std::string preference_id;
    std::string reviewer_class;
    std::string rubric_version;
    std::string decision;
    bool blind = false;
    bool domain_expert = false;
    bool conflict_recorded = false;
};

struct ReviewSummary {
    std::size_t review_count = 0;
    std::size_t pass_count = 0;
    std::size_t conflict_count = 0;
    std::size_t expert_review_count = 0;
    double pass_rate = 0.0;
    bool blind_protocol_valid = false;
    bool domain_expert_coverage = false;
    bool disagreement_visible = false;
};

ReviewSummary validate_blind_reviews(const PreferenceManifest& manifest,
                                     const std::vector<BlindReviewRecord>& reviews);

}  // namespace cct
