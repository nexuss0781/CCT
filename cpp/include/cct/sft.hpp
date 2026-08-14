#pragma once

#include "cct/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class SftTaskKind : std::uint8_t {
    Classification = 0,
    StructuredExtraction = 1,
    GroundedQuestionAnswering = 2,
    Summarization = 3,
    CodeUnderstanding = 4,
    WorkflowDrafting = 5
};

enum class SftOutputKind : std::uint8_t {
    Label = 0,
    Json = 1,
    Grounded = 2,
    BoundedText = 3,
    Draft = 4
};

class SftError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct SftTaskSchema {
    std::string task_id;
    SftTaskKind kind = SftTaskKind::Classification;
    std::string schema_version = "v1";
    SftOutputKind output_kind = SftOutputKind::Label;
    std::vector<std::string> labels;
    std::size_t maximum_output_bytes = 256;
    bool requires_citations = false;
    bool allows_abstention = true;
    std::string policy_class = "bounded";
};

struct SftInstructionExample {
    std::string example_id;
    std::string task_id;
    std::string schema_version;
    std::string input;
    std::string target;
    std::string target_label;
    std::string input_provenance;
    std::string target_provenance;
    std::string policy_class;
    std::string split;
    std::string evaluator_owner;
    std::string source_hash;
    std::string target_hash;
    std::string example_hash;
    std::string citation_id;
    std::size_t source_span_start = 0;
    std::size_t source_span_end = 0;
    bool training_allowed = false;
    bool evaluation_allowed = false;
    bool evaluator_only = false;
};

struct SftManifest {
    std::string manifest_hash;
    std::vector<SftInstructionExample> examples;

    static SftManifest build(const std::vector<SftInstructionExample>& examples,
                             const std::vector<SftTaskSchema>& schemas);
    std::vector<SftInstructionExample> training_examples() const;
    std::vector<SftInstructionExample> evaluation_examples() const;
    bool contains_evaluator_training() const;
    std::string serialize() const;
    static SftManifest deserialize(const std::string& serialized);
};

struct FormattedInstruction {
    std::string example_id;
    std::string serialized;
    std::vector<TokenId> token_ids;
    std::vector<std::uint8_t> loss_mask;
    std::size_t target_token_start = 0;
    std::size_t target_token_end = 0;
};

class SftFormatter {
public:
    static FormattedInstruction format(const SftInstructionExample& example, const SftTaskSchema& schema,
                                       const Tokenizer& tokenizer);
    static std::string mask_policy_name();
};

struct SftAdapterSpec {
    std::string adapter_id;
    std::string task_id;
    std::string domain;
    std::string version = "adapter-v1";
    std::size_t rank = 1;
    std::string target_module = "output_projection";
    std::string base_checkpoint_hash;
    std::string training_manifest_hash;
    std::vector<std::string> permissions;
};

struct SftModelConfig {
    std::string base_checkpoint_hash;
    std::string task_id;
    std::size_t feature_dim = 8;
    std::size_t label_count = 2;
    std::uint64_t seed = 0;
};

struct SftOptimizerConfig {
    double learning_rate = 0.05;
    double clip_norm = 2.0;
    double weight_decay = 0.0;
    std::size_t total_steps = 40;
};

struct SftPrediction {
    std::string task_id;
    std::string label;
    std::string output;
    double confidence = 0.0;
    std::string citation_id;
    bool schema_valid = false;
    bool citation_valid = false;
    bool abstained = false;
};

struct SftEvaluation {
    double cross_entropy = 0.0;
    double accuracy = 0.0;
    double schema_validity = 0.0;
    double citation_integrity = 0.0;
    double abstention_rate = 0.0;
    std::size_t example_count = 0;
    std::size_t correct_count = 0;
    std::size_t valid_schema_count = 0;
    std::size_t valid_citation_count = 0;
    std::size_t abstention_count = 0;
    double elapsed_seconds = 0.0;
    bool finite = false;
};

class SftAdapter;

class SftModel {
public:
    explicit SftModel(SftModelConfig config);

    const SftModelConfig& config() const noexcept { return config_; }
    std::size_t parameter_count() const noexcept { return parameters_.size(); }
    std::vector<double> parameter_vector() const { return parameters_; }
    void set_parameter_vector(const std::vector<double>& values);
    std::string parameter_checksum() const;
    std::string name() const;

    SftPrediction predict(const SftInstructionExample& example, const SftTaskSchema& schema,
                          const SftAdapter* adapter = nullptr) const;
    SftEvaluation evaluate(const std::vector<SftInstructionExample>& examples,
                           const SftTaskSchema& schema, const SftAdapter* adapter = nullptr) const;
    double loss(const SftInstructionExample& example, const SftTaskSchema& schema,
                const SftAdapter* adapter = nullptr) const;
    std::vector<double> gradients(const SftInstructionExample& example, const SftTaskSchema& schema) const;
    void apply_gradient(const std::vector<double>& gradient, const SftOptimizerConfig& optimizer);

    void save(std::ostream& stream) const;
    static SftModel load(std::istream& stream);
    SftModel merged(const SftAdapter& adapter) const;

private:
    SftModelConfig config_;
    std::vector<double> parameters_;

    void validate() const;
    std::vector<double> features(const std::string& input) const;
    std::vector<double> logits(const std::vector<double>& features, const SftAdapter* adapter) const;
    static std::size_t label_index(const std::string& label, const SftTaskSchema& schema);
    static std::string json_output(const SftInstructionExample& example, const SftTaskSchema& schema,
                                   const std::string& label, double confidence);
};

class SftAdapter {
public:
    explicit SftAdapter(SftAdapterSpec spec, SftModelConfig base_config);

    const SftAdapterSpec& spec() const noexcept { return spec_; }
    const SftModelConfig& base_config() const noexcept { return base_config_; }
    std::size_t parameter_count() const noexcept { return parameters_.size(); }
    std::vector<double> parameter_vector() const { return parameters_; }
    std::string parameter_checksum() const;
    void set_parameter_vector(const std::vector<double>& values);
    std::vector<double> gradients(const SftModel& base, const SftInstructionExample& example,
                                  const SftTaskSchema& schema) const;
    void apply_gradient(const std::vector<double>& gradient, const SftOptimizerConfig& optimizer);
    void save(std::ostream& stream) const;
    static SftAdapter load(std::istream& stream);

private:
    SftAdapterSpec spec_;
    SftModelConfig base_config_;
    std::vector<double> parameters_;
};

class SftAdapterRegistry {
public:
    void register_adapter(const SftAdapter& adapter);
    bool authorize(const std::string& adapter_id, const std::string& task_id, const std::string& domain,
                   const std::string& base_hash, const std::string& training_manifest_hash, const std::string& permission) const;
    const SftAdapter& load_authorized(const std::string& adapter_id, const std::string& task_id, const std::string& domain,
                                       const std::string& base_hash, const std::string& training_manifest_hash,
                                       const std::string& permission) const;
    const std::vector<SftAdapter>& adapters() const noexcept { return adapters_; }
    std::string serialize() const;
    static SftAdapterRegistry deserialize(const std::string& serialized);

private:
    std::vector<SftAdapter> adapters_;
};

class StructuredDecoder {
public:
    static SftPrediction validate(const SftPrediction& prediction, const SftInstructionExample& example,
                                  const SftTaskSchema& schema);
};

struct SftRetentionReport {
    SftEvaluation base;
    SftEvaluation adapted;
    double relative_loss_change = 0.0;
    bool safety_boundary_preserved = false;
    bool within_budget = false;
};

std::string sft_task_kind_name(SftTaskKind kind);
std::string sft_output_kind_name(SftOutputKind kind);
std::string sft_hash(const std::string& serialized);
std::string sft_example_hash(const SftInstructionExample& example);

}  // namespace cct
