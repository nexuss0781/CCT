#pragma once

#include "cct/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class NlpModelKind : std::uint8_t {
    Track1CctRecurrence = 0,
    DenseCausalAttention = 1,
    GRU = 2,
    DiagonalSSM = 3
};

class NlpTrainingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct NlpSequence {
    std::string sequence_id;
    std::string record_id;
    std::vector<TokenId> input_ids;
    std::vector<TokenId> target_ids;
    std::vector<std::uint8_t> loss_mask;
};

struct NlpDataset {
    std::string tokenizer_hash;
    std::string dataset_hash;
    std::size_t context_length = 0;
    std::vector<NlpSequence> train;
    std::vector<NlpSequence> validation;
    std::size_t train_tokens = 0;
    std::size_t validation_tokens = 0;

    static NlpDataset build(const std::vector<EncodedDocument>& train_documents,
                            const std::vector<EncodedDocument>& validation_documents,
                            const std::string& tokenizer_hash, std::size_t context_length);
};

struct NlpModelConfig {
    NlpModelKind kind = NlpModelKind::Track1CctRecurrence;
    std::size_t vocabulary_size = 0;
    std::size_t embedding_dim = 8;
    std::size_t hidden_dim = 8;
    std::size_t context_length = 32;
    std::uint64_t seed = 0;
    bool compact_vocabulary = false;
    TokenId token_id_limit = 0;

};

struct NlpOptimizerConfig {
    double learning_rate = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 1e-4;
    double clip_norm = 1.0;
    std::size_t warmup_steps = 2;
    std::size_t batch_size = 1;
    std::size_t total_steps = 100;
    std::size_t validation_interval_steps = 1;
};

struct NlpEvaluation {
    double cross_entropy = 0.0;
    double perplexity = 0.0;
    double token_accuracy = 0.0;
    double gradient_norm = 0.0;
    std::size_t token_count = 0;
    double elapsed_seconds = 0.0;
    double tokens_per_second = 0.0;
    bool finite = false;
};

struct NlpGradientResult {
    double cross_entropy = 0.0;
    std::size_t token_count = 0;
    double gradient_norm = 0.0;
    std::vector<double> gradients;
};

struct NlpPreferencePair {
    NlpSequence preferred;
    NlpSequence rejected;
};

struct NlpTrainingPoint {
    std::size_t step = 0;
    std::size_t data_cursor = 0;
    double learning_rate = 0.0;
    double train_loss = 0.0;
    double validation_loss = 0.0;
    double validation_perplexity = 0.0;
    double gradient_norm = 0.0;
    std::size_t token_count = 0;
    bool validation_performed = false;
    double training_elapsed_seconds = 0.0;
    double validation_elapsed_seconds = 0.0;
};

struct NlpTrainerState {
    std::size_t optimizer_step = 0;
    std::size_t data_cursor = 0;
    std::uint64_t seed = 0;
    std::string rng_state;
};

class NextTokenModel {
public:
    explicit NextTokenModel(NlpModelConfig config);

    const NlpModelConfig& config() const noexcept { return config_; }
    NlpModelKind kind() const noexcept { return config_.kind; }
    std::string name() const;
    std::size_t parameter_count() const noexcept { return parameters_.size(); }
    std::size_t state_memory_bytes() const noexcept;
    std::vector<double> parameter_vector() const { return parameters_; }
    void set_parameter_vector(const std::vector<double>& values);

    NlpGradientResult loss_and_gradients(const NlpSequence& sequence) const;
    std::vector<double> next_logits(const std::vector<TokenId>& context) const;
    TokenId token_id_from_logit_slot(std::size_t slot) const;
    std::size_t logit_slot_for_token_id(TokenId token) const;
    NlpEvaluation evaluate(const std::vector<NlpSequence>& sequences) const;
    double loss_only(const NlpSequence& sequence) const;
    void apply_gradient(const std::vector<double>& gradients, const NlpOptimizerConfig& optimizer,
                        NlpTrainerState& state, double* applied_learning_rate = nullptr);

    void save_model(std::ostream& stream) const;
    static NextTokenModel load_model(std::istream& stream);

private:
    NlpModelConfig config_;
    std::vector<double> parameters_;

    void validate_config() const;
    void initialize();
    std::vector<double> embedding(const TokenId id) const;
    void validate_sequence(const NlpSequence& sequence) const;
    std::size_t embedding_offset() const noexcept;
    std::size_t cct_offset() const noexcept;
    std::size_t gru_offset() const noexcept;
    std::size_t ssm_offset() const noexcept;
    std::size_t dense_offset() const noexcept;
    std::size_t head_offset() const noexcept;
    std::size_t skip_offset() const noexcept;
    std::size_t bias_offset() const noexcept;
    std::size_t expected_parameter_count() const noexcept;
};

struct NlpCheckpointInfo {
    std::string tokenizer_hash;
    std::string dataset_hash;
    std::string training_contract_hash;
    std::string checkpoint_hash;
    std::string session_id;
    std::string parent_checkpoint_hash;
    std::size_t optimizer_step = 0;
    std::size_t data_cursor = 0;
};

class NlpTrainer {
public:
    NlpTrainer(NlpModelConfig model_config, NlpOptimizerConfig optimizer_config,
               std::string tokenizer_hash, std::string dataset_hash);

    NextTokenModel& model() noexcept { return model_; }
    const NextTokenModel& model() const noexcept { return model_; }
    const NlpOptimizerConfig& optimizer_config() const noexcept { return optimizer_config_; }
    const NlpTrainerState& state() const noexcept { return state_; }
    const std::vector<NlpTrainingPoint>& history() const noexcept { return history_; }
    const NlpCheckpointInfo& checkpoint_info() const noexcept { return checkpoint_info_; }

    NlpEvaluation evaluate(const std::vector<NlpSequence>& sequences) const;
    NlpTrainingPoint train_step(const NlpDataset& dataset);
    NlpTrainingPoint train_preference_step(const NlpPreferencePair& pair, double margin = 0.0);
    std::vector<NlpTrainingPoint> train_steps(const NlpDataset& dataset, std::size_t steps);
    std::vector<NlpTrainingPoint> train_preference_steps(const std::vector<NlpPreferencePair>& pairs, std::size_t steps, double margin = 0.0);
    void save_checkpoint(const std::string& path) const;
    void begin_continuation(const std::string& dataset_hash, const std::string& session_id,
                            const std::string& parent_checkpoint_hash, std::size_t session_steps);
    const std::string& tokenizer_hash() const noexcept { return tokenizer_hash_; }
    const std::string& dataset_hash() const noexcept { return dataset_hash_; }
    static NlpTrainer load_checkpoint(const std::string& path, const std::string& expected_tokenizer_hash = {},
                                      const std::string& expected_dataset_hash = {});

private:
    NextTokenModel model_;
    NlpOptimizerConfig optimizer_config_;
    std::string tokenizer_hash_;
    std::string dataset_hash_;
    std::string training_contract_hash_;
    std::string session_id_;
    std::string parent_checkpoint_hash_;
    NlpTrainerState state_;
    std::vector<double> first_moment_;
    std::vector<double> second_moment_;
    std::vector<NlpTrainingPoint> history_;
    mutable NlpCheckpointInfo checkpoint_info_;

    double scheduled_learning_rate() const;
    void validate_optimizer() const;
    void validate_checkpoint_identity(const std::string& tokenizer_hash, const std::string& dataset_hash) const;
};

std::string nlp_model_kind_name(NlpModelKind kind);
std::string nlp_checkpoint_hash(const std::string& serialized_checkpoint);

}  // namespace cct
