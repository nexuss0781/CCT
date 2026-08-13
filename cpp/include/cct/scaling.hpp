#pragma once

#include "cct/baselines.hpp"
#include "cct/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cct {

enum class Stage5ModelKind : std::uint8_t {
    DenseCausalAttention = 0,
    GRU = 1,
    DiagonalSSM = 2,
    CCTNoMemory = 3,
    CCTFrozenMemory = 4
};

struct Stage5ModelConfig {
    std::size_t input_dim = 32;
    std::size_t hidden_dim = 8;
    std::size_t output_dim = 32;
    std::uint64_t seed = 0;
    Stage5ModelKind kind = Stage5ModelKind::CCTNoMemory;
};

struct Stage5TrainConfig {
    std::size_t epochs = 1;
    double learning_rate = 0.05;
    double clip_norm = 5.0;
    std::uint64_t data_cursor = 0;
    std::uint64_t manifest_fingerprint = 0;
};

struct Stage5Evaluation {
    double mean_squared_loss = 0.0;
    double cross_entropy = 0.0;
    double token_accuracy = 0.0;
    std::size_t token_count = 0;
};

class Stage5Vocabulary {
public:
    static constexpr std::size_t kByteVocabularySize = 258;
    static constexpr std::size_t kUnknownToken = 256;
    static constexpr std::size_t kEndOfTextToken = 257;

    static std::vector<std::size_t> encode_bytes(const std::string& text, bool append_end = false);
    static std::string decode_bytes(const std::vector<std::size_t>& tokens);
    static std::vector<std::size_t> compact_encode(const std::string& text, const std::string& alphabet,
                                                   std::size_t unknown_token);
    static std::string compact_decode(const std::vector<std::size_t>& tokens, const std::string& alphabet,
                                      std::size_t unknown_token);
};

class Stage5LanguageModel {
public:
    explicit Stage5LanguageModel(Stage5ModelConfig config);

    const Stage5ModelConfig& config() const noexcept { return config_; }
    Stage5ModelKind kind() const noexcept { return config_.kind; }
    std::string name() const;
    bool uses_memory() const noexcept { return config_.kind == Stage5ModelKind::CCTFrozenMemory; }

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) const;
    Stage5Evaluation evaluate(const std::vector<std::vector<std::vector<double>>>& input_batch,
                               const std::vector<std::vector<std::vector<double>>>& target_batch,
                               const std::vector<std::vector<std::uint8_t>>& masks = {}) const;
    void train(const std::vector<std::vector<std::vector<double>>>& input_batch,
               const std::vector<std::vector<std::vector<double>>>& target_batch,
               const std::vector<std::vector<std::uint8_t>>& masks,
               const Stage5TrainConfig& config);
    /** Common deterministic finite-difference reference path for matched architecture comparisons. */
    void train_reference_finite_difference(const std::vector<std::vector<std::vector<double>>>& input_batch,
                                           const std::vector<std::vector<std::vector<double>>>& target_batch,
                                           const std::vector<std::vector<std::uint8_t>>& masks,
                                           const Stage5TrainConfig& config,
                                           double finite_difference_epsilon = 1e-5);

    std::size_t parameter_count() const noexcept;
    std::size_t state_memory_bytes() const noexcept;
    std::vector<double> parameter_vector() const;
    void set_parameter_vector(const std::vector<double>& values);
    std::uint64_t optimizer_step() const noexcept { return optimizer_step_; }
    std::uint64_t data_cursor() const noexcept { return data_cursor_; }
    std::uint64_t manifest_fingerprint() const noexcept { return manifest_fingerprint_; }

    void save_checkpoint(const std::string& path) const;
    static Stage5LanguageModel load_checkpoint(const std::string& path);

private:
    Stage5ModelConfig config_;
    std::unique_ptr<MatchedBaseline> baseline_;
    std::unique_ptr<SelectiveSequenceCore> cct_;
    std::uint64_t optimizer_step_ = 0;
    std::uint64_t data_cursor_ = 0;
    std::uint64_t manifest_fingerprint_ = 0;

    void initialize();
};

struct Stage5MemoryEvaluation {
    std::size_t no_memory_hits = 0;
    std::size_t memory_hits = 0;
    double retrieval_latency_ms = 0.0;
    bool evidence_ids_attributed = false;
};

Stage5MemoryEvaluation evaluate_stage5_memory_augmentation(PersistentMemory& memory);

}  // namespace cct
