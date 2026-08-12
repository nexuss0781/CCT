#pragma once

#include "cct/nlp_trainer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace cct {

enum class ScalingBackend : std::uint8_t {
    CpuReference = 0,
    CpuFused = 1,
    CudaUnavailable = 2,
    HipUnavailable = 3
};

std::string scaling_backend_name(ScalingBackend backend);

struct BackendCapabilities {
    bool cpu_reference = false;
    bool cpu_fused = false;
    bool cuda_available = false;
    bool hip_available = false;
    std::string hardware_class;
    std::string compiler_class;
};

struct ScalingPointConfig {
    ScalingBackend backend = ScalingBackend::CpuReference;
    NlpModelConfig model;
    NlpOptimizerConfig optimizer;
    std::string tokenizer_hash;
    std::string dataset_hash;
    std::size_t context_length = 0;
    std::size_t training_horizon = 0;
    std::size_t worker_count = 1;
};

struct ResourceProfile {
    double wall_seconds = 0.0;
    double cpu_seconds = 0.0;
    double tokens_per_second = 0.0;
    std::size_t peak_resident_bytes = 0;
    std::size_t state_memory_bytes = 0;
    std::size_t parameter_memory_bytes = 0;
};

struct ScalingPoint {
    ScalingPointConfig config;
    double initial_train_loss = 0.0;
    double final_train_loss = 0.0;
    double initial_validation_loss = 0.0;
    double final_validation_loss = 0.0;
    double final_perplexity = 0.0;
    std::size_t training_tokens = 0;
    std::size_t optimizer_steps = 0;
    std::size_t parameter_count = 0;
    ResourceProfile resources;
    std::string parameter_checksum;
    bool finite = false;
};

struct AtomicCheckpointResult {
    std::string committed_path;
    std::string checkpoint_hash;
    bool committed = false;
    bool temporary_interruption_preserved_commit = false;
};

class ScalingRunner {
public:
    static BackendCapabilities probe_capabilities();
    static ScalingPoint run(const ScalingPointConfig& config, const NlpDataset& dataset);
    static std::string parameter_checksum(const NextTokenModel& model);
    static AtomicCheckpointResult save_atomic(const NlpTrainer& trainer, const std::string& path);
    static NlpTrainer load_verified(const std::string& path, const std::string& tokenizer_hash,
                                    const std::string& dataset_hash);
};

}  // namespace cct
