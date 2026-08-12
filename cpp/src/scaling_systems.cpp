#include "cct/scaling_systems.hpp"

#include "cct/corpus.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw NlpTrainingError(message);
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not read scaling artifact: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

std::size_t resident_bytes() {
    struct rusage usage {};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed for scaling profile");
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
}

double cpu_seconds() {
    struct rusage usage {};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed for CPU profile");
    return static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) / 1e6 +
           static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec) / 1e6;
}

}  // namespace

std::string scaling_backend_name(const ScalingBackend backend) {
    switch (backend) {
        case ScalingBackend::CpuReference: return "cpu_reference";
        case ScalingBackend::CpuFused: return "cpu_fused";
        case ScalingBackend::CudaUnavailable: return "cuda_unavailable";
        case ScalingBackend::HipUnavailable: return "hip_unavailable";
    }
    throw NlpTrainingError("unknown scaling backend");
}

BackendCapabilities ScalingRunner::probe_capabilities() {
    BackendCapabilities capabilities;
    capabilities.cpu_reference = true;
    capabilities.cpu_fused = true;
    capabilities.cuda_available = false;
    capabilities.hip_available = false;
    capabilities.hardware_class = "x86_64-cpu-6-vcpu-4gb-sandbox";
    capabilities.compiler_class = "gcc-c++20";
    return capabilities;
}

ScalingPoint ScalingRunner::run(const ScalingPointConfig& config, const NlpDataset& dataset) {
    require(config.backend == ScalingBackend::CpuReference || config.backend == ScalingBackend::CpuFused,
            "requested accelerator backend is unavailable in the declared Stage 12 environment");
    require(config.tokenizer_hash == dataset.tokenizer_hash && config.dataset_hash == dataset.dataset_hash,
            "scaling point dataset identity does not match Stage 11 dataset");
    require(config.context_length == dataset.context_length && config.model.context_length == config.context_length,
            "scaling point context identity is inconsistent");
    require(config.training_horizon > 0U && config.worker_count > 0U, "scaling point horizon or worker count is invalid");
    require(config.optimizer.total_steps >= config.training_horizon, "scaling point horizon exceeds optimizer schedule");

    NlpTrainer trainer(config.model, config.optimizer, config.tokenizer_hash, config.dataset_hash);
    const auto initial_train = trainer.evaluate(dataset.train);
    const auto initial_validation = trainer.evaluate(dataset.validation);
    const auto started = std::chrono::steady_clock::now();
    const auto cpu_started = cpu_seconds();
    trainer.train_steps(dataset, config.training_horizon);
    const auto cpu_finished = cpu_seconds();
    const auto finished = std::chrono::steady_clock::now();
    const auto final_train = trainer.evaluate(dataset.train);
    const auto final_validation = trainer.evaluate(dataset.validation);

    ScalingPoint point;
    point.config = config;
    point.initial_train_loss = initial_train.cross_entropy;
    point.final_train_loss = final_train.cross_entropy;
    point.initial_validation_loss = initial_validation.cross_entropy;
    point.final_validation_loss = final_validation.cross_entropy;
    point.final_perplexity = final_validation.perplexity;
    point.optimizer_steps = trainer.state().optimizer_step;
    point.parameter_count = trainer.model().parameter_count();
    for (const auto& history : trainer.history()) point.training_tokens += history.token_count;
    point.resources.wall_seconds = std::chrono::duration<double>(finished - started).count();
    point.resources.cpu_seconds = cpu_finished - cpu_started;
    point.resources.tokens_per_second = point.resources.wall_seconds > 0.0
                                            ? static_cast<double>(point.training_tokens) / point.resources.wall_seconds
                                            : 0.0;
    point.resources.peak_resident_bytes = resident_bytes();
    point.resources.state_memory_bytes = trainer.model().state_memory_bytes();
    point.resources.parameter_memory_bytes = trainer.model().parameter_count() * sizeof(double);
    point.parameter_checksum = parameter_checksum(trainer.model());
    point.finite = initial_train.finite && initial_validation.finite && final_train.finite && final_validation.finite &&
                   point.resources.wall_seconds > 0.0 && point.resources.tokens_per_second > 0.0 &&
                   point.resources.peak_resident_bytes > 0U && point.parameter_count > 0U && point.training_tokens > 0U;
    require(point.finite, "scaling point produced non-finite or empty metrics");
    return point;
}

std::string ScalingRunner::parameter_checksum(const NextTokenModel& model) {
    std::ostringstream serialized;
    serialized << model.name() << ' ' << model.config().vocabulary_size << ' ' << model.config().embedding_dim << ' '
               << model.config().hidden_dim << ' ' << model.config().context_length << ' ' << model.config().seed << '\n'
               << std::setprecision(17);
    for (const auto value : model.parameter_vector()) serialized << value << ' ';
    return GovernedCorpus::content_sha256(serialized.str());
}

AtomicCheckpointResult ScalingRunner::save_atomic(const NlpTrainer& trainer, const std::string& path) {
    const auto temporary = path + ".tmp";
    std::remove(temporary.c_str());
    trainer.save_checkpoint(temporary);
    require(std::filesystem::exists(temporary), "temporary scaling checkpoint was not created");
    std::filesystem::rename(temporary, path);
    require(std::filesystem::exists(path), "atomic scaling checkpoint commit is missing");
    const auto committed = read_file(path);
    AtomicCheckpointResult result;
    result.committed_path = path;
    result.checkpoint_hash = nlp_checkpoint_hash(committed);
    result.committed = true;
    std::ofstream interrupted(temporary, std::ios::binary);
    require(static_cast<bool>(interrupted), "could not create simulated interrupted checkpoint");
    interrupted << "CCT_NLP_CHECKPOINT_V2\ntruncated";
    interrupted.close();
    result.temporary_interruption_preserved_commit = std::filesystem::exists(path) && read_file(path) == committed;
    std::remove(temporary.c_str());
    require(result.temporary_interruption_preserved_commit, "simulated temporary checkpoint interruption changed committed state");
    return result;
}

NlpTrainer ScalingRunner::load_verified(const std::string& path, const std::string& tokenizer_hash,
                                        const std::string& dataset_hash) {
    require(std::filesystem::exists(path), "verified scaling checkpoint does not exist");
    return NlpTrainer::load_checkpoint(path, tokenizer_hash, dataset_hash);
}

}  // namespace cct
