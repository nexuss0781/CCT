#include "cct/scaling_systems.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

NlpSequence sequence(const std::string& id, const std::vector<TokenId>& input) {
    NlpSequence result;
    result.sequence_id = id;
    result.record_id = id;
    result.input_ids = input;
    result.target_ids.assign(input.size(), Tokenizer::kPadId);
    result.loss_mask.assign(input.size(), 0U);
    for (std::size_t index = 0; index + 1U < input.size(); ++index) {
        result.target_ids[index] = input[index + 1U];
        result.loss_mask[index] = 1U;
    }
    return result;
}

NlpDataset dataset() {
    NlpDataset result;
    result.tokenizer_hash = "stage12-tokenizer";
    result.dataset_hash = "stage12-dataset";
    result.context_length = 8;
    result.train = {sequence("train-0", {256, 257, 258, 259, 260, 261}), sequence("train-1", {256, 258, 260, 262, 264, 266})};
    result.validation = {sequence("validation-0", {256, 257, 258, 259, 260, 261})};
    result.train_tokens = 10;
    result.validation_tokens = 5;
    return result;
}

ScalingPointConfig point_config(const ScalingBackend backend, const std::size_t width, const std::size_t context,
                                const std::size_t horizon, const std::uint64_t seed) {
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.01;
    optimizer.warmup_steps = 1;
    optimizer.total_steps = horizon;
    optimizer.clip_norm = 1.0;
    optimizer.weight_decay = 0.0;
    return {backend, {NlpModelKind::Track1CctRecurrence, 768, width, width, context, seed}, optimizer,
            "stage12-tokenizer", "stage12-dataset", context, horizon, 1};
}

void test_capabilities_and_backend_boundary() {
    const auto capabilities = ScalingRunner::probe_capabilities();
    require(capabilities.cpu_reference && capabilities.cpu_fused, "CPU scaling paths are not available");
    require(!capabilities.cuda_available && !capabilities.hip_available, "unavailable accelerator was falsely reported");
    require(scaling_backend_name(ScalingBackend::CpuReference) == "cpu_reference" &&
                scaling_backend_name(ScalingBackend::CpuFused) == "cpu_fused" &&
                scaling_backend_name(ScalingBackend::CudaUnavailable) == "cuda_unavailable",
            "backend names are not stable");
    bool rejected = false;
    try {
        auto config = point_config(ScalingBackend::CudaUnavailable, 2, 8, 2, 3);
        static_cast<void>(ScalingRunner::run(config, dataset()));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "unavailable CUDA backend was accepted for execution");
}

void test_reference_fused_parity_and_resources() {
    const auto data = dataset();
    const auto reference = ScalingRunner::run(point_config(ScalingBackend::CpuReference, 2, 8, 4, 7), data);
    const auto fused = ScalingRunner::run(point_config(ScalingBackend::CpuFused, 2, 8, 4, 7), data);
    require(reference.finite && fused.finite, "reference/fused scaling point is not finite");
    require(std::abs(reference.final_train_loss - fused.final_train_loss) <= 1e-12 &&
                std::abs(reference.final_validation_loss - fused.final_validation_loss) <= 1e-12 &&
                reference.parameter_checksum == fused.parameter_checksum,
            "reference/fused deterministic parity failed");
    require(reference.training_tokens == 20U && reference.optimizer_steps == 4U && reference.resources.tokens_per_second > 0.0 &&
                reference.resources.peak_resident_bytes > 0U && reference.resources.state_memory_bytes > 0U &&
                reference.resources.parameter_memory_bytes > 0U,
            "scaling resource or accounting fields are incomplete");
}

void test_scaling_matrix_points_and_repeatability() {
    const auto data = dataset();
    for (const auto width : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        for (const auto context : {std::size_t{8}, std::size_t{16}}) {
            auto point_data = data;
            point_data.context_length = context;
            const auto first = ScalingRunner::run(point_config(ScalingBackend::CpuFused, width, context, 4, 11), point_data);
            const auto second = ScalingRunner::run(point_config(ScalingBackend::CpuFused, width, context, 4, 11), point_data);
            require(first.finite && second.finite && first.parameter_checksum == second.parameter_checksum &&
                        std::abs(first.final_validation_loss - second.final_validation_loss) <= 1e-12,
                    "same scaling point did not reproduce");
        }
    }
}

void test_atomic_checkpoint_and_recovery() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.01;
    optimizer.total_steps = 4;
    optimizer.warmup_steps = 1;
    optimizer.weight_decay = 0.0;
    NlpTrainer trainer({NlpModelKind::Track1CctRecurrence, 768, 2, 2, 8, 17}, optimizer, data.tokenizer_hash, data.dataset_hash);
    trainer.train_steps(data, 2);
    const auto path = std::string("artifacts/stage-12-test-checkpoint.bin");
    const auto result = ScalingRunner::save_atomic(trainer, path);
    require(result.committed && result.temporary_interruption_preserved_commit && !result.checkpoint_hash.empty(),
            "atomic checkpoint did not preserve committed state");
    const auto restored = ScalingRunner::load_verified(path, data.tokenizer_hash, data.dataset_hash);
    require(restored.state().optimizer_step == 2U && restored.state().data_cursor == 2U,
            "atomic checkpoint did not restore cursor/step");
    std::filesystem::remove(path);
    bool rejected = false;
    try {
        static_cast<void>(ScalingRunner::load_verified(path, data.tokenizer_hash, data.dataset_hash));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "missing committed checkpoint was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"capabilities_and_backend_boundary", test_capabilities_and_backend_boundary},
        {"reference_fused_parity_and_resources", test_reference_fused_parity_and_resources},
        {"scaling_matrix_points_and_repeatability", test_scaling_matrix_points_and_repeatability},
        {"atomic_checkpoint_and_recovery", test_atomic_checkpoint_and_recovery}};
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
            ++passed;
        } catch (const std::exception& error) {
            std::cout << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << "SUMMARY " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
