#include "cct/nlp_trainer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

NlpModelConfig model_config(const NlpModelKind kind, const std::uint64_t seed = 7) {
    return {kind, 768, 3, 3, 8, seed};
}

NlpSequence sequence(const std::string& id, const std::vector<TokenId>& input) {
    require(input.size() >= 3U, "test sequence is too short");
    NlpSequence result;
    result.sequence_id = id;
    result.record_id = id;
    result.input_ids = input;
    result.target_ids.resize(input.size(), Tokenizer::kPadId);
    result.loss_mask.assign(input.size(), 0U);
    for (std::size_t index = 0; index + 1U < input.size(); ++index) {
        result.target_ids[index] = input[index + 1U];
        result.loss_mask[index] = 1U;
    }
    return result;
}

NlpDataset dataset() {
    NlpDataset result;
    result.tokenizer_hash = "tokenizer-hash";
    result.dataset_hash = "dataset-hash";
    result.context_length = 8;
    result.train = {sequence("train-0", {256, 257, 258, 259, 260}), sequence("train-1", {256, 258, 260, 262, 264}),
                    sequence("train-2", {257, 259, 261, 263, 265})};
    result.validation = {sequence("validation-0", {256, 257, 258, 259, 260}), sequence("validation-1", {256, 258, 260, 262, 264})};
    result.train_tokens = 12;
    result.validation_tokens = 8;
    return result;
}

double relative_error(const double analytical, const double numerical) {
    return std::abs(analytical - numerical) / std::max({1.0, std::abs(analytical), std::abs(numerical)});
}

void test_objective_mask_and_finite_metrics() {
    auto model = NextTokenModel(model_config(NlpModelKind::Track1CctRecurrence));
    const auto item = sequence("objective", {256, 257, 258, 259, 260});
    const auto gradient = model.loss_and_gradients(item);
    require(std::isfinite(gradient.cross_entropy) && gradient.token_count == 4U && std::isfinite(gradient.gradient_norm),
            "Track 1 categorical objective is not finite or mask count is wrong");
    const auto evaluation = model.evaluate({item});
    require(evaluation.finite && std::isfinite(evaluation.perplexity) && evaluation.token_count == 4U,
            "Track 1 evaluation metrics are not finite");
    const auto masked = NlpSequence{"masked", item.record_id, item.input_ids, item.target_ids, {1, 0, 0, 0, 0}};
    require(model.loss_and_gradients(masked).token_count == 1U, "loss mask did not exclude inactive targets");
}

void test_track1_recurrence_gradient_finite_difference() {
    const auto config = model_config(NlpModelKind::Track1CctRecurrence, 11);
    const auto item = sequence("gradient", {256, 257, 258, 259, 260});
    const auto model = NextTokenModel(config);
    const auto analytical = model.loss_and_gradients(item);
    const auto original = model.parameter_vector();
    const std::vector<std::size_t> indices{256U * 3U, 768U * 3U, 768U * 3U + 3U * 3U + 3U,
                                           768U * 3U + 4U * 3U * 3U + 3U + 768U * 3U};
    for (const auto index : indices) {
        require(index < original.size(), "gradient spot-check index exceeds model parameters");
        auto plus_values = original;
        auto minus_values = original;
        plus_values[index] += 1e-5;
        minus_values[index] -= 1e-5;
        auto plus = NextTokenModel(config);
        auto minus = NextTokenModel(config);
        plus.set_parameter_vector(plus_values);
        minus.set_parameter_vector(minus_values);
        const auto numerical = (plus.loss_only(item) - minus.loss_only(item)) / 2e-5;
        require(relative_error(analytical.gradients[index], numerical) <= 1e-4,
                "Track 1 analytic gradient disagrees with finite difference at parameter " + std::to_string(index));
    }
}

void test_track1_model_identity_and_serialization() {
    const auto model = NextTokenModel(model_config(NlpModelKind::Track1CctRecurrence, 29));
    require(model.kind() == NlpModelKind::Track1CctRecurrence, "Track 1 model kind changed at construction");
    require(model.name() == "track1_cct_recurrence", "Track 1 model identity name is ambiguous");
    std::ostringstream serialized;
    model.save_model(serialized);
    const auto content = serialized.str();
    require(content.rfind("NLP_MODEL_V3\n", 0U) == 0U, "new model serialization did not publish V3 identity");
    std::istringstream current_stream(content);
    const auto restored = NextTokenModel::load_model(current_stream);
    require(restored.kind() == NlpModelKind::Track1CctRecurrence && restored.name() == model.name(),
            "V3 model identity did not survive serialization");
    require(restored.parameter_vector() == model.parameter_vector(), "V3 model parameters changed across serialization");
    auto legacy = content;
    legacy.replace(0U, 12U, "NLP_MODEL_V2");
    std::istringstream legacy_stream(legacy);
    const auto legacy_restored = NextTokenModel::load_model(legacy_stream);
    require(legacy_restored.kind() == NlpModelKind::Track1CctRecurrence &&
                legacy_restored.name() == "track1_cct_recurrence",
            "legacy model checkpoint did not map to the explicit Track 1 identity");
}

void test_optimizer_direction_and_schedule() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.02;
    optimizer.warmup_steps = 2;
    optimizer.total_steps = 10;
    optimizer.validation_interval_steps = 2;
    NlpTrainer trainer(model_config(NlpModelKind::Track1CctRecurrence, 13), optimizer, data.tokenizer_hash, data.dataset_hash);
    const auto before = trainer.evaluate(data.validation).cross_entropy;
    const auto first = trainer.train_step(data);
    const auto second = trainer.train_step(data);
    require(first.learning_rate < second.learning_rate && first.step == 1U && second.step == 2U,
            "NLP warmup schedule or optimizer step is incorrect");
    require(std::isfinite(before) && std::isfinite(first.train_loss) && trainer.state().data_cursor == 2U,
            "NLP optimizer produced invalid state");
    require(!first.validation_performed && second.validation_performed && first.training_elapsed_seconds >= 0.0 &&
                second.validation_elapsed_seconds >= 0.0,
            "NLP validation cadence or timing evidence is incorrect");
}

void test_deterministic_initialization_and_controls() {
    const auto data = dataset();
    for (const auto kind : {NlpModelKind::Track1CctRecurrence, NlpModelKind::DenseCausalAttention, NlpModelKind::GRU, NlpModelKind::DiagonalSSM}) {
        const auto first = NextTokenModel(model_config(kind, 19));
        const auto second = NextTokenModel(model_config(kind, 19));
        require(first.parameter_vector() == second.parameter_vector(), "same seed did not initialize deterministically");
        require(first.parameter_count() > 0U && first.state_memory_bytes() > 0U, "model accounting is empty");
        const auto evaluation = first.evaluate(data.validation);
        require(evaluation.finite && evaluation.tokens_per_second > 0.0, "matched control evaluation is not finite");
    }
}

void test_checkpoint_resume_exactness_and_fail_closed() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.01;
    optimizer.warmup_steps = 1;
    optimizer.total_steps = 8;
    auto uninterrupted = NlpTrainer(model_config(NlpModelKind::Track1CctRecurrence, 23), optimizer, data.tokenizer_hash, data.dataset_hash);
    uninterrupted.train_steps(data, 5);
    auto interrupted = NlpTrainer(model_config(NlpModelKind::Track1CctRecurrence, 23), optimizer, data.tokenizer_hash, data.dataset_hash);
    interrupted.train_steps(data, 2);
    interrupted.save_checkpoint("artifacts/stage-11-test-checkpoint.bin");
    const auto restored = NlpTrainer::load_checkpoint("artifacts/stage-11-test-checkpoint.bin", data.tokenizer_hash, data.dataset_hash);
    require(restored.state().optimizer_step == 2U && restored.state().data_cursor == 2U,
            "checkpoint did not preserve optimizer step and data cursor");
    auto resumed = restored;
    resumed.train_steps(data, 3);
    const auto left = uninterrupted.model().parameter_vector();
    const auto right = resumed.model().parameter_vector();
    require(left.size() == right.size(), "resume parameter size changed");
    for (std::size_t index = 0; index < left.size(); ++index) {
        require(std::abs(left[index] - right[index]) <= 1e-12, "interrupted/resumed parameters diverged");
    }
    bool rejected = false;
    try {
        static_cast<void>(NlpTrainer::load_checkpoint("artifacts/stage-11-test-checkpoint.bin", "wrong-tokenizer", data.dataset_hash));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "wrong tokenizer checkpoint identity was accepted");
    std::ifstream checkpoint_input("artifacts/stage-11-test-checkpoint.bin");
    std::ostringstream checkpoint_text;
    checkpoint_text << checkpoint_input.rdbuf();
    const auto contract_start = checkpoint_text.str().find("training_contract_hash=");
    require(contract_start != std::string::npos, "training contract digest was not serialized");
    const auto contract_end = checkpoint_text.str().find('\n', contract_start);
    auto tampered = checkpoint_text.str();
    tampered.replace(contract_start, contract_end - contract_start, "training_contract_hash=" + std::string(64U, '0'));
    std::ofstream tampered_output("artifacts/stage-11-tampered-contract.bin", std::ios::trunc);
    tampered_output << tampered;
    tampered_output.close();
    rejected = false;
    try {
        static_cast<void>(NlpTrainer::load_checkpoint("artifacts/stage-11-tampered-contract.bin"));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "tampered training contract identity was accepted");
    std::ofstream malformed("artifacts/stage-11-malformed-checkpoint.bin");
    malformed << "CCT_NLP_CHECKPOINT_V2\ntruncated";
    malformed.close();
    rejected = false;
    try {
        static_cast<void>(NlpTrainer::load_checkpoint("artifacts/stage-11-malformed-checkpoint.bin"));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "malformed checkpoint was accepted");
}

void test_invalid_masks_and_nonfinite_parameters_fail_closed() {
    auto model = NextTokenModel(model_config(NlpModelKind::Track1CctRecurrence));
    auto invalid = sequence("invalid", {256, 257, 258});
    invalid.loss_mask = {0, 0, 0};
    bool rejected = false;
    try {
        static_cast<void>(model.loss_and_gradients(invalid));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "all-false loss mask was accepted");
    auto parameters = model.parameter_vector();
    parameters.front() = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        model.set_parameter_vector(parameters);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "non-finite model parameters were accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"objective_mask_and_finite_metrics", test_objective_mask_and_finite_metrics},
        {"track1_recurrence_gradient_finite_difference", test_track1_recurrence_gradient_finite_difference},
        {"track1_model_identity_and_serialization", test_track1_model_identity_and_serialization},
        {"optimizer_direction_and_schedule", test_optimizer_direction_and_schedule},
        {"deterministic_initialization_and_controls", test_deterministic_initialization_and_controls},
        {"checkpoint_resume_exactness_and_fail_closed", test_checkpoint_resume_exactness_and_fail_closed},
        {"invalid_masks_and_nonfinite_parameters_fail_closed", test_invalid_masks_and_nonfinite_parameters_fail_closed}};
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
