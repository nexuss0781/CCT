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

void test_upgraded_baseline_gradients() {
    const auto item = sequence("baseline-gradient", {256, 257, 258, 259, 260});
    for (const auto kind : {NlpModelKind::GRU, NlpModelKind::DiagonalSSM, NlpModelKind::DenseCausalAttention}) {
        const auto config = model_config(kind, 17);
        const auto model = NextTokenModel(config);
        const auto analytical = model.loss_and_gradients(item);
        const auto original = model.parameter_vector();
        const std::vector<std::size_t> indices{0U, config.vocabulary_size * config.embedding_dim, original.size() / 2U, original.size() - 1U};
        for (const auto index : indices) {
            require(index < original.size(), "baseline gradient spot-check index exceeds model parameters");
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
                    "upgraded baseline analytic gradient disagrees with finite difference");
        }
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

void test_compact_vocabulary_mapping_and_serialization() {
    auto config = model_config(NlpModelKind::Track1CctRecurrence, 43);
    config.vocabulary_size = 513U;
    config.compact_vocabulary = true;
    config.token_id_limit = 767U;
    const auto model = NextTokenModel(config);
    require(model.logit_slot_for_token_id(Tokenizer::kEosId) == 0U && model.logit_slot_for_token_id(256U) == 1U &&
                model.token_id_from_logit_slot(0U) == Tokenizer::kEosId && model.token_id_from_logit_slot(1U) == 256U,
            "compact vocabulary token-slot mapping is incorrect");
    require(model.parameter_count() < NextTokenModel(model_config(NlpModelKind::Track1CctRecurrence, 43)).parameter_count(),
            "compact vocabulary did not reduce parameter allocation");
    const auto item = sequence("compact", {256, 257, 258, 259, 260});
    require(std::isfinite(model.loss_only(item)) && model.next_logits(item.input_ids).size() == 513U,
            "compact vocabulary model did not score or decode a valid sequence");
    std::ostringstream serialized;
    model.save_model(serialized);
    require(serialized.str().rfind("NLP_MODEL_V4\n", 0U) == 0U, "compact vocabulary did not publish V4 model identity");
    std::istringstream stream(serialized.str());
    const auto restored = NextTokenModel::load_model(stream);
    require(restored.config().compact_vocabulary && restored.config().token_id_limit == 767U &&
                restored.parameter_vector() == model.parameter_vector(),
            "compact vocabulary metadata did not survive V4 serialization");
    auto invalid = item;
    invalid.sequence_id = "compact-invalid";
    invalid.input_ids[0] = 100U;
    bool rejected = false;
    try {
        static_cast<void>(model.loss_only(invalid));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "compact vocabulary accepted an unavailable control token");
}

void test_track1_initialization_and_state_accounting() {
    const auto config = model_config(NlpModelKind::Track1CctRecurrence, 31);
    const auto model = NextTokenModel(config);
    const auto parameters = model.parameter_vector();
    const auto recurrent_offset = config.vocabulary_size * config.embedding_dim;
    const auto retain_bias_offset = recurrent_offset + 4U * config.hidden_dim * config.embedding_dim + config.hidden_dim;
    for (std::size_t index = 0U; index < config.hidden_dim; ++index) {
        require(std::abs(parameters[retain_bias_offset + index] - 2.0) <= 1e-12,
                "Track 1 retain bias was not initialized at its declared bias offset");
    }
    require(model.state_memory_bytes() == (config.embedding_dim + config.hidden_dim) * sizeof(double),
            "Track 1 recurrent state accounting omitted the previous-input vector");
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

void test_minibatch_training_contract() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.01;
    optimizer.warmup_steps = 1;
    optimizer.batch_size = 2U;
    optimizer.total_steps = 2U;
    optimizer.validation_interval_steps = 1U;
    NlpTrainer trainer(model_config(NlpModelKind::GRU, 37), optimizer, data.tokenizer_hash, data.dataset_hash);
    const auto point = trainer.train_step(data);
    require(point.step == 1U && point.data_cursor == 2U && point.token_count == 8U && std::isfinite(point.train_loss) &&
                std::isfinite(point.gradient_norm),
            "native mini-batch training did not consume or report the configured batch");
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

void test_minibatch_checkpoint_resume() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.01;
    optimizer.batch_size = 2U;
    optimizer.total_steps = 4U;
    optimizer.validation_interval_steps = 1U;
    NlpTrainer trainer(model_config(NlpModelKind::GRU, 41), optimizer, data.tokenizer_hash, data.dataset_hash);
    trainer.train_step(data);
    trainer.save_checkpoint("artifacts/stage-11-batch-checkpoint.bin");
    const auto restored = NlpTrainer::load_checkpoint("artifacts/stage-11-batch-checkpoint.bin", data.tokenizer_hash, data.dataset_hash);
    require(restored.optimizer_config().batch_size == 2U && restored.state().data_cursor == 2U && restored.state().optimizer_step == 1U,
            "mini-batch optimizer configuration or cursor was not preserved in checkpoint");
}

void test_continuation_lineage_and_dataset_rebinding() {
    const auto data = dataset();
    auto next_data = data;
    next_data.dataset_hash = "next-dataset-hash";
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.01;
    optimizer.warmup_steps = 1U;
    optimizer.total_steps = 2U;
    optimizer.validation_interval_steps = 1U;
    NlpTrainer trainer(model_config(NlpModelKind::Track1CctRecurrence, 43), optimizer, data.tokenizer_hash, data.dataset_hash);
    trainer.begin_continuation(data.dataset_hash, "level-0-session", "GENESIS", 2U);
    trainer.train_steps(data, 2U);
    trainer.save_checkpoint("artifacts/stage-11-continuation-session-0.bin");
    const auto parent_hash = trainer.checkpoint_info().checkpoint_hash;
    require(trainer.checkpoint_info().session_id == "level-0-session" && trainer.checkpoint_info().parent_checkpoint_hash == "GENESIS",
            "initial continuation lineage metadata was not persisted in memory");
    auto restored = NlpTrainer::load_checkpoint("artifacts/stage-11-continuation-session-0.bin", data.tokenizer_hash, data.dataset_hash);
    require(restored.checkpoint_info().session_id == "level-0-session" &&
                restored.checkpoint_info().parent_checkpoint_hash == "GENESIS" && restored.state().optimizer_step == 2U,
            "V3 checkpoint lineage or optimizer state did not reload");
    bool rejected = false;
    try {
        restored.begin_continuation(next_data.dataset_hash, "level-1-session", "wrong-parent", 2U);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "incorrect continuation parent checkpoint was accepted");
    restored.begin_continuation(next_data.dataset_hash, "level-1-session", parent_hash, 2U);
    require(restored.dataset_hash() == next_data.dataset_hash && restored.state().data_cursor == 0U &&
                restored.optimizer_config().total_steps == 4U,
            "dataset rebinding did not reset the chunk cursor or extend the global optimizer budget");
    restored.train_steps(next_data, 2U);
    restored.save_checkpoint("artifacts/stage-11-continuation-session-1.bin");
    const auto reloaded = NlpTrainer::load_checkpoint("artifacts/stage-11-continuation-session-1.bin", data.tokenizer_hash, next_data.dataset_hash);
    require(reloaded.checkpoint_info().session_id == "level-1-session" &&
                reloaded.checkpoint_info().parent_checkpoint_hash == parent_hash && reloaded.state().optimizer_step == 4U,
            "continued checkpoint lineage or global optimizer step was not durable");
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
    invalid = sequence("invalid-binary-mask", {256, 257, 258});
    invalid.loss_mask[1] = 2U;
    rejected = false;
    try {
        static_cast<void>(model.loss_only(invalid));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "non-binary loss mask was accepted");
    invalid = sequence("invalid-identity", {256, 257, 258});
    invalid.sequence_id.clear();
    rejected = false;
    try {
        static_cast<void>(model.loss_only(invalid));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "sequence without identity was accepted");
    auto parameters = model.parameter_vector();
    parameters.front() = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        model.set_parameter_vector(parameters);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "non-finite model parameters were accepted");

    NlpTrainerState state;
    auto before = model.parameter_vector();
    auto gradients = std::vector<double>(before.size(), 0.0);
    gradients.front() = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        model.apply_gradient(gradients, NlpOptimizerConfig{}, state);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected && model.parameter_vector() == before && state.optimizer_step == 0U,
            "non-finite simple optimizer update was not rejected atomically");

    NlpModelConfig invalid_kind = model_config(NlpModelKind::Track1CctRecurrence);
    invalid_kind.kind = static_cast<NlpModelKind>(99);
    rejected = false;
    try {
        NextTokenModel invalid_model(invalid_kind);
        static_cast<void>(invalid_model);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "unsupported model kind was accepted");

    NlpModelConfig oversized = model_config(NlpModelKind::Track1CctRecurrence);
    oversized.vocabulary_size = 1'000'000U;
    oversized.embedding_dim = 4096U;
    oversized.hidden_dim = 4096U;
    rejected = false;
    try {
        NextTokenModel invalid_model(oversized);
        static_cast<void>(invalid_model);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "oversized model parameter allocation was accepted");
}

void test_incremental_inference_equivalence() {
    const std::vector<NlpModelKind> kinds{NlpModelKind::Track1CctRecurrence, NlpModelKind::GRU, NlpModelKind::DiagonalSSM,
                                          NlpModelKind::DenseCausalAttention};
    const std::vector<TokenId> tokens{256U, 257U, 258U, 259U, 260U, 261U};
    for (const auto kind : kinds) {
        const auto model = NextTokenModel(model_config(kind, 101U + static_cast<std::uint64_t>(kind)));
        auto state = model.create_inference_state();
        std::vector<TokenId> prefix;
        std::vector<double> incremental;
        for (const auto token : tokens) {
            prefix.push_back(token);
            incremental = model.next_logits_incremental(token, state);
            const auto reference = model.next_logits(prefix);
            require(incremental.size() == reference.size(), "incremental logits shape mismatch");
            double maximum_error = 0.0;
            for (std::size_t index = 0U; index < reference.size(); ++index) maximum_error = std::max(maximum_error, std::abs(incremental[index] - reference[index]));
            require(maximum_error <= 1e-10, "incremental logits diverged from full-context logits for " + model.name());
        }
        require(state.valid_length == tokens.size(), "incremental state length did not advance");
    }
}

void test_parallel_optimizer_determinism() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.total_steps = 2U;
    optimizer.warmup_steps = 1U;
    optimizer.batch_size = 3U;
    optimizer.worker_count = 2U;
    NlpTrainer first(model_config(NlpModelKind::GRU, 109U), optimizer, data.tokenizer_hash, data.dataset_hash);
    NlpTrainer second(model_config(NlpModelKind::GRU, 109U), optimizer, data.tokenizer_hash, data.dataset_hash);
    static_cast<void>(first.train_steps(data, 2U));
    static_cast<void>(second.train_steps(data, 2U));
    const auto first_parameters = first.model().parameter_vector();
    const auto second_parameters = second.model().parameter_vector();
    require(first_parameters == second_parameters, "parallel optimizer execution was not deterministic");
    require(first.history().size() == second.history().size() && first.history().back().train_loss == second.history().back().train_loss,
            "parallel optimizer history was not deterministic");
}

void test_worker_contract_and_fail_closed_zero_workers() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.total_steps = 2U;
    optimizer.warmup_steps = 1U;
    optimizer.worker_count = 2U;
    NlpTrainer trainer(model_config(NlpModelKind::Track1CctRecurrence, 103U), optimizer, data.tokenizer_hash, data.dataset_hash);
    static_cast<void>(trainer.train_step(data));
    require(trainer.checkpoint_info().training_contract_hash.size() == 64U, "worker-aware training contract hash is missing");
    optimizer.worker_count = 0U;
    bool rejected = false;
    try {
        NlpTrainer invalid(model_config(NlpModelKind::Track1CctRecurrence, 107U), optimizer, data.tokenizer_hash, data.dataset_hash);
        static_cast<void>(invalid);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "zero worker count was accepted");
}

void test_dataset_context_budget_and_optimizer_domain_fail_closed() {
    const auto data = dataset();
    NlpOptimizerConfig optimizer;
    optimizer.warmup_steps = 1U;
    optimizer.total_steps = 1U;
    auto trainer = NlpTrainer(model_config(NlpModelKind::Track1CctRecurrence, 37), optimizer, data.tokenizer_hash, data.dataset_hash);
    auto mismatched = data;
    mismatched.context_length = 7U;
    bool rejected = false;
    try {
        static_cast<void>(trainer.train_step(mismatched));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "dataset/model context mismatch was accepted");
    static_cast<void>(trainer.train_step(data));
    rejected = false;
    try {
        static_cast<void>(trainer.train_step(data));
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "optimizer budget overrun was accepted");
    NlpOptimizerConfig invalid_optimizer;
    invalid_optimizer.learning_rate = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        NlpTrainer invalid_trainer(model_config(NlpModelKind::Track1CctRecurrence, 39), invalid_optimizer, data.tokenizer_hash, data.dataset_hash);
        static_cast<void>(invalid_trainer);
    } catch (const NlpTrainingError&) {
        rejected = true;
    }
    require(rejected, "non-finite optimizer configuration was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"objective_mask_and_finite_metrics", test_objective_mask_and_finite_metrics},
        {"track1_recurrence_gradient_finite_difference", test_track1_recurrence_gradient_finite_difference},
        {"upgraded_baseline_gradients", test_upgraded_baseline_gradients},
        {"track1_model_identity_and_serialization", test_track1_model_identity_and_serialization},
        {"track1_initialization_and_state_accounting", test_track1_initialization_and_state_accounting},
        {"compact_vocabulary_mapping_and_serialization", test_compact_vocabulary_mapping_and_serialization},
        {"optimizer_direction_and_schedule", test_optimizer_direction_and_schedule},
        {"minibatch_training_contract", test_minibatch_training_contract},
        {"deterministic_initialization_and_controls", test_deterministic_initialization_and_controls},
        {"checkpoint_resume_exactness_and_fail_closed", test_checkpoint_resume_exactness_and_fail_closed},
        {"minibatch_checkpoint_resume", test_minibatch_checkpoint_resume},
        {"continuation_lineage_and_dataset_rebinding", test_continuation_lineage_and_dataset_rebinding},
        {"invalid_masks_and_nonfinite_parameters_fail_closed", test_invalid_masks_and_nonfinite_parameters_fail_closed},
        {"incremental_inference_equivalence", test_incremental_inference_equivalence},
        {"parallel_optimizer_determinism", test_parallel_optimizer_determinism},
        {"worker_contract_and_fail_closed_zero_workers", test_worker_contract_and_fail_closed_zero_workers},
        {"dataset_context_budget_and_optimizer_domain_fail_closed", test_dataset_context_budget_and_optimizer_domain_fail_closed}};
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
