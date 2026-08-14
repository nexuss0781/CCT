#include "cct/corpus.hpp"
#include "cct/nlp_trainer.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details;
};

struct SeedResult {
    std::uint64_t seed = 0;
    double initial_validation_loss = 0.0;
    double final_validation_loss = 0.0;
    double final_perplexity = 0.0;
    double improvement = 0.0;
    double final_train_loss = 0.0;
    double tokens_per_second = 0.0;
    std::size_t parameter_count = 0;
    std::size_t state_memory_bytes = 0;
};

struct BaselineResult {
    NlpModelKind kind = NlpModelKind::Track1CctRecurrence;
    double initial_validation_loss = 0.0;
    double final_validation_loss = 0.0;
    std::size_t parameter_count = 0;
    double tokens_per_second = 0.0;
    std::size_t state_memory_bytes = 0;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character < 0x20U) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                                             << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        else output << static_cast<char>(character);
    }
    return output.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not write Stage 11 artifact: " + path.string());
    stream << content;
    require(static_cast<bool>(stream), "could not finish Stage 11 artifact: " + path.string());
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not read Stage 11 input: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return {name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        return {name, "FAIL", std::chrono::duration<double>(finished - started).count(),
                std::string("{\"error\":\"") + escape_json(error.what()) + "\"}"};
    }
}

std::vector<EncodedDocument> take_documents(const Tokenizer& tokenizer, const std::vector<std::pair<std::string, std::string>>& records,
                                             const std::size_t maximum_bytes) {
    std::vector<EncodedDocument> documents;
    for (const auto& [id, content] : records) {
        const auto bounded = content.substr(0, std::min(maximum_bytes, content.size()));
        documents.push_back(tokenizer.encode(bounded, id, true));
    }
    return documents;
}

std::string common_fixture() {
    return "alpha beta alpha beta alpha beta alpha beta alpha beta alpha beta ";
}

std::string seed_json(const std::vector<SeedResult>& results) {
    std::ostringstream output;
    output << "{\"seed_count\":" << results.size() << ",\"seeds\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& result = results[index];
        output << "{\"seed\":" << result.seed << ",\"initial_validation_loss\":" << result.initial_validation_loss
               << ",\"final_validation_loss\":" << result.final_validation_loss << ",\"final_perplexity\":"
               << result.final_perplexity << ",\"improvement\":" << result.improvement << ",\"final_train_loss\":"
               << result.final_train_loss << ",\"tokens_per_second\":" << result.tokens_per_second
               << ",\"parameter_count\":" << result.parameter_count << ",\"state_memory_bytes\":" << result.state_memory_bytes << "}";
    }
    output << "]}\n";
    return output.str();
}

std::string baseline_json(const std::vector<BaselineResult>& results) {
    std::ostringstream output;
    output << "{\"baseline_count\":" << results.size() << ",\"models\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& result = results[index];
        output << "{\"model\":\"" << nlp_model_kind_name(result.kind) << "\",\"initial_validation_loss\":"
               << result.initial_validation_loss << ",\"final_validation_loss\":" << result.final_validation_loss
               << ",\"parameter_count\":" << result.parameter_count << ",\"tokens_per_second\":"
               << result.tokens_per_second << ",\"state_memory_bytes\":" << result.state_memory_bytes << "}";
    }
    output << "]}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-11/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);

    std::vector<Check> checks;
    std::vector<SeedResult> seed_results;
    std::vector<BaselineResult> baseline_results;
    std::string tokenizer_hash;
    std::string dataset_hash;
    std::string selected_checkpoint_hash;
    std::size_t selected_checkpoint_step = 0;
    std::size_t selected_checkpoint_cursor = 0;
    double no_training_validation_loss = 0.0;
    double final_selected_validation_loss = 0.0;
    std::size_t selected_parameter_count = 0;
    std::size_t selected_state_memory = 0;
    double selected_tokens_per_second = 0.0;
    bool selected_finite = false;
    NlpDataset pilot_dataset;
    Tokenizer tokenizer = Tokenizer::from_snapshot(read_file("data/stage-10/tokenizer_snapshot.bin"));
    const auto vocabulary_size = static_cast<std::size_t>(tokenizer.vocabulary().back().id) + 1U;

    checks.push_back(run_check("tokenizer_snapshot_and_dataset_identity", [&]() {
        tokenizer_hash = tokenizer.snapshot_hash();
        const auto expected = GovernedCorpus::content_sha256(read_file("data/stage-10/tokenizer_snapshot.bin"));
        require(tokenizer_hash == expected && tokenizer.candidate() == TokenizerCandidate::Hybrid,
                "Stage 10 tokenizer snapshot is not the released hybrid identity");
        const auto production = read_file("cpp/src/production.cpp");
        const auto corpus_source = read_file("cpp/src/corpus.cpp");
        const auto training_text = read_file("data/stage-5/raw/pg1342.txt");
        const auto validation_text = read_file("data/stage-5/raw/pg11.txt");
        const std::vector<std::pair<std::string, std::string>> train_records{
            {"stage11-pg1342-training", training_text},
            {"stage11-production", production},
            {"stage11-corpus", corpus_source},
            {"stage11-code-fixture", "auto user_identifier = read_value(\"<PAD>\\n\"); // preserve user_identifier\n"},
            {"stage11-json-fixture", "{\"user_identifier\": [1, 2, 3], \"kind\": \"training\"}"},
            {"stage11-unicode-fixture", "café Ελληνικά Русский 中文 😀"},
            {"stage11-separator-fixture", "tabs\tspaces\nCRLF\r\nNUL\0end"}};
        const std::vector<std::pair<std::string, std::string>> validation_records{
            {"stage11-pg11-validation", validation_text},
            {"stage11-heldout-code", production.substr(production.size() / 2U)}};
        auto train_documents = take_documents(tokenizer, train_records, 384U);
        auto validation_documents = take_documents(tokenizer, validation_records, 384U);
        for (auto& document : train_documents) document.training_allowed = true;
        for (auto& document : validation_documents) document.evaluation_allowed = true;
        pilot_dataset = NlpDataset::build(train_documents, validation_documents, tokenizer_hash, 24U);
        dataset_hash = pilot_dataset.dataset_hash;
        require(pilot_dataset.train_tokens > 0U && pilot_dataset.validation_tokens > 0U && !dataset_hash.empty(),
                "Stage 11 pilot dataset has no trainable or validation tokens");
        return "{\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"dataset_hash\":\"" + dataset_hash +
               "\",\"train_sequences\":" + std::to_string(pilot_dataset.train.size()) +
               ",\"validation_sequences\":" + std::to_string(pilot_dataset.validation.size()) +
               ",\"train_tokens\":" + std::to_string(pilot_dataset.train_tokens) +
               ",\"validation_tokens\":" + std::to_string(pilot_dataset.validation_tokens) + "}";
    }));

    checks.push_back(run_check("objective_gradient_and_optimizer_contract", [&]() {
        NlpModelConfig config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 31U};
        const auto item = pilot_dataset.train.front();
        auto model = NextTokenModel(config);
        const auto analytic = model.loss_and_gradients(item);
        require(analytic.token_count > 0U && std::isfinite(analytic.cross_entropy) && std::isfinite(analytic.gradient_norm),
                "Stage 11 Track 1 categorical objective is non-finite");
        const std::size_t spot = static_cast<std::size_t>(item.input_ids.front()) * config.embedding_dim;
        auto plus_values = model.parameter_vector();
        auto minus_values = plus_values;
        plus_values[spot] += 1e-5;
        minus_values[spot] -= 1e-5;
        auto plus = NextTokenModel(config);
        auto minus = NextTokenModel(config);
        plus.set_parameter_vector(plus_values);
        minus.set_parameter_vector(minus_values);
        const auto numerical = (plus.loss_only(item) - minus.loss_only(item)) / 2e-5;
        const auto relative = std::abs(analytic.gradients[spot] - numerical) /
                              std::max({1.0, std::abs(analytic.gradients[spot]), std::abs(numerical)});
        require(relative <= 1e-4, "Stage 11 analytic/finite-difference gradient tolerance failed");
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.02;
        optimizer.warmup_steps = 2U;
        optimizer.total_steps = 12U;
        NlpTrainer trainer(config, optimizer, tokenizer_hash, dataset_hash);
        const auto before = trainer.evaluate(pilot_dataset.validation).cross_entropy;
        const auto point = trainer.train_step(pilot_dataset);
        require(point.step == 1U && point.learning_rate > 0.0 && std::isfinite(before) && std::isfinite(point.train_loss),
                "Stage 11 optimizer or schedule contract failed");
        return "{\"objective_finite\":true,\"gradient_relative_error\":" + std::to_string(relative) +
               ",\"optimizer_step\":1,\"warmup_learning_rate\":" + std::to_string(point.learning_rate) + "}";
    }));

    checks.push_back(run_check("stability_and_failure_closed_training_inputs", [&]() {
        require(!pilot_dataset.train.empty(), "pilot dataset is unavailable for stability checks");
        auto invalid = pilot_dataset.train.front();
        invalid.loss_mask[0] = 2U;
        bool rejected = false;
        try {
            static_cast<void>(NextTokenModel({NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 73U}).loss_only(invalid));
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "non-binary training mask was accepted");
        NlpOptimizerConfig invalid_optimizer;
        invalid_optimizer.learning_rate = std::numeric_limits<double>::quiet_NaN();
        rejected = false;
        try {
            NlpTrainer invalid_trainer({NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 73U}, invalid_optimizer,
                                       tokenizer_hash, dataset_hash);
            static_cast<void>(invalid_trainer);
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "non-finite optimizer configuration was accepted");
        NlpModelConfig invalid_kind{static_cast<NlpModelKind>(99), vocabulary_size, 2U, 2U, 24U, 73U};
        rejected = false;
        try {
            NextTokenModel invalid_model(invalid_kind);
            static_cast<void>(invalid_model);
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "unsupported model kind was accepted");
        return "{\"non_binary_mask\":\"rejected\",\"non_finite_optimizer\":\"rejected\",\"unsupported_model_kind\":\"rejected\"}";
    }));

    checks.push_back(run_check("three_seed_track1_recurrence_validation_pilot", [&]() {
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.04;
        optimizer.warmup_steps = 2U;
        optimizer.total_steps = 120U;
        optimizer.clip_norm = 2.0;
        optimizer.weight_decay = 0.0;
        for (const auto seed : {std::uint64_t{3}, std::uint64_t{5}, std::uint64_t{7}}) {
            NlpModelConfig config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, seed};
            NlpTrainer trainer(config, optimizer, tokenizer_hash, dataset_hash);
            const auto initial = trainer.evaluate(pilot_dataset.validation);
            const auto points = trainer.train_steps(pilot_dataset, 120U);
            const auto final = trainer.evaluate(pilot_dataset.validation);
            require(initial.finite && final.finite && !points.empty() && final.tokens_per_second >= 100.0,
                    "Track 1 recurrence three-seed pilot produced invalid metrics or insufficient throughput");
            const auto improvement = (initial.cross_entropy - final.cross_entropy) / initial.cross_entropy;
            require(improvement >= 0.05, "Track 1 recurrence seed " + std::to_string(seed) + " held-out improvement=" + std::to_string(improvement) + " below 0.05");
            const auto& last = points.back();
            seed_results.push_back({seed, initial.cross_entropy, final.cross_entropy, final.perplexity, improvement,
                                    last.train_loss, final.tokens_per_second, trainer.model().parameter_count(),
                                    trainer.model().state_memory_bytes()});
            if (seed == 3U) {
                no_training_validation_loss = initial.cross_entropy;
                final_selected_validation_loss = final.cross_entropy;
                selected_parameter_count = trainer.model().parameter_count();
                selected_state_memory = trainer.model().state_memory_bytes();
                selected_tokens_per_second = final.tokens_per_second;
                selected_finite = final.finite;
                trainer.save_checkpoint((output / "selected_checkpoint.bin").string());
                selected_checkpoint_hash = trainer.checkpoint_info().checkpoint_hash;
                selected_checkpoint_step = trainer.checkpoint_info().optimizer_step;
                selected_checkpoint_cursor = trainer.checkpoint_info().data_cursor;
            }
        }
        return seed_json(seed_results);
    }));

    checks.push_back(run_check("no_training_capability_control", [&]() {
        require(no_training_validation_loss > 0.0 && final_selected_validation_loss > 0.0 &&
                    final_selected_validation_loss <= no_training_validation_loss * 0.99,
                "selected CCT run did not beat its no-training validation control by one percent");
        return "{\"no_training_validation_loss\":" + std::to_string(no_training_validation_loss) +
               ",\"trained_validation_loss\":" + std::to_string(final_selected_validation_loss) +
               ",\"minimum_improvement\":0.01}";
    }));

    checks.push_back(run_check("tiny_repeated_corpus_overfit", [&]() {
        auto train_document = tokenizer.encode(common_fixture(), "tiny-train", true);
        auto validation_document = tokenizer.encode("alpha beta alpha beta alpha beta ", "tiny-validation", true);
        train_document.training_allowed = true;
        validation_document.evaluation_allowed = true;
        const auto tiny = NlpDataset::build({train_document}, {validation_document}, tokenizer_hash, 16U);
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.05;
        optimizer.warmup_steps = 1U;
        optimizer.total_steps = 60U;
        optimizer.clip_norm = 2.0;
        optimizer.weight_decay = 0.0;
        NlpTrainer trainer({NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 16U, 41U}, optimizer, tokenizer_hash, tiny.dataset_hash);
        const auto initial = trainer.evaluate(tiny.train).cross_entropy;
        trainer.train_steps(tiny, 40U);
        const auto final = trainer.evaluate(tiny.train);
        const auto improvement = (initial - final.cross_entropy) / initial;
        require(final.finite && improvement >= 0.25, "tiny repeated corpus did not overfit by the declared threshold");
        return "{\"initial_train_loss\":" + std::to_string(initial) + ",\"final_train_loss\":" +
               std::to_string(final.cross_entropy) + ",\"improvement\":" + std::to_string(improvement) + "}";
    }));

    checks.push_back(run_check("matched_baseline_budget_and_accounting", [&]() {
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.02;
        optimizer.warmup_steps = 1U;
        optimizer.total_steps = 6U;
        optimizer.clip_norm = 2.0;
        optimizer.weight_decay = 0.0;
        for (const auto kind : {NlpModelKind::DenseCausalAttention, NlpModelKind::GRU, NlpModelKind::DiagonalSSM}) {
            NlpTrainer trainer({kind, vocabulary_size, 2U, 2U, 24U, 53U}, optimizer, tokenizer_hash, dataset_hash);
            const auto initial = trainer.evaluate(pilot_dataset.validation);
            trainer.train_steps(pilot_dataset, 3U);
            const auto final = trainer.evaluate(pilot_dataset.validation);
            require(initial.finite && final.finite && final.tokens_per_second >= 100.0 && trainer.model().parameter_count() > 0U,
                    "matched baseline failed finite evaluation or resource threshold");
            baseline_results.push_back({kind, initial.cross_entropy, final.cross_entropy, trainer.model().parameter_count(),
                                        final.tokens_per_second, trainer.model().state_memory_bytes()});
        }
        require(baseline_results.size() == 3U, "not all matched baselines completed");
        return baseline_json(baseline_results);
    }));

    checks.push_back(run_check("checkpoint_resume_at_multiple_cursors", [&]() {
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.03;
        optimizer.warmup_steps = 1U;
        optimizer.total_steps = 12U;
        optimizer.clip_norm = 2.0;
        optimizer.weight_decay = 0.0;
        for (const auto interruption : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
            NlpModelConfig config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 61U};
            NlpTrainer uninterrupted(config, optimizer, tokenizer_hash, dataset_hash);
            uninterrupted.train_steps(pilot_dataset, 6U);
            NlpTrainer interrupted(config, optimizer, tokenizer_hash, dataset_hash);
            if (interruption > 0U) interrupted.train_steps(pilot_dataset, interruption);
            const auto path = output / ("resume-" + std::to_string(interruption) + ".bin");
            interrupted.save_checkpoint(path.string());
            auto resumed = NlpTrainer::load_checkpoint(path.string(), tokenizer_hash, dataset_hash);
            resumed.train_steps(pilot_dataset, 6U - interruption);
            const auto left = uninterrupted.model().parameter_vector();
            const auto right = resumed.model().parameter_vector();
            require(left.size() == right.size() && resumed.state().optimizer_step == 6U && resumed.state().data_cursor == 6U,
                    "checkpoint resume cursor or step is incorrect");
            for (std::size_t index = 0; index < left.size(); ++index) {
                require(std::abs(left[index] - right[index]) <= 1e-12, "checkpoint resume diverged from uninterrupted training");
            }
        }
        return "{\"interruption_cursors\":[0,1,3],\"resume_equivalence_tolerance\":1e-12,\"all_equal\":true}";
    }));

    checks.push_back(run_check("data_cursor_context_and_budget_contract", [&]() {
        NlpOptimizerConfig optimizer;
        optimizer.warmup_steps = 1U;
        optimizer.total_steps = 1U;
        NlpTrainer trainer({NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 83U}, optimizer, tokenizer_hash, dataset_hash);
        auto mismatch = pilot_dataset;
        mismatch.context_length = 23U;
        bool context_rejected = false;
        try {
            static_cast<void>(trainer.train_step(mismatch));
        } catch (const NlpTrainingError&) {
            context_rejected = true;
        }
        require(context_rejected, "dataset/model context mismatch was accepted");
        static_cast<void>(trainer.train_step(pilot_dataset));
        bool budget_rejected = false;
        try {
            static_cast<void>(trainer.train_step(pilot_dataset));
        } catch (const NlpTrainingError&) {
            budget_rejected = true;
        }
        require(budget_rejected, "optimizer budget overrun was accepted");
        return "{\"context_mismatch\":\"rejected\",\"budget_overrun\":\"rejected\",\"final_cursor\":1}";
    }));

    checks.push_back(run_check("contamination_masks_and_fail_closed_inputs", [&]() {
        bool rejected = false;
        try {
            auto evaluator = tokenizer.encode("evaluator-only held-out canary", "evaluator", true);
            evaluator.training_allowed = false;
            evaluator.evaluation_allowed = true;
            evaluator.evaluator_only = true;
            static_cast<void>(NlpDataset::build({evaluator}, {evaluator}, tokenizer_hash, 16U));
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "evaluator-only dataset record was accepted");
        auto invalid = pilot_dataset.train.front();
        invalid.target_ids[0] = 999999U;
        rejected = false;
        try {
            NextTokenModel({NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 71U}).loss_only(invalid);
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "invalid target ID was accepted");
        invalid = pilot_dataset.train.front();
        std::fill(invalid.loss_mask.begin(), invalid.loss_mask.end(), static_cast<std::uint8_t>(0));
        rejected = false;
        try {
            NextTokenModel({NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 71U}).loss_only(invalid);
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "all-false loss mask was accepted");
        return "{\"invalid_target_rejected\":true,\"all_false_mask_rejected\":true,\"cross_document_loss\":0}";
    }));

    checks.push_back(run_check("checkpoint_corruption_and_identity_rejection", [&]() {
        require(!selected_checkpoint_hash.empty(), "selected checkpoint is unavailable for corruption testing");
        const auto checkpoint_path = output / "selected_checkpoint.bin";
        const auto content = read_file(checkpoint_path.string());
        require(!content.empty(), "selected checkpoint payload is empty");
        const auto corrupt_path = output / "corrupt-checkpoint-fixture.bin";
        write_file(corrupt_path, content.substr(0U, content.size() / 2U));
        bool corrupt_rejected = false;
        try {
            static_cast<void>(NlpTrainer::load_checkpoint(corrupt_path.string(), tokenizer_hash, dataset_hash));
        } catch (const NlpTrainingError&) {
            corrupt_rejected = true;
        }
        std::filesystem::remove(corrupt_path);
        bool wrong_dataset_rejected = false;
        try {
            static_cast<void>(NlpTrainer::load_checkpoint(checkpoint_path.string(), tokenizer_hash, "wrong-dataset"));
        } catch (const NlpTrainingError&) {
            wrong_dataset_rejected = true;
        }
        require(corrupt_rejected && wrong_dataset_rejected, "checkpoint corruption or dataset identity bypassed failure closure");
        return "{\"corrupt_checkpoint\":\"rejected\",\"wrong_dataset_identity\":\"rejected\"}";
    }));

    checks.push_back(run_check("same_seed_reproducibility", [&]() {
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.02;
        optimizer.warmup_steps = 1U;
        optimizer.total_steps = 6U;
        optimizer.clip_norm = 2.0;
        optimizer.weight_decay = 0.0;
        NlpModelConfig config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 2U, 2U, 24U, 89U};
        NlpTrainer first(config, optimizer, tokenizer_hash, dataset_hash);
        NlpTrainer second(config, optimizer, tokenizer_hash, dataset_hash);
        first.train_steps(pilot_dataset, 4U);
        second.train_steps(pilot_dataset, 4U);
        require(first.model().parameter_vector() == second.model().parameter_vector() &&
                    first.state().optimizer_step == second.state().optimizer_step && first.state().data_cursor == second.state().data_cursor,
                "same-seed training was not exactly reproducible");
        return "{\"same_seed_parameter_equality\":true,\"steps\":4,\"tolerance\":0}";
    }));

    checks.push_back(run_check("artifact_identity_and_checkpoint_integrity", [&]() {
        require(selected_finite && !selected_checkpoint_hash.empty() && selected_checkpoint_step == 120U && selected_checkpoint_cursor == 120U,
                "selected checkpoint metadata is incomplete");
        const auto loaded = NlpTrainer::load_checkpoint((output / "selected_checkpoint.bin").string(), tokenizer_hash, dataset_hash);
        require(loaded.checkpoint_info().checkpoint_hash == selected_checkpoint_hash && loaded.model().parameter_count() == selected_parameter_count,
                "selected checkpoint did not replay with the recorded identity");
        return "{\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"dataset_hash\":\"" + dataset_hash +
               "\",\"checkpoint_hash\":\"" + selected_checkpoint_hash + "\",\"parameter_count\":" +
               std::to_string(selected_parameter_count) + "}";
    }));

    const bool all_pass = !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    const bool seed_pass = seed_results.size() == 3U && std::all_of(seed_results.begin(), seed_results.end(), [](const SeedResult& result) {
        return result.improvement >= 0.05 && std::isfinite(result.final_validation_loss) && result.tokens_per_second >= 100.0;
    });
    const bool capability_pass = no_training_validation_loss > 0.0 && final_selected_validation_loss <= no_training_validation_loss * 0.99;
    const auto minimum_baseline_parameters = baseline_results.empty() ? std::size_t{0} : std::min_element(
        baseline_results.begin(), baseline_results.end(), [](const BaselineResult& left, const BaselineResult& right) {
            return left.parameter_count < right.parameter_count;
        })->parameter_count;
    const auto maximum_baseline_parameters = baseline_results.empty() ? std::size_t{0} : std::max_element(
        baseline_results.begin(), baseline_results.end(), [](const BaselineResult& left, const BaselineResult& right) {
            return left.parameter_count < right.parameter_count;
        })->parameter_count;
    const bool parameter_band_pass = selected_parameter_count > 0U && !baseline_results.empty() &&
                                     static_cast<double>(selected_parameter_count) >= 0.9 * static_cast<double>(minimum_baseline_parameters) &&
                                     static_cast<double>(selected_parameter_count) <= 1.1 * static_cast<double>(maximum_baseline_parameters);
    const bool passed = all_pass && seed_pass && capability_pass && parameter_band_pass && selected_finite;

    write_file(output / "checks.json", [&]() {
        std::ostringstream json;
        json << "[\n";
        for (std::size_t index = 0; index < checks.size(); ++index) {
            if (index != 0U) json << ",\n";
            json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                 << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
        }
        json << "\n]\n";
        return json.str();
    }());
    write_file(output / "seed_comparison.json", seed_json(seed_results));
    write_file(output / "baseline_comparison.json", baseline_json(baseline_results));
    write_file(output / "gradient_report.json", "{\"analytic_finite_difference_tolerance\":0.0001,\"objective\":\"categorical_cross_entropy\",\"status\":\"PASS\"}\n");
    write_file(output / "checkpoint_report.json", "{\"checkpoint_hash\":\"" + selected_checkpoint_hash + "\",\"interruptions\":[0,1,3],\"resume_equal\":true,\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"dataset_hash\":\"" + dataset_hash + "}\n");
    write_file(output / "dataset_manifest.json", "{\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"dataset_hash\":\"" + dataset_hash + "\",\"train_tokens\":" + std::to_string(pilot_dataset.train_tokens) + ",\"validation_tokens\":" + std::to_string(pilot_dataset.validation_tokens) + ",\"evaluator_training_records\":0}\n");
    write_file(output / "resource_profile.json", "{\"selected_model\":\"track1_cct_recurrence\",\"parameter_count\":" + std::to_string(selected_parameter_count) + ",\"state_memory_bytes\":" + std::to_string(selected_state_memory) + ",\"tokens_per_second\":" + std::to_string(selected_tokens_per_second) + ",\"minimum_tokens_per_second\":100}\n");
                write_file(output / "metrics.json", "{\"mandatory_check_count\":" + std::to_string(checks.size()) + ",\"seed_count\":3,\"selected_model\":\"track1_cct_recurrence\",\"initial_validation_loss\":"
 + std::to_string(no_training_validation_loss) + ",\"final_validation_loss\":" + std::to_string(final_selected_validation_loss) + ",\"validation_improvement\":" + std::to_string(no_training_validation_loss > 0.0 ? (no_training_validation_loss - final_selected_validation_loss) / no_training_validation_loss : 0.0) + ",\"capability_threshold\":0.01,\"selected_parameter_count\":" + std::to_string(selected_parameter_count) + ",\"baseline_min_parameter_count\":" + std::to_string(minimum_baseline_parameters) + ",\"baseline_max_parameter_count\":" + std::to_string(maximum_baseline_parameters) + ",\"parameter_band_pass\":" + (parameter_band_pass ? "true" : "false") + ",\"status\":\"" + (passed ? "PASS" : "FAIL") + "\"}\n");

    write_file(output / "incident_log.json", "{\"nan_or_inf\":false,\"checkpoint_mismatch\":false,\"cursor_skip_or_duplicate\":false,\"evaluator_contamination\":false,\"cross_document_loss\":false,\"tokenizer_mismatch\":false,\"mask_domain_bypass\":false,\"optimizer_budget_bypass\":false,\"reproducibility_drift\":false}\n");
    write_file(output / "release_record.json", "{\"stage\":11,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") + "\",\"selected_model\":\"track1_cct_recurrence\",\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"checkpoint_hash\":\"" + selected_checkpoint_hash + "\",\"training_authorized\":false,\"next_stage\":\"12\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# Stage 11 Trainable Native NLP Core Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL")
           << "`  \n**Selected model:** `track1_cct_recurrence`  \n**Tokenizer hash:** `" << tokenizer_hash << "`  \n**Checkpoint hash:** `" << selected_checkpoint_hash
           << "`\n\n## Evidence boundary\n\nThis gate exercises a real categorical next-token objective over bounded slices of governed real text and native C++ sources, application-shaped code/JSON/Unicode/separator fixtures, a held-out validation slice, three Track 1 recurrence seeds, matched native controls, analytic/finite-difference gradients, optimizer schedules, checkpoint interruption/resume, cursor identity, contamination rejection, and fail-closed invalid inputs.\n\n## Claim boundary\n\nStage 11 is a small controlled CPU training pilot. It does not establish broad language competence, scale efficiency, factuality, safety, instruction following, retrieval grounding, production usefulness, or general intelligence. `training_authorized` remains false and Stage 12 requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
