#include "cct/nlp_trainer.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cct::EncodedDocument;
using cct::NlpDataset;
using cct::NlpEvaluation;
using cct::NlpModelConfig;
using cct::NlpModelKind;
using cct::NlpOptimizerConfig;
using cct::NlpSequence;
using cct::NlpTrainer;
using cct::TokenId;
using cct::Tokenizer;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read benchmark input: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    require(static_cast<bool>(input) || input.eof(), "cannot finish benchmark input: " + path.string());
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write benchmark report: " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish benchmark report: " + path.string());
}

std::string argument_path(const int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

std::size_t argument_size(const int argc, char** argv, const std::string& name, const std::size_t fallback) {
    const auto value = argument_path(argc, argv, name, std::to_string(fallback));
    try {
        std::size_t consumed = 0U;
        const auto parsed = std::stoull(value, &consumed, 10);
        require(consumed == value.size() && parsed > 0U, "invalid benchmark numeric argument: " + name);
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid benchmark numeric argument: " + name);
    }
}

std::string model_name(const NlpModelKind kind) { return cct::nlp_model_kind_name(kind); }

std::size_t count_targets(const std::vector<NlpSequence>& sequences) {
    std::size_t total = 0U;
    for (const auto& sequence : sequences) {
        total += static_cast<std::size_t>(std::count(sequence.loss_mask.begin(), sequence.loss_mask.end(), static_cast<std::uint8_t>(1)));
    }
    return total;
}

void cap_sequences(std::vector<NlpSequence>& sequences, const std::size_t maximum) {
    if (sequences.size() > maximum) sequences.resize(maximum);
}

EncodedDocument encode_document(const Tokenizer& tokenizer, const std::filesystem::path& path, const std::string& id,
                                const bool training_allowed, const bool evaluation_allowed) {
    auto document = tokenizer.encode(read_file(path), id, false);
    document.training_allowed = training_allowed;
    document.evaluation_allowed = evaluation_allowed;
    document.evaluator_only = false;
    require(document.tokens.size() >= 2U, "benchmark document is too short: " + path.string());
    return document;
}

struct DecodeMeasurement {
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    std::size_t generated_tokens = 0U;
    bool finite = false;
};

std::vector<TokenId> context_window(const std::vector<TokenId>& context, const std::size_t maximum) {
    require(!context.empty() && maximum > 0U, "benchmark context is empty");
    const auto start = context.size() > maximum ? context.size() - maximum : 0U;
    return std::vector<TokenId>(context.begin() + static_cast<std::ptrdiff_t>(start), context.end());
}

DecodeMeasurement measure_decode(const cct::NextTokenModel& model, const Tokenizer& tokenizer, const std::string& prompt,
                                 const std::size_t maximum_tokens) {
    const auto encoded = tokenizer.encode(prompt, "optimization-benchmark-prompt", false);
    std::vector<TokenId> context;
    context.reserve(encoded.tokens.size() + maximum_tokens);
    for (const auto& token : encoded.tokens) context.push_back(token.id);
    context = context_window(context, model.config().context_length);

    const auto prefill_started = std::chrono::steady_clock::now();
    auto logits = model.next_logits(context);
    const auto prefill_finished = std::chrono::steady_clock::now();
    const auto decode_started = prefill_finished;
    std::size_t generated = 0U;
    for (std::size_t step = 0U; step < maximum_tokens; ++step) {
        const auto slot = static_cast<std::size_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        const auto token = model.token_id_from_logit_slot(slot);
        if (token == Tokenizer::kEosId) break;
        context.push_back(token);
        ++generated;
        if (context.size() > model.config().context_length) context.erase(context.begin());
        if (step + 1U < maximum_tokens) logits = model.next_logits(context);
    }
    const auto finished = std::chrono::steady_clock::now();
    const auto prefill_seconds = std::chrono::duration<double>(prefill_finished - prefill_started).count();
    const auto decode_seconds = std::chrono::duration<double>(finished - decode_started).count();
    return {prefill_seconds, decode_seconds, generated, std::isfinite(prefill_seconds) && std::isfinite(decode_seconds)};
}

struct BenchmarkResult {
    NlpModelKind kind = NlpModelKind::Track1CctRecurrence;
    std::size_t parameter_count = 0U;
    std::size_t parameter_bytes = 0U;
    std::size_t state_memory_bytes = 0U;
    std::size_t serialized_model_bytes = 0U;
    double train_seconds = 0.0;
    std::size_t train_tokens = 0U;
    double train_tokens_per_second = 0.0;
    double evaluation_seconds = 0.0;
    std::size_t evaluation_tokens = 0U;
    double evaluation_tokens_per_second = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    std::size_t generated_tokens = 0U;
    double decode_tokens_per_second = 0.0;
    double end_to_end_tokens_per_second = 0.0;
    double validation_loss_before = 0.0;
    double validation_loss_after = 0.0;
    double test_loss_before = 0.0;
    double test_loss_after = 0.0;
    bool finite = false;
};

BenchmarkResult run_model(const NlpModelKind kind, const NlpModelConfig& model_config, const NlpOptimizerConfig& optimizer,
                          const NlpDataset& dataset, const std::vector<NlpSequence>& test_sequences, const Tokenizer& tokenizer,
                          const std::size_t decode_tokens, const std::size_t repeats) {
    NlpTrainer trainer(model_config, optimizer, dataset.tokenizer_hash, dataset.dataset_hash);
    const auto before = trainer.evaluate(dataset.validation);
    const auto test_before = trainer.evaluate(test_sequences);
    const auto train_started = std::chrono::steady_clock::now();
    static_cast<void>(trainer.train_steps(dataset, optimizer.total_steps));
    const auto train_finished = std::chrono::steady_clock::now();
    const auto after = trainer.evaluate(dataset.validation);
    const auto test_after = trainer.evaluate(test_sequences);

    std::size_t train_tokens = 0U;
    for (const auto& point : trainer.history()) train_tokens += point.token_count;
    require(train_tokens > 0U, "benchmark training produced no target tokens");

    double evaluation_seconds = 0.0;
    std::size_t evaluation_tokens = 0U;
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto measured = trainer.evaluate(dataset.validation);
        evaluation_seconds += measured.elapsed_seconds;
        evaluation_tokens += measured.token_count;
    }

    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    std::size_t generated_tokens = 0U;
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto measured = measure_decode(trainer.model(), tokenizer, "The scientist observed", decode_tokens);
        require(measured.finite, "benchmark decode timing is non-finite");
        prefill_seconds += measured.prefill_seconds;
        decode_seconds += measured.decode_seconds;
        generated_tokens += measured.generated_tokens;
    }
    std::ostringstream model_stream;
    trainer.model().save_model(model_stream);
    const auto train_seconds = std::chrono::duration<double>(train_finished - train_started).count();
    const auto average_evaluation_seconds = evaluation_seconds / static_cast<double>(repeats);
    const auto average_prefill_seconds = prefill_seconds / static_cast<double>(repeats);
    const auto average_decode_seconds = decode_seconds / static_cast<double>(repeats);
    const auto average_generated_tokens = generated_tokens / repeats;
    const auto total_decode_seconds = average_prefill_seconds + average_decode_seconds;
    BenchmarkResult result;
    result.kind = kind;
    result.parameter_count = trainer.model().parameter_count();
    result.parameter_bytes = result.parameter_count * sizeof(double);
    result.state_memory_bytes = trainer.model().state_memory_bytes();
    result.serialized_model_bytes = model_stream.str().size();
    result.train_seconds = train_seconds;
    result.train_tokens = train_tokens;
    result.train_tokens_per_second = static_cast<double>(train_tokens) / train_seconds;
    result.evaluation_seconds = average_evaluation_seconds;
    result.evaluation_tokens = evaluation_tokens / repeats;
    result.evaluation_tokens_per_second = static_cast<double>(result.evaluation_tokens) / average_evaluation_seconds;
    result.prefill_seconds = average_prefill_seconds;
    result.decode_seconds = average_decode_seconds;
    result.generated_tokens = average_generated_tokens;
    result.decode_tokens_per_second = average_decode_seconds > 0.0 && average_generated_tokens > 1U
                                          ? static_cast<double>(average_generated_tokens - 1U) / average_decode_seconds
                                          : 0.0;
    result.end_to_end_tokens_per_second = total_decode_seconds > 0.0 ? static_cast<double>(average_generated_tokens) / total_decode_seconds : 0.0;
    result.validation_loss_before = before.cross_entropy;
    result.validation_loss_after = after.cross_entropy;
    result.test_loss_before = test_before.cross_entropy;
    result.test_loss_after = test_after.cross_entropy;
    result.finite = before.finite && after.finite && test_before.finite && test_after.finite && std::isfinite(result.train_tokens_per_second) &&
                    std::isfinite(result.evaluation_tokens_per_second) && std::isfinite(result.end_to_end_tokens_per_second);
    require(result.finite, "benchmark produced non-finite result");
    return result;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto train_path = argument_path(argc, argv, "--train", "artifacts/track1/real-release/data/pretrain_train.txt");
        const auto validation_path = argument_path(argc, argv, "--validation", "artifacts/track1/real-release/data/pretrain_validation.txt");
        const auto test_path = argument_path(argc, argv, "--test", "artifacts/track1/real-release/data/pretrain_test.txt");
        const auto tokenizer_path = argument_path(argc, argv, "--tokenizer", "data/stage-10/tokenizer_snapshot.bin");
        const auto output_path = argument_path(argc, argv, "--output", "/tmp/cct-optimization/benchmark.json");
        const auto context_length = argument_size(argc, argv, "--context", 128U);
        const auto embedding_dim = argument_size(argc, argv, "--embedding", 16U);
        const auto hidden_dim = argument_size(argc, argv, "--hidden", 16U);
        const auto steps = argument_size(argc, argv, "--steps", 20U);
        const auto batch_size = argument_size(argc, argv, "--batch", 4U);
        const auto train_sequence_limit = argument_size(argc, argv, "--train-sequences", 64U);
        const auto evaluation_sequence_limit = argument_size(argc, argv, "--eval-sequences", 32U);
        const auto decode_tokens = argument_size(argc, argv, "--decode-tokens", 64U);
        const auto repeats = argument_size(argc, argv, "--repeats", 3U);
        const auto vocabulary_mode = argument_path(argc, argv, "--vocab-mode", "compact");
        require(vocabulary_mode == "compact" || vocabulary_mode == "legacy", "benchmark vocabulary mode must be compact or legacy");

        const auto tokenizer = Tokenizer::from_snapshot(read_file(tokenizer_path));
        const auto tokenizer_hash = tokenizer.snapshot_hash();
        const auto token_id_limit = tokenizer.vocabulary().back().id;
        const auto train_document = encode_document(tokenizer, train_path, "optimization-train", true, false);
        const auto validation_document = encode_document(tokenizer, validation_path, "optimization-validation", false, true);
        const auto test_document = encode_document(tokenizer, test_path, "optimization-test", false, true);
        auto dataset = NlpDataset::build({train_document}, {validation_document}, tokenizer_hash, context_length);
        auto test_dataset = NlpDataset::build({train_document}, {test_document}, tokenizer_hash, context_length);
        cap_sequences(dataset.train, train_sequence_limit);
        cap_sequences(dataset.validation, evaluation_sequence_limit);
        cap_sequences(test_dataset.validation, evaluation_sequence_limit);
        dataset.train_tokens = count_targets(dataset.train);
        dataset.validation_tokens = count_targets(dataset.validation);
        test_dataset.validation_tokens = count_targets(test_dataset.validation);
        const auto compact_vocabulary = vocabulary_mode == "compact";
        const auto vocabulary_size = compact_vocabulary ? 2U + static_cast<std::size_t>(token_id_limit - Tokenizer::kByteFirstId)
                                                        : static_cast<std::size_t>(token_id_limit) + 1U;
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.001;
        optimizer.beta1 = 0.9;
        optimizer.beta2 = 0.999;
        optimizer.epsilon = 1e-8;
        optimizer.weight_decay = 1e-4;
        optimizer.clip_norm = 1.0;
        optimizer.warmup_steps = std::min<std::size_t>(10U, steps);
        optimizer.batch_size = batch_size;
        optimizer.total_steps = steps;
        optimizer.validation_interval_steps = steps;
        const std::vector<NlpModelKind> kinds{NlpModelKind::Track1CctRecurrence, NlpModelKind::GRU,
                                              NlpModelKind::DiagonalSSM, NlpModelKind::DenseCausalAttention};
        std::vector<BenchmarkResult> results;
        for (const auto kind : kinds) {
            const NlpModelConfig config{kind, vocabulary_size, embedding_dim, hidden_dim, context_length, 1701U, compact_vocabulary, token_id_limit};
            results.push_back(run_model(kind, config, optimizer, dataset, test_dataset.validation, tokenizer, decode_tokens, repeats));
        }
        std::ostringstream report;
        report << std::setprecision(12) << "{\"status\":\"COMPLETE\",\"contract\":{\"train_tokens\":" << dataset.train_tokens
               << ",\"validation_tokens\":" << dataset.validation_tokens << ",\"test_tokens\":" << test_dataset.validation_tokens
               << ",\"context_length\":" << context_length << ",\"embedding_dim\":" << embedding_dim << ",\"hidden_dim\":" << hidden_dim
               << ",\"steps\":" << steps << ",\"batch_size\":" << batch_size << ",\"repeats\":" << repeats
               << ",\"decode_tokens\":" << decode_tokens << ",\"train_sequences_used\":" << dataset.train.size()
               << ",\"evaluation_sequences_used\":" << dataset.validation.size() << ",\"tokenizer_hash\":\"" << tokenizer_hash
               << "\",\"active_vocabulary_size\":" << vocabulary_size << ",\"vocabulary_mode\":\"" << vocabulary_mode << "\"},\"results\":[";
        for (std::size_t index = 0U; index < results.size(); ++index) {
            if (index > 0U) report << ',';
            const auto& result = results[index];
            report << "{\"model\":\"" << model_name(result.kind) << "\",\"parameter_count\":" << result.parameter_count
                   << ",\"parameter_bytes\":" << result.parameter_bytes << ",\"state_memory_bytes\":" << result.state_memory_bytes
                   << ",\"serialized_model_bytes\":" << result.serialized_model_bytes << ",\"train_seconds\":" << result.train_seconds
                   << ",\"train_tokens\":" << result.train_tokens << ",\"train_tokens_per_second\":" << result.train_tokens_per_second
                   << ",\"evaluation_seconds\":" << result.evaluation_seconds << ",\"evaluation_tokens\":" << result.evaluation_tokens
                   << ",\"evaluation_tokens_per_second\":" << result.evaluation_tokens_per_second << ",\"prefill_seconds\":" << result.prefill_seconds
                   << ",\"decode_seconds\":" << result.decode_seconds << ",\"generated_tokens\":" << result.generated_tokens
                   << ",\"decode_tokens_per_second\":" << result.decode_tokens_per_second
                   << ",\"end_to_end_tokens_per_second\":" << result.end_to_end_tokens_per_second
                   << ",\"validation_loss_before\":" << result.validation_loss_before << ",\"validation_loss_after\":" << result.validation_loss_after
                   << ",\"test_loss_before\":" << result.test_loss_before << ",\"test_loss_after\":" << result.test_loss_after
                   << ",\"finite\":" << (result.finite ? "true" : "false") << "}";
        }
        report << "]}\n";
        write_file(output_path, report.str());
        std::cout << report.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "optimization benchmark error: " << error.what() << '\n';
        return 2;
    }
}
