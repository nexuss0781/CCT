#include "cct/nlp_trainer.hpp"

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
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using cct::EncodedDocument;
using cct::NlpDataset;
using cct::NlpModelConfig;
using cct::NlpModelKind;
using cct::NlpOptimizerConfig;
using cct::NlpTrainer;
using cct::TokenId;
using cct::Tokenizer;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read qualification input: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    require(static_cast<bool>(input) || input.eof(), "cannot finish qualification input: " + path.string());
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write qualification output: " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish qualification output: " + path.string());
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const auto character : value) {
        if (character == '\\') escaped += "\\\\";
        else if (character == '"') escaped += "\\\"";
        else if (character == '\n') escaped += "\\n";
        else if (character == '\r') escaped += "\\r";
        else if (character == '\t') escaped += "\\t";
        else escaped.push_back(character);
    }
    return escaped;
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
        require(consumed == value.size() && parsed > 0U, "invalid qualification numeric argument: " + name);
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid qualification numeric argument: " + name);
    }
}

std::uint64_t argument_seed(const int argc, char** argv, const std::string& name, const std::uint64_t fallback) {
    const auto value = argument_path(argc, argv, name, std::to_string(fallback));
    try {
        std::size_t consumed = 0U;
        const auto parsed = std::stoull(value, &consumed, 10);
        require(consumed == value.size(), "invalid qualification seed argument");
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid qualification seed argument: " + name);
    }
}

std::string model_name(const NlpModelKind kind) { return cct::nlp_model_kind_name(kind); }

std::size_t count_targets(const std::vector<cct::NlpSequence>& sequences) {
    std::size_t total = 0U;
    for (const auto& sequence : sequences) for (const auto mask : sequence.loss_mask) total += mask != 0U ? 1U : 0U;
    return total;
}

void cap_sequences(std::vector<cct::NlpSequence>& sequences, const std::size_t maximum) {
    if (sequences.size() > maximum) sequences.resize(maximum);
}

EncodedDocument encode_document(const Tokenizer& tokenizer, const std::filesystem::path& path, const std::string& id,
                                const bool training_allowed, const bool evaluation_allowed) {
    auto document = tokenizer.encode(read_file(path), id, false);
    document.training_allowed = training_allowed;
    document.evaluation_allowed = evaluation_allowed;
    document.evaluator_only = false;
    require(document.tokens.size() >= 2U, "qualification document is too short: " + path.string());
    return document;
}

struct GenerationDiagnostic {
    std::string prompt;
    std::string output;
    std::string greedy_output;
    std::size_t generated_tokens = 0U;
    bool repetitive = false;
    bool greedy_repetitive = false;
};

std::vector<TokenId> generate_tokens(const cct::NextTokenModel& model, const Tokenizer& tokenizer, const std::string& prompt,
                                     const std::size_t maximum_tokens, const bool constrained) {
    const auto encoded = tokenizer.encode(prompt, "qualification-prompt", false);
    std::vector<TokenId> context;
    context.reserve(encoded.tokens.size() + maximum_tokens);
    for (const auto& token : encoded.tokens) context.push_back(token.id);
    std::vector<TokenId> generated;
    for (std::size_t step = 0U; step < maximum_tokens; ++step) {
        const auto window_start = context.size() > model.config().context_length ? context.size() - model.config().context_length : 0U;
        const std::vector<TokenId> window(context.begin() + static_cast<std::ptrdiff_t>(window_start), context.end());
        const auto logits = model.next_logits(window);
        std::size_t selected_slot = 0U;
        if (!constrained) {
            selected_slot = static_cast<std::size_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        } else {
            std::vector<std::size_t> candidates(logits.size());
            std::iota(candidates.begin(), candidates.end(), 0U);
            constexpr std::size_t kCandidateLimit = 64U;
            const auto candidate_count = std::min(kCandidateLimit, candidates.size());
            std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count), candidates.end(),
                              [&logits](const std::size_t left, const std::size_t right) { return logits[left] > logits[right]; });
            selected_slot = candidates.front();
            for (std::size_t candidate_index = 0U; candidate_index < candidate_count; ++candidate_index) {
                const auto candidate_slot = candidates[candidate_index];
                const auto candidate = model.token_id_from_logit_slot(candidate_slot);
                if (candidate == Tokenizer::kEosId) {
                    selected_slot = candidate_slot;
                    break;
                }
                bool forbidden = generated.size() >= 1U && candidate == generated.back();
                forbidden = forbidden || (generated.size() >= 2U && candidate == generated[generated.size() - 2U]);
                if (!forbidden && generated.size() >= 2U) {
                    for (std::size_t start = 0U; start + 2U < generated.size(); ++start) {
                        if (generated[start] == generated[generated.size() - 2U] && generated[start + 1U] == generated.back() &&
                            generated[start + 2U] == candidate) {
                            forbidden = true;
                            break;
                        }
                    }
                }
                if (!forbidden) {
                    selected_slot = candidate_slot;
                    break;
                }
            }
        }
        const auto token = model.token_id_from_logit_slot(selected_slot);
        if (token == Tokenizer::kEosId) break;
        generated.push_back(token);
        context.push_back(token);
    }
    return generated;
}

bool is_repetitive(const std::vector<TokenId>& generated, const std::string& output) {
    for (std::size_t index = 2U; index < generated.size(); ++index) {
        if (generated[index] == generated[index - 1U] && generated[index] == generated[index - 2U]) return true;
        if (index >= 3U && generated[index] == generated[index - 2U] && generated[index - 1U] == generated[index - 3U]) return true;
    }
    if (generated.size() >= 8U) {
        const std::unordered_set<TokenId> unique_tokens(generated.begin(), generated.end());
        if (unique_tokens.size() * 4U <= generated.size()) return true;
    }
    if (!output.empty()) {
        const auto has_non_whitespace = std::any_of(output.begin(), output.end(), [](const char character) {
            return character != ' ' && character != '\t' && character != '\n' && character != '\r';
        });
        if (!has_non_whitespace) return true;
    }
    return false;
}

GenerationDiagnostic generate(const cct::NextTokenModel& model, const Tokenizer& tokenizer, const std::string& prompt,
                              const std::size_t maximum_tokens) {
    const auto greedy = generate_tokens(model, tokenizer, prompt, maximum_tokens, false);
    const auto constrained = generate_tokens(model, tokenizer, prompt, maximum_tokens, true);
    GenerationDiagnostic result;
    result.prompt = prompt;
    result.output = tokenizer.decode(constrained, true);
    result.greedy_output = tokenizer.decode(greedy, true);
    result.generated_tokens = constrained.size();
    result.repetitive = is_repetitive(constrained, result.output);
    result.greedy_repetitive = is_repetitive(greedy, result.greedy_output);
    return result;
}

struct TrialResult {
    NlpModelKind kind = NlpModelKind::Track1CctRecurrence;
    std::size_t parameter_count = 0U;
    std::size_t state_memory_bytes = 0U;
    std::size_t checkpoint_model_bytes = 0U;
    double control_validation_loss = 0.0;
    double final_validation_loss = 0.0;
    double control_test_loss = 0.0;
    double final_test_loss = 0.0;
    double train_seconds = 0.0;
    double target_tokens_per_second = 0.0;
    bool finite = false;
    bool validation_improved = false;
    bool test_improved = false;
    std::vector<GenerationDiagnostic> generations;
};

TrialResult run_trial(const NlpModelKind kind, const NlpModelConfig& model_config, const NlpOptimizerConfig& optimizer,
                      const NlpDataset& dataset, const std::vector<cct::NlpSequence>& test_sequences,
                       const Tokenizer& tokenizer, const std::string& tokenizer_hash) {
    NlpTrainer trainer(model_config, optimizer, tokenizer_hash, dataset.dataset_hash);
    const auto control_validation = trainer.evaluate(dataset.validation);
    const auto control_test = trainer.evaluate(test_sequences);
    const auto started = std::chrono::steady_clock::now();
    static_cast<void>(trainer.train_steps(dataset, optimizer.total_steps));
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto final_validation = trainer.evaluate(dataset.validation);
    const auto final_test = trainer.evaluate(test_sequences);
    std::size_t target_tokens = 0U;
    for (const auto& point : trainer.history()) target_tokens += point.token_count;
    std::ostringstream model_stream;
    trainer.model().save_model(model_stream);
    TrialResult result;
    result.kind = kind;
    result.parameter_count = trainer.model().parameter_count();
    result.state_memory_bytes = trainer.model().state_memory_bytes();
    result.checkpoint_model_bytes = model_stream.str().size();
    result.control_validation_loss = control_validation.cross_entropy;
    result.final_validation_loss = final_validation.cross_entropy;
    result.control_test_loss = control_test.cross_entropy;
    result.final_test_loss = final_test.cross_entropy;
    result.train_seconds = elapsed;
    result.target_tokens_per_second = elapsed > 0.0 ? static_cast<double>(target_tokens) / elapsed : 0.0;
    result.finite = control_validation.finite && control_test.finite && final_validation.finite && final_test.finite &&
                    std::isfinite(result.target_tokens_per_second);
    result.validation_improved = result.final_validation_loss < result.control_validation_loss;
    result.test_improved = result.final_test_loss < result.control_test_loss;
    const std::vector<std::string> prompts{"The scientist observed", "In the morning, the child", "A good explanation"};
    for (std::size_t prompt_index = 0U; prompt_index < prompts.size(); ++prompt_index) {
        result.generations.push_back(generate(trainer.model(), tokenizer, prompts[prompt_index], 24U));
    }
    return result;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto train_path = argument_path(argc, argv, "--train", "/tmp/cct-track1-2m/data/pretrain_train.txt");
        const auto validation_path = argument_path(argc, argv, "--validation", "/tmp/cct-track1-2m/data/pretrain_validation.txt");
        const auto test_path = argument_path(argc, argv, "--test", "/tmp/cct-track1-2m/data/pretrain_test.txt");
        const auto tokenizer_path = argument_path(argc, argv, "--tokenizer", "data/stage-10/tokenizer_snapshot.bin");
        const auto output_path = argument_path(argc, argv, "--output", "/tmp/cct-architecture-qualification/report.json");
        const auto context_length = argument_size(argc, argv, "--context", 128U);
        const auto embedding_dim = argument_size(argc, argv, "--embedding", 16U);
        const auto hidden_dim = argument_size(argc, argv, "--hidden", 16U);
        const auto steps = argument_size(argc, argv, "--steps", 100U);
        const auto batch_size = argument_size(argc, argv, "--batch", 4U);
        const auto train_sequence_limit = argument_size(argc, argv, "--train-sequences", 256U);
        const auto evaluation_sequence_limit = argument_size(argc, argv, "--eval-sequences", 64U);
        const auto vocabulary_mode = argument_path(argc, argv, "--vocab-mode", "compact");
        require(vocabulary_mode == "compact" || vocabulary_mode == "legacy", "qualification vocabulary mode must be compact or legacy");
        const auto seed = argument_seed(argc, argv, "--seed", 1701U);
        const auto tokenizer = Tokenizer::from_snapshot(read_file(tokenizer_path));
        require(!tokenizer.vocabulary().empty(), "qualification tokenizer vocabulary is empty");
        const auto token_id_limit = tokenizer.vocabulary().back().id;
        require(token_id_limit >= Tokenizer::kByteFirstId, "qualification tokenizer has no byte vocabulary");
        const auto tokenizer_hash = tokenizer.snapshot_hash();
        const auto train_document = encode_document(tokenizer, train_path, "qualification-train", true, false);
        const auto validation_document = encode_document(tokenizer, validation_path, "qualification-validation", false, true);
        const auto test_document = encode_document(tokenizer, test_path, "qualification-test", false, true);
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
        std::vector<TrialResult> results;
        for (const auto kind : kinds) {
            NlpModelConfig config{kind, vocabulary_size, embedding_dim, hidden_dim, context_length, seed, compact_vocabulary, token_id_limit};
            results.push_back(run_trial(kind, config, optimizer, dataset, test_dataset.validation, tokenizer, tokenizer_hash));
        }
        std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
        std::ostringstream report;
        report << std::setprecision(10) << "{\"status\":\"COMPLETE\",\"contract\":{\"train_bytes\":"
               << std::filesystem::file_size(train_path) << ",\"train_model_tokens\":" << dataset.train_tokens
               << ",\"validation_model_tokens\":" << dataset.validation_tokens << ",\"test_model_tokens\":" << test_dataset.validation_tokens
               << ",\"context_length\":" << context_length << ",\"embedding_dim\":" << embedding_dim << ",\"hidden_dim\":" << hidden_dim
               << ",\"steps\":" << steps << ",\"batch_size\":" << batch_size << ",\"train_sequences_used\":" << dataset.train.size()
               << ",\"evaluation_sequences_used\":" << dataset.validation.size() << ",\"seed\":" << seed
               << ",\"tokenizer_hash\":\"" << tokenizer_hash << "\",\"token_id_limit\":" << token_id_limit
               << ",\"vocabulary_mode\":\"" << vocabulary_mode << "\",\"active_vocabulary_size\":" << vocabulary_size
               << ",\"decoding\":\"greedy_and_deterministic_no_repeat_2gram_3gram_top64\"},\"results\":[";
        for (std::size_t result_index = 0U; result_index < results.size(); ++result_index) {
            if (result_index > 0U) report << ',';
            const auto& result = results[result_index];
            report << "{\"model\":\"" << model_name(result.kind) << "\",\"parameter_count\":" << result.parameter_count
                   << ",\"state_memory_bytes\":" << result.state_memory_bytes << ",\"model_bytes\":" << result.checkpoint_model_bytes
                   << ",\"control_validation_loss\":" << result.control_validation_loss << ",\"final_validation_loss\":" << result.final_validation_loss
                   << ",\"control_test_loss\":" << result.control_test_loss << ",\"final_test_loss\":" << result.final_test_loss
                   << ",\"train_seconds\":" << result.train_seconds << ",\"target_tokens_per_second\":" << result.target_tokens_per_second
                   << ",\"finite\":" << (result.finite ? "true" : "false") << ",\"validation_improved\":" << (result.validation_improved ? "true" : "false")
                   << ",\"test_improved\":" << (result.test_improved ? "true" : "false") << ",\"generations\":[";
            for (std::size_t generation_index = 0U; generation_index < result.generations.size(); ++generation_index) {
                if (generation_index > 0U) report << ',';
                const auto& generation = result.generations[generation_index];
                report << "{\"prompt\":\"" << json_escape(generation.prompt) << "\",\"output\":\"" << json_escape(generation.output)
                       << "\",\"greedy_output\":\"" << json_escape(generation.greedy_output) << "\",\"generated_tokens\":"
                       << generation.generated_tokens << ",\"repetitive\":" << (generation.repetitive ? "true" : "false")
                       << ",\"greedy_repetitive\":" << (generation.greedy_repetitive ? "true" : "false") << "}";
            }
            report << "]}";
        }
        report << "]}\n";
        write_file(output_path, report.str());
        std::cout << report.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "architecture qualification error: " << error.what() << '\n';
        return 2;
    }
}
