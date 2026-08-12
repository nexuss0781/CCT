#include "cct/corpus.hpp"
#include "cct/nlp_trainer.hpp"
#include "cct/sft.hpp"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cct::EncodedDocument;
using cct::GovernedCorpus;
using cct::NlpDataset;
using cct::NlpEvaluation;
using cct::NlpModelConfig;
using cct::NlpModelKind;
using cct::NlpOptimizerConfig;
using cct::NlpTrainer;
using cct::SftFormatter;
using cct::SftInstructionExample;
using cct::SftOutputKind;
using cct::SftTaskKind;
using cct::SftTaskSchema;
using cct::Tokenizer;

struct Config {
    std::filesystem::path input_root = "artifacts/track1";
    std::filesystem::path output_root = "artifacts/track1/training";
    std::filesystem::path tokenizer_snapshot = "data/stage-10/tokenizer_snapshot.bin";
    std::size_t pretrain_steps = 200U;
    std::size_t sft_steps = 120U;
    std::size_t context_length = 32U;
    std::size_t embedding_dim = 4U;
    std::size_t hidden_dim = 4U;
    std::size_t pretrain_selection_validation_limit = 256U;
    std::size_t sft_selection_evaluation_limit = 64U;
    std::size_t final_test_limit = 0U;
    std::size_t sft_context_bytes = 1024U;
    std::uint64_t seed = 1701U;
};

struct SquadRecord {
    std::string id;
    std::string context;
    std::string question;
    std::string answer;
    bool answerable = false;
};

struct PhaseResult {
    NlpEvaluation before;
    NlpEvaluation after;
    NlpEvaluation held_out;
    std::size_t train_sequences = 0U;
    std::size_t selection_sequences = 0U;
    std::size_t held_out_sequences = 0U;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary);
    require(static_cast<bool>(output), "cannot write " + temporary);
    output << content;
    output.close();
    require(static_cast<bool>(output), "cannot finish " + temporary);
    std::filesystem::rename(temporary, path);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        if (character == '\\') output << "\\\\";
        else if (character == '"') output << "\\\"";
        else if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character < 0x20U) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        else output << static_cast<char>(character);
    }
    return output.str();
}

unsigned int hex_digit(const char value) {
    if (value >= '0' && value <= '9') return static_cast<unsigned int>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<unsigned int>(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return static_cast<unsigned int>(value - 'A' + 10);
    throw std::runtime_error("invalid JSON hex digit");
}

void append_codepoint(std::string& output, const unsigned int codepoint) {
    require(codepoint <= 0x10FFFFU && !(codepoint >= 0xD800U && codepoint <= 0xDFFFU), "invalid JSON Unicode codepoint");
    if (codepoint <= 0x7FU) output.push_back(static_cast<char>(codepoint));
    else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

std::string json_string(const std::string& line, const std::string& key) {
    const auto marker = "\"" + key + "\":";
    const auto marker_position = line.find(marker);
    require(marker_position != std::string::npos, "JSONL field is missing: " + key);
    std::size_t position = marker_position + marker.size();
    while (position < line.size() && (line[position] == ' ' || line[position] == '\t')) ++position;
    require(position < line.size() && line[position] == '"', "JSONL string field is invalid: " + key);
    ++position;
    std::string value;
    while (position < line.size()) {
        const auto character = line[position++];
        if (character == '"') return value;
        if (character != '\\') {
            value.push_back(character);
            continue;
        }
        require(position < line.size(), "truncated JSON escape");
        const auto escaped = line[position++];
        if (escaped == '"' || escaped == '\\' || escaped == '/') value.push_back(escaped);
        else if (escaped == 'b') value.push_back('\b');
        else if (escaped == 'f') value.push_back('\f');
        else if (escaped == 'n') value.push_back('\n');
        else if (escaped == 'r') value.push_back('\r');
        else if (escaped == 't') value.push_back('\t');
        else if (escaped == 'u') {
            const auto parse_code_unit = [&](std::size_t& cursor) {
                require(cursor + 4U <= line.size(), "truncated JSON Unicode escape");
                unsigned int code_unit = 0U;
                for (std::size_t index = 0U; index < 4U; ++index) code_unit = (code_unit << 4U) | hex_digit(line[cursor++]);
                return code_unit;
            };
            const auto codepoint = parse_code_unit(position);
            if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                require(position + 2U <= line.size() && line[position] == '\\' && line[position + 1U] == 'u', "missing JSON low surrogate");
                position += 2U;
                const auto low = parse_code_unit(position);
                require(low >= 0xDC00U && low <= 0xDFFFU, "invalid JSON low surrogate");
                append_codepoint(value, 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U));
            } else {
                append_codepoint(value, codepoint);
            }
        } else throw std::runtime_error("unsupported JSON escape");
    }
    throw std::runtime_error("unterminated JSON string field: " + key);
}

bool json_bool(const std::string& line, const std::string& key) {
    const auto marker = "\"" + key + "\":";
    const auto position = line.find(marker);
    require(position != std::string::npos, "JSONL boolean field is missing: " + key);
    return line.compare(position + marker.size(), 4U, "true") == 0;
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

std::string truncate_utf8(const std::string& value, const std::size_t maximum) {
    if (value.size() <= maximum) return value;
    std::size_t end = maximum;
    while (end > 0U && (static_cast<unsigned char>(value[end]) & 0xC0U) == 0x80U) --end;
    return value.substr(0U, end);
}

std::vector<std::pair<std::string, std::string>> read_text_records(const std::filesystem::path& path, const std::size_t maximum_records) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read text records from " + path.string());
    std::vector<std::pair<std::string, std::string>> records;
    std::string line;
    std::size_t index = 0U;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.size() < 8U) continue;
        records.emplace_back(path.filename().string() + ":" + std::to_string(index++), std::move(line));
        if (maximum_records != 0U && records.size() >= maximum_records) break;
    }
    require(!records.empty(), "no usable text records in " + path.string());
    return records;
}

std::vector<SquadRecord> read_squad_records(const std::filesystem::path& path, const std::size_t maximum_records) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read SQuAD records from " + path.string());
    std::vector<SquadRecord> records;
    std::string line;
    while (std::getline(input, line)) {
        if (trim(line).empty()) continue;
        SquadRecord record;
        record.id = json_string(line, "id");
        record.context = json_string(line, "context");
        record.question = json_string(line, "question");
        record.answer = json_string(line, "answer");
        record.answerable = json_bool(line, "answerable");
        require(!record.id.empty() && !record.context.empty() && !record.question.empty(), "SQuAD training record is incomplete");
        require(record.answerable || record.answer.empty(), "unanswerable SQuAD record has an answer");
        records.push_back(std::move(record));
        if (maximum_records != 0U && records.size() >= maximum_records) break;
    }
    require(!records.empty(), "no usable SQuAD records in " + path.string());
    return records;
}

std::string format_squad(const SquadRecord& record, const std::size_t context_bytes) {
    const auto answer = record.answerable ? record.answer : std::string("<NO_ANSWER>");
    return "<CCT_TASK_V1>\n<CONTEXT>\n" + truncate_utf8(record.context, context_bytes) + "\n<QUESTION>\n" + record.question +
           "\n<TARGET>\n" + answer + "\n<END>";
}

std::vector<EncodedDocument> encode_text_records(const Tokenizer& tokenizer, const std::vector<std::pair<std::string, std::string>>& records,
                                                 const bool training) {
    std::vector<EncodedDocument> documents;
    documents.reserve(records.size());
    for (const auto& [id, content] : records) {
        auto document = tokenizer.encode(content, id, true);
        document.training_allowed = training;
        document.evaluation_allowed = !training;
        documents.push_back(std::move(document));
    }
    return documents;
}

SftTaskSchema squad_schema() {
    SftTaskSchema schema;
    schema.task_id = "track1_squad_v2";
    schema.kind = SftTaskKind::GroundedQuestionAnswering;
    schema.schema_version = "v1";
    schema.output_kind = SftOutputKind::BoundedText;
    schema.labels = {"answer"};
    schema.maximum_output_bytes = 512U;
    schema.requires_citations = false;
    schema.allows_abstention = true;
    schema.policy_class = "track1-governed";
    return schema;
}

SftInstructionExample squad_example(const SquadRecord& record, const std::size_t context_bytes) {
    SftInstructionExample example;
    example.example_id = record.id;
    example.task_id = "track1_squad_v2";
    example.schema_version = "v1";
    example.input = "<CONTEXT>\n" + truncate_utf8(record.context, context_bytes) + "\n<QUESTION>\n" + record.question;
    example.target = record.answerable ? record.answer : std::string("<NO_ANSWER>");
    example.target_label = "answer";
    example.input_provenance = "GEM/squad_v2";
    example.target_provenance = "SQuAD-v2-answer";
    example.policy_class = "track1-governed";
    example.split = "track1";
    example.evaluator_owner = "track1";
    example.source_hash = GovernedCorpus::content_sha256(record.context + "\n" + record.question);
    example.target_hash = GovernedCorpus::content_sha256(example.target);
    example.example_hash = GovernedCorpus::content_sha256(record.id + "\n" + example.source_hash + "\n" + example.target_hash);
    example.citation_id = record.id;
    example.source_span_start = 0U;
    example.source_span_end = record.context.size();
    return example;
}

void append_sft_sequences(const Tokenizer& tokenizer, const std::vector<SquadRecord>& records, const std::size_t context_bytes,
                          const std::size_t context_length, const std::string& split, std::vector<cct::NlpSequence>& destination,
                          std::size_t& token_count, std::ostringstream& identity) {
    const auto schema = squad_schema();
    for (const auto& record : records) {
        const auto formatted = SftFormatter::format(squad_example(record, context_bytes), schema, tokenizer);
        require(formatted.token_ids.size() == formatted.loss_mask.size() && formatted.token_ids.size() >= 2U,
                "formatted SQuAD example has invalid token/mask lengths");
        std::size_t start = 0U;
        std::size_t chunk = 0U;
        while (start + 1U < formatted.token_ids.size()) {
            const auto end = std::min(formatted.token_ids.size(), start + context_length);
            if (end - start < 2U) break;
            cct::NlpSequence sequence;
            sequence.sequence_id = split + ":" + record.id + ":" + std::to_string(chunk++);
            sequence.record_id = record.id;
            sequence.input_ids.insert(sequence.input_ids.end(), formatted.token_ids.begin() + static_cast<std::ptrdiff_t>(start),
                                      formatted.token_ids.begin() + static_cast<std::ptrdiff_t>(end));
            sequence.target_ids.resize(end - start, Tokenizer::kPadId);
            sequence.loss_mask.assign(end - start, 0U);
            std::size_t active_tokens = 0U;
            for (std::size_t index = start; index + 1U < end; ++index) {
                sequence.target_ids[index - start] = formatted.token_ids[index + 1U];
                sequence.loss_mask[index - start] = formatted.loss_mask[index + 1U];
                active_tokens += static_cast<std::size_t>(sequence.loss_mask[index - start]);
            }
            if (active_tokens > 0U) {
                token_count += active_tokens;
                destination.push_back(std::move(sequence));
            }
            start = end;
        }
        identity << record.id << ':' << record.answerable << ':' << GovernedCorpus::content_sha256(format_squad(record, context_bytes)) << '\n';
    }
}

NlpDataset build_sft_dataset(const Tokenizer& tokenizer, const std::vector<SquadRecord>& train_records,
                             const std::vector<SquadRecord>& evaluation_records, const std::string& tokenizer_hash,
                             const std::size_t context_bytes, const std::size_t context_length) {
    require(!train_records.empty() && !evaluation_records.empty(), "SQuAD SFT datasets are empty");
    NlpDataset dataset;
    dataset.tokenizer_hash = tokenizer_hash;
    dataset.context_length = context_length;
    std::ostringstream identity;
    identity << "track1-sft-target-span-only-v1\ntrain\n";
    append_sft_sequences(tokenizer, train_records, context_bytes, context_length, "train", dataset.train, dataset.train_tokens, identity);
    identity << "evaluation\n";
    append_sft_sequences(tokenizer, evaluation_records, context_bytes, context_length, "evaluation", dataset.validation, dataset.validation_tokens, identity);
    require(!dataset.train.empty() && !dataset.validation.empty() && dataset.train_tokens > 0U && dataset.validation_tokens > 0U,
            "SQuAD SFT dataset produced no answer-target tokens");
    dataset.dataset_hash = GovernedCorpus::content_sha256(tokenizer_hash + "\ncontext=" + std::to_string(context_length) + "\n" + identity.str());
    return dataset;
}

std::vector<cct::NlpSequence> build_sft_evaluation_sequences(const Tokenizer& tokenizer, const std::vector<SquadRecord>& records,
                                                              const std::size_t context_bytes, const std::size_t context_length,
                                                              std::size_t& token_count) {
    std::vector<cct::NlpSequence> sequences;
    std::ostringstream identity;
    append_sft_sequences(tokenizer, records, context_bytes, context_length, "frozen_final_test", sequences, token_count, identity);
    require(!sequences.empty() && token_count > 0U, "frozen SQuAD final test produced no answer-target tokens");
    return sequences;
}

NlpDataset evaluation_dataset(const std::vector<EncodedDocument>& documents, const std::string& tokenizer_hash, const std::size_t context_length) {
    require(!documents.empty(), "held-out evaluation documents are empty");
    auto anchor = documents.front();
    anchor.record_id = "evaluation-anchor";
    anchor.training_allowed = true;
    anchor.evaluation_allowed = false;
    return NlpDataset::build({anchor}, documents, tokenizer_hash, context_length);
}

std::string evaluation_json(const NlpEvaluation& evaluation) {
    std::ostringstream output;
    output << std::setprecision(10) << "{\"cross_entropy\":" << evaluation.cross_entropy << ",\"perplexity\":" << evaluation.perplexity
           << ",\"token_accuracy\":" << evaluation.token_accuracy << ",\"token_count\":" << evaluation.token_count
           << ",\"tokens_per_second\":" << evaluation.tokens_per_second << ",\"finite\":" << (evaluation.finite ? "true" : "false") << "}";
    return output.str();
}

PhaseResult run_phase(NlpTrainer& trainer, const NlpDataset& training, const std::vector<cct::NlpSequence>& held_out, const std::size_t steps) {
    PhaseResult result;
    result.train_sequences = training.train.size();
    result.selection_sequences = training.validation.size();
    result.held_out_sequences = held_out.size();
    result.before = trainer.evaluate(training.validation);
    static_cast<void>(trainer.train_steps(training, steps));
    result.after = trainer.evaluate(training.validation);
    result.held_out = trainer.evaluate(held_out);
    require(result.before.finite && result.after.finite && result.held_out.finite, "Track 1 phase evaluation is non-finite");
    return result;
}

Config parse_arguments(const int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        const auto value = [&]() {
            require(index + 1 < argc, "missing value for " + key);
            return std::string(argv[++index]);
        };
        if (key == "--input") config.input_root = value();
        else if (key == "--output") config.output_root = value();
        else if (key == "--tokenizer") config.tokenizer_snapshot = value();
        else if (key == "--pretrain-steps") config.pretrain_steps = std::stoull(value());
        else if (key == "--sft-steps") config.sft_steps = std::stoull(value());
        else if (key == "--context") config.context_length = std::stoull(value());
        else if (key == "--embedding") config.embedding_dim = std::stoull(value());
        else if (key == "--hidden") config.hidden_dim = std::stoull(value());
        else if (key == "--pretrain-selection-validation-limit") config.pretrain_selection_validation_limit = std::stoull(value());
        else if (key == "--sft-selection-evaluation-limit") config.sft_selection_evaluation_limit = std::stoull(value());
        else if (key == "--final-test-limit") config.final_test_limit = std::stoull(value());
        else if (key == "--sft-context-bytes") config.sft_context_bytes = std::stoull(value());
        else if (key == "--seed") config.seed = std::stoull(value());
        else if (key == "--help") {
            std::cout << "track1_train --input PATH --output PATH --tokenizer PATH --pretrain-steps N --sft-steps N --context N --embedding N --hidden N "
                         "--pretrain-selection-validation-limit N --sft-selection-evaluation-limit N --final-test-limit N --sft-context-bytes N --seed N\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument " + key);
    }
    require(config.pretrain_steps > 0U && config.sft_steps > 0U && config.context_length >= 2U && config.embedding_dim > 0U &&
                config.hidden_dim > 0U && config.sft_context_bytes >= 64U,
            "invalid Track 1 training configuration");
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse_arguments(argc, argv);
        const auto started = std::chrono::steady_clock::now();
        const auto tokenizer = Tokenizer::from_snapshot(read_file(config.tokenizer_snapshot));
        const auto tokenizer_hash = tokenizer.snapshot_hash();
        const auto vocabulary_size = static_cast<std::size_t>(tokenizer.vocabulary().back().id) + 1U;
        const auto data = config.input_root / "data";
        std::filesystem::create_directories(config.output_root);

        const auto pretrain_train_records = read_text_records(data / "pretrain_train.txt", 0U);
        const auto pretrain_validation_records = read_text_records(data / "pretrain_validation.txt", config.pretrain_selection_validation_limit);
        const auto pretrain_test_records = read_text_records(data / "pretrain_test.txt", 0U);
        const auto pretrain_train = encode_text_records(tokenizer, pretrain_train_records, true);
        const auto pretrain_validation = encode_text_records(tokenizer, pretrain_validation_records, false);
        const auto pretrain_test = encode_text_records(tokenizer, pretrain_test_records, false);
        const auto pretrain_dataset = NlpDataset::build(pretrain_train, pretrain_validation, tokenizer_hash, config.context_length);
        const auto pretrain_test_dataset = evaluation_dataset(pretrain_test, tokenizer_hash, config.context_length);

        NlpModelConfig model{NlpModelKind::CCT, vocabulary_size, config.embedding_dim, config.hidden_dim, config.context_length, config.seed};
        NlpOptimizerConfig pretrain_optimizer;
        pretrain_optimizer.learning_rate = 0.01;
        pretrain_optimizer.weight_decay = 1e-5;
        pretrain_optimizer.clip_norm = 1.0;
        pretrain_optimizer.warmup_steps = std::min<std::size_t>(10U, config.pretrain_steps);
        pretrain_optimizer.total_steps = config.pretrain_steps;
        NlpTrainer pretrainer(model, pretrain_optimizer, tokenizer_hash, pretrain_dataset.dataset_hash);
        const auto pretrain = run_phase(pretrainer, pretrain_dataset, pretrain_test_dataset.validation, config.pretrain_steps);
        const auto pretrain_checkpoint = config.output_root / "pretrain_checkpoint.bin";
        pretrainer.save_checkpoint(pretrain_checkpoint.string());

        const auto sft_train_records = read_squad_records(data / "squad_sft_train.jsonl", 0U);
        const auto sft_evaluation_records = read_squad_records(data / "squad_sft_evaluation.jsonl", config.sft_selection_evaluation_limit);
        const auto final_records = read_squad_records(data / "squad_final_test.jsonl", config.final_test_limit);
        const auto sft_dataset = build_sft_dataset(tokenizer, sft_train_records, sft_evaluation_records, tokenizer_hash,
                                                   config.sft_context_bytes, config.context_length);
        std::size_t final_target_tokens = 0U;
        const auto final_sequences = build_sft_evaluation_sequences(tokenizer, final_records, config.sft_context_bytes,
                                                                     config.context_length, final_target_tokens);

        NlpOptimizerConfig sft_optimizer;
        sft_optimizer.learning_rate = 0.005;
        sft_optimizer.weight_decay = 1e-5;
        sft_optimizer.clip_norm = 1.0;
        sft_optimizer.warmup_steps = std::min<std::size_t>(10U, config.sft_steps);
        sft_optimizer.total_steps = config.sft_steps;
        NlpTrainer sft_trainer(model, sft_optimizer, tokenizer_hash, sft_dataset.dataset_hash);
        sft_trainer.model().set_parameter_vector(pretrainer.model().parameter_vector());
        const auto sft = run_phase(sft_trainer, sft_dataset, final_sequences, config.sft_steps);
        const auto sft_checkpoint = config.output_root / "sft_checkpoint.bin";
        sft_trainer.save_checkpoint(sft_checkpoint.string());

        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto pretrain_checkpoint_hash = cct::nlp_checkpoint_hash(read_file(pretrain_checkpoint));
        const auto sft_checkpoint_hash = cct::nlp_checkpoint_hash(read_file(sft_checkpoint));
        std::ostringstream report;
        report << std::setprecision(10) << "{\"status\":\"PASS\",\"track\":\"track1\",\"backend\":\"native-c++20-cct\",\"tokenizer_hash\":\""
               << tokenizer_hash << "\",\"pretrain_checkpoint\":\"" << json_escape(pretrain_checkpoint.string()) << "\",\"pretrain_checkpoint_hash\":\""
               << pretrain_checkpoint_hash << "\",\"sft_checkpoint\":\"" << json_escape(sft_checkpoint.string()) << "\",\"sft_checkpoint_hash\":\""
               << sft_checkpoint_hash << "\",\"pretrain\":{\"before_selection\":" << evaluation_json(pretrain.before)
               << ",\"after_selection\":" << evaluation_json(pretrain.after) << ",\"held_out_test\":" << evaluation_json(pretrain.held_out)
               << ",\"train_sequences\":" << pretrain.train_sequences << ",\"selection_sequences\":" << pretrain.selection_sequences
               << ",\"held_out_sequences\":" << pretrain.held_out_sequences << "},\"sft\":{\"before_selection\":" << evaluation_json(sft.before)
               << ",\"after_selection\":" << evaluation_json(sft.after) << ",\"frozen_final_test\":" << evaluation_json(sft.held_out)
               << ",\"train_sequences\":" << sft.train_sequences << ",\"selection_sequences\":" << sft.selection_sequences
               << ",\"final_test_sequences\":" << sft.held_out_sequences << ",\"final_test_target_tokens\":" << final_target_tokens
               << "},\"selection_policy\":\"validation slices only; frozen final test scored once after SFT\""
               << ",\"sft_mask_policy\":\"target-span-only-v1\",\"evaluation_scope\":\"answer-target next-token held-out metrics; answer exact-match and F1 are not claimed by this runner\",\"elapsed_seconds\":" << elapsed << "}";
        write_file(config.output_root / "training_report.json", report.str() + "\n");
        std::cout << report.str() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "track1_train error: " << error.what() << '\n';
        return 2;
    }
}
