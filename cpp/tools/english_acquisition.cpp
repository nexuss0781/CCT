#include "cct/corpus.hpp"
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
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
using cct::NlpSequence;
using cct::NlpTrainer;
using cct::TokenId;
using cct::Tokenizer;

constexpr std::size_t kMaximumTextBytes = 1U << 20U;
constexpr std::size_t kMaximumBlimpFileBytes = 4U << 20U;
constexpr std::size_t kMaximumGeneratedTokens = 128U;

struct Config {
    std::filesystem::path input_root = "artifacts/track1/real-release";
    std::filesystem::path output_root = "artifacts/english/acquisition";
    std::filesystem::path tokenizer_snapshot = "data/stage-10/tokenizer_snapshot.bin";
    std::filesystem::path blimp_data = "artifacts/english/raw/blimp-master/data";
    std::filesystem::path blimp_archive = "artifacts/english/raw/BLiMP.zip";
    std::filesystem::path cola_train = "artifacts/english/raw/cola/CoLA/train.tsv";
    std::filesystem::path cola_validation = "artifacts/english/raw/cola/CoLA/dev.tsv";
    std::filesystem::path evaluate_checkpoint;
    std::size_t pretrain_steps = 200U;
    std::size_t grammar_steps = 1000U;
    double learning_rate = 0.001;
    double grammar_learning_rate = 0.0005;
    std::size_t context_length = 32U;
    std::size_t embedding_dim = 4U;
    std::size_t hidden_dim = 4U;
    std::size_t validation_records = 256U;
    std::size_t test_records = 0U;
    std::size_t blimp_pairs_per_file = 100U;
    std::size_t generation_tokens = 32U;
    std::uint64_t seed = 1701U;
};

struct TextRecord {
    std::string id;
    std::string text;
};

struct LabeledTextRecord {
    TextRecord record;
    bool acceptable = false;
};

struct SentenceScore {
    double loss = 0.0;
    std::size_t tokens = 0U;
};

struct BlimpBucket {
    std::size_t pairs = 0U;
    std::size_t preferred = 0U;
};

struct BlimpResult {
    std::size_t files = 0U;
    std::size_t pairs = 0U;
    std::size_t preferred = 0U;
    std::size_t malformed = 0U;
    std::map<std::string, BlimpBucket> fields;
    std::map<std::string, BlimpBucket> files_by_name;
};

struct GenerationResult {
    std::size_t prompts = 0U;
    std::size_t nonempty = 0U;
    std::size_t valid_utf8 = 0U;
    std::size_t repetitive = 0U;
    std::size_t stopped = 0U;
    std::size_t invalid_token_ids = 0U;
    std::size_t total_tokens = 0U;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path, const std::size_t maximum_bytes = 16U * 1024U * 1024U) {
    const auto size = std::filesystem::file_size(path);
    require(size <= maximum_bytes, "file exceeds configured byte budget: " + path.string());
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read file: " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    require(static_cast<bool>(input) || input.eof(), "cannot finish reading file: " + path.string());
    return output.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "cannot write artifact: " + path.string());
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        require(static_cast<bool>(output), "cannot finish artifact: " + path.string());
    }
    std::filesystem::rename(temporary, path);
}

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

std::vector<TextRecord> read_text_records(const std::filesystem::path& path, const std::size_t maximum_records) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read English text split: " + path.string());
    std::vector<TextRecord> records;
    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        const auto clean = trim(line);
        if (clean.size() < 2U) {
            ++line_number;
            continue;
        }
        require(clean.size() <= kMaximumTextBytes, "English text record exceeds byte budget");
        records.push_back({path.filename().string() + ":" + std::to_string(line_number), clean});
        ++line_number;
        if (maximum_records != 0U && records.size() >= maximum_records) break;
    }
    require(!records.empty(), "English text split is empty: " + path.string());
    return records;
}

std::vector<LabeledTextRecord> read_cola_labeled_records(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read CoLA file: " + path.string());
    std::vector<LabeledTextRecord> records;
    std::string line;
    std::size_t line_number = 0U;
    while (std::getline(input, line)) {
        const auto first = line.find('\t');
        const auto second = first == std::string::npos ? std::string::npos : line.find('\t', first + 1U);
        const auto third = second == std::string::npos ? std::string::npos : line.find('\t', second + 1U);
        require(first != std::string::npos && second != std::string::npos && third != std::string::npos, "CoLA row has fewer than four TSV fields");
        const auto label = line.substr(first + 1U, second - first - 1U);
        const auto sentence = trim(line.substr(third + 1U));
        require(label == "0" || label == "1", "CoLA label is outside {0,1}");
        require(!sentence.empty() && sentence.size() <= kMaximumTextBytes, "CoLA sentence is empty or oversized");
        records.push_back({{path.filename().string() + ":" + std::to_string(line_number), sentence}, label == "1"});
        ++line_number;
    }
    require(!records.empty(), "CoLA contains no labeled sentences: " + path.string());
    return records;
}

std::vector<EncodedDocument> encode_text_records(const Tokenizer& tokenizer, const std::vector<TextRecord>& records, const bool training) {
    std::vector<EncodedDocument> documents;
    documents.reserve(records.size());
    for (const auto& record : records) {
        auto document = tokenizer.encode(record.text, record.id, false);
        require(!document.tokens.empty(), "English record tokenized to empty content: " + record.id);
        document.training_allowed = training;
        document.evaluation_allowed = !training;
        documents.push_back(std::move(document));
    }
    return documents;
}

NlpDataset make_dataset(const Tokenizer& tokenizer, const std::vector<TextRecord>& training_records, const std::vector<TextRecord>& validation_records,
                       const std::string& tokenizer_hash, const std::size_t context_length) {
    return NlpDataset::build(encode_text_records(tokenizer, training_records, true), encode_text_records(tokenizer, validation_records, false),
                             tokenizer_hash, context_length);
}

NlpSequence score_sequence(const std::vector<TokenId>& ids, const std::size_t start, const std::size_t context_length) {
    const auto end = std::min(ids.size(), start + context_length);
    require(end > start + 1U, "English scoring window has fewer than two tokens");
    NlpSequence sequence;
    sequence.sequence_id = "english-score:" + std::to_string(start);
    sequence.record_id = sequence.sequence_id;
    sequence.input_ids.assign(ids.begin() + static_cast<std::ptrdiff_t>(start), ids.begin() + static_cast<std::ptrdiff_t>(end));
    sequence.target_ids.assign(sequence.input_ids.size(), Tokenizer::kPadId);
    sequence.loss_mask.assign(sequence.input_ids.size(), 0U);
    for (std::size_t index = 0U; index + 1U < sequence.input_ids.size(); ++index) {
        sequence.target_ids[index] = sequence.input_ids[index + 1U];
        sequence.loss_mask[index] = 1U;
    }
    return sequence;
}

NlpSequence causal_sentence_sequence(const Tokenizer& tokenizer, const TextRecord& record, const std::size_t context_length) {
    const auto encoded = tokenizer.encode(record.text, record.id, false);
    require(encoded.tokens.size() >= 2U && encoded.tokens.size() <= context_length, "CoLA sentence is outside the causal context budget: " + record.id);
    std::vector<TokenId> ids;
    ids.reserve(encoded.tokens.size());
    for (const auto& token : encoded.tokens) ids.push_back(token.id);
    return score_sequence(ids, 0U, context_length);
}

std::vector<cct::NlpPreferencePair> cola_preference_pairs(const Tokenizer& tokenizer, const std::filesystem::path& path,
                                                           const std::size_t context_length) {
    const auto labeled = read_cola_labeled_records(path);
    std::vector<TextRecord> acceptable;
    std::vector<TextRecord> unacceptable;
    for (const auto& item : labeled) {
        if (item.acceptable) acceptable.push_back(item.record);
        else unacceptable.push_back(item.record);
    }
    require(!acceptable.empty() && !unacceptable.empty(), "CoLA requires both acceptable and unacceptable sentences");
    std::vector<cct::NlpPreferencePair> pairs;
    const auto count = std::max(acceptable.size(), unacceptable.size());
    pairs.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        pairs.push_back({causal_sentence_sequence(tokenizer, acceptable[index % acceptable.size()], context_length),
                         causal_sentence_sequence(tokenizer, unacceptable[index % unacceptable.size()], context_length)});
    }
    return pairs;
}

std::size_t preference_correct(const cct::NextTokenModel& model, const std::vector<cct::NlpPreferencePair>& pairs) {
    std::size_t correct = 0U;
    for (const auto& pair : pairs) {
        if (model.loss_only(pair.preferred) < model.loss_only(pair.rejected)) ++correct;
    }
    return correct;
}

SentenceScore score_sentence(const Tokenizer& tokenizer, const cct::NextTokenModel& model, const std::string& sentence) {
    const auto encoded = tokenizer.encode(sentence, "english-evaluation", false);
    std::vector<TokenId> ids;
    ids.reserve(encoded.tokens.size());
    for (const auto& token : encoded.tokens) ids.push_back(token.id);
    require(ids.size() >= 2U, "English evaluation sentence tokenized to fewer than two tokens");
    double loss_sum = 0.0;
    std::size_t token_count = 0U;
    for (std::size_t start = 0U; start + 1U < ids.size(); start += model.config().context_length) {
        const auto sequence = score_sequence(ids, start, model.config().context_length);
        const auto loss = model.loss_only(sequence);
        const auto active = sequence.input_ids.size() - 1U;
        require(std::isfinite(loss) && active > 0U, "English sentence score is non-finite");
        loss_sum += loss * static_cast<double>(active);
        token_count += active;
        if (start + model.config().context_length >= ids.size()) break;
    }
    return {loss_sum / static_cast<double>(token_count), token_count};
}

unsigned int hex_digit(const char value) {
    if (value >= '0' && value <= '9') return static_cast<unsigned int>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<unsigned int>(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return static_cast<unsigned int>(value - 'A' + 10);
    throw std::runtime_error("invalid JSON Unicode escape");
}

void append_codepoint(std::string& output, const unsigned int codepoint) {
    require(codepoint <= 0x10FFFFU && !(codepoint >= 0xD800U && codepoint <= 0xDFFFU), "invalid JSON codepoint");
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

std::string json_string_field(const std::string& line, const std::string& key) {
    const auto marker = "\"" + key + "\":";
    const auto marker_position = line.find(marker);
    require(marker_position != std::string::npos, "BLiMP field missing: " + key);
    std::size_t position = marker_position + marker.size();
    while (position < line.size() && (line[position] == ' ' || line[position] == '\t')) ++position;
    require(position < line.size() && line[position] == '"', "BLiMP field is not a JSON string: " + key);
    ++position;
    std::string value;
    while (position < line.size()) {
        const auto character = line[position++];
        if (character == '"') return value;
        if (character != '\\') {
            value.push_back(character);
            continue;
        }
        require(position < line.size(), "BLiMP string escape is truncated");
        const auto escaped = line[position++];
        if (escaped == '"' || escaped == '\\' || escaped == '/') value.push_back(escaped);
        else if (escaped == 'n') value.push_back('\n');
        else if (escaped == 'r') value.push_back('\r');
        else if (escaped == 't') value.push_back('\t');
        else if (escaped == 'b') value.push_back('\b');
        else if (escaped == 'f') value.push_back('\f');
        else if (escaped == 'u') {
            require(position + 4U <= line.size(), "BLiMP Unicode escape is truncated");
            unsigned int high = 0U;
            for (std::size_t index = 0U; index < 4U; ++index) high = (high << 4U) | hex_digit(line[position++]);
            if (high >= 0xD800U && high <= 0xDBFFU) {
                require(position + 6U <= line.size() && line[position] == '\\' && line[position + 1U] == 'u', "BLiMP low surrogate missing");
                position += 2U;
                unsigned int low = 0U;
                for (std::size_t index = 0U; index < 4U; ++index) low = (low << 4U) | hex_digit(line[position++]);
                require(low >= 0xDC00U && low <= 0xDFFFU, "BLiMP low surrogate invalid");
                append_codepoint(value, 0x10000U + ((high - 0xD800U) << 10U) + low - 0xDC00U);
            } else append_codepoint(value, high);
        } else throw std::runtime_error("unsupported BLiMP JSON escape");
    }
    throw std::runtime_error("unterminated BLiMP JSON string: " + key);
}

std::string field_name(const std::string& field) {
    if (field == "morphology" || field == "syntax" || field == "syntax-semantics" || field == "semantics") return field;
    return "unknown";
}

BlimpResult score_blimp(const Config& config, const Tokenizer& tokenizer, const cct::NextTokenModel& model) {
    require(std::filesystem::exists(config.blimp_data), "BLiMP data directory is missing");
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(config.blimp_data)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jsonl") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    require(files.size() == 67U, "BLiMP file count is not the pinned 67-file benchmark");
    BlimpResult result;
    result.files = files.size();
    for (const auto& path : files) {
        require(std::filesystem::file_size(path) <= kMaximumBlimpFileBytes, "BLiMP file exceeds byte budget");
        std::ifstream input(path);
        require(static_cast<bool>(input), "cannot open BLiMP file: " + path.string());
        std::string line;
        std::size_t file_pairs = 0U;
        auto& file_bucket = result.files_by_name[path.filename().string()];
        while (std::getline(input, line)) {
            if (trim(line).empty()) continue;
            if (config.blimp_pairs_per_file != 0U && file_pairs >= config.blimp_pairs_per_file) break;
            try {
                const auto good = json_string_field(line, "sentence_good");
                const auto bad = json_string_field(line, "sentence_bad");
                const auto field = field_name(json_string_field(line, "field"));
                const auto good_score = score_sentence(tokenizer, model, good);
                const auto bad_score = score_sentence(tokenizer, model, bad);
                const auto preferred = good_score.loss < bad_score.loss;
                ++result.pairs;
                ++file_pairs;
                if (preferred) ++result.preferred;
                ++file_bucket.pairs;
                if (preferred) ++file_bucket.preferred;
                ++result.fields[field].pairs;
                if (preferred) ++result.fields[field].preferred;
            } catch (const std::exception&) {
                ++result.malformed;
            }
        }
        require(file_pairs > 0U, "BLiMP file produced no scored pairs: " + path.string());
    }
    require(result.pairs > 0U && result.malformed == 0U, "BLiMP scoring encountered malformed evaluation rows");
    return result;
}

bool valid_utf8(const std::string& value) {
    for (std::size_t index = 0U; index < value.size();) {
        const auto byte = static_cast<unsigned char>(value[index]);
        std::size_t width = 0U;
        if (byte <= 0x7FU) width = 1U;
        else if (byte >= 0xC2U && byte <= 0xDFU) width = 2U;
        else if (byte >= 0xE0U && byte <= 0xEFU) width = 3U;
        else if (byte >= 0xF0U && byte <= 0xF4U) width = 4U;
        else return false;
        if (index + width > value.size()) return false;
        for (std::size_t offset = 1U; offset < width; ++offset) {
            if ((static_cast<unsigned char>(value[index + offset]) & 0xC0U) != 0x80U) return false;
        }
        index += width;
    }
    return true;
}

std::size_t constrained_generation_slot(const std::vector<double>& logits, const cct::NextTokenModel& model,
                                        const std::vector<TokenId>& generated) {
    std::vector<std::size_t> candidates(logits.size());
    std::iota(candidates.begin(), candidates.end(), 0U);
    constexpr std::size_t kCandidateLimit = 64U;
    const auto candidate_count = std::min(kCandidateLimit, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count), candidates.end(),
                      [&logits](const std::size_t left, const std::size_t right) { return logits[left] > logits[right]; });
    for (std::size_t candidate_index = 0U; candidate_index < candidate_count; ++candidate_index) {
        const auto slot = candidates[candidate_index];
        const auto candidate = model.token_id_from_logit_slot(slot);
        if (candidate == Tokenizer::kEosId) return slot;
        if (candidate < Tokenizer::kByteFirstId) continue;
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
        if (!forbidden) return slot;
    }
    return candidates.front();
}

GenerationResult generation_diagnostics(const Config& config, const Tokenizer& tokenizer, const cct::NextTokenModel& model) {
    const std::vector<std::string> prompts{"The scientist observed", "In the morning, the child", "A good explanation", "The city decided", "Although the weather"};
    GenerationResult result;
    result.prompts = prompts.size();
    for (const auto& prompt : prompts) {
        auto encoded = tokenizer.encode(prompt, "generation-prompt", false);
        std::vector<TokenId> context;
        for (const auto& token : encoded.tokens) context.push_back(token.id);
        std::vector<TokenId> generated;
        generated.reserve(config.generation_tokens);
        for (std::size_t step = 0U; step < config.generation_tokens; ++step) {
            const auto context_start = context.size() > model.config().context_length ? context.size() - model.config().context_length : 0U;
            std::vector<TokenId> window(context.begin() + static_cast<std::ptrdiff_t>(context_start), context.end());
            const auto logits = model.next_logits(window);
            const auto selected_slot = constrained_generation_slot(logits, model, generated);
            const auto selected = model.token_id_from_logit_slot(selected_slot);
            if (selected != Tokenizer::kEosId && selected < Tokenizer::kByteFirstId) ++result.invalid_token_ids;
            generated.push_back(selected);
            context.push_back(selected);
            if (selected == Tokenizer::kEosId) {
                ++result.stopped;
                break;
            }
        }
        const auto text = tokenizer.decode(generated, false);
        if (!text.empty()) ++result.nonempty;
        if (valid_utf8(text)) ++result.valid_utf8;
        if (generated.size() >= 8U) {
            const auto tail_start = generated.size() - 8U;
            const auto tail = std::vector<TokenId>(generated.begin() + static_cast<std::ptrdiff_t>(tail_start), generated.end());
            bool repeated = false;
            for (std::size_t left = 0U; left + 4U <= tail.size() && !repeated; ++left) {
                for (std::size_t right = left + 1U; right + 4U <= tail.size(); ++right) {
                    if (std::equal(tail.begin() + static_cast<std::ptrdiff_t>(left), tail.begin() + static_cast<std::ptrdiff_t>(left + 4U),
                                   tail.begin() + static_cast<std::ptrdiff_t>(right))) {
                        repeated = true;
                        break;
                    }
                }
            }
            for (std::size_t index = 3U; index < generated.size() && !repeated; ++index) {
                repeated = generated[index] == generated[index - 2U] && generated[index - 1U] == generated[index - 3U];
            }
            if (!repeated) {
                std::unordered_set<TokenId> unique_tokens(generated.begin(), generated.end());
                repeated = unique_tokens.size() * 4U <= generated.size();
            }
            if (!repeated) repeated = std::all_of(text.begin(), text.end(), [](const char character) {
                return character == ' ' || character == '\t' || character == '\n' || character == '\r';
            });
            if (repeated) ++result.repetitive;
        }
        result.total_tokens += generated.size();
    }
    return result;
}

std::string evaluation_json(const NlpEvaluation& result) {
    std::ostringstream output;
    output << std::setprecision(10) << "{\"cross_entropy\":" << result.cross_entropy << ",\"perplexity\":" << result.perplexity
           << ",\"token_accuracy\":" << result.token_accuracy << ",\"token_count\":" << result.token_count
           << ",\"tokens_per_second\":" << result.tokens_per_second << ",\"finite\":" << (result.finite ? "true" : "false") << "}";
    return output.str();
}

std::string blimp_json(const BlimpResult& result) {
    std::ostringstream output;
    output << std::setprecision(10) << "{\"files\":" << result.files << ",\"pairs\":" << result.pairs << ",\"preferred\":" << result.preferred
           << ",\"accuracy\":" << static_cast<double>(result.preferred) / static_cast<double>(result.pairs) << ",\"malformed\":" << result.malformed << ",\"fields\":{";
    bool first = true;
    for (const auto& [name, bucket] : result.fields) {
        if (!first) output << ',';
        first = false;
        output << "\"" << name << "\":{\"pairs\":" << bucket.pairs << ",\"preferred\":" << bucket.preferred
               << ",\"accuracy\":" << static_cast<double>(bucket.preferred) / static_cast<double>(bucket.pairs) << "}";
    }
    output << "}}";
    return output.str();
}

std::string generation_json(const GenerationResult& result) {
    std::ostringstream output;
    output << "{\"prompts\":" << result.prompts << ",\"nonempty\":" << result.nonempty << ",\"valid_utf8\":" << result.valid_utf8
           << ",\"repetitive\":" << result.repetitive << ",\"stopped\":" << result.stopped
           << ",\"invalid_token_ids\":" << result.invalid_token_ids << ",\"generated_tokens\":" << result.total_tokens << "}";
    return output.str();
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
        else if (key == "--blimp-data") config.blimp_data = value();
        else if (key == "--blimp-archive") config.blimp_archive = value();
        else if (key == "--cola-train") config.cola_train = value();
        else if (key == "--cola-validation") config.cola_validation = value();
        else if (key == "--evaluate-checkpoint") config.evaluate_checkpoint = value();
        else if (key == "--pretrain-steps") config.pretrain_steps = std::stoull(value());
        else if (key == "--grammar-steps") config.grammar_steps = std::stoull(value());
        else if (key == "--learning-rate") config.learning_rate = std::stod(value());
        else if (key == "--grammar-learning-rate") config.grammar_learning_rate = std::stod(value());
        else if (key == "--context") config.context_length = std::stoull(value());
        else if (key == "--embedding") config.embedding_dim = std::stoull(value());
        else if (key == "--hidden") config.hidden_dim = std::stoull(value());
        else if (key == "--validation-records") config.validation_records = std::stoull(value());
        else if (key == "--test-records") config.test_records = std::stoull(value());
        else if (key == "--blimp-pairs-per-file") config.blimp_pairs_per_file = std::stoull(value());
        else if (key == "--generation-tokens") config.generation_tokens = std::stoull(value());
        else if (key == "--seed") config.seed = std::stoull(value());
        else if (key == "--help") {
            std::cout << "english_acquisition --input PATH --output PATH --tokenizer PATH --blimp-data PATH --blimp-archive PATH --cola-train PATH --cola-validation PATH --evaluate-checkpoint PATH "
                         "--pretrain-steps N --grammar-steps N --learning-rate X --grammar-learning-rate X --context N --embedding N --hidden N --validation-records N --test-records N --blimp-pairs-per-file N --generation-tokens N --seed N\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument " + key);
    }
    require(config.pretrain_steps > 0U && config.grammar_steps > 0U && config.context_length >= 2U && config.embedding_dim > 0U && config.hidden_dim > 0U &&
                std::isfinite(config.learning_rate) && config.learning_rate > 0.0 && std::isfinite(config.grammar_learning_rate) && config.grammar_learning_rate > 0.0 &&
                config.generation_tokens <= kMaximumGeneratedTokens && config.validation_records > 0U,
            "invalid English acquisition configuration");
    return config;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto config = parse_arguments(argc, argv);
        const auto started = std::chrono::steady_clock::now();
        const auto tokenizer = Tokenizer::from_snapshot(read_file(config.tokenizer_snapshot));
        const auto tokenizer_hash = tokenizer.snapshot_hash();
        const auto vocabulary_size = static_cast<std::size_t>(tokenizer.vocabulary().back().id) + 1U;
        const auto data_root = config.input_root / "data";
        const auto manifest_path = config.input_root / "manifest.json";
        require(std::filesystem::exists(manifest_path), "English acquisition manifest is missing");
        const auto manifest_text = read_file(manifest_path);
        const auto manifest_hash = GovernedCorpus::content_sha256(manifest_text);
        const auto blimp_archive_hash = GovernedCorpus::content_sha256(read_file(config.blimp_archive, 64U * 1024U * 1024U));
        const auto training_records = read_text_records(data_root / "pretrain_train.txt", 0U);
        const auto validation_records = read_text_records(data_root / "pretrain_validation.txt", config.validation_records);
        const auto test_records = read_text_records(data_root / "pretrain_test.txt", config.test_records);
        const auto dataset = make_dataset(tokenizer, training_records, validation_records, tokenizer_hash, config.context_length);
        const auto cola_train_labeled = read_cola_labeled_records(config.cola_train);
        const auto cola_validation_labeled = read_cola_labeled_records(config.cola_validation);
        std::vector<TextRecord> cola_train_records;
        std::vector<TextRecord> cola_validation_records;
        for (const auto& item : cola_train_labeled) {
            if (item.acceptable) cola_train_records.push_back(item.record);
        }
        for (const auto& item : cola_validation_labeled) {
            if (item.acceptable) cola_validation_records.push_back(item.record);
        }
        const auto cola_dataset = make_dataset(tokenizer, cola_train_records, cola_validation_records, tokenizer_hash, config.context_length);
        const auto cola_train_pairs = cola_preference_pairs(tokenizer, config.cola_train, config.context_length);
        const auto cola_validation_pairs = cola_preference_pairs(tokenizer, config.cola_validation, config.context_length);
        const auto cola_archive = config.cola_train.parent_path().parent_path() / "CoLA.zip";
        const auto cola_archive_hash = std::filesystem::exists(cola_archive) ? GovernedCorpus::content_sha256(read_file(cola_archive, 8U * 1024U * 1024U)) : std::string{};
        auto test_documents = encode_text_records(tokenizer, test_records, false);
        test_documents.erase(std::remove_if(test_documents.begin(), test_documents.end(), [](const EncodedDocument& document) {
                                 return document.tokens.size() < 2U;
                             }),
                             test_documents.end());
        require(!test_documents.empty(), "frozen English test has no usable causal windows");
        auto test_anchor = test_documents.front();
        test_anchor.record_id = "english-test-anchor";
        test_anchor.training_allowed = true;
        test_anchor.evaluation_allowed = false;
        const auto test_dataset = NlpDataset::build({test_anchor}, test_documents, tokenizer_hash, config.context_length);
        NlpModelConfig model_config{NlpModelKind::Track1CctRecurrence, vocabulary_size, config.embedding_dim, config.hidden_dim, config.context_length, config.seed};
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = config.learning_rate;
        optimizer.weight_decay = 1e-5;
        optimizer.clip_norm = 1.0;
        optimizer.warmup_steps = std::min<std::size_t>(10U, config.pretrain_steps);
        optimizer.total_steps = config.pretrain_steps;
        optimizer.validation_interval_steps = config.pretrain_steps;
        NlpTrainer control(model_config, optimizer, tokenizer_hash, dataset.dataset_hash);
        const auto control_validation = control.evaluate(dataset.validation);
        const auto control_test = control.evaluate(test_dataset.validation);
        NlpTrainer trainer(model_config, optimizer, tokenizer_hash, dataset.dataset_hash);
        const auto before_validation = trainer.evaluate(dataset.validation);
        std::filesystem::create_directories(config.output_root);
        NlpEvaluation after_pretrain_validation;
        NlpEvaluation pretrain_test;
        const bool evaluation_only = !config.evaluate_checkpoint.empty();
        if (evaluation_only) {
            require(std::filesystem::exists(config.evaluate_checkpoint), "English evaluation checkpoint is missing");
            after_pretrain_validation = control_validation;
            pretrain_test = control_test;
        } else {
            static_cast<void>(trainer.train_steps(dataset, config.pretrain_steps));
            trainer.save_checkpoint((config.output_root / "pretraining_checkpoint.bin").string());
            after_pretrain_validation = trainer.evaluate(dataset.validation);
            pretrain_test = trainer.evaluate(test_dataset.validation);
        }
        NlpOptimizerConfig grammar_optimizer = optimizer;
        grammar_optimizer.learning_rate = config.grammar_learning_rate;
        grammar_optimizer.total_steps = config.grammar_steps;
        grammar_optimizer.warmup_steps = std::min<std::size_t>(10U, config.grammar_steps);
        grammar_optimizer.validation_interval_steps = config.grammar_steps;
        NlpTrainer grammar_trainer(model_config, grammar_optimizer, tokenizer_hash, cola_dataset.dataset_hash);
        std::vector<std::string> preference_checkpoint_names;
        if (evaluation_only) {
            grammar_trainer = NlpTrainer::load_checkpoint(config.evaluate_checkpoint.string(), tokenizer_hash, cola_dataset.dataset_hash);
        } else {
            grammar_trainer.model().set_parameter_vector(trainer.model().parameter_vector());
            const std::vector<std::size_t> requested_milestones{1000U, 5000U, config.grammar_steps};
            std::size_t completed_steps = 0U;
            for (const auto milestone : requested_milestones) {
                const auto target_step = std::min(milestone, config.grammar_steps);
                if (target_step <= completed_steps) continue;
                static_cast<void>(grammar_trainer.train_preference_steps(cola_train_pairs, target_step - completed_steps, 0.05));
                completed_steps = target_step;
                const auto checkpoint_name = "preference_step_" + std::to_string(completed_steps) + ".bin";
                grammar_trainer.save_checkpoint((config.output_root / checkpoint_name).string());
                preference_checkpoint_names.push_back(checkpoint_name);
            }
        }
        const auto after_validation = grammar_trainer.evaluate(dataset.validation);
        const auto trained_test = grammar_trainer.evaluate(test_dataset.validation);
        const auto control_blimp = score_blimp(config, tokenizer, control.model());
        const auto pretrain_blimp = score_blimp(config, tokenizer, trainer.model());
        const auto trained_blimp = score_blimp(config, tokenizer, grammar_trainer.model());
        const auto control_cola_correct = preference_correct(control.model(), cola_validation_pairs);
        const auto pretrain_cola_correct = preference_correct(trainer.model(), cola_validation_pairs);
        const auto trained_cola_correct = preference_correct(grammar_trainer.model(), cola_validation_pairs);
        const auto generation = generation_diagnostics(config, tokenizer, grammar_trainer.model());
        const bool finite_metrics = control_validation.finite && control_test.finite && before_validation.finite && after_pretrain_validation.finite &&
                                    pretrain_test.finite && after_validation.finite && trained_test.finite;
        const bool language_improves = after_validation.cross_entropy < control_validation.cross_entropy && trained_test.cross_entropy < control_test.cross_entropy;
        const bool grammar_improves = trained_blimp.pairs == control_blimp.pairs && trained_blimp.preferred > control_blimp.preferred &&
                                      trained_blimp.preferred * 2U >= trained_blimp.pairs && trained_cola_correct > control_cola_correct;
        const bool generation_valid = generation.nonempty == generation.prompts && generation.valid_utf8 == generation.prompts &&
                                      generation.repetitive == 0U && generation.invalid_token_ids == 0U;
        const bool passed = finite_metrics && language_improves && grammar_improves && generation_valid;
        std::filesystem::create_directories(config.output_root);
        const auto checkpoint_path = config.output_root / "english_checkpoint.bin";
        grammar_trainer.save_checkpoint(checkpoint_path.string());
        const auto checkpoint_hash = cct::nlp_checkpoint_hash(read_file(checkpoint_path));
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::size_t pretraining_target_tokens_processed = 0U;
        for (const auto& point : trainer.history()) pretraining_target_tokens_processed += point.token_count;
        std::size_t preference_target_tokens_per_update = 0U;
        if (!cola_train_pairs.empty()) {
            for (const auto& token : cola_train_pairs.front().preferred.loss_mask) preference_target_tokens_per_update += token != 0U ? 1U : 0U;
            for (const auto& token : cola_train_pairs.front().rejected.loss_mask) preference_target_tokens_per_update += token != 0U ? 1U : 0U;
        }
        const auto status = passed ? "PASS" : "FAIL";
        std::ostringstream report;
        report << std::setprecision(10) << "{\"status\":\"" << status << "\",\"milestone\":\"l1-english-acquisition\",\"backend\":\"native-c++20-track1-cct-recurrence\",\"manifest_hash\":\""
               << manifest_hash << "\",\"tokenizer_hash\":\"" << tokenizer_hash << "\",\"blimp_archive_hash\":\"" << blimp_archive_hash
               << "\",\"cola_archive_hash\":\"" << cola_archive_hash << "\",\"dataset_hash\":\"" << dataset.dataset_hash
               << "\",\"cola_dataset_hash\":\"" << cola_dataset.dataset_hash << "\",\"training_steps\":" << config.pretrain_steps << ",\"grammar_steps\":" << config.grammar_steps
               << ",\"learning_rate\":" << config.learning_rate << ",\"grammar_learning_rate\":" << config.grammar_learning_rate << ",\"seed\":" << config.seed
               << ",\"data_contract\":{\"pretrain_bytes\":" << std::filesystem::file_size(data_root / "pretrain_train.txt")
               << ",\"pretrain_model_tokens\":" << dataset.train_tokens << ",\"validation_model_tokens\":" << dataset.validation_tokens
               << ",\"test_model_tokens\":" << test_dataset.validation_tokens << ",\"train_sequences\":" << dataset.train.size()
               << ",\"validation_sequences\":" << dataset.validation.size() << ",\"pretraining_target_tokens_processed\":" << pretraining_target_tokens_processed
               << ",\"preference_target_tokens_per_update\":" << preference_target_tokens_per_update << "}"
               << ",\"model\":{\"vocabulary_size\":" << vocabulary_size << ",\"embedding_dim\":" << config.embedding_dim << ",\"hidden_dim\":" << config.hidden_dim
               << ",\"context_length\":" << config.context_length << ",\"parameter_count\":" << trainer.model().parameter_count() << "}"
               << ",\"control_validation\":" << evaluation_json(control_validation) << ",\"before_validation\":" << evaluation_json(before_validation)
               << ",\"after_validation\":" << evaluation_json(after_validation) << ",\"control_test\":" << evaluation_json(control_test)
               << ",\"trained_test\":" << evaluation_json(trained_test) << ",\"control_blimp\":" << blimp_json(control_blimp)
               << ",\"pretrain_blimp\":" << blimp_json(pretrain_blimp) << ",\"trained_blimp\":" << blimp_json(trained_blimp)
               << ",\"cola_preference\":{\"control_correct\":" << control_cola_correct << ",\"pretrain_correct\":" << pretrain_cola_correct
               << ",\"adapted_correct\":" << trained_cola_correct << ",\"evaluation_pairs\":" << cola_validation_pairs.size() << "},\"generation\":" << generation_json(generation)
               << ",\"checkpoint\":{\"reference\":\"english_checkpoint.bin\",\"sha256\":\"" << checkpoint_hash << "\",\"bytes\":"
               << std::filesystem::file_size(checkpoint_path) << ",\"pretraining_reference\":\"pretraining_checkpoint.bin\",\"preference_milestones\":[";
        for (std::size_t index = 0U; index < preference_checkpoint_names.size(); ++index) {
            if (index > 0U) report << ',';
            report << '\"' << preference_checkpoint_names[index] << '\"';
        }
        report << "]},\"gate\":{\"finite_metrics\":" << (finite_metrics ? "true" : "false")
               << ",\"language_improves\":" << (language_improves ? "true" : "false") << ",\"grammar_improves\":" << (grammar_improves ? "true" : "false")
               << ",\"generation_valid\":" << (generation_valid ? "true" : "false") << "},\"external_actions\":false,\"evaluation_only\":" << (evaluation_only ? "true" : "false") << ",\"elapsed_seconds\":" << elapsed << "}\n";
        write_file(config.output_root / "training_report.json", report.str());
        std::ostringstream human;
        human << "# L1 English Acquisition Report\n\n**Status:** `" << status << "`  \n**Backend:** native C++20 CCT recurrence  \n**Training steps:** " << config.pretrain_steps
               << "  \n**BLiMP files/pairs:** " << trained_blimp.files << "/" << trained_blimp.pairs << "  \n**Checkpoint:** `" << checkpoint_hash << "`\n\n"
               << "The run compares a trained model with a matched no-training control on fixed WikiText validation/test slices and BLiMP minimal pairs. It records generation validity diagnostics and retains external actions disabled. This is an English acquisition experiment with explicit token-level and grammar-probe evidence; it is not a claim of human-speaker equivalence or broad language competence.\n";
        write_file(config.output_root / "report.md", human.str());
        write_file(config.output_root / "control_blimp.json", blimp_json(control_blimp) + "\n");
        write_file(config.output_root / "trained_blimp.json", blimp_json(trained_blimp) + "\n");
        write_file(config.output_root / "generation_report.json", generation_json(generation) + "\n");
        std::cout << report.str();
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "english_acquisition error: " << error.what() << '\n';
        return 2;
    }
}
