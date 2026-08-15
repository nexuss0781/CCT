#include "cct/corpus.hpp"
#include "cct/nlp_trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw NlpTrainingError(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U || character >= 0x80U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::vector<std::string> read_prompts(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read prompt file " + path.string());
    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) prompts.push_back(line);
    }
    require(!prompts.empty(), "prompt file contains no non-empty prompts: " + path.string());
    return prompts;
}

std::size_t number(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0U;
        const auto parsed = std::stoull(value, &consumed);
        require(consumed == value.size(), "invalid numeric value for " + name);
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw NlpTrainingError("invalid numeric value for " + name);
    }
}

struct Options {
    std::filesystem::path checkpoint;
    std::filesystem::path tokenizer;
    std::filesystem::path prompts;
    std::filesystem::path output;
    std::size_t max_new_tokens = 64U;
};

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        const auto value = [&]() {
            require(index + 1 < argc, "missing value for " + key);
            return std::string(argv[++index]);
        };
        if (key == "--checkpoint") options.checkpoint = value();
        else if (key == "--tokenizer") options.tokenizer = value();
        else if (key == "--prompts") options.prompts = value();
        else if (key == "--output") options.output = value();
        else if (key == "--max-new-tokens") options.max_new_tokens = number(value(), key);
        else if (key == "--help") {
            std::cout << "cct_curriculum_inspect --checkpoint PATH --tokenizer PATH --prompts PATH --output PATH [--max-new-tokens N]\n";
            std::exit(0);
        } else {
            throw NlpTrainingError("unknown argument " + key);
        }
    }
    require(!options.checkpoint.empty() && !options.tokenizer.empty() && !options.prompts.empty() && !options.output.empty() &&
                options.max_new_tokens > 0U && options.max_new_tokens <= 4096U,
            "inspection configuration is invalid");
    return options;
}

bool valid_utf8(const std::string& value) {
    for (std::size_t index = 0U; index < value.size();) {
        const auto first = static_cast<unsigned char>(value[index]);
        if (first <= 0x7FU) {
            ++index;
            continue;
        }
        std::size_t length = 0U;
        std::uint32_t codepoint = 0U;
        std::uint32_t minimum = 0U;
        if (first >= 0xC2U && first <= 0xDFU) {
            length = 2U;
            codepoint = first & 0x1FU;
            minimum = 0x80U;
        } else if (first >= 0xE0U && first <= 0xEFU) {
            length = 3U;
            codepoint = first & 0x0FU;
            minimum = 0x800U;
        } else if (first >= 0xF0U && first <= 0xF4U) {
            length = 4U;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        } else {
            return false;
        }
        if (index + length > value.size()) return false;
        for (std::size_t offset = 1U; offset < length; ++offset) {
            const auto continuation = static_cast<unsigned char>(value[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) return false;
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) return false;
        index += length;
    }
    return true;
}

std::size_t max_same_token_run(const std::vector<TokenId>& ids) {
    if (ids.empty()) return 0U;
    std::size_t maximum = 1U;
    std::size_t current = 1U;
    for (std::size_t index = 1U; index < ids.size(); ++index) {
        if (ids[index] == ids[index - 1U]) {
            ++current;
            maximum = std::max(maximum, current);
        } else {
            current = 1U;
        }
    }
    return maximum;
}

bool repeated_tail(const std::vector<TokenId>& ids) {
    constexpr std::size_t kPatternLength = 3U;
    if (ids.size() < 2U * kPatternLength) return false;
    const auto start = ids.size() - 2U * kPatternLength;
    for (std::size_t index = 0U; index < kPatternLength; ++index) {
        if (ids[start + index] != ids[start + kPatternLength + index]) return false;
    }
    return true;
}

std::size_t punctuation_count(const std::string& value) {
    constexpr std::string_view punctuation = ".,?!:;\"'()-";
    return static_cast<std::size_t>(std::count_if(value.begin(), value.end(), [&](const char character) {
        return punctuation.find(character) != std::string_view::npos;
    }));
}

std::size_t count_spaces(const std::string& value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), ' '));
}

std::size_t best_slot(const std::vector<double>& logits) {
    require(!logits.empty(), "model returned empty logits");
    std::size_t best = 0U;
    require(std::isfinite(logits[0]), "model returned a non-finite logit");
    for (std::size_t index = 1U; index < logits.size(); ++index) {
        require(std::isfinite(logits[index]), "model returned a non-finite logit");
        if (logits[index] > logits[best]) best = index;
    }
    return best;
}

bool contains_token(const std::vector<VocabularyEntry>& vocabulary, const TokenId token) {
    return std::any_of(vocabulary.begin(), vocabulary.end(), [&](const VocabularyEntry& entry) { return entry.id == token; });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto tokenizer_content = read_file(options.tokenizer);
        const auto tokenizer_hash = GovernedCorpus::content_sha256(tokenizer_content);
        const auto tokenizer = Tokenizer::from_snapshot(tokenizer_content, tokenizer_hash);
        const auto trainer = NlpTrainer::load_checkpoint(options.checkpoint.string(), tokenizer_hash);
        const auto prompts = read_prompts(options.prompts);
        std::filesystem::create_directories(options.output.parent_path());
        std::ofstream output(options.output, std::ios::trunc);
        require(static_cast<bool>(output), "cannot write inspection output " + options.output.string());

        output << std::setprecision(12);
        std::size_t valid_outputs = 0U;
        for (std::size_t prompt_index = 0U; prompt_index < prompts.size(); ++prompt_index) {
            const auto encoded = tokenizer.encode(prompts[prompt_index], "inspection-prompt-" + std::to_string(prompt_index), false);
            require(!encoded.tokens.empty(), "inspection prompt tokenized to zero tokens");
            std::vector<TokenId> prompt_ids;
            prompt_ids.reserve(encoded.tokens.size());
            for (const auto& token : encoded.tokens) prompt_ids.push_back(token.id);
            std::vector<TokenId> context = prompt_ids;
            const bool prompt_context_truncated = context.size() > trainer.model().config().context_length;
            if (prompt_context_truncated) {
                context.erase(context.begin(), context.end() - static_cast<std::ptrdiff_t>(trainer.model().config().context_length));
            }
            std::vector<TokenId> generated;
            generated.reserve(options.max_new_tokens);
            bool token_valid = true;
            bool emitted_eos = false;
            for (std::size_t step = 0U; step < options.max_new_tokens; ++step) {
                const auto logits = trainer.model().next_logits(context);
                const auto slot = best_slot(logits);
                const auto token = trainer.model().token_id_from_logit_slot(slot);
                token_valid = token_valid && contains_token(tokenizer.vocabulary(), token);
                if (!token_valid) break;
                if (token == Tokenizer::kEosId) {
                    emitted_eos = true;
                    break;
                }
                generated.push_back(token);
                context.push_back(token);
                if (context.size() > trainer.model().config().context_length) context.erase(context.begin());
            }
            const auto continuation = tokenizer.decode(generated, true);
            const auto full_ids = [&]() {
                auto ids = prompt_ids;
                ids.insert(ids.end(), generated.begin(), generated.end());
                return ids;
            }();
            const auto full_text = tokenizer.decode(full_ids, true);
            const auto utf8 = valid_utf8(continuation);
            const auto finite_text = utf8 && token_valid;
            if (finite_text) ++valid_outputs;
            output << "{\"prompt_index\":" << prompt_index << ",\"prompt\":\"" << json_escape(prompts[prompt_index])
                   << "\",\"continuation\":\"" << json_escape(continuation) << "\",\"full_text\":\"" << json_escape(full_text)
                   << "\",\"prompt_context_truncated\":" << (prompt_context_truncated ? "true" : "false")
                   << ",\"context_tokens\":" << context.size() << ",\"generated_tokens\":" << generated.size()
                   << ",\"token_valid\":" << (token_valid ? "true" : "false") << ",\"valid_utf8\":" << (utf8 ? "true" : "false")
                   << ",\"emitted_eos\":" << (emitted_eos ? "true" : "false") << ",\"max_same_token_run\":"
                   << max_same_token_run(generated) << ",\"repeated_tail\":" << (repeated_tail(generated) ? "true" : "false")
                   << ",\"continuation_spaces\":" << count_spaces(continuation)
                   << ",\"continuation_punctuation\":" << punctuation_count(continuation) << "}\n";
        }
        output.close();
        require(static_cast<bool>(output), "cannot finish inspection output");
        std::cout << "{\"status\":\"PASS\",\"checkpoint_hash\":\"" << trainer.checkpoint_info().checkpoint_hash
                  << "\",\"prompt_count\":" << prompts.size() << ",\"valid_outputs\":" << valid_outputs << ",\"output\":\""
                  << json_escape(options.output.string()) << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"FAIL\",\"error\":\"" << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
