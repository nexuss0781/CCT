#include "cct/track1.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw Track1Error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void write_file_atomic(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "cannot write " + temporary);
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        require(static_cast<bool>(output), "cannot finish " + temporary);
    }
    std::filesystem::rename(temporary, path);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character == '\b') output << "\\b";
        else if (character == '\f') output << "\\f";
        else if (character < 0x20U) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                                             << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        else output << static_cast<char>(character);
    }
    return output.str();
}

std::size_t utf8_byte_offset(const std::string& text, const std::size_t codepoint_offset) {
    std::size_t position = 0U;
    for (std::size_t index = 0U; index < codepoint_offset; ++index) {
        require(position < text.size(), "SQuAD source answer offset exceeds UTF-8 context");
        const auto byte = static_cast<unsigned char>(text[position]);
        std::size_t width = 0U;
        if (byte <= 0x7FU) width = 1U;
        else if ((byte & 0xE0U) == 0xC0U) width = 2U;
        else if ((byte & 0xF0U) == 0xE0U) width = 3U;
        else if ((byte & 0xF8U) == 0xF0U) width = 4U;
        else throw Track1Error("invalid UTF-8 context before SQuAD answer offset");
        require(position + width <= text.size(), "truncated UTF-8 context before SQuAD answer offset");
        for (std::size_t continuation = 1U; continuation < width; ++continuation)
            require((static_cast<unsigned char>(text[position + continuation]) & 0xC0U) == 0x80U, "invalid UTF-8 continuation in SQuAD context");
        position += width;
    }
    return position;
}

std::string json_array(const std::vector<std::string>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << json_escape(values[index]) << '"';
    }
    output << ']';
    return output.str();
}

std::size_t skip_space(const std::string& text, std::size_t position) {
    while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position])) != 0) ++position;
    return position;
}

void append_codepoint(std::string& output, const unsigned int codepoint) {
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

unsigned int hex_digit(const char character) {
    if (character >= '0' && character <= '9') return static_cast<unsigned int>(character - '0');
    if (character >= 'a' && character <= 'f') return static_cast<unsigned int>(character - 'a' + 10);
    if (character >= 'A' && character <= 'F') return static_cast<unsigned int>(character - 'A' + 10);
    throw Track1Error("invalid JSON unicode escape");
}

std::string parse_json_string(const std::string& text, std::size_t& position) {
    position = skip_space(text, position);
    require(position < text.size() && text[position] == '"', "expected JSON string");
    ++position;
    std::string value;
    while (position < text.size()) {
        const char character = text[position++];
        if (character == '"') return value;
        if (character != '\\') {
            value.push_back(character);
            continue;
        }
        require(position < text.size(), "truncated JSON escape");
        const char escaped = text[position++];
        if (escaped == '"' || escaped == '\\' || escaped == '/') value.push_back(escaped);
        else if (escaped == 'b') value.push_back('\b');
        else if (escaped == 'f') value.push_back('\f');
        else if (escaped == 'n') value.push_back('\n');
        else if (escaped == 'r') value.push_back('\r');
        else if (escaped == 't') value.push_back('\t');
        else if (escaped == 'u') {
            require(position + 4U <= text.size(), "truncated JSON unicode escape");
            unsigned int codepoint = 0U;
            for (std::size_t index = 0U; index < 4U; ++index) codepoint = (codepoint << 4U) | hex_digit(text[position++]);
            append_codepoint(value, codepoint);
        } else throw Track1Error("unsupported JSON escape");
    }
    throw Track1Error("unterminated JSON string");
}

std::size_t matching_delimiter(const std::string& text, const std::size_t start, const char open, const char close) {
    require(start < text.size() && text[start] == open, "JSON delimiter start is invalid");
    std::size_t depth = 0U;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t position = start; position < text.size(); ++position) {
        const char character = text[position];
        if (in_string) {
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') in_string = false;
            continue;
        }
        if (character == '"') in_string = true;
        else if (character == open) ++depth;
        else if (character == close) {
            require(depth > 0U, "JSON delimiter depth underflow");
            --depth;
            if (depth == 0U) return position;
        }
    }
    throw Track1Error("unterminated JSON delimiter");
}

std::string field_string(const std::string& object, const std::string& key, const bool required = true) {
    const auto marker = '"' + key + '"';
    const auto key_position = object.find(marker);
    if (key_position == std::string::npos) {
        require(!required, "missing JSON field " + key);
        return {};
    }
    auto position = skip_space(object, object.find(':', key_position + marker.size()) + 1U);
    return parse_json_string(object, position);
}

std::string nested_object(const std::string& object, const std::string& key) {
    const auto marker = '"' + key + '"';
    const auto key_position = object.find(marker);
    require(key_position != std::string::npos, "missing JSON object field " + key);
    const auto colon = object.find(':', key_position + marker.size());
    require(colon != std::string::npos, "missing JSON object colon " + key);
    const auto start = object.find('{', colon + 1U);
    require(start != std::string::npos, "missing JSON object value " + key);
    return object.substr(start, matching_delimiter(object, start, '{', '}') - start + 1U);
}

std::string first_array_string(const std::string& object, const std::string& key) {
    const auto marker = '"' + key + '"';
    const auto key_position = object.find(marker);
    require(key_position != std::string::npos, "missing JSON array field " + key);
    const auto colon = object.find(':', key_position + marker.size());
    const auto start = skip_space(object, object.find('[', colon + 1U) + 1U);
    if (start < object.size() && object[start] == ']') return {};
    auto position = start;
    return parse_json_string(object, position);
}

std::size_t first_array_integer(const std::string& object, const std::string& key) {
    const auto marker = '"' + key + '"';
    const auto key_position = object.find(marker);
    require(key_position != std::string::npos, "missing JSON array field " + key);
    const auto colon = object.find(':', key_position + marker.size());
    const auto start = skip_space(object, object.find('[', colon + 1U) + 1U);
    if (start < object.size() && object[start] == ']') return 0U;
    const auto end = object.find_first_not_of("0123456789", start);
    require(end != start, "invalid JSON array integer");
    return static_cast<std::size_t>(std::stoull(object.substr(start, end - start)));
}

std::vector<std::string> row_objects(const std::string& page) {
    const auto rows_key = page.find("\"rows\"");
    require(rows_key != std::string::npos, "Hugging Face response has no rows field");
    const auto array_start = page.find('[', rows_key);
    require(array_start != std::string::npos, "Hugging Face rows field is not an array");
    const auto array_end = matching_delimiter(page, array_start, '[', ']');
    std::vector<std::string> rows;
    std::size_t position = array_start + 1U;
    while (position < array_end) {
        position = skip_space(page, position);
        if (position >= array_end) break;
        if (page[position] == ',') { ++position; continue; }
        require(page[position] == '{', "Hugging Face row is not an object");
        const auto end = matching_delimiter(page, position, '{', '}');
        rows.push_back(page.substr(position, end - position + 1U));
        position = end + 1U;
    }
    return rows;
}

template <typename Callback>
void for_each_flat_data_object(const std::string& document, Callback&& callback) {
    const auto data_key = document.find("\"data\"");
    require(data_key != std::string::npos, "GEM SQuAD response has no data field");
    const auto array_start = document.find('[', data_key);
    require(array_start != std::string::npos, "GEM SQuAD data field is not an array");
    const auto array_end = matching_delimiter(document, array_start, '[', ']');
    std::size_t position = array_start + 1U;
    std::size_t index = 0U;
    while (position < array_end) {
        position = skip_space(document, position);
        if (position >= array_end) break;
        if (document[position] == ',') { ++position; continue; }
        require(document[position] == '{', "GEM SQuAD data item is not an object");
        const auto end = matching_delimiter(document, position, '{', '}');
        callback(index, document.substr(position, end - position + 1U));
        ++index;
        position = end + 1U;
    }
}

std::string run_curl(const std::string& url, const std::filesystem::path& path, const bool acquire_remote) {
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0U) return read_file(path);
    require(acquire_remote, "cached Hugging Face page is missing: " + path.string());
    std::filesystem::create_directories(path.parent_path());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    const auto command = "curl --fail --location --silent --show-error --retry 12 --retry-all-errors --retry-max-time 900 "
                         "--connect-timeout 30 --max-time 960 --user-agent \"CCT-ASE-Track1/1.0\" --output \"" +
                         path.string() + "\" \"" + url + "\"";
    require(std::system(command.c_str()) == 0, "Hugging Face acquisition failed for " + url);
    require(std::filesystem::exists(path) && std::filesystem::file_size(path) > 0U, "Hugging Face acquisition wrote an empty page for " + url);
    return read_file(path);
}

std::string page_url(const Track1Source& source, const std::size_t offset, const std::size_t length) {
    return source.row_api_url + "&offset=" + std::to_string(offset) + "&length=" + std::to_string(length) + "&revision=" + source.revision;
}

Track1Example parse_squad_example(const std::string& object, const Track1Source& source, const Track1Split split) {
    Track1Example example;
    example.id = field_string(object, "id");
    example.title = field_string(object, "title");
    example.context = field_string(object, "context");
    example.question = field_string(object, "question");
    const auto answers = nested_object(object, "answers");
    example.answer = first_array_string(answers, "text");
    example.source_answer_start = first_array_integer(answers, "answer_start");
    example.answerable = !example.answer.empty();
    if (example.answerable) {
        example.answer_start = utf8_byte_offset(example.context, example.source_answer_start);
        require(example.answer_start < example.context.size() && example.context.compare(example.answer_start, example.answer.size(), example.answer) == 0,
                "SQuAD answer offset does not match context");
    } else {
        require(example.source_answer_start == 0U, "unanswerable SQuAD example has a non-empty answer offset");
    }
    example.source_id = source.source_id;
    example.split = split;
    example.content_digest = GovernedCorpus::content_sha256(example.id + "|" + example.context + "|" + example.question + "|" + example.answer);
    return example;
}

std::string source_path_component(const Track1Source& source) {
    return source.source_id + "_" + source.split + ".json";
}

std::uint64_t stable_key(const std::string& id, const std::uint64_t seed) {
    const auto digest = GovernedCorpus::content_sha256(id + "|" + std::to_string(seed));
    return std::stoull(digest.substr(0U, 16U), nullptr, 16);
}

std::string manifest_body(const Track1Manifest& manifest) {
    std::ostringstream output;
    output << "{\"manifest_version\":\"" << json_escape(manifest.manifest_version) << "\",\"tokenizer_snapshot\":\""
           << json_escape(manifest.tokenizer_snapshot) << "\",\"selection_policy\":\"" << json_escape(manifest.selection_policy)
           << "\",\"selection_seed\":" << manifest.selection_seed << ",\"pretrain_train_tokens\":" << manifest.pretrain_train_tokens
           << ",\"pretrain_validation_tokens\":" << manifest.pretrain_validation_tokens << ",\"pretrain_test_tokens\":" << manifest.pretrain_test_tokens
           << ",\"sft_train_examples\":" << manifest.sft_train_examples << ",\"sft_evaluation_examples\":" << manifest.sft_evaluation_examples
           << ",\"final_test_examples\":" << manifest.final_test_examples << ",\"train_ids\":" << json_array(manifest.train_ids)
           << ",\"evaluation_ids\":" << json_array(manifest.evaluation_ids) << ",\"final_test_ids\":" << json_array(manifest.final_test_ids) << ",\"sources\":[";
    for (std::size_t index = 0U; index < manifest.sources.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& source = manifest.sources[index];
        output << "{\"source_id\":\"" << json_escape(source.source_id) << "\",\"dataset_id\":\"" << json_escape(source.dataset_id)
               << "\",\"config\":\"" << json_escape(source.config) << "\",\"split\":\"" << json_escape(source.split)
               << "\",\"revision\":\"" << source.revision << "\",\"license\":\"" << json_escape(source.license)
               << "\",\"row_api_url\":\"" << json_escape(source.row_api_url) << "\",\"total_rows\":" << source.total_rows
               << ",\"raw_digest\":\"" << source.raw_digest << "\",\"upstream_dataset_id\":\""
               << json_escape(source.upstream_dataset_id) << "\",\"acquisition_type\":\"" << json_escape(source.acquisition_type)
               << "\",\"raw_file_url\":\"" << json_escape(source.raw_file_url) << "\"}";
    }
    output << "]}";
    return output.str();
}

std::string example_json(const Track1Example& example) {
    std::ostringstream output;
    output << "{\"id\":\"" << json_escape(example.id) << "\",\"title\":\"" << json_escape(example.title)
           << "\",\"context\":\"" << json_escape(example.context) << "\",\"question\":\"" << json_escape(example.question)
           << "\",\"answer\":\"" << json_escape(example.answer) << "\",\"answer_start\":" << example.answer_start
           << ",\"source_answer_start\":" << example.source_answer_start << ",\"answerable\":" << (example.answerable ? "true" : "false") << ",\"source_id\":\"" << json_escape(example.source_id)
           << "\",\"split\":\"" << track1_split_name(example.split) << "\",\"content_digest\":\"" << example.content_digest << "\"}";
    return output.str();
}

struct Candidate {
    std::uint64_t key = 0U;
    Track1Example example;
};

void keep_candidate(std::vector<Candidate>& candidates, Candidate candidate, const std::size_t limit) {
    if (candidates.size() < limit) {
        candidates.push_back(std::move(candidate));
        return;
    }
    const auto worst = std::max_element(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) { return left.key < right.key; });
    if (worst != candidates.end() && candidate.key < worst->key) *worst = std::move(candidate);
}

void sort_candidates(std::vector<Candidate>& candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        if (left.key != right.key) return left.key < right.key;
        return left.example.id < right.example.id;
    });
}

const Track1Source& source_at(const std::vector<Track1Source>& sources, const std::string& id) {
    const auto found = std::find_if(sources.begin(), sources.end(), [&](const auto& source) { return source.source_id == id; });
    require(found != sources.end(), "Track 1 source is missing: " + id);
    return *found;
}

}  // namespace

std::string track1_split_name(const Track1Split split) {
    if (split == Track1Split::PretrainTrain) return "pretrain_train";
    if (split == Track1Split::PretrainValidation) return "pretrain_validation";
    if (split == Track1Split::PretrainTest) return "pretrain_test";
    if (split == Track1Split::SftTrain) return "sft_train";
    if (split == Track1Split::SftEvaluation) return "sft_evaluation";
    return "final_test";
}

Track1Pipeline::Track1Pipeline(Track1Config config) : config_(std::move(config)) {
    require(config_.page_length > 0U && config_.page_length <= 100U, "Track 1 page length must be 1..100 for Hugging Face rows API");
    const auto minimum_examples = config_.allow_small_fixture ? 2U : 1000U;
    require(config_.pretrain_token_cap > 0U && config_.sft_examples >= minimum_examples && config_.sft_eval_examples > 0U &&
                config_.sft_eval_examples < config_.sft_examples && config_.sft_examples % 2U == 0U && config_.sft_eval_examples % 2U == 0U,
            "Track 1 size configuration is invalid");
    manifest_.selection_seed = config_.selection_seed;
    manifest_.sources = {
        {"wikitext2_pretrain_train", "Salesforce/wikitext", "wikitext-2-raw-v1", "train", "b08601e04326c79dfdd32d625aee71d232d685c3", "CC BY-SA 3.0; GFDL metadata", "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=train", 36718, {}, {}, "hf_rows", {}},
        {"wikitext2_pretrain_validation", "Salesforce/wikitext", "wikitext-2-raw-v1", "validation", "b08601e04326c79dfdd32d625aee71d232d685c3", "CC BY-SA 3.0; GFDL metadata", "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=validation", 3760, {}, {}, "hf_rows", {}},
        {"wikitext2_pretrain_test", "Salesforce/wikitext", "wikitext-2-raw-v1", "test", "b08601e04326c79dfdd32d625aee71d232d685c3", "CC BY-SA 3.0; GFDL metadata", "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=test", 4358, {}, {}, "hf_rows", {}},
        {"squad2_sft_train_source", "GEM/squad_v2", "gem_data_split", "train", "67199807729e631955056c71c258b7acbee548a3", "CC BY-SA 4.0", "https://datasets-server.huggingface.co/rows?dataset=rajpurkar%2Fsquad_v2&config=squad_v2&split=train", 116397, {}, {}, "hf_gem_flat_file", {}},
        {"squad2_final_test_source", "GEM/squad_v2", "gem_data_split", "validation", "67199807729e631955056c71c258b7acbee548a3", "CC BY-SA 4.0", "https://datasets-server.huggingface.co/rows?dataset=rajpurkar%2Fsquad_v2&config=squad_v2&split=validation", 11873, {}, {}, "hf_gem_flat_file", {}}};
    auto& squad_train = const_cast<Track1Source&>(source_at(manifest_.sources, "squad2_sft_train_source"));
    auto& squad_final = const_cast<Track1Source&>(source_at(manifest_.sources, "squad2_final_test_source"));
    squad_train.upstream_dataset_id = "rajpurkar/squad_v2";
    squad_final.upstream_dataset_id = "rajpurkar/squad_v2";
    squad_train.acquisition_type = "hf_gem_flat_file";
    squad_final.acquisition_type = "hf_gem_flat_file";
    squad_train.raw_file_url = "https://huggingface.co/datasets/GEM/squad_v2/resolve/67199807729e631955056c71c258b7acbee548a3/gem_data_split/train.json?download=true";
    squad_final.raw_file_url = "https://huggingface.co/datasets/GEM/squad_v2/resolve/67199807729e631955056c71c258b7acbee548a3/gem_data_split/validation.json?download=true";
}

const Track1Config& Track1Pipeline::config() const noexcept { return config_; }
const Track1Manifest& Track1Pipeline::manifest() const noexcept { return manifest_; }
const Track1PreparationReport& Track1Pipeline::report() const noexcept { return report_; }
const Track1EvaluationContract& Track1Pipeline::evaluation_contract() const noexcept { return evaluation_contract_; }

void Track1Pipeline::prepare_wikitext() {
    const std::vector<std::pair<Track1Split, std::string>> split_paths{
        {Track1Split::PretrainTrain, "wikitext2_pretrain_train"}, {Track1Split::PretrainValidation, "wikitext2_pretrain_validation"}, {Track1Split::PretrainTest, "wikitext2_pretrain_test"}};
    for (const auto& [split, source_id] : split_paths) {
        auto& source = const_cast<Track1Source&>(source_at(manifest_.sources, source_id));
        const auto raw_path = std::filesystem::path(config_.output_root) / "raw" / source_path_component(source);
        std::ostringstream prepared;
        std::string raw_digest_material;
        std::size_t tokens = 0U;
        std::size_t rows = 0U;
        const auto source_rows = config_.source_row_limit == 0U ? source.total_rows : std::min(source.total_rows, config_.source_row_limit);
        for (std::size_t offset = 0U; offset < source_rows; offset += config_.page_length) {
            const auto page = run_curl(page_url(source, offset, config_.page_length), raw_path.string() + "." + std::to_string(offset), config_.acquire_remote);
            raw_digest_material += page;
            ++report_.source_pages;
            for (const auto& wrapper : row_objects(page)) {
                const auto row = nested_object(wrapper, "row");
                const auto text = field_string(row, "text");
                ++report_.source_rows;
                const auto cap = split == Track1Split::PretrainTrain ? config_.pretrain_token_cap : std::numeric_limits<std::size_t>::max();
                if (tokens >= cap) continue;
                const auto take = std::min(text.size(), cap - tokens);
                prepared.write(text.data(), static_cast<std::streamsize>(take));
                prepared.put('\n');
                tokens += take;
                ++rows;
            }
        }
        source.raw_digest = GovernedCorpus::content_sha256(raw_digest_material);
        const auto filename = track1_split_name(split) + ".txt";
        write_file_atomic(std::filesystem::path(config_.output_root) / "data" / filename, prepared.str());
        if (split == Track1Split::PretrainTrain) manifest_.pretrain_train_tokens = tokens;
        else if (split == Track1Split::PretrainValidation) manifest_.pretrain_validation_tokens = tokens;
        else manifest_.pretrain_test_tokens = tokens;
        report_.pretrain_rows += rows;
        if (split == Track1Split::PretrainTrain) report_.pretrain_tokens = tokens;
    }
}

void Track1Pipeline::prepare_squad() {
    auto& train_source = const_cast<Track1Source&>(source_at(manifest_.sources, "squad2_sft_train_source"));
    auto& final_source = const_cast<Track1Source&>(source_at(manifest_.sources, "squad2_final_test_source"));
    std::vector<Candidate> answerable;
    std::vector<Candidate> unanswerable;
    require(config_.squad_train_row_offset < train_source.total_rows && config_.squad_final_test_row_offset < final_source.total_rows,
            "Track 1 SQuAD source offset exceeds the pinned split");
    const std::size_t category_limit = config_.sft_examples / 2U;
    std::string train_raw_digest;
    std::string final_raw_digest;
    std::vector<Track1Example> final_examples;
    const auto train_available = train_source.total_rows - config_.squad_train_row_offset;
    const auto final_available = final_source.total_rows - config_.squad_final_test_row_offset;
    const auto train_rows = config_.source_row_limit == 0U ? train_available : std::min(train_available, config_.source_row_limit);
    const auto final_rows = config_.source_row_limit == 0U ? final_available : std::min(final_available, config_.source_row_limit);
    const auto process_train_object = [&](const std::string& object, const bool wrapped) {
        ++report_.source_rows;
        try {
            const auto row = wrapped ? nested_object(object, "row") : object;
            auto example = parse_squad_example(row, train_source, Track1Split::SftTrain);
            const Candidate candidate{stable_key(example.id, config_.selection_seed), std::move(example)};
            if (candidate.example.answerable) keep_candidate(answerable, candidate, category_limit);
            else { ++report_.unanswerable_rows; keep_candidate(unanswerable, candidate, category_limit); }
        } catch (const std::exception& error) {
            ++report_.malformed_rows;
            throw Track1Error(std::string("malformed SQuAD training row: ") + error.what());
        }
    };
    const auto process_final_object = [&](const std::string& object, const bool wrapped) {
        ++report_.source_rows;
        try {
            const auto row = wrapped ? nested_object(object, "row") : object;
            final_examples.push_back(parse_squad_example(row, final_source, Track1Split::FinalTest));
        } catch (const std::exception& error) {
            ++report_.malformed_rows;
            throw Track1Error(std::string("malformed SQuAD final-test row: ") + error.what());
        }
    };
    const auto train_direct_path = std::filesystem::path(config_.output_root) / "raw" / source_path_component(train_source);
    const auto final_direct_path = std::filesystem::path(config_.output_root) / "raw" / source_path_component(final_source);
    const bool use_train_direct_file = train_source.acquisition_type == "hf_gem_flat_file" &&
                                       (config_.acquire_remote || (std::filesystem::exists(train_direct_path) && std::filesystem::file_size(train_direct_path) > 0U));
    const bool use_final_direct_file = final_source.acquisition_type == "hf_gem_flat_file" &&
                                       (config_.acquire_remote || (std::filesystem::exists(final_direct_path) && std::filesystem::file_size(final_direct_path) > 0U));
    if (use_train_direct_file) {
        const auto path = train_direct_path;
        const auto document = run_curl(train_source.raw_file_url, path, config_.acquire_remote);
        train_raw_digest = document;
        ++report_.source_pages;
        for_each_flat_data_object(document, [&](const std::size_t index, const std::string& object) {
            if (index >= config_.squad_train_row_offset && index < config_.squad_train_row_offset + train_rows) process_train_object(object, false);
        });
    } else {
        for (std::size_t relative_offset = 0U; relative_offset < train_rows; relative_offset += config_.page_length) {
            const auto offset = config_.squad_train_row_offset + relative_offset;
            const auto page = run_curl(page_url(train_source, offset, config_.page_length), (std::filesystem::path(config_.output_root) / "raw" / source_path_component(train_source)).string() + "." + std::to_string(offset), config_.acquire_remote);
            train_raw_digest += page;
            ++report_.source_pages;
            for (const auto& wrapper : row_objects(page)) process_train_object(wrapper, true);
        }
    }
    if (use_final_direct_file) {
        const auto path = final_direct_path;
        const auto document = run_curl(final_source.raw_file_url, path, config_.acquire_remote);
        final_raw_digest = document;
        ++report_.source_pages;
        for_each_flat_data_object(document, [&](const std::size_t index, const std::string& object) {
            if (index >= config_.squad_final_test_row_offset && index < config_.squad_final_test_row_offset + final_rows) process_final_object(object, false);
        });
    } else {
        for (std::size_t relative_offset = 0U; relative_offset < final_rows; relative_offset += config_.page_length) {
            const auto offset = config_.squad_final_test_row_offset + relative_offset;
            const auto page = run_curl(page_url(final_source, offset, config_.page_length), (std::filesystem::path(config_.output_root) / "raw" / source_path_component(final_source)).string() + "." + std::to_string(offset), config_.acquire_remote);
            final_raw_digest += page;
            ++report_.source_pages;
            for (const auto& wrapper : row_objects(page)) process_final_object(wrapper, true);
        }
    }
    require(answerable.size() == category_limit && unanswerable.size() == category_limit, "SQuAD stable selection did not produce balanced categories");
    sort_candidates(answerable);
    sort_candidates(unanswerable);
    std::vector<Track1Example> sft_train;
    std::vector<Track1Example> sft_eval;
    const auto split_candidates = [&](std::vector<Candidate>& candidates) {
        const auto evaluation_count = config_.sft_eval_examples / 2U;
        for (std::size_t index = 0U; index < candidates.size(); ++index) {
            auto example = candidates[index].example;
            example.split = index < evaluation_count ? Track1Split::SftEvaluation : Track1Split::SftTrain;
            (index < evaluation_count ? sft_eval : sft_train).push_back(std::move(example));
        }
    };
    require(config_.sft_eval_examples % 2U == 0U, "SQuAD evaluation count must be even for balanced answerability");
    split_candidates(answerable);
    split_candidates(unanswerable);
    std::sort(final_examples.begin(), final_examples.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
    auto write_jsonl = [&](const std::string& filename, const std::vector<Track1Example>& examples) {
        std::ostringstream content;
        for (const auto& example : examples) content << example_json(example) << '\n';
        write_file_atomic(std::filesystem::path(config_.output_root) / "data" / filename, content.str());
    };
    write_jsonl("squad_sft_train.jsonl", sft_train);
    write_jsonl("squad_sft_evaluation.jsonl", sft_eval);
    write_jsonl("squad_final_test.jsonl", final_examples);
    train_source.raw_digest = GovernedCorpus::content_sha256(train_raw_digest);
    final_source.raw_digest = GovernedCorpus::content_sha256(final_raw_digest);
    for (const auto& example : sft_train) manifest_.train_ids.push_back(example.id);
    for (const auto& example : sft_eval) manifest_.evaluation_ids.push_back(example.id);
    for (const auto& example : final_examples) manifest_.final_test_ids.push_back(example.id);
    manifest_.sft_train_examples = sft_train.size();
    manifest_.sft_evaluation_examples = sft_eval.size();
    manifest_.final_test_examples = final_examples.size();
    report_.sft_rows = sft_train.size() + sft_eval.size();
    report_.sft_train_rows = sft_train.size();
    report_.sft_evaluation_rows = sft_eval.size();
    report_.final_test_rows = final_examples.size();
}

void Track1Pipeline::validate_manifest() const {
    require(manifest_.pretrain_train_tokens == config_.pretrain_token_cap, "pretraining cap was not exactly applied");
    require(manifest_.sft_train_examples == config_.sft_examples - config_.sft_eval_examples && manifest_.sft_evaluation_examples == config_.sft_eval_examples,
            "SQuAD train/evaluation counts do not match contract");
    require(manifest_.final_test_examples > 0U && manifest_.train_ids.size() == manifest_.sft_train_examples &&
                manifest_.evaluation_ids.size() == manifest_.sft_evaluation_examples && manifest_.final_test_ids.size() == manifest_.final_test_examples,
            "Track 1 manifest counts are incomplete");
    std::map<std::string, Track1Split> seen;
    for (const auto& id : manifest_.train_ids) require(seen.emplace(id, Track1Split::SftTrain).second, "duplicate or overlapping Track 1 train ID");
    for (const auto& id : manifest_.evaluation_ids) require(seen.emplace(id, Track1Split::SftEvaluation).second, "duplicate or overlapping Track 1 evaluation ID");
    for (const auto& id : manifest_.final_test_ids) require(seen.emplace(id, Track1Split::FinalTest).second, "duplicate or overlapping Track 1 final-test ID");
    require(!manifest_.manifest_digest.empty() && manifest_.manifest_digest == GovernedCorpus::content_sha256(manifest_body(manifest_)),
            "Track 1 manifest digest is missing or inconsistent");
}

void Track1Pipeline::write_artifacts() const {
    write_file_atomic(std::filesystem::path(config_.output_root) / "manifest.json", serialize_manifest());
    write_file_atomic(std::filesystem::path(config_.output_root) / "preparation_report.json", serialize_report());
    write_file_atomic(std::filesystem::path(config_.output_root) / "evaluation_contract.json", serialize_evaluation_contract());
}

void Track1Pipeline::prepare() {
    report_ = {};
    manifest_.train_ids.clear();
    manifest_.evaluation_ids.clear();
    manifest_.final_test_ids.clear();
    prepare_wikitext();
    prepare_squad();
    manifest_.manifest_digest = GovernedCorpus::content_sha256(manifest_body(manifest_));
    validate_manifest();
    report_.passed = true;
    report_.manifest_path = (std::filesystem::path(config_.output_root) / "manifest.json").string();
    report_.report_path = (std::filesystem::path(config_.output_root) / "preparation_report.json").string();
    write_artifacts();
}

void Track1Pipeline::validate_existing() const {
    const auto manifest_path = std::filesystem::path(config_.output_root) / "manifest.json";
    require(std::filesystem::exists(manifest_path), "Track 1 manifest is missing");
    require(std::filesystem::exists(std::filesystem::path(config_.output_root) / "data" / "pretrain_train.txt"), "Track 1 pretraining stream is missing");
    require(std::filesystem::exists(std::filesystem::path(config_.output_root) / "data" / "squad_final_test.jsonl"), "Track 1 final test stream is missing");
    require(!read_file(manifest_path).empty(), "Track 1 manifest is empty");
}

std::string Track1Pipeline::serialize_manifest() const {
    return manifest_body(manifest_).substr(0U, manifest_body(manifest_).size() - 1U) + ",\"manifest_digest\":\"" + manifest_.manifest_digest + "\"}\n";
}

std::string Track1Pipeline::serialize_report() const {
    std::ostringstream output;
    output << "{\"passed\":" << (report_.passed ? "true" : "false") << ",\"source_pages\":" << report_.source_pages
           << ",\"source_rows\":" << report_.source_rows << ",\"pretrain_rows\":" << report_.pretrain_rows << ",\"pretrain_tokens\":" << report_.pretrain_tokens
           << ",\"sft_rows\":" << report_.sft_rows << ",\"sft_train_rows\":" << report_.sft_train_rows << ",\"sft_evaluation_rows\":" << report_.sft_evaluation_rows
           << ",\"final_test_rows\":" << report_.final_test_rows << ",\"duplicate_ids\":" << report_.duplicate_ids << ",\"overlap_ids\":" << report_.overlap_ids
           << ",\"malformed_rows\":" << report_.malformed_rows << ",\"unanswerable_rows\":" << report_.unanswerable_rows
           << ",\"manifest_path\":\"" << json_escape(report_.manifest_path) << "\",\"report_path\":\"" << json_escape(report_.report_path) << "\"}\n";
    return output.str();
}

std::string Track1Pipeline::serialize_evaluation_contract() const {
    std::ostringstream output;
    output << "{\"metric_version\":\"" << evaluation_contract_.metric_version << "\",\"pretrain_metrics\":" << json_array(evaluation_contract_.pretrain_metrics)
           << ",\"qa_metrics\":" << json_array(evaluation_contract_.qa_metrics) << ",\"required_slices\":" << json_array(evaluation_contract_.required_slices)
           << ",\"forbidden_behaviors\":" << json_array(evaluation_contract_.forbidden_behaviors) << "}\n";
    return output.str();
}

}  // namespace cct
