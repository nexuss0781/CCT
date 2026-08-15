#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

class PreparationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw PreparationError(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish writing " + path.string());
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
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (const char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    result += '\'';
    return result;
}

std::string url_encode(const std::string& value) {
    static constexpr char digits[] = "0123456789ABCDEF";
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (std::isalnum(character) != 0 || character == '-' || character == '_' || character == '.' || character == '~') {
            output << static_cast<char>(character);
        } else {
            output << '%' << digits[character >> 4U] << digits[character & 0x0FU];
        }
    }
    return output.str();
}

std::string sha256_hex(const std::string& value) {
    const auto temporary = std::filesystem::temp_directory_path() / ("cct-curriculum-digest-" + std::to_string(::getpid()) + ".txt");
    write_file(temporary, value);
    const auto command = "sha256sum " + shell_quote(temporary.string());
    FILE* pipe = ::popen(command.c_str(), "r");
    require(pipe != nullptr, "cannot calculate source digest");
    char buffer[256]{};
    const auto read = std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe);
    const auto status = ::pclose(pipe);
    std::filesystem::remove(temporary);
    require(read != nullptr && status == 0, "source digest command failed");
    const std::string line(buffer);
    const auto separator = line.find(' ');
    require(separator == 64U, "source digest output is malformed");
    return line.substr(0U, separator);
}

class JsonParser {
public:
    explicit JsonParser(const std::string& content) : content_(content) {}

    std::map<std::string, std::string> object() {
        skip_space();
        require(position_ < content_.size() && content_[position_] == '{', "JSON object is missing opening brace");
        ++position_;
        std::map<std::string, std::string> fields;
        skip_space();
        if (position_ < content_.size() && content_[position_] == '}') {
            ++position_;
            return fields;
        }
        while (position_ < content_.size()) {
            const auto key = string();
            skip_space();
            require(position_ < content_.size() && content_[position_] == ':', "JSON object key is missing colon");
            ++position_;
            skip_space();
            fields[key] = value();
            skip_space();
            require(position_ < content_.size() && (content_[position_] == ',' || content_[position_] == '}'),
                    "JSON object separator is invalid");
            if (content_[position_] == '}') {
                ++position_;
                return fields;
            }
            ++position_;
            skip_space();
        }
        throw PreparationError("JSON object is unterminated");
    }

private:
    const std::string& content_;
    std::size_t position_ = 0U;

    void skip_space() {
        while (position_ < content_.size() && std::isspace(static_cast<unsigned char>(content_[position_])) != 0) ++position_;
    }

    static void append_utf8(std::string& output, const unsigned int codepoint) {
        if (codepoint <= 0x7FU) output.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0x10FFFFU) {
            output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else {
            throw PreparationError("JSON Unicode escape exceeds Unicode range");
        }
    }

    std::string string() {
        require(position_ < content_.size() && content_[position_] == '"', "JSON string is missing opening quote");
        ++position_;
        std::string output;
        while (position_ < content_.size()) {
            const auto character = static_cast<unsigned char>(content_[position_++]);
            if (character == '"') return output;
            if (character != '\\') {
                require(character >= 0x20U, "JSON string contains a control character");
                output.push_back(static_cast<char>(character));
                continue;
            }
            require(position_ < content_.size(), "JSON string escape is truncated");
            const auto escaped = content_[position_++];
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    require(position_ + 4U <= content_.size(), "JSON Unicode escape is truncated");
                    unsigned int codepoint = 0U;
                    for (std::size_t index = 0U; index < 4U; ++index) {
                        const auto digit = content_[position_++];
                        codepoint <<= 4U;
                        if (digit >= '0' && digit <= '9') codepoint += static_cast<unsigned int>(digit - '0');
                        else if (digit >= 'a' && digit <= 'f') codepoint += static_cast<unsigned int>(digit - 'a' + 10);
                        else if (digit >= 'A' && digit <= 'F') codepoint += static_cast<unsigned int>(digit - 'A' + 10);
                        else throw PreparationError("JSON Unicode escape contains a non-hex digit");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default: throw PreparationError("JSON string contains an unsupported escape");
            }
        }
        throw PreparationError("JSON string is unterminated");
    }

    std::string value() {
        const auto start = position_;
        if (position_ < content_.size() && content_[position_] == '"') return string();
        if (position_ < content_.size() && (content_[position_] == '{' || content_[position_] == '[')) {
            const auto opening = content_[position_++];
            const auto closing = opening == '{' ? '}' : ']';
            std::size_t depth = 1U;
            bool in_string = false;
            bool escaped = false;
            while (position_ < content_.size() && depth > 0U) {
                const auto character = content_[position_++];
                if (in_string) {
                    if (escaped) escaped = false;
                    else if (character == '\\') escaped = true;
                    else if (character == '"') in_string = false;
                } else if (character == '"') {
                    in_string = true;
                } else if (character == opening) {
                    ++depth;
                } else if (character == closing) {
                    --depth;
                }
            }
            require(depth == 0U && !in_string, "JSON composite value is unterminated");
            return content_.substr(start, position_ - start);
        }
        while (position_ < content_.size() && content_[position_] != ',' && content_[position_] != '}' && content_[position_] != ']') ++position_;
        auto result = content_.substr(start, position_ - start);
        while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back())) != 0) result.pop_back();
        return result;
    }
};

std::vector<std::map<std::string, std::string>> array_objects(const std::string& raw) {
    JsonParser parser(raw);
    std::vector<std::map<std::string, std::string>> objects;
    std::size_t position = 0U;
    while (position < raw.size() && std::isspace(static_cast<unsigned char>(raw[position])) != 0) ++position;
    require(position < raw.size() && raw[position] == '[', "JSON rows field is not an array");
    ++position;
    while (position < raw.size()) {
        while (position < raw.size() && (std::isspace(static_cast<unsigned char>(raw[position])) != 0 || raw[position] == ',')) ++position;
        if (position >= raw.size() || raw[position] == ']') break;
        require(raw[position] == '{', "rows array contains a non-object value");
        std::size_t depth = 0U;
        bool in_string = false;
        bool escaped = false;
        const auto start = position;
        while (position < raw.size()) {
            const auto character = raw[position++];
            if (in_string) {
                if (escaped) escaped = false;
                else if (character == '\\') escaped = true;
                else if (character == '"') in_string = false;
            } else if (character == '"') {
                in_string = true;
            } else if (character == '{') {
                ++depth;
            } else if (character == '}') {
                require(depth > 0U, "rows object depth underflow");
                --depth;
                if (depth == 0U) break;
            }
        }
        require(depth == 0U && !in_string, "rows array object is unterminated");
        const auto object_text = raw.substr(start, position - start);
        objects.push_back(JsonParser(object_text).object());
    }
    return objects;
}

std::vector<std::map<std::string, std::string>> fetch_page(const std::string& dataset, const std::string& config,
                                                           const std::string& revision, const std::size_t offset,
                                                           const std::size_t length, const std::filesystem::path& temporary) {
    const auto url = "https://datasets-server.huggingface.co/rows?dataset=" + url_encode(dataset) + "&config=" + url_encode(config) +
                     "&split=train&offset=" + std::to_string(offset) + "&length=" + std::to_string(length) + "&revision=" + revision;
    const auto command = "curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 20 --max-time 180 -sS --output " +
                         shell_quote(temporary.string()) + " " + shell_quote(url);
    require(std::system(command.c_str()) == 0, "failed to download dataset rows at offset " + std::to_string(offset));
    const auto response = read_file(temporary);
    const auto top = JsonParser(response).object();
    require(top.contains("rows"), "dataset rows response is missing rows");
    return array_objects(top.at("rows"));
}

struct RowRecord {
    std::string id;
    std::string text;
    std::string language;
    std::string role;
    double score = 0.0;
    bool deleted = false;
    bool review_result = true;
};

RowRecord parse_row(const std::map<std::string, std::string>& wrapper, const std::string& source_name) {
    require(wrapper.contains("row"), source_name + " row wrapper is missing row object");
    const auto fields = JsonParser(wrapper.at("row")).object();
    RowRecord result;
    if (fields.contains("id")) result.id = fields.at("id");
    if (fields.contains("message_id")) result.id = fields.at("message_id");
    if (fields.contains("text")) result.text = fields.at("text");
    if (fields.contains("language")) result.language = fields.at("language");
    if (fields.contains("lang")) result.language = fields.at("lang");
    if (fields.contains("role")) result.role = fields.at("role");
    if (fields.contains("score")) result.score = std::stod(fields.at("score"));
    if (fields.contains("deleted")) result.deleted = fields.at("deleted") == "true";
    if (fields.contains("review_result")) result.review_result = fields.at("review_result") == "true";
    require(!result.id.empty() && !result.text.empty(), source_name + " row lacks stable ID or text");
    return result;
}

std::string normalize_record(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    bool pending_space = false;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (std::isspace(character) != 0) {
            pending_space = !result.empty();
        } else {
            if (pending_space) result.push_back(' ');
            result.push_back(static_cast<char>(character));
            pending_space = false;
        }
    }
    while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back())) != 0) result.pop_back();
    return result;
}

struct Options {
    std::filesystem::path output = "artifacts/curriculum/current";
    std::size_t pretrain_offset = 0U;
    std::size_t pretrain_rows = 100U;
    std::size_t validation_offset = 1000U;
    std::size_t validation_rows = 40U;
    std::size_t test_offset = 2000U;
    std::size_t test_rows = 40U;
    std::size_t sft_offset = 0U;
    std::size_t sft_rows = 100U;
    std::size_t sft_validation_offset = 1000U;
    std::size_t sft_validation_rows = 40U;
    std::size_t page_length = 100U;
    double minimum_education_score = 2.0;
    std::string fineweb_revision = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9";
    std::string oasst_revision = "fdf72ae0827c1cda404aff25b6603abec9e3399b";
};

std::size_t number(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0U;
        const auto parsed = std::stoull(value, &consumed);
        require(consumed == value.size(), "invalid numeric value for " + name);
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw PreparationError("invalid numeric value for " + name);
    }
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        const auto value = [&]() {
            require(index + 1 < argc, "missing value for " + key);
            return std::string(argv[++index]);
        };
        if (key == "--output") options.output = value();
        else if (key == "--pretrain-offset") options.pretrain_offset = number(value(), key);
        else if (key == "--pretrain-rows") options.pretrain_rows = number(value(), key);
        else if (key == "--validation-offset") options.validation_offset = number(value(), key);
        else if (key == "--validation-rows") options.validation_rows = number(value(), key);
        else if (key == "--test-offset") options.test_offset = number(value(), key);
        else if (key == "--test-rows") options.test_rows = number(value(), key);
        else if (key == "--sft-offset") options.sft_offset = number(value(), key);
        else if (key == "--sft-rows") options.sft_rows = number(value(), key);
        else if (key == "--sft-validation-offset") options.sft_validation_offset = number(value(), key);
        else if (key == "--sft-validation-rows") options.sft_validation_rows = number(value(), key);
        else if (key == "--page-length") options.page_length = number(value(), key);
        else if (key == "--minimum-education-score") options.minimum_education_score = std::stod(value());
        else if (key == "--fineweb-revision") options.fineweb_revision = value();
        else if (key == "--oasst-revision") options.oasst_revision = value();
        else if (key == "--help") {
            std::cout << "cct_curriculum_prepare --output PATH --pretrain-offset N --pretrain-rows N --validation-offset N --validation-rows N "
                         "--test-offset N --test-rows N --sft-offset N --sft-rows N --sft-validation-offset N --sft-validation-rows N --minimum-education-score N "
                         "[--fineweb-revision SHA] [--oasst-revision SHA]\n";
            std::exit(0);
        } else throw PreparationError("unknown argument " + key);
    }
    require(options.page_length > 0U && options.page_length <= 100U && options.pretrain_rows > 0U && options.validation_rows > 0U &&
                options.test_rows > 0U && options.sft_rows > 0U && options.sft_validation_rows > 0U && std::isfinite(options.minimum_education_score),
            "curriculum range configuration is invalid");
    return options;
}

struct Collection {
    std::vector<RowRecord> rows;
    std::vector<std::string> source_ids;
    std::size_t requested_rows = 0U;
    std::size_t pages = 0U;
};

Collection collect(const std::string& dataset, const std::string& config, const std::string& revision,
                   const std::size_t offset, const std::size_t requested, const std::string& source_name,
                   const Options& options) {
    Collection collection;
    collection.requested_rows = requested;
    std::unordered_set<std::string> ids;
    const auto temporary = options.output / (source_name + ".page.json");
    for (std::size_t consumed = 0U; consumed < requested; consumed += options.page_length) {
        const auto length = std::min(options.page_length, requested - consumed);
        const auto page = fetch_page(dataset, config, revision, offset + consumed, length, temporary);
        ++collection.pages;
        for (const auto& wrapper : page) {
            const auto row = parse_row(wrapper, source_name);
            if (ids.insert(row.id).second) collection.rows.push_back(row);
        }
        if (page.size() < length) break;
    }
    std::filesystem::remove(temporary);
    return collection;
}

void write_records(const std::filesystem::path& path, const std::vector<RowRecord>& rows, const bool fineweb,
                   const double minimum_education_score, const std::size_t maximum_records, std::vector<std::string>& selected_ids) {
    std::ofstream output(path, std::ios::trunc);
    require(static_cast<bool>(output), "cannot write curriculum record file " + path.string());
    for (const auto& row : rows) {
        const auto text = normalize_record(row.text);
        const bool accepted = fineweb ? row.language == "en" && row.score >= minimum_education_score :
                                       row.language == "en" && !row.deleted && row.review_result && (row.role == "assistant" || row.role == "prompter");
        if (!accepted || text.size() < 20U || selected_ids.size() >= maximum_records) continue;
        output << (fineweb ? text : (row.role + ": " + text)) << '\n';
        selected_ids.push_back(row.id);
    }
    require(selected_ids.size() == maximum_records,
            "curriculum filter selected " + std::to_string(selected_ids.size()) + " records but required " + std::to_string(maximum_records) +
                " for " + path.string());
}

std::string json_ids(const std::vector<std::string>& ids) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0U; index < ids.size(); ++index) {
        if (index > 0U) output << ',';
        output << '"' << json_escape(ids[index]) << '"';
    }
    output << ']';
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::filesystem::create_directories(options.output);
        const auto fineweb_train = collect("HuggingFaceFW/fineweb-edu", "sample-10BT", options.fineweb_revision,
                                           options.pretrain_offset, options.pretrain_rows, "fineweb_pretrain", options);
        const auto fineweb_validation = collect("HuggingFaceFW/fineweb-edu", "sample-10BT", options.fineweb_revision,
                                                options.validation_offset, options.validation_rows, "fineweb_validation", options);
        const auto fineweb_test = collect("HuggingFaceFW/fineweb-edu", "sample-10BT", options.fineweb_revision,
                                          options.test_offset, options.test_rows, "fineweb_test", options);
        require(options.sft_rows <= std::numeric_limits<std::size_t>::max() / 100U &&
                    options.sft_validation_rows <= std::numeric_limits<std::size_t>::max() / 100U,
                "OpenAssistant scan range would overflow");
        const auto oasst_train = collect("OpenAssistant/oasst1", "default", options.oasst_revision,
                                         options.sft_offset, options.sft_rows * 100U, "oasst_sft", options);
        const auto oasst_validation = collect("OpenAssistant/oasst1", "default", options.oasst_revision,
                                              options.sft_validation_offset, options.sft_validation_rows * 100U, "oasst_validation", options);
        const auto pretrain_train_path = options.output / "pretrain_train.txt";
        const auto pretrain_validation_path = options.output / "pretrain_validation.txt";
        const auto pretrain_test_path = options.output / "pretrain_test.txt";
        const auto sft_train_path = options.output / "sft_train.txt";
        const auto sft_validation_path = options.output / "sft_validation.txt";
        std::vector<std::string> pretrain_train_ids;
        std::vector<std::string> pretrain_validation_ids;
        std::vector<std::string> pretrain_test_ids;
        std::vector<std::string> sft_train_ids;
        std::vector<std::string> sft_validation_ids;
        write_records(pretrain_train_path, fineweb_train.rows, true, options.minimum_education_score, options.pretrain_rows, pretrain_train_ids);
        write_records(pretrain_validation_path, fineweb_validation.rows, true, options.minimum_education_score, options.validation_rows, pretrain_validation_ids);
        write_records(pretrain_test_path, fineweb_test.rows, true, options.minimum_education_score, options.test_rows, pretrain_test_ids);
        write_records(sft_train_path, oasst_train.rows, false, options.minimum_education_score, options.sft_rows, sft_train_ids);
        write_records(sft_validation_path, oasst_validation.rows, false, options.minimum_education_score, options.sft_validation_rows, sft_validation_ids);
        std::unordered_set<std::string> all_ids(pretrain_train_ids.begin(), pretrain_train_ids.end());
        for (const auto& id : pretrain_validation_ids) require(all_ids.insert(id).second, "FineWeb pretraining and validation IDs overlap");
        for (const auto& id : pretrain_test_ids) require(all_ids.insert(id).second, "FineWeb pretraining and test IDs overlap");
        require(std::all_of(sft_train_ids.begin(), sft_train_ids.end(), [&](const auto& id) {
            return std::find(sft_validation_ids.begin(), sft_validation_ids.end(), id) == sft_validation_ids.end();
        }), "OpenAssistant SFT and validation IDs overlap");
        const auto manifest = [&]() {
            std::ostringstream output;
            output << "{\n  \"status\":\"PASS\",\n  \"fineweb\":{\"dataset\":\"HuggingFaceFW/fineweb-edu\",\"config\":\"sample-10BT\",\"split\":\"train\",\"revision\":\""
                   << json_escape(options.fineweb_revision) << "\",\"pretrain_offset\":" << options.pretrain_offset << ",\"pretrain_requested_rows\":"
                   << options.pretrain_rows << ",\"validation_offset\":" << options.validation_offset << ",\"validation_requested_rows\":"
                   << options.validation_rows << ",\"test_offset\":" << options.test_offset << ",\"test_requested_rows\":" << options.test_rows
                   << ",\"minimum_education_score\":" << std::setprecision(17) << options.minimum_education_score
                   << ",\"pretrain_selected_ids\":" << json_ids(pretrain_train_ids) << ",\"validation_selected_ids\":" << json_ids(pretrain_validation_ids)
                   << ",\"test_selected_ids\":" << json_ids(pretrain_test_ids) << "},\n"
                   << "  \"oasst\":{\"dataset\":\"OpenAssistant/oasst1\",\"config\":\"default\",\"split\":\"train\",\"revision\":\""
                   << json_escape(options.oasst_revision) << "\",\"sft_offset\":" << options.sft_offset << ",\"sft_requested_rows\":"
                   << options.sft_rows << ",\"sft_validation_offset\":" << options.sft_validation_offset << ",\"sft_validation_requested_rows\":"
                   << options.sft_validation_rows << ",\"sft_selected_ids\":" << json_ids(sft_train_ids) << ",\"sft_validation_selected_ids\":"
                   << json_ids(sft_validation_ids) << "},\n"
                   << "  \"files\":{\"pretrain_train\":\"pretrain_train.txt\",\"pretrain_validation\":\"pretrain_validation.txt\",\"pretrain_test\":\"pretrain_test.txt\",\"sft_train\":\"sft_train.txt\",\"sft_validation\":\"sft_validation.txt\"}\n}\n";
            return output.str();
        }();
        write_file(options.output / "manifest.json", manifest);
        write_file(options.output / "source_digest.txt", "manifest_sha256=" + sha256_hex(manifest) + "\npretrain_train_sha256=" +
                                                            sha256_hex(read_file(pretrain_train_path)) + "\npretrain_validation_sha256=" +
                                                            sha256_hex(read_file(pretrain_validation_path)) + "\npretrain_test_sha256=" +
                                                            sha256_hex(read_file(pretrain_test_path)) + "\nsft_train_sha256=" +
                                                            sha256_hex(read_file(sft_train_path)) + "\nsft_validation_sha256=" +
                                                            sha256_hex(read_file(sft_validation_path)) + "\n");
        std::cout << "{\"status\":\"PASS\",\"output\":\"" << json_escape(options.output.string()) << "\",\"pretrain_train_records\":"
                  << pretrain_train_ids.size() << ",\"pretrain_validation_records\":" << pretrain_validation_ids.size()
                  << ",\"pretrain_test_records\":" << pretrain_test_ids.size() << ",\"sft_train_records\":" << sft_train_ids.size() << ",\"sft_validation_records\":" << sft_validation_ids.size() << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"FAIL\",\"error\":\"" << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
