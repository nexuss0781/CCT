#include "cct/track1.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cerrno>
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
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

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

std::filesystem::path unique_temporary_path(const std::filesystem::path& target) {
    std::filesystem::create_directories(target.parent_path());
    std::string pattern = target.string() + ".tmp.XXXXXX";
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const auto descriptor = mkstemp(mutable_pattern.data());
    require(descriptor >= 0, "cannot create unique temporary file for " + target.string());
    require(close(descriptor) == 0, "cannot close unique temporary file for " + target.string());
    return std::filesystem::path(mutable_pattern.data());
}

void sync_file(const std::filesystem::path& path) {
    const auto descriptor = open(path.c_str(), O_RDONLY);
    require(descriptor >= 0, "cannot open file for durable sync: " + path.string());
    require(fsync(descriptor) == 0, "cannot sync file: " + path.string());
    require(close(descriptor) == 0, "cannot close synced file: " + path.string());
}

void sync_directory(const std::filesystem::path& path) {
    const auto directory = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const auto descriptor = open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    require(descriptor >= 0, "cannot open directory for durable sync: " + directory.string());
    require(fsync(descriptor) == 0, "cannot sync directory: " + directory.string());
    require(close(descriptor) == 0, "cannot close synced directory: " + directory.string());
}

void publish_file_atomically(const std::filesystem::path& temporary, const std::filesystem::path& destination) {
    sync_file(temporary);
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    require(!error, "cannot atomically publish " + destination.string() + ": " + error.message());
    sync_directory(destination);
}

void write_file_atomic(const std::filesystem::path& path, const std::string& content) {
    const auto temporary = unique_temporary_path(path);
    try {
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(output), "cannot write " + temporary.string());
            output.write(content.data(), static_cast<std::streamsize>(content.size()));
            require(static_cast<bool>(output), "cannot finish " + temporary.string());
        }
        publish_file_atomically(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

int run_process(const std::vector<std::string>& arguments, const std::filesystem::path& stdout_path) {
    require(!arguments.empty() && !arguments.front().empty(), "native process arguments are empty");
    const auto descriptor = open(stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    require(descriptor >= 0, "cannot open native process output " + stdout_path.string());
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1U);
    for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    const auto child = fork();
    require(child >= 0, "cannot fork native process for " + arguments.front());
    if (child == 0) {
        if (dup2(descriptor, STDOUT_FILENO) < 0) _exit(126);
        if (close(descriptor) < 0) _exit(126);
        execvp(argv.front(), argv.data());
        _exit(127);
    }
    require(close(descriptor) == 0, "cannot close native process output " + stdout_path.string());
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) throw Track1Error("cannot wait for native process " + arguments.front());
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 125;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
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

void require_valid_utf8(const std::string& value) {
    for (std::size_t position = 0U; position < value.size();) {
        const auto byte = static_cast<unsigned char>(value[position]);
        if (byte <= 0x7FU) {
            ++position;
            continue;
        }
        std::size_t width = 0U;
        unsigned int codepoint = 0U;
        if (byte >= 0xC2U && byte <= 0xDFU) {
            width = 2U;
            codepoint = byte & 0x1FU;
        } else if (byte >= 0xE0U && byte <= 0xEFU) {
            width = 3U;
            codepoint = byte & 0x0FU;
        } else if (byte >= 0xF0U && byte <= 0xF4U) {
            width = 4U;
            codepoint = byte & 0x07U;
        } else {
            throw Track1Error("invalid UTF-8 leading byte in JSON string");
        }
        require(position + width <= value.size(), "truncated UTF-8 sequence in JSON string");
        for (std::size_t offset = 1U; offset < width; ++offset) {
            const auto continuation = static_cast<unsigned char>(value[position + offset]);
            require((continuation & 0xC0U) == 0x80U, "invalid UTF-8 continuation in JSON string");
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        require(!(width == 3U && codepoint < 0x800U) && !(width == 4U && codepoint < 0x10000U) &&
                    !(codepoint >= 0xD800U && codepoint <= 0xDFFFU) && codepoint <= 0x10FFFFU,
                "invalid UTF-8 code point in JSON string");
        position += width;
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
        if (character == '"') {
            require_valid_utf8(value);
            return value;
        }
        if (character != '\\') {
            require(static_cast<unsigned char>(character) >= 0x20U, "unescaped JSON control character");
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
            const auto parse_escape_code_unit = [&](std::size_t& cursor) {
                require(cursor + 4U <= text.size(), "truncated JSON unicode escape");
                unsigned int code_unit = 0U;
                for (std::size_t index = 0U; index < 4U; ++index) code_unit = (code_unit << 4U) | hex_digit(text[cursor++]);
                return code_unit;
            };
            const auto codepoint = parse_escape_code_unit(position);
            if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                require(position + 2U <= text.size() && text[position] == '\\' && text[position + 1U] == 'u', "missing JSON low surrogate");
                position += 2U;
                const auto low_surrogate = parse_escape_code_unit(position);
                require(low_surrogate >= 0xDC00U && low_surrogate <= 0xDFFFU, "invalid JSON low surrogate");
                append_codepoint(value, 0x10000U + ((codepoint - 0xD800U) << 10U) + (low_surrogate - 0xDC00U));
            } else {
                require(codepoint < 0xDC00U || codepoint > 0xDFFFU, "unexpected JSON low surrogate");
                append_codepoint(value, codepoint);
            }
        } else throw Track1Error("unsupported JSON escape");
    }
    throw Track1Error("unterminated JSON string");
}

std::size_t matching_delimiter(const std::string& text, const std::size_t start, const char open, const char close) {
    require(start < text.size() && text[start] == open, "JSON delimiter start is invalid");
    std::vector<char> delimiters{open};
    constexpr std::size_t maximum_json_depth = 128U;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t position = start + 1U; position < text.size(); ++position) {
        const char character = text[position];
        if (in_string) {
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') in_string = false;
            continue;
        }
        if (character == '"') {
            in_string = true;
        } else if (character == '{' || character == '[') {
            require(delimiters.size() < maximum_json_depth, "JSON nesting depth exceeds limit");
            delimiters.push_back(character);
        } else if (character == '}' || character == ']') {
            require(!delimiters.empty() && ((delimiters.back() == '{' && character == '}') ||
                                            (delimiters.back() == '[' && character == ']')),
                    "mismatched JSON delimiter");
            delimiters.pop_back();
            if (delimiters.empty()) {
                require(character == close, "JSON delimiter closed with the wrong type");
                return position;
            }
        }
    }
    throw Track1Error("unterminated JSON delimiter");
}

struct JsonSpan {
    bool found = false;
    std::size_t start = 0U;
    std::size_t end = 0U;
};

bool valid_json_number(const std::string& token) {
    std::size_t position = 0U;
    if (position < token.size() && token[position] == '-') ++position;
    if (position >= token.size()) return false;
    if (token[position] == '0') {
        ++position;
        if (position < token.size() && std::isdigit(static_cast<unsigned char>(token[position])) != 0) return false;
    } else {
        if (std::isdigit(static_cast<unsigned char>(token[position])) == 0) return false;
        while (position < token.size() && std::isdigit(static_cast<unsigned char>(token[position])) != 0) ++position;
    }
    if (position < token.size() && token[position] == '.') {
        ++position;
        const auto fraction_start = position;
        while (position < token.size() && std::isdigit(static_cast<unsigned char>(token[position])) != 0) ++position;
        if (position == fraction_start) return false;
    }
    if (position < token.size() && (token[position] == 'e' || token[position] == 'E')) {
        ++position;
        if (position < token.size() && (token[position] == '+' || token[position] == '-')) ++position;
        const auto exponent_start = position;
        while (position < token.size() && std::isdigit(static_cast<unsigned char>(token[position])) != 0) ++position;
        if (position == exponent_start) return false;
    }
    return position == token.size();
}

std::size_t skip_json_value(const std::string& text, std::size_t position) {
    position = skip_space(text, position);
    require(position < text.size(), "JSON value is missing");
    if (text[position] == '"') {
        static_cast<void>(parse_json_string(text, position));
        return position;
    }
    if (text[position] == '{' || text[position] == '[') {
        const auto close = text[position] == '{' ? '}' : ']';
        return matching_delimiter(text, position, text[position], close) + 1U;
    }
    const auto start = position;
    while (position < text.size() && text[position] != ',' && text[position] != '}' && text[position] != ']' &&
           std::isspace(static_cast<unsigned char>(text[position])) == 0)
        ++position;
    require(position > start, "JSON primitive value is empty");
    const auto primitive = text.substr(start, position - start);
    require(primitive == "true" || primitive == "false" || primitive == "null" || valid_json_number(primitive),
            "unsupported JSON primitive value");
    return position;
}

JsonSpan json_field_span(const std::string& object, const std::string& key, const bool required = true) {
    auto position = skip_space(object, 0U);
    require(position < object.size() && object[position] == '{', "JSON object is missing");
    const auto object_end = matching_delimiter(object, position, '{', '}');
    ++position;
    JsonSpan result;
    bool expect_member = true;
    while (position < object_end) {
        position = skip_space(object, position);
        require(position < object_end, "JSON object has a trailing comma");
        require(expect_member && object[position] == '"', "JSON object member key is invalid");
        const auto member_key = parse_json_string(object, position);
        position = skip_space(object, position);
        require(position < object_end && object[position] == ':', "JSON object member colon is missing");
        ++position;
        const auto value_start = skip_space(object, position);
        const auto value_end = skip_json_value(object, value_start);
        if (member_key == key) {
            require(!result.found, "duplicate JSON object member: " + key);
            result = {true, value_start, value_end};
        }
        position = skip_space(object, value_end);
        if (position == object_end) {
            expect_member = false;
            break;
        }
        require(position < object_end && object[position] == ',', "JSON object member separator is missing");
        ++position;
        expect_member = true;
    }
    require(!expect_member, "JSON object is empty or malformed");
    require(result.found || !required, "missing JSON field " + key);
    return result;
}

void validate_json_document(const std::string& document) {
    constexpr std::size_t maximum_json_bytes = 512U * 1024U * 1024U;
    require(document.size() <= maximum_json_bytes, "JSON document exceeds the Track 1 size limit");
    const auto start = skip_space(document, 0U);
    require(start < document.size() && document[start] == '{', "JSON document root must be an object");
    const auto end = matching_delimiter(document, start, '{', '}');
    require(skip_space(document, end + 1U) == document.size(), "JSON document has trailing data");
}

std::string field_string(const std::string& object, const std::string& key, const bool required = true) {
    const auto field = json_field_span(object, key, required);
    if (!field.found) return {};
    auto position = field.start;
    const auto value = parse_json_string(object, position);
    require(position == field.end, "JSON string field has trailing value data: " + key);
    return value;
}

std::string nested_object(const std::string& object, const std::string& key) {
    const auto field = json_field_span(object, key);
    require(field.start < object.size() && object[field.start] == '{', "JSON field is not an object: " + key);
    require(matching_delimiter(object, field.start, '{', '}') + 1U == field.end, "JSON object field boundary is invalid: " + key);
    return object.substr(field.start, field.end - field.start);
}

std::string first_array_string(const std::string& object, const std::string& key) {
    const auto field = json_field_span(object, key);
    require(field.start < object.size() && object[field.start] == '[', "JSON field is not an array: " + key);
    const auto array_end = matching_delimiter(object, field.start, '[', ']');
    require(array_end + 1U == field.end, "JSON array field boundary is invalid: " + key);
    auto position = skip_space(object, field.start + 1U);
    if (position == array_end) return {};
    std::string first;
    bool first_value = true;
    while (position < array_end) {
        require(object[position] == '"', "JSON array string element is invalid: " + key);
        const auto value = parse_json_string(object, position);
        if (first_value) first = value;
        first_value = false;
        position = skip_space(object, position);
        if (position == array_end) break;
        require(object[position] == ',', "JSON array string separator is invalid: " + key);
        position = skip_space(object, position + 1U);
    }
    return first;
}

std::size_t first_array_integer(const std::string& object, const std::string& key) {
    const auto field = json_field_span(object, key);
    require(field.start < object.size() && object[field.start] == '[', "JSON field is not an array: " + key);
    const auto array_end = matching_delimiter(object, field.start, '[', ']');
    require(array_end + 1U == field.end, "JSON array field boundary is invalid: " + key);
    auto position = skip_space(object, field.start + 1U);
    if (position == array_end) return 0U;
    std::size_t first = 0U;
    bool first_value = true;
    while (position < array_end) {
        const auto start = position;
        while (position < array_end && std::isdigit(static_cast<unsigned char>(object[position])) != 0) ++position;
        require(position > start, "JSON array integer element is invalid: " + key);
        const auto value = static_cast<std::size_t>(std::stoull(object.substr(start, position - start)));
        if (first_value) first = value;
        first_value = false;
        position = skip_space(object, position);
        if (position == array_end) break;
        require(object[position] == ',', "JSON array integer separator is invalid: " + key);
        position = skip_space(object, position + 1U);
    }
    return first;
}

std::vector<std::string> row_objects(const std::string& page) {
    validate_json_document(page);
    const auto rows_field = json_field_span(page, "rows");
    require(rows_field.start < page.size() && page[rows_field.start] == '[', "Hugging Face rows field is not an array");
    const auto array_start = rows_field.start;
    const auto array_end = matching_delimiter(page, array_start, '[', ']');
    require(array_end + 1U == rows_field.end, "Hugging Face rows field boundary is invalid");
    std::vector<std::string> rows;
    std::size_t position = array_start + 1U;
    bool expect_value = true;
    while (position < array_end) {
        position = skip_space(page, position);
        if (position >= array_end) break;
        if (!expect_value) {
            require(page[position] == ',', "Hugging Face row separator is invalid");
            ++position;
            expect_value = true;
            continue;
        }
        require(page[position] == '{', "Hugging Face row is not an object");
        const auto end = matching_delimiter(page, position, '{', '}');
        rows.push_back(page.substr(position, end - position + 1U));
        position = end + 1U;
        expect_value = false;
    }
    require(!expect_value, "Hugging Face rows array has a trailing comma");
    return rows;
}

template <typename Callback>
std::size_t for_each_flat_data_object(const std::string& document, Callback&& callback) {
    validate_json_document(document);
    const auto data = json_field_span(document, "data");
    require(data.start < document.size() && document[data.start] == '[', "GEM SQuAD data field is not an array");
    const auto array_start = data.start;
    const auto array_end = matching_delimiter(document, array_start, '[', ']');
    require(array_end + 1U == data.end, "GEM SQuAD data field boundary is invalid");
    std::size_t position = array_start + 1U;
    std::size_t index = 0U;
    bool expect_value = true;
    while (position < array_end) {
        position = skip_space(document, position);
        if (position >= array_end) break;
        if (!expect_value) {
            require(document[position] == ',', "GEM SQuAD data separator is invalid");
            ++position;
            expect_value = true;
            continue;
        }
        require(document[position] == '{', "GEM SQuAD data item is not an object");
        const auto end = matching_delimiter(document, position, '{', '}');
        callback(index, document.substr(position, end - position + 1U));
        ++index;
        position = end + 1U;
        expect_value = false;
    }
    require(!expect_value, "GEM SQuAD data array has a trailing comma");
    return index;
}

std::string run_curl(const std::string& url, const std::filesystem::path& path, const bool acquire_remote) {
    const auto digest_path = std::filesystem::path(path.string() + ".sha256");
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0U) {
        if (std::filesystem::exists(digest_path) && std::filesystem::file_size(digest_path) > 0U) {
            std::istringstream digest_input(read_file(digest_path));
            std::string declared_digest;
            digest_input >> declared_digest;
            require(declared_digest == GovernedCorpus::content_sha256(read_file(path)),
                    "cached Hugging Face page digest mismatch: " + path.string());
            return read_file(path);
        }
        if (!acquire_remote) return read_file(path);
    }
    require(acquire_remote, "cached Hugging Face page is missing or lacks an integrity sidecar: " + path.string());
    require(url.rfind("https://", 0U) == 0U, "Track 1 acquisition URL must use HTTPS");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    const auto temporary = unique_temporary_path(path);
    const auto transfer_status = unique_temporary_path(std::filesystem::path(path.string() + ".status"));
    const std::vector<std::string> arguments{
        "curl", "--fail", "--location", "--silent", "--show-error", "--retry", "12", "--retry-all-errors",
        "--retry-max-time", "900", "--connect-timeout", "30", "--max-time", "960", "--user-agent",
        "CCT-ASE-Track1/1.0", "--output", temporary.string(), "--write-out", "%{http_code} %{size_download}\n", url};
    try {
        require(run_process(arguments, transfer_status) == 0, "Hugging Face acquisition failed for " + url);
        require(std::filesystem::exists(temporary) && std::filesystem::file_size(temporary) > 0U,
                "Hugging Face acquisition wrote an empty temporary file for " + url);
        std::istringstream status_input(read_file(transfer_status));
        unsigned int status_code = 0U;
        std::uintmax_t downloaded_bytes = 0U;
        require(static_cast<bool>(status_input >> status_code >> downloaded_bytes) && status_code >= 200U && status_code < 300U,
                "Hugging Face acquisition returned an invalid HTTP status for " + url);
        require(downloaded_bytes == std::filesystem::file_size(temporary), "Hugging Face byte count mismatch for " + url);
        const auto digest = GovernedCorpus::content_sha256(read_file(temporary));
        publish_file_atomically(temporary, path);
        write_file_atomic(digest_path, digest + "\n");
        std::error_code ignored;
        std::filesystem::remove(transfer_status, ignored);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        std::filesystem::remove(transfer_status, ignored);
        throw;
    }
    return read_file(path);
}

std::string extract_archive_member(const std::filesystem::path& archive, const std::string& member, const std::filesystem::path& path) {
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0U) return read_file(path);
    require(std::filesystem::exists(archive) && std::filesystem::file_size(archive) > 0U, "WikiText archive cache is missing: " + archive.string());
    require(!member.empty(), "WikiText archive member is missing");
    static constexpr std::array<std::string_view, 3U> allowed_members{
        "wikitext-2-raw/wiki.train.raw", "wikitext-2-raw/wiki.valid.raw", "wikitext-2-raw/wiki.test.raw"};
    require(std::find(allowed_members.begin(), allowed_members.end(), member) != allowed_members.end(),
            "WikiText archive member is not in the pinned allowlist");
    const auto temporary = unique_temporary_path(path);
    const std::vector<std::string> arguments{"unzip", "-p", archive.string(), member};
    try {
        require(run_process(arguments, temporary) == 0, "cannot extract WikiText archive member " + member);
        require(std::filesystem::exists(temporary) && std::filesystem::file_size(temporary) > 0U,
                "WikiText archive member is empty: " + member);
        publish_file_atomically(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
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
                "SQuAD answer offset does not match context for id " + example.id + " at codepoint " + std::to_string(example.source_answer_start));
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

std::string source_attestation_digest(const Track1Source& source) {
    return GovernedCorpus::content_sha256(source.source_id + "|" + source.dataset_id + "|" + source.config + "|" + source.split +
                                          "|" + source.revision + "|" + source.license + "|" + source.upstream_dataset_id +
                                          "|" + source.acquisition_type + "|" + source.raw_file_url + "|" + source.archive_member +
                                          "|" + source.raw_digest);
}

std::uint64_t stable_key(const std::string& id, const std::uint64_t seed) {
    const auto digest = GovernedCorpus::content_sha256(id + "|" + std::to_string(seed));
    return std::stoull(digest.substr(0U, 16U), nullptr, 16);
}

std::string manifest_body(const Track1Manifest& manifest) {
    std::ostringstream output;
    output << "{\"manifest_version\":\"" << json_escape(manifest.manifest_version) << "\",\"tokenizer_snapshot\":\""
           << json_escape(manifest.tokenizer_snapshot) << "\",\"pretrain_token_count_mode\":\"" << json_escape(manifest.pretrain_token_count_mode)
           << "\",\"selection_policy\":\"" << json_escape(manifest.selection_policy)
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
               << "\",\"raw_file_url\":\"" << json_escape(source.raw_file_url) << "\",\"archive_member\":\""
               << json_escape(source.archive_member) << "\",\"attestation_digest\":\"" << source.attestation_digest << "\"}";
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

Track1Source& source_at(std::vector<Track1Source>& sources, const std::string& id) {
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
        {"wikitext2_pretrain_train", "Salesforce/wikitext", "wikitext-2-raw-v1", "train", "b08601e04326c79dfdd32d625aee71d232d685c3", "CC BY-SA 3.0; GFDL metadata", "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=train", 36718, {}, {}, "hf_rows", {}, {}, {}},
        {"wikitext2_pretrain_validation", "Salesforce/wikitext", "wikitext-2-raw-v1", "validation", "b08601e04326c79dfdd32d625aee71d232d685c3", "CC BY-SA 3.0; GFDL metadata", "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=validation", 3760, {}, {}, "hf_rows", {}, {}, {}},
        {"wikitext2_pretrain_test", "Salesforce/wikitext", "wikitext-2-raw-v1", "test", "b08601e04326c79dfdd32d625aee71d232d685c3", "CC BY-SA 3.0; GFDL metadata", "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=test", 4358, {}, {}, "hf_rows", {}, {}, {}},
        {"squad2_sft_train_source", "GEM/squad_v2", "gem_data_split", "train", "67199807729e631955056c71c258b7acbee548a3", "CC BY-SA 4.0", "https://datasets-server.huggingface.co/rows?dataset=rajpurkar%2Fsquad_v2&config=squad_v2&split=train", 116397, {}, {}, "hf_gem_flat_file", {}, {}, {}},
        {"squad2_final_test_source", "GEM/squad_v2", "gem_data_split", "validation", "67199807729e631955056c71c258b7acbee548a3", "CC BY-SA 4.0", "https://datasets-server.huggingface.co/rows?dataset=rajpurkar%2Fsquad_v2&config=squad_v2&split=validation", 11873, {}, {}, "hf_gem_flat_file", {}, {}, {}}};
    const std::array<std::pair<std::string, std::string>, 3U> wikitext_members{{
        {"wikitext2_pretrain_train", "wikitext-2-raw/wiki.train.raw"},
        {"wikitext2_pretrain_validation", "wikitext-2-raw/wiki.valid.raw"},
        {"wikitext2_pretrain_test", "wikitext-2-raw/wiki.test.raw"}}};
    for (const auto& [source_id, archive_member] : wikitext_members) {
        auto& source = source_at(manifest_.sources, source_id);
        source.upstream_dataset_id = "Salesforce/wikitext";
        source.acquisition_type = "hf_zip_member";
        source.raw_file_url = "https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip?download=true";
        source.archive_member = archive_member;
    }
    auto& squad_train = source_at(manifest_.sources, "squad2_sft_train_source");
    auto& squad_final = source_at(manifest_.sources, "squad2_final_test_source");
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
        auto& source = source_at(manifest_.sources, source_id);
        const auto raw_path = std::filesystem::path(config_.output_root) / "raw" / source_path_component(source);
        const auto archive_path = std::filesystem::path(config_.output_root) / "raw" / "wikitext-2-raw-v1.zip";
        const bool use_archive = source.acquisition_type == "hf_zip_member" &&
                                 (config_.acquire_remote || (std::filesystem::exists(archive_path) && std::filesystem::file_size(archive_path) > 0U));
        std::ostringstream prepared;
        std::string raw_digest_material;
        std::size_t tokens = 0U;
        std::size_t rows = 0U;
        const auto source_rows = config_.source_row_limit == 0U ? source.total_rows : std::min(source.total_rows, config_.source_row_limit);
        const auto append_text = [&](const std::string& text) {
            ++report_.source_rows;
            const auto cap = split == Track1Split::PretrainTrain ? config_.pretrain_token_cap : std::numeric_limits<std::size_t>::max();
            if (tokens >= cap) return;
            // Track 1 V1 uses the byte-fallback tokenizer: each retained byte and the line delimiter are one token.
            const auto take = std::min(text.size(), cap - tokens);
            prepared.write(text.data(), static_cast<std::streamsize>(take));
            tokens += take;
            if (tokens < cap) {
                prepared.put('\n');
                ++tokens;
            }
            ++rows;
        };
        if (use_archive) {
            require(!source.raw_file_url.empty(), "WikiText archive URL is missing");
            static_cast<void>(run_curl(source.raw_file_url, archive_path, config_.acquire_remote));
            const auto text = extract_archive_member(archive_path, source.archive_member, raw_path.string() + ".txt");
            raw_digest_material = text;
            ++report_.source_pages;
            std::istringstream lines(text);
            std::string line;
            std::size_t line_index = 0U;
            while (line_index < source_rows && std::getline(lines, line)) {
                append_text(line);
                ++line_index;
            }
            require(line_index == source_rows, "WikiText archive member ended before its declared split size");
        } else {
            for (std::size_t offset = 0U; offset < source_rows; offset += config_.page_length) {
                const auto page = run_curl(page_url(source, offset, config_.page_length), raw_path.string() + "." + std::to_string(offset), config_.acquire_remote);
                raw_digest_material += page;
                ++report_.source_pages;
                for (const auto& wrapper : row_objects(page)) append_text(field_string(nested_object(wrapper, "row"), "text"));
            }
        }
        source.raw_digest = GovernedCorpus::content_sha256(raw_digest_material);
        source.attestation_digest = source_attestation_digest(source);
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
    auto& train_source = source_at(manifest_.sources, "squad2_sft_train_source");
    auto& final_source = source_at(manifest_.sources, "squad2_final_test_source");
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
        const auto observed_rows = for_each_flat_data_object(document, [&](const std::size_t index, const std::string& object) {
            if (index >= config_.squad_train_row_offset && index < config_.squad_train_row_offset + train_rows) process_train_object(object, false);
        });
        const auto expected_rows = config_.allow_small_fixture && config_.source_row_limit > 0U ? config_.source_row_limit : train_source.total_rows;
        require(observed_rows == expected_rows, "SQuAD direct training file row count mismatch: expected " + std::to_string(expected_rows) +
                                                ", observed " + std::to_string(observed_rows));
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
        const auto observed_rows = for_each_flat_data_object(document, [&](const std::size_t index, const std::string& object) {
            if (index >= config_.squad_final_test_row_offset && index < config_.squad_final_test_row_offset + final_rows) process_final_object(object, false);
        });
        const auto expected_rows = config_.allow_small_fixture && config_.source_row_limit > 0U ? config_.source_row_limit : final_source.total_rows;
        require(observed_rows == expected_rows, "SQuAD direct final-test file row count mismatch: expected " + std::to_string(expected_rows) +
                                                ", observed " + std::to_string(observed_rows));
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
    train_source.attestation_digest = source_attestation_digest(train_source);
    final_source.attestation_digest = source_attestation_digest(final_source);
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
    for (const auto& source : manifest_.sources) {
        require(!source.raw_digest.empty() && source.attestation_digest == source_attestation_digest(source),
                "Track 1 source attestation is missing or inconsistent for " + source.source_id);
    }
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
    output << "{\"metric_version\":\"" << evaluation_contract_.metric_version << "\",\"pretrain_task\":\"" << evaluation_contract_.pretrain_task
           << "\",\"pretrain_evaluation_scope\":\"" << evaluation_contract_.pretrain_evaluation_scope << "\",\"qa_task\":\"" << evaluation_contract_.qa_task
           << "\",\"pretrain_metrics\":" << json_array(evaluation_contract_.pretrain_metrics)
           << ",\"qa_metrics\":" << json_array(evaluation_contract_.qa_metrics) << ",\"required_slices\":" << json_array(evaluation_contract_.required_slices)
           << ",\"forbidden_behaviors\":" << json_array(evaluation_contract_.forbidden_behaviors) << "}\n";
    return output.str();
}

}  // namespace cct
