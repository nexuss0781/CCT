#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kBoundary = 8U;
constexpr std::uint32_t kByteFirst = 256U;
constexpr std::uint64_t kMagic = 0x314750544343ULL; // CCTGP1

struct Writer {
    std::ofstream stream;
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
    std::uint64_t token_limit = 0;
    explicit Writer(const std::string& path, const std::uint64_t limit) : stream(path, std::ios::binary), token_limit(limit) {
        if (!stream) throw std::runtime_error("cannot open output " + path);
        stream.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }
    void token(const std::uint32_t value) {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
        if (!stream) throw std::runtime_error("dataset output write failed");
        ++count;
        bytes += sizeof(value);
    }
    bool bytes_text(const std::string& text) {
        const auto required = static_cast<std::uint64_t>(text.size()) + 2U;
        if (token_limit != 0U && count + required > token_limit) return false;
        token(kBoundary);
        for (const unsigned char byte : text) token(kByteFirst + byte);
        token(kBoundary);
        return true;
    }
    void close() {
        stream.seekp(static_cast<std::streamoff>(sizeof(kMagic)), std::ios::beg);
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        stream.close();
    }
};

std::uint64_t fnv1a(const std::string_view value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string trim(std::string value) {
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\r' || value.back() == '\n')) value.pop_back();
    std::size_t start = 0;
    while (start < value.size() && (value[start] == ' ' || value[start] == '\t' || value[start] == '\r' || value[start] == '\n')) ++start;
    return value.substr(start);
}

std::string strip_xml(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    bool tag = false;
    for (const char character : value) {
        if (character == '<') { tag = true; continue; }
        if (character == '>') { tag = false; result.push_back(' '); continue; }
        if (!tag) result.push_back(character);
    }
    const std::vector<std::pair<std::string, std::string>> entities{
        {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}
    };
    for (const auto& [from, to] : entities) {
        std::size_t position = 0;
        while ((position = result.find(from, position)) != std::string::npos) {
            result.replace(position, from.size(), to);
            position += to.size();
        }
    }
    return trim(result);
}

std::string json_string(const std::string& line, const std::string& key) {
    const std::string marker = "\"" + key + "\":";
    const auto marker_position = line.find(marker);
    if (marker_position == std::string::npos) return {};
    std::size_t position = marker_position + marker.size();
    while (position < line.size() && (line[position] == ' ' || line[position] == '\t')) ++position;
    if (position >= line.size() || line[position] != '\"') return {};
    ++position;
    std::string result;
    bool escape = false;
    for (; position < line.size(); ++position) {
        const char character = line[position];
        if (escape) {
            switch (character) {
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                default: result.push_back(character); break;
            }
            escape = false;
        } else if (character == '\\') {
            escape = true;
        } else if (character == '"') {
            break;
        } else {
            result.push_back(character);
        }
    }
    return result;
}

bool json_bool(const std::string& line, const std::string& key, const bool fallback) {
    const std::string marker = "\"" + key + "\":";
    const auto position = line.find(marker);
    if (position == std::string::npos) return fallback;
    return line.find("true", position + marker.size()) == position + marker.size();
}

void write_manifest(const std::string& path, const std::string& source, const std::string& train_path,
                    const std::string& valid_path, const std::string& test_path, const Writer& train,
                    const Writer& valid, const Writer& test, const std::uint64_t documents) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write manifest " + path);
    output << "{\n  \"source\":\"" << source << "\",\n  \"token_encoding\":\"stage10-byte-fallback-v1\",\n"
           << "  \"boundary_id\":8,\n  \"byte_first_id\":256,\n  \"documents\":" << documents << ",\n"
           << "  \"train\":{\"path\":\"" << train_path << "\",\"tokens\":" << train.count << "},\n"
           << "  \"validation\":{\"path\":\"" << valid_path << "\",\"tokens\":" << valid.count << "},\n"
           << "  \"test\":{\"path\":\"" << test_path << "\",\"tokens\":" << test.count << "}\n}\n";
}

int prepare_wiki(const std::string& prefix, const std::uint64_t max_train, const std::uint64_t max_valid,
                 const std::uint64_t max_test) {
    Writer train(prefix + ".train.bin", max_train);
    Writer valid(prefix + ".validation.bin", max_valid);
    Writer test(prefix + ".test.bin", max_test);
    std::string line;
    std::string article;
    bool inside_text = false;
    std::uint64_t documents = 0;
    std::uint64_t candidates = 0;
    std::uint64_t max_documents = std::numeric_limits<std::uint64_t>::max();
    while (std::getline(std::cin, line)) {
        const auto open = line.find("<text");
        if (open != std::string::npos) {
            inside_text = true;
            article.clear();
            const auto end = line.find('>', open);
            if (end != std::string::npos) article += line.substr(end + 1U);
        } else if (inside_text) {
            article += '\n';
            article += line;
        }
        if (inside_text) {
            const auto close = article.find("</text>");
            if (close != std::string::npos) {
                const auto text = strip_xml(article.substr(0, close));
                if (text.size() >= 128U && !text.empty()) {
                    const auto bucket = fnv1a(std::to_string(candidates)) % 100U;
                    ++candidates;
                    bool accepted = false;
                    if (bucket < 90U) accepted = train.bytes_text(text);
                    else if (bucket < 95U) accepted = valid.bytes_text(text);
                    else accepted = test.bytes_text(text);
                    if (accepted) ++documents;
                }
                inside_text = false;
                if (documents >= max_documents) break;
                if (max_train != 0U && max_valid != 0U && max_test != 0U &&
                    train.count >= max_train && valid.count >= max_valid && test.count >= max_test) break;
            }
        }
    }
    train.close(); valid.close(); test.close();
    write_manifest(prefix + ".manifest.json", "Wikimedia English multistream XML shard", prefix + ".train.bin", prefix + ".validation.bin", prefix + ".test.bin", train, valid, test, documents);
    std::cerr << "prepared Wikimedia documents=" << documents << " train_tokens=" << train.count << " validation_tokens=" << valid.count << " test_tokens=" << test.count << '\n';
    return 0;
}

int prepare_oasst(const std::string& prefix, const std::uint64_t max_train, const std::uint64_t max_valid,
                  const std::uint64_t max_test) {
    Writer train(prefix + ".train.bin", max_train);
    Writer valid(prefix + ".validation.bin", max_valid);
    Writer test(prefix + ".test.bin", max_test);
    std::string line;
    std::uint64_t documents = 0;
    while (std::getline(std::cin, line)) {
        const auto role = json_string(line, "role");
        const auto language = json_string(line, "lang");
        const auto text = trim(json_string(line, "text"));
        if (role != "assistant" || language != "en" || text.size() < 16U || json_bool(line, "deleted", false) || json_bool(line, "synthetic", false)) continue;
        const auto message_id = json_string(line, "message_id");
        const auto bucket = fnv1a(message_id.empty() ? std::to_string(documents) : message_id) % 100U;
        const auto formatted = std::string("<assistant> ") + text;
        bool accepted = false;
        if (bucket < 80U) accepted = train.bytes_text(formatted);
        else if (bucket < 90U) accepted = valid.bytes_text(formatted);
        else accepted = test.bytes_text(formatted);
        if (accepted) ++documents;
        if (max_train != 0U && max_valid != 0U && max_test != 0U &&
            train.count >= max_train && valid.count >= max_valid && test.count >= max_test) break;
    }
    train.close(); valid.close(); test.close();
    write_manifest(prefix + ".manifest.json", "OpenAssistant OASST1 pinned ready messages JSONL", prefix + ".train.bin", prefix + ".validation.bin", prefix + ".test.bin", train, valid, test, documents);
    std::cerr << "prepared OASST assistant messages=" << documents << " train_tokens=" << train.count << " validation_tokens=" << valid.count << " test_tokens=" << test.count << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 6) throw std::runtime_error("usage: prepare <wiki|oasst> <output-prefix> [max-train max-validation max-test]");
        std::uint64_t max_train = 0U;
        std::uint64_t max_valid = 0U;
        std::uint64_t max_test = 0U;
        if (argc == 6) {
            max_train = std::stoull(argv[3]);
            max_valid = std::stoull(argv[4]);
            max_test = std::stoull(argv[5]);
        }
        if (std::string(argv[1]) == "wiki") return prepare_wiki(argv[2], max_train, max_valid, max_test);
        if (std::string(argv[1]) == "oasst") return prepare_oasst(argv[2], max_train, max_valid, max_test);
        throw std::runtime_error("unknown preparation mode");
    } catch (const std::exception& error) {
        std::cerr << "prepare error: " << error.what() << '\n';
        return 2;
    }
}
