#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t kMagic = 0x314750544343ULL;
constexpr std::uint32_t kBoundary = 8U;
constexpr std::uint32_t kVocab = 512U;
struct Summary { std::uint64_t tokens = 0; std::uint64_t boundaries = 0; };
Summary validate_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::uint64_t magic = 0;
    std::uint64_t declared = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    input.read(reinterpret_cast<char*>(&declared), sizeof(declared));
    if (magic != kMagic) throw std::runtime_error("invalid stream magic " + path);
    Summary summary;
    for (std::uint64_t index = 0; index < declared; ++index) {
        std::uint32_t token = 0;
        input.read(reinterpret_cast<char*>(&token), sizeof(token));
        if (!input) throw std::runtime_error("truncated stream " + path);
        if (token >= kVocab) throw std::runtime_error("token outside vocabulary " + path);
        ++summary.tokens;
        if (token == kBoundary) ++summary.boundaries;
    }
    if (summary.tokens < 3U || summary.boundaries < 2U) throw std::runtime_error("stream has no complete document sequence " + path);
    return summary;
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: validate stream...");
        for (int index = 1; index < argc; ++index) {
            const auto result = validate_file(argv[index]);
            std::cout << argv[index] << " tokens=" << result.tokens << " boundaries=" << result.boundaries << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "validate error: " << error.what() << '\n';
        return 2;
    }
}
