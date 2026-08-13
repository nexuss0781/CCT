#include "cct/corpus.hpp"

#include <chrono>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character < 0x20U) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                                             << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        else output << static_cast<char>(character);
    }
    return output.str();
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void atomic_write(const std::filesystem::path& path, const std::string& content) {
    const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    std::filesystem::create_directories(parent);
    const auto temporary = path.string() + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create gate envelope temporary file");
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.flush();
        if (!output) throw std::runtime_error("cannot write gate envelope temporary file");
    }
    const auto descriptor = ::open(temporary.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0 || ::fsync(descriptor) != 0 || ::close(descriptor) != 0 || ::rename(temporary.c_str(), path.c_str()) != 0) {
        static_cast<void>(::unlink(temporary.c_str()));
        throw std::runtime_error("cannot publish gate envelope");
    }
    const auto directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory < 0 || ::fsync(directory) != 0 || ::close(directory) != 0) throw std::runtime_error("cannot sync gate envelope directory");
}

int run_child(const std::vector<std::string>& arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1U);
    for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    const auto child = ::fork();
    if (child < 0) throw std::runtime_error("cannot fork gate executable");
    if (child == 0) {
        ::execvp(argv.front(), argv.data());
        _exit(127);
    }
    int status = 0;
    if (::waitpid(child, &status, 0) < 0) throw std::runtime_error("cannot wait for gate executable");
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 125;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || std::string(argv[1]) != "--output" || std::string(argv[3]) != "--") return 125;
    const std::filesystem::path output(argv[2]);
    std::vector<std::string> command;
    for (int index = 4; index < argc; ++index) command.emplace_back(argv[index]);
    const auto started = std::chrono::system_clock::now();
    int exit_code = 125;
    try {
        exit_code = run_child(command);
        const auto executable = std::filesystem::absolute(command.front());
        const auto binary = read_file(executable);
        const auto checks = output / "checks.json";
        std::ostringstream envelope;
        envelope << "{\"schema\":\"cct-gate-envelope-v1\",\"status\":\"" << (exit_code == 0 ? "PASS" : "FAIL")
                 << "\",\"source_commit\":\"" << json_escape(CCT_SOURCE_COMMIT) << "\",\"compiler\":\""
                 << json_escape(std::string(CCT_COMPILER_ID) + " " + CCT_COMPILER_VERSION) << "\",\"build_type\":\""
                 << json_escape(CCT_BUILD_TYPE) << "\",\"test_binary\":\"" << json_escape(executable.string())
                 << "\",\"test_binary_sha256\":\"" << cct::GovernedCorpus::content_sha256(binary) << "\",\"output\":\""
                 << json_escape(output.string()) << "\",\"checks\":\"" << json_escape(checks.string()) << "\",\"exit_code\":"
                 << exit_code << ",\"started_epoch_milliseconds\":"
                 << std::chrono::duration_cast<std::chrono::milliseconds>(started.time_since_epoch()).count() << "}\n";
        atomic_write(output / "gate_envelope.json", envelope.str());
    } catch (...) {
        return 125;
    }
    return exit_code;
}
