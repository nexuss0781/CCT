#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "required documentation file is missing: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void test_authority_and_stage_contracts() {
    const auto root = std::filesystem::current_path();
    const auto goal = read_file(root / "SPEC" / "Goal.md");
    const auto todo = read_file(root / "SPEC" / "Todo.md");
    const auto status = read_file(root / "SPEC" / "Status.md");
    const auto architecture = read_file(root / "Architecture.md");
    require(goal.find("Native C++20") != std::string::npos && goal.find("Level 1") != std::string::npos,
            "canonical goal does not declare the native Level 1 contract");
    require(todo.find("SPEC/Goal.md") != std::string::npos && todo.find("A checkbox is not evidence") != std::string::npos,
            "canonical todo does not point to the goal evidence rule");
    require(status.find("CCT Level 1 Status Authority") != std::string::npos && status.find("gate_envelope.json") != std::string::npos,
            "status authority does not declare the identity envelope");
    require(architecture.find("C++20") != std::string::npos && architecture.find("checkpoint-backed") != std::string::npos,
            "architecture does not describe the current native implementation");
    require(todo.find("Completed remediation register") != std::string::npos && todo.find("Build and sanitizer validation") != std::string::npos,
            "canonical todo does not retain the completed remediation authority");
    for (unsigned int stage = 0U; stage <= 17U; ++stage) {
        const auto prefix = (stage < 10U ? "0" : "") + std::to_string(stage) + "_";
        bool found = false;
        for (const auto& entry : std::filesystem::directory_iterator(root / "Stages")) {
            if (entry.path().filename().string().find(prefix) == 0U) {
                found = true;
                break;
            }
        }
        require(found, "stage contract is missing for Stage " + std::to_string(stage));
        require(status.find("| " + std::to_string(stage) + " |") != std::string::npos,
                "status authority is missing Stage " + std::to_string(stage));
    }
}

}  // namespace

int main() {
    try {
        test_authority_and_stage_contracts();
        std::cout << "PASS authority_and_stage_contracts\nSUMMARY 1/1 passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "FAIL authority_and_stage_contracts: " << error.what() << "\nSUMMARY 0/1 passed\n";
        return 1;
    }
}
