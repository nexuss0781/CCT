#include "cct/nlp_trainer.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Check {
    std::string name;
    bool passed = false;
    std::string detail;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read gate input: " + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    require(static_cast<bool>(input) || input.eof(), "cannot finish gate input: " + path.string());
    return output.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write gate output: " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish gate output: " + path.string());
}

std::string object_for(const std::string& json, const std::string& key) {
    const auto marker = "\"" + key + "\":{";
    const auto start = json.find(marker);
    require(start != std::string::npos, "JSON object is missing: " + key);
    const auto content_start = start + marker.size() - 1U;
    std::size_t depth = 0U;
    bool quoted = false;
    bool escaped = false;
    for (std::size_t index = content_start; index < json.size(); ++index) {
        const auto character = json[index];
        if (quoted) {
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') quoted = false;
            continue;
        }
        if (character == '"') quoted = true;
        else if (character == '{') ++depth;
        else if (character == '}') {
            require(depth > 0U, "JSON object depth underflow");
            --depth;
            if (depth == 0U) return json.substr(content_start, index - content_start + 1U);
        }
    }
    throw std::runtime_error("JSON object is unterminated: " + key);
}

std::string string_value(const std::string& json, const std::string& key) {
    const auto marker = "\"" + key + "\":\"";
    const auto start = json.find(marker);
    require(start != std::string::npos, "JSON string is missing: " + key);
    const auto value_start = start + marker.size();
    const auto end = json.find('"', value_start);
    require(end != std::string::npos, "JSON string is unterminated: " + key);
    return json.substr(value_start, end - value_start);
}

double number_value(const std::string& json, const std::string& key) {
    const auto marker = "\"" + key + "\":";
    const auto start = json.find(marker);
    require(start != std::string::npos, "JSON number is missing: " + key);
    const auto value_start = start + marker.size();
    std::size_t end = value_start;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n') ++end;
    try {
        std::size_t consumed = 0U;
        const auto value = std::stod(json.substr(value_start, end - value_start), &consumed);
        require(consumed > 0U, "JSON number is empty: " + key);
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("JSON number is invalid: " + key);
    }
}

bool bool_value(const std::string& json, const std::string& key) {
    const auto marker = "\"" + key + "\":";
    const auto start = json.find(marker);
    require(start != std::string::npos, "JSON boolean is missing: " + key);
    const auto value_start = start + marker.size();
    if (json.compare(value_start, 4U, "true") == 0) return true;
    if (json.compare(value_start, 5U, "false") == 0) return false;
    throw std::runtime_error("JSON boolean is invalid: " + key);
}

std::filesystem::path argument_path(const int argc, char** argv, const std::string& expected, const std::filesystem::path& fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == expected) return argv[index + 1];
    }
    return fallback;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto report_path = argument_path(argc, argv, "--report", "artifacts/english/acquisition/evaluation_report.json");
        const auto checkpoint_path = argument_path(argc, argv, "--checkpoint", "artifacts/english/acquisition/english_checkpoint.bin");
        const auto output_root = argument_path(argc, argv, "--output", "artifacts/english/acquisition/gate");
        const auto report = read_file(report_path);
        const auto checkpoint = read_file(checkpoint_path);
        std::vector<Check> checks;
        const auto add = [&](const std::string& name, const bool passed, const std::string& detail) {
            checks.push_back({name, passed, detail});
        };
        const auto trained_blimp = object_for(report, "trained_blimp");
        const auto control_blimp = object_for(report, "control_blimp");
        const auto cola = object_for(report, "cola_preference");
        const auto control_validation = object_for(report, "control_validation");
        const auto after_validation = object_for(report, "after_validation");
        const auto control_test = object_for(report, "control_test");
        const auto trained_test = object_for(report, "trained_test");
        const auto checkpoint_meta = object_for(report, "checkpoint");
        add("status_pass", string_value(report, "status") == "PASS", "runner status is PASS");
        add("native_backend", string_value(report, "backend") == "native-c++20-track1-cct-recurrence", "native C++ backend is recorded");
        add("source_identity", !string_value(report, "manifest_hash").empty() && !string_value(report, "tokenizer_hash").empty() &&
                                   !string_value(report, "dataset_hash").empty() && !string_value(report, "cola_dataset_hash").empty(),
            "source, tokenizer, WikiText, and CoLA identities are present");
        add("full_blimp_coverage", number_value(trained_blimp, "files") == 67.0 && number_value(trained_blimp, "pairs") >= 6700.0,
            "all 67 BLiMP files and at least 6,700 pairs are scored");
        add("blimp_finite_above_chance", std::isfinite(number_value(trained_blimp, "accuracy")) && number_value(trained_blimp, "accuracy") >= 0.50,
            "trained BLiMP accuracy is finite and at least chance");
        add("blimp_beats_control", number_value(trained_blimp, "preferred") > number_value(control_blimp, "preferred"),
            "trained BLiMP preference count beats the matched no-training control");
        add("cola_beats_control", number_value(cola, "adapted_correct") > number_value(cola, "control_correct") &&
                                      number_value(cola, "adapted_correct") * 2.0 >= number_value(cola, "evaluation_pairs"),
            "adapted CoLA preference beats control and remains above chance");
        add("validation_loss_improves", bool_value(after_validation, "finite") &&
                                         number_value(after_validation, "cross_entropy") < number_value(control_validation, "cross_entropy"),
            "held-out WikiText validation loss improves");
        add("frozen_test_improves", bool_value(trained_test, "finite") &&
                                      number_value(trained_test, "cross_entropy") < number_value(control_test, "cross_entropy"),
            "frozen WikiText test loss improves");
        add("checkpoint_hash", cct::nlp_checkpoint_hash(checkpoint) == string_value(checkpoint_meta, "sha256"), "checkpoint SHA-256 matches report identity");
        add("checkpoint_nonempty", !checkpoint.empty() && std::filesystem::file_size(checkpoint_path) > 0U, "checkpoint is durable and non-empty");
        add("side_effect_isolation", bool_value(report, "external_actions") == false, "external actions are disabled");
        add("evaluation_only_boundary", bool_value(report, "evaluation_only") == true, "final report was produced through evaluation-only scoring");
        add("generation_validity", number_value(object_for(report, "generation"), "nonempty") == number_value(object_for(report, "generation"), "prompts") &&
                                       number_value(object_for(report, "generation"), "valid_utf8") == number_value(object_for(report, "generation"), "prompts"),
            "bounded generation outputs are non-empty and valid UTF-8");
        std::size_t passed = 0U;
        for (const auto& check : checks) if (check.passed) ++passed;
        const bool all_passed = passed == checks.size();
        std::ostringstream checks_json;
        checks_json << "{\"status\":\"" << (all_passed ? "PASS" : "FAIL") << "\",\"passed\":" << passed << ",\"total\":" << checks.size() << ",\"checks\":[";
        for (std::size_t index = 0U; index < checks.size(); ++index) {
            if (index > 0U) checks_json << ',';
            checks_json << "{\"name\":\"" << checks[index].name << "\",\"status\":\"" << (checks[index].passed ? "PASS" : "FAIL")
                        << "\",\"detail\":\"" << checks[index].detail << "\"}";
        }
        checks_json << "]}\n";
        write_file(output_root / "checks.json", checks_json.str());
        std::ostringstream markdown;
        markdown << "# L1 English Acquisition Gate\n\n**Status:** `" << (all_passed ? "PASS" : "FAIL") << "`\n**Checks:** " << passed << "/" << checks.size() << "\n\n";
        for (const auto& check : checks) markdown << "- `" << (check.passed ? "PASS" : "FAIL") << "` **" << check.name << "** — " << check.detail << "\n";
        write_file(output_root / "report.md", markdown.str());
        std::cout << checks_json.str();
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "l1 english gate error: " << error.what() << '\n';
        return 2;
    }
}
