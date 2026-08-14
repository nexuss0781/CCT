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

struct Check { std::string name; bool passed = false; std::string detail; };

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read architecture qualification report: " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    require(static_cast<bool>(input) || input.eof(), "cannot finish architecture qualification report");
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write architecture qualification gate: " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish architecture qualification gate");
}

std::string argument_path(const int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int index = 1; index + 1 < argc; ++index) if (argv[index] == name) return argv[index + 1];
    return fallback;
}

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
    std::size_t count = 0U;
    std::size_t position = 0U;
    while ((position = text.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

double number_value(const std::string& text, const std::string& key) {
    const auto marker = "\"" + key + "\":";
    const auto start = text.find(marker);
    require(start != std::string::npos, "qualification report is missing numeric field: " + key);
    const auto value_start = start + marker.size();
    std::size_t end = value_start;
    while (end < text.size() && text[end] != ',' && text[end] != '}' && text[end] != '\n') ++end;
    try {
        std::size_t consumed = 0U;
        const auto value = std::stod(text.substr(value_start, end - value_start), &consumed);
        require(consumed > 0U, "qualification numeric field is empty: " + key);
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("qualification numeric field is invalid: " + key);
    }
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const auto report_path = argument_path(argc, argv, "--report", "/tmp/cct-architecture-qualification/report.json");
        const auto output_root = argument_path(argc, argv, "--output", "/tmp/cct-architecture-qualification/gate");
        const auto report = read_file(report_path);
        std::vector<Check> checks;
        const auto add = [&](const std::string& name, const bool passed, const std::string& detail) { checks.push_back({name, passed, detail}); };
        add("report_complete", contains(report, "\"status\":\"COMPLETE\""), "qualification report completed all requested model trials");
        add("real_data_contract", number_value(report, "train_bytes") > 0.0 && number_value(report, "train_model_tokens") > 0.0 &&
                                  number_value(report, "validation_model_tokens") > 0.0 && number_value(report, "test_model_tokens") > 0.0,
            "raw bytes and frozen-tokenizer model-token counts are recorded");
        add("matched_contract", number_value(report, "steps") > 0.0 && number_value(report, "batch_size") > 0.0 &&
                                number_value(report, "context_length") > 1.0 && number_value(report, "embedding_dim") > 0.0 &&
                                number_value(report, "hidden_dim") > 0.0 && number_value(report, "seed") >= 0.0,
            "steps, batch, context, widths, and seed are recorded");
        add("compact_vocab_accounting", contains(report, "\"vocabulary_mode\":\"compact\"") &&
                                         number_value(report, "active_vocabulary_size") < number_value(report, "token_id_limit") + 1.0,
            "compact vocabulary mode records a smaller active slot allocation");
        add("all_architectures_present", count_occurrences(report, "\"model\":\"") == 4U && contains(report, "track1_cct_recurrence") &&
                                             contains(report, "gru") && contains(report, "diagonal_ssm") && contains(report, "dense_causal_attention"),
            "CCT, GRU, diagonal SSM, and causal attention are all evaluated");
        add("finite_results", count_occurrences(report, "\"finite\":true") == 4U, "all four model trials have finite metrics");
        add("validation_improvement", count_occurrences(report, "\"validation_improved\":true") == 4U, "all four models improve held-out validation loss");
        add("test_improvement", count_occurrences(report, "\"test_improved\":true") == 4U, "all four models improve frozen test loss");
        add("efficiency_metrics", count_occurrences(report, "\"parameter_count\":") == 4U && count_occurrences(report, "\"state_memory_bytes\":") == 4U &&
                                  count_occurrences(report, "\"target_tokens_per_second\":") == 4U,
            "parameter, state-memory, and target-token throughput metrics are recorded for every model");
        add("coherent_generation", !contains(report, "\"repetitive\":true") &&
                                      count_occurrences(report, "\"generated_tokens\":24") == 12U &&
                                      contains(report, "\"decoding\":\"greedy_and_deterministic_no_repeat_2gram_3gram_top64\""),
            "every production deterministic no-repeat continuation is full-length and non-repetitive; greedy baselines remain reported");
        std::size_t passed = 0U;
        for (const auto& check : checks) if (check.passed) ++passed;
        const bool all_passed = passed == checks.size();
        std::ostringstream checks_json;
        checks_json << "{\"status\":\"" << (all_passed ? "PASS" : "FAIL") << "\",\"passed\":" << passed << ",\"total\":" << checks.size() << ",\"checks\":[";
        std::ostringstream markdown;
        markdown << "# Architecture Qualification Gate\n\n**Status:** `" << (all_passed ? "PASS" : "FAIL") << "`\n**Checks:** " << passed << "/" << checks.size() << "\n\n";
        for (std::size_t index = 0U; index < checks.size(); ++index) {
            if (index > 0U) checks_json << ',';
            const auto& check = checks[index];
            checks_json << "{\"name\":\"" << check.name << "\",\"status\":\"" << (check.passed ? "PASS" : "FAIL")
                        << "\",\"detail\":\"" << check.detail << "\"}";
            markdown << "- `" << (check.passed ? "PASS" : "FAIL") << "` **" << check.name << "** — " << check.detail << "\n";
        }
        checks_json << "]}\n";
        const auto output_directory = std::filesystem::path(output_root);
        write_file(output_directory / "checks.json", checks_json.str());
        write_file(output_directory / "report.md", markdown.str());
        std::cout << checks_json.str();
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "architecture qualification gate error: " << error.what() << '\n';
        return 2;
    }
}
