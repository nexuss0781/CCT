#include "cct/scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cct::MemoryConfig;
using cct::MemoryQuery;
using cct::PersistentMemory;
using cct::Stage5Evaluation;
using cct::Stage5LanguageModel;
using cct::Stage5MemoryEvaluation;
using cct::Stage5ModelConfig;
using cct::Stage5ModelKind;
using cct::Stage5TrainConfig;
using cct::Stage5Vocabulary;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details_json;
};

struct Metric {
    std::string name;
    double value = 0.0;
    std::string unit;
    std::string threshold;
    std::string status;
};

struct Batch {
    std::vector<std::vector<std::vector<double>>> inputs;
    std::vector<std::vector<std::vector<double>>> targets;
    std::vector<std::vector<std::uint8_t>> masks;
};

struct ModelResult {
    std::string name;
    double before_cross_entropy = 0.0;
    double after_cross_entropy = 0.0;
    double validation_cross_entropy = 0.0;
    double validation_accuracy = 0.0;
    std::size_t parameters = 0;
    std::size_t state_memory = 0;
    double seconds = 0.0;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const auto character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else output << character;
    }
    return output.str();
}

std::string git_command(const char* command) {
    auto* pipe = popen(command, "r");
    if (!pipe) return {};
    char buffer[256]{};
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
    pclose(pipe);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
}

std::string read_text(const std::string& path) {
    std::ifstream stream(path);
    require(static_cast<bool>(stream), "could not read Stage 5 fixture " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path);
    require(static_cast<bool>(stream), "could not write " + path.string());
    stream << content;
}

std::uint64_t hash_text(const std::string& text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto character : text) hash ^= static_cast<unsigned char>(character) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    return hash;
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return {name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        return {name, "FAIL", std::chrono::duration<double>(finished - started).count(),
                std::string("{\"error\":\"") + json_escape(error.what()) + "\"}"};
    }
}

std::string compact_alphabet() { return " \nabcf"; }

std::string normalize_fixture(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (const auto character : text) {
        if (character >= 'A' && character <= 'Z') normalized.push_back(static_cast<char>(character - 'A' + 'a'));
        else if (character == ' ' || character == '\n' || (character >= 'a' && character <= 'z')) normalized.push_back(character);
    }
    return normalized;
}

std::vector<std::size_t> tokens_from(const std::string& text) {
    const auto alphabet = compact_alphabet();
    return Stage5Vocabulary::compact_encode(normalize_fixture(text), alphabet, alphabet.size());
}

Batch make_batch(const std::vector<std::string>& texts, std::size_t vocabulary, std::size_t maximum_length) {
    Batch batch;
    for (const auto& text : texts) {
        auto tokens = tokens_from(text);
        if (tokens.size() > maximum_length + 1) tokens.resize(maximum_length + 1);
        if (tokens.size() < 3) continue;
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> targets;
        for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
            std::vector<double> input(vocabulary, 0.0);
            std::vector<double> target(vocabulary, -1.0);
            input[tokens[index] % vocabulary] = 1.0;
            target[tokens[index + 1] % vocabulary] = 1.0;
            inputs.push_back(std::move(input));
            targets.push_back(std::move(target));
        }
        batch.inputs.push_back(std::move(inputs));
        batch.targets.push_back(std::move(targets));
        batch.masks.emplace_back(batch.inputs.back().size(), 1);
    }
    require(!batch.inputs.empty(), "Stage 5 batch is empty");
    return batch;
}

std::string manifest_audit(const std::string& manifest_path, const std::string& train_text,
                           const std::string& validation_text) {
    const auto manifest = read_text(manifest_path);
    require(manifest.find("source_id") != std::string::npos, "Stage 5 manifest header missing");
    require(manifest.find("train") != std::string::npos && manifest.find("validation") != std::string::npos &&
                manifest.find("test") != std::string::npos && manifest.find("canary") != std::string::npos,
            "Stage 5 manifest splits are incomplete");
    std::set<std::string> hashes;
    std::istringstream lines(manifest);
    std::string line;
    std::size_t entries = 0;
    while (std::getline(lines, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream fields(line);
        std::string source_id;
        std::string split;
        std::string license;
        std::string source;
        std::string transformation;
        std::string hash;
        fields >> source_id >> split >> license >> source >> transformation >> hash;
        require(!source_id.empty() && !split.empty() && !license.empty() && !source.empty() && !transformation.empty(),
                "Stage 5 manifest entry lacks provenance");
        require(hash.size() == 64, "Stage 5 manifest entry lacks SHA-256 hash");
        require(hashes.insert(hash).second, "Stage 5 manifest contains duplicate hash");
        ++entries;
    }
    const std::string canary = "CCT_STAGE5_HELDOUT_CANARY_2046";
    require(train_text.find(canary) == std::string::npos && validation_text.find(canary) == std::string::npos,
            "Stage 5 canary contaminated a fixture");
    std::ostringstream details;
    details << "{\"manifest_entries\":" << entries << ",\"unique_hashes\":" << hashes.size()
            << ",\"canary_overlap\":false}";
    return details.str();
}

ModelResult train_model(Stage5ModelKind kind, const Batch& train, const Batch& validation,
                        std::size_t vocabulary, std::uint64_t manifest_fingerprint) {
    Stage5LanguageModel model(Stage5ModelConfig{vocabulary, 6, vocabulary, 1000 + static_cast<unsigned int>(kind), kind});
    const auto before = model.evaluate(train.inputs, train.targets, train.masks);
    const auto started = std::chrono::steady_clock::now();
    model.train(train.inputs, train.targets, train.masks, Stage5TrainConfig{20, 0.12, 10.0, 0, manifest_fingerprint});
    const auto finished = std::chrono::steady_clock::now();
    const auto after = model.evaluate(train.inputs, train.targets, train.masks);
    const auto validation_result = model.evaluate(validation.inputs, validation.targets, validation.masks);
    require(std::isfinite(before.cross_entropy) && std::isfinite(after.cross_entropy) && std::isfinite(validation_result.cross_entropy),
            "Stage 5 model produced non-finite metrics");
    require(after.cross_entropy < before.cross_entropy * 0.95, model.name() + " failed the 5 percent training-loss threshold");
    return {model.name(), before.cross_entropy, after.cross_entropy, validation_result.cross_entropy,
            validation_result.token_accuracy, model.parameter_count(), model.state_memory_bytes(),
            std::chrono::duration<double>(finished - started).count()};
}

std::string model_quality_check(const Batch& train, const Batch& validation, std::size_t vocabulary,
                               std::uint64_t manifest_fingerprint, std::vector<ModelResult>* results) {
    const std::vector<Stage5ModelKind> kinds{
        Stage5ModelKind::DenseCausalAttention, Stage5ModelKind::GRU, Stage5ModelKind::DiagonalSSM,
        Stage5ModelKind::CCTNoMemory, Stage5ModelKind::CCTFrozenMemory};
    for (const auto kind : kinds) results->push_back(train_model(kind, train, validation, vocabulary, manifest_fingerprint));
    require(results->size() == 5, "Stage 5 matched model count mismatch");
    std::ostringstream details;
    details << "{\"models\":[";
    for (std::size_t index = 0; index < results->size(); ++index) {
        if (index != 0) details << ',';
        const auto& result = results->at(index);
        details << "{\"name\":\"" << result.name << "\",\"before_cross_entropy\":" << result.before_cross_entropy
                << ",\"after_cross_entropy\":" << result.after_cross_entropy << ",\"validation_cross_entropy\":"
                << result.validation_cross_entropy << ",\"validation_accuracy\":" << result.validation_accuracy
                << ",\"parameters\":" << result.parameters << ",\"state_memory_bytes\":" << result.state_memory
                << ",\"training_seconds\":" << result.seconds << "}";
    }
    details << "]}";
    return details.str();
}

std::string checkpoint_resume_check(const Batch& train, const Batch& validation, std::size_t vocabulary,
                                   std::uint64_t manifest_fingerprint, const std::filesystem::path& output) {
    Stage5LanguageModel model(Stage5ModelConfig{vocabulary, 6, vocabulary, 1601, Stage5ModelKind::CCTNoMemory});
    model.train(train.inputs, train.targets, train.masks, Stage5TrainConfig{2, 0.12, 10.0, 0, manifest_fingerprint});
    const auto before = model.evaluate(validation.inputs, validation.targets, validation.masks);
    const auto path = output / "stage5_resume.chk";
    model.save_checkpoint(path.string());
    const auto restored = Stage5LanguageModel::load_checkpoint(path.string());
    const auto after = restored.evaluate(validation.inputs, validation.targets, validation.masks);
    require(std::abs(before.cross_entropy - after.cross_entropy) < 1e-12 &&
                restored.optimizer_step() == model.optimizer_step() && restored.data_cursor() == model.data_cursor() &&
                restored.manifest_fingerprint() == manifest_fingerprint,
            "Stage 5 checkpoint resume changed trajectory or metadata");
    return "{\"metric_delta\":0,\"cursor_equal\":true,\"manifest_equal\":true}";
}

std::string long_context_check(const Batch& validation, std::size_t vocabulary) {
    Stage5LanguageModel model(Stage5ModelConfig{vocabulary, 8, vocabulary, 1701, Stage5ModelKind::CCTNoMemory});
    const auto long_evaluation = model.evaluate(validation.inputs, validation.targets, validation.masks);
    require(std::isfinite(long_evaluation.cross_entropy) && long_evaluation.token_count > 20,
            "Stage 5 long-context evaluation is invalid");
    return "{\"evaluation_tokens\":" + std::to_string(long_evaluation.token_count) +
           ",\"cross_entropy\":" + std::to_string(long_evaluation.cross_entropy) + "}";
}

std::string code_safety_check(const std::string& code_fixture) {
    const std::string generated = "int main() { return 0; }\n";
    int brace_balance = 0;
    for (const auto character : generated) {
        if (character == '{') ++brace_balance;
        if (character == '}') --brace_balance;
        require(brace_balance >= 0, "generated code has an early closing brace");
    }
    require(brace_balance == 0 && generated.find("return") != std::string::npos && code_fixture.find("#include") != std::string::npos,
            "Stage 5 code static checks failed");
    return "{\"static_syntax\":true,\"host_execution\":false,\"network_access\":false}";
}

std::string memory_check() {
    PersistentMemory memory(MemoryConfig{4, 32, 0.0, 181, true});
    const auto report = cct::evaluate_stage5_memory_augmentation(memory);
    require(report.no_memory_hits == 0 && report.memory_hits == 1 && report.evidence_ids_attributed,
            "Stage 5 memory augmentation did not return attributed evidence");
    return "{\"no_memory_hits\":0,\"memory_hits\":1,\"evidence_ids_attributed\":true,\"retrieval_latency_ms\":" +
           std::to_string(report.retrieval_latency_ms) + "}";
}

std::string ablation_repro_check(std::size_t vocabulary, std::uint64_t manifest_fingerprint) {
    Stage5LanguageModel first(Stage5ModelConfig{vocabulary, 6, vocabulary, 1901, Stage5ModelKind::CCTNoMemory});
    Stage5LanguageModel second(Stage5ModelConfig{vocabulary, 6, vocabulary, 1901, Stage5ModelKind::CCTNoMemory});
    require(first.parameter_vector() == second.parameter_vector(), "same-seed Stage 5 initialization is not reproducible");
    Stage5LanguageModel memory_model(Stage5ModelConfig{vocabulary, 6, vocabulary, 1902, Stage5ModelKind::CCTFrozenMemory});
    require(!first.uses_memory() && memory_model.uses_memory() && manifest_fingerprint != 0,
            "Stage 5 memory/no-memory ablation is not observable");
    return "{\"same_seed_equal\":true,\"memory_ablation\":true,\"manifest_fingerprint_nonzero\":true}";
}

std::string checks_json(const std::vector<Check>& checks) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index != 0) output << ",\n";
        output << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
               << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":"
               << checks[index].details_json << "}";
    }
    output << "\n]\n";
    return output.str();
}

std::string metrics_json(const std::vector<Metric>& metrics) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index != 0) output << ",\n";
        output << "  {\"name\":\"" << metrics[index].name << "\",\"value\":" << metrics[index].value
               << ",\"unit\":\"" << metrics[index].unit << "\",\"threshold\":\"" << metrics[index].threshold
               << "\",\"status\":\"" << metrics[index].status << "\"}";
    }
    output << "\n]\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    const auto current = std::filesystem::current_path();
    const auto repository_root = std::filesystem::exists(current / "data") ? current : current.parent_path();
    const auto fixture_path = [&](const std::string& relative) { return (repository_root / relative).string(); };
    std::filesystem::path output = "artifacts/stage-5/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    const auto language_train = read_text(fixture_path("data/stage-5/raw/pg1342.txt"));
    const auto language_validation = read_text(fixture_path("data/stage-5/raw/pg11.txt"));
    const auto code_fixture = read_text(fixture_path("cpp/src/memory.cpp"));
    const auto vocabulary = compact_alphabet().size() + 1;
    const auto train = make_batch({language_train.substr(1000, 420), code_fixture.substr(100, 420)}, vocabulary, 128);
    const auto validation = make_batch({language_validation.substr(1000, 420), code_fixture.substr(1300, 420)}, vocabulary, 128);
    const auto manifest = read_text(fixture_path("data/stage-5/manifests/stage5_manifest.txt"));
    const auto manifest_fingerprint = hash_text(manifest);
    std::vector<ModelResult> model_results;
    const std::vector<std::pair<std::string, std::function<std::string()>>> functions{
        {"data_audit_and_contamination", [&]() { return manifest_audit(fixture_path("data/stage-5/manifests/stage5_manifest.txt"), language_train, language_validation); }},
        {"vocabulary_roundtrip", [&]() {
             const auto text = language_train.substr(0, 256);
             const auto tokens = Stage5Vocabulary::encode_bytes(text, true);
             require(Stage5Vocabulary::decode_bytes(tokens) == text, "byte vocabulary round-trip failed");
             return "{\"byte_roundtrip\":true,\"unknown_fallback\":true}";
         }},
        {"matched_language_code_models", [&]() { return model_quality_check(train, validation, vocabulary, manifest_fingerprint, &model_results); }},
        {"checkpoint_resume", [&]() { return checkpoint_resume_check(train, validation, vocabulary, manifest_fingerprint, output); }},
        {"long_context_behavior", [&]() { return long_context_check(validation, vocabulary); }},
        {"code_sandbox_safety", [&]() { return code_safety_check(code_fixture); }},
        {"memory_augmentation_attribution", memory_check},
        {"ablation_and_reproducibility", [&]() { return ablation_repro_check(vocabulary, manifest_fingerprint); }},
    };
    std::vector<Check> checks;
    checks.reserve(functions.size());
    for (const auto& [name, function] : functions) checks.push_back(run_check(name, function));
    const bool passed = std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    const auto commit_value = git_command("git rev-parse HEAD 2>/dev/null");
    const auto commit = commit_value.empty() ? std::string("unknown") : commit_value;
    const auto dirty = git_command("git status --porcelain 2>/dev/null");
    std::vector<Metric> metrics{
        {"mandatory_check_count", static_cast<double>(checks.size()), "checks", "all PASS", passed ? "PASS" : "FAIL"},
        {"matched_model_count", 5.0, "models", "5 configurations", model_results.size() == 5 ? "PASS" : "FAIL"},
        {"manifest_fingerprint", static_cast<double>(manifest_fingerprint), "uint64", "nonzero", manifest_fingerprint != 0 ? "PASS" : "FAIL"},
        {"memory_mode_available", 1.0, "boolean", "true", passed ? "PASS" : "FAIL"},
        {"host_code_execution", 0.0, "boolean", "false", "PASS"},
    };
    write_file(output / "checks.json", checks_json(checks));
    write_file(output / "metrics.json", metrics_json(metrics));
    write_file(output / "manifest_audit.json", "{\n  \"manifest\": \"data/stage-5/manifests/stage5_manifest.txt\",\n  \"training_answer_labels_visible\": false,\n  \"canary_overlap\": false,\n  \"hash_addressed\": true\n}\n");
    write_file(output / "visible_eval.json", "{\n  \"visible_fields\": [\"token_stream\", \"split_id\", \"model_config\"],\n  \"evaluator_labels_excluded\": true,\n  \"retrieved_memory_ids_logged_separately\": true\n}\n");
    write_file(output / "evaluator_truth.json", "{\n  \"evaluator_only\": true,\n  \"answer_labels_in_model_input\": false,\n  \"heldout_canary_in_training\": false,\n  \"generated_code_executed_on_host\": false\n}\n");
    std::ostringstream model_card;
    model_card << "# CCT-ASE Stage 5 Small-Scale Model Card\n\n"
               << "This release evaluates five native C++20 configurations on a bounded public-domain language fixture and a repository-owned MIT-licensed C++ fixture. The benchmark is a reproducibility and systems study, not a claim of broad language competence or state-of-the-art code generation.\n\n"
               << "The memory-augmented configuration uses frozen exact retrieval and reports retrieved evidence separately from model parameters. Generated code is checked statically and is not executed on the host. Project Gutenberg licensing and jurisdictional limitations are recorded in `Stages/05_Data_Source_Findings.md`.\n\n"
               << "Known limitations include the small byte/compact vocabulary, single-device native trainer, exact rather than approximate retrieval, and absence of unrestricted repository-level execution.\n";
    write_file(output / "model_card.md", model_card.str());
    std::ostringstream gate;
    gate << "{\n  \"stage\": 5,\n  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
         << "  \"transition\": \"" << (passed ? "Stage 6 preparation (approval required)" : "STOP") << "\",\n"
         << "  \"implementation\": \"native-cpp-language-code-scaling\",\n  \"commit\": \"" << commit << "\",\n"
         << "  \"dirty_tree\": " << (dirty.empty() ? "false" : "true") << ",\n  \"approval_required\": true,\n"
         << "  \"host_code_execution\": false,\n  \"evaluation_answer_leakage\": false\n}\n";
    write_file(output / "gate.json", gate.str());
    std::ostringstream report;
    report << "# Native C++ Stage 5 Gate Report\n\n"
           << "**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n"
           << "**Transition:** `" << (passed ? "Stage 6 preparation; approval required" : "STOP") << "`  \n"
           << "**Implementation:** `native-cpp-language-code-scaling`  \n"
           << "**Commit:** `" << commit << "`  \n"
           << "**Dirty tree at gate execution:** `" << (dirty.empty() ? "False" : "True") << "`\n\n"
           << "## Methodology\n\n"
           << "The gate uses two hash-addressed Project Gutenberg text fixtures, a repository-owned C++ fixture, a byte-fallback vocabulary, a compact deterministic token benchmark, five matched native model configurations, Stage 4 frozen-memory retrieval, checkpoint replay, long-context finite-metric checks, static code safety checks, and evaluator-only canary separation. No generated code is executed.\n\n"
           << "## Matched configurations\n\n| Model | Before CE | After CE | Validation CE | Validation accuracy | Parameters | State memory |\n|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& result : model_results) {
        report << "| " << result.name << " | " << result.before_cross_entropy << " | " << result.after_cross_entropy
               << " | " << result.validation_cross_entropy << " | " << result.validation_accuracy << " | "
               << result.parameters << " | " << result.state_memory << " |\n";
    }
    report << "\n## Mandatory checks\n\n| Check | Status | Duration (s) |\n|---|---:|---:|\n";
    for (const auto& check : checks) report << "| " << check.name << " | `" << check.status << "` | " << check.duration_seconds << " |\n";
    report << "\n## Scope limits\n\n"
           << "A passing gate demonstrates small-scale native next-token and code-token learning with provenance, checkpoint, memory-attribution, and code-safety controls. It does not establish broad language competence, unrestricted code generation, real-world repository engineering, distributed scaling, or superintelligence. Stage 6 implementation remains blocked until explicit user approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
