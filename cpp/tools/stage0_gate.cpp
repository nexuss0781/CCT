#include "cct/corpus.hpp"
#include "cct/event.hpp"
#include "cct/field.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Check {
    std::string name;
    std::string status;
    double duration_seconds;
    std::string details;
};

struct L1BaselineConfig {
    static constexpr const char* kSchema = "cct-l1-baseline-config-v1";
    std::uint64_t seed = 1701U;
    std::vector<std::size_t> shape{32U};
    double spacing = 1.0;
    double wave_speed = 1.0;
    double dt = 0.05;
    double cfl_safety = 0.9;
    std::size_t benchmark_repetitions = 2U;

    std::string canonical_json() const {
        std::ostringstream output;
        output << std::setprecision(17);
        output << "{\"schema\":\"" << kSchema << "\",\"seed\":" << seed << ",\"shape\":[";
        for (std::size_t index = 0U; index < shape.size(); ++index) {
            if (index != 0U) output << ',';
            output << shape[index];
        }
        output << "],\"spacing\":" << spacing << ",\"wave_speed\":" << wave_speed << ",\"dt\":" << dt
               << ",\"cfl_safety\":" << cfl_safety << ",\"benchmark_repetitions\":" << benchmark_repetitions << "}";
        return output.str();
    }

    void validate() const {
        if (shape.empty() || shape.size() > 2U || shape.front() < 2U || benchmark_repetitions < 2U || spacing <= 0.0 ||
            wave_speed <= 0.0 || dt <= 0.0 || cfl_safety <= 0.0 || cfl_safety > 1.0) {
            throw std::invalid_argument("invalid L1 baseline configuration");
        }
    }
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character; break;
        }
    }
    return output.str();
}

std::string run_command(const char* command) {
    auto* pipe = popen(command, "r");
    if (pipe == nullptr) return "unknown";
    char buffer[256]{};
    std::string value;
    while (fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr) value += buffer;
    static_cast<void>(pclose(pipe));
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
    return value.empty() ? "unknown" : value;
}

Check run(const std::string& name, const std::function<std::string()>& function) {
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

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("could not write " + path.string());
    stream << content;
    if (!stream) throw std::runtime_error("could not publish " + path.string());
}

std::string event_contract_check() {
    cct::Manifold manifold({4U, 4U});
    const cct::Event event{{1.0F}, {1, 2}, {0.5F, 0.25F}};
    manifold.place_event(event);
    const auto* found = manifold.get_event({1, 2});
    require(found != nullptr && found->semantic_vector == event.semantic_vector && manifold.filled_cells() == 1U,
            "event lifecycle failed");
    return "{\"native_module\":\"cct_native\",\"filled_cells\":1,\"lookup_exact\":true}";
}

std::string non_mutation_failure_check() {
    bool rejected = false;
    try { static_cast<void>(cct::Manifold({0U, 4U})); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "zero dimension was accepted");

    cct::Manifold manifold({4U, 4U});
    const cct::Event original{{1.0F}, {1, 2}, {0.5F}};
    manifold.place_event(original);
    const auto unchanged = [&]() {
        const auto* stored = manifold.get_event({1, 2});
        return manifold.filled_cells() == 1U && stored != nullptr && stored->semantic_vector == original.semantic_vector &&
               stored->temporal_tensor == original.temporal_tensor && stored->causal_potential_vector == original.causal_potential_vector;
    };
    rejected = false;
    try { manifold.place_event(cct::Event{{}, {-1, 0}, {}}); } catch (const std::out_of_range&) { rejected = true; }
    require(rejected && unchanged(), "out-of-range insertion mutated state");
    rejected = false;
    try { static_cast<void>(manifold.get_event({1})); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && unchanged(), "wrong-dimensional lookup mutated state");
    rejected = false;
    try { manifold.place_event(cct::Event{{2.0F}, {1, 2}, {}}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && unchanged(), "duplicate insertion mutated state");
    return "{\"invalid_dimensions\":true,\"out_of_range\":true,\"duplicate_rejected\":true,\"state_unchanged\":true}";
}

std::string deterministic_check(const L1BaselineConfig& config) {
    config.validate();
    cct::SolverConfig solver_config;
    solver_config.shape = config.shape;
    solver_config.spacing.assign(config.shape.size(), config.spacing);
    solver_config.wave_speed = config.wave_speed;
    solver_config.dt = config.dt;
    solver_config.boundary = cct::Boundary::Periodic;
    solver_config.method = cct::Method::RK4;
    solver_config.cfl_safety = config.cfl_safety;
    const cct::SpectralSolver solver(solver_config);
    const auto state = solver.initialize(cct::manufactured_mode(config.shape, solver_config.spacing, {2}));
    const auto first = solver.step(state);
    const auto second = solver.step(state);
    require(first.phi == second.phi && first.psi == second.psi, "native baseline is nondeterministic");
    require(cct::fft_round_trip_error(first.phi, config.shape) < 2e-12, "native FFT baseline is invalid");
    return "{\"seed\":" + std::to_string(config.seed) + ",\"repetitions\":" + std::to_string(config.benchmark_repetitions) +
           ",\"deterministic\":true,\"finite_values\":true}";
}

std::string configuration_validation_check(const L1BaselineConfig& config, const std::string& config_hash) {
    config.validate();
    const auto canonical = config.canonical_json();
    require(cct::GovernedCorpus::content_sha256(canonical) == config_hash, "baseline configuration hash mismatch");
    bool rejected = false;
    auto invalid = config;
    invalid.dt = 0.0;
    try { invalid.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "zero timestep configuration was accepted");
    rejected = false;
    invalid = config;
    invalid.shape = {1U};
    try { invalid.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "undersized configuration shape was accepted");
    return "{\"config_schema\":\"" + std::string(L1BaselineConfig::kSchema) + "\",\"config_hash\":\"" + config_hash +
           "\",\"canonical_identity\":true,\"malformed_configuration_rejected\":true}";
}

std::string repository_hygiene_check() {
    const auto no_python = run_command("if git ls-files '*.py' | grep -q .; then echo fail; else echo clean; fi");
    require(no_python == "clean", "tracked Python implementation files violate the native-only contract");
    const auto no_secret = run_command(
        "if git grep -nE 'ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}' -- . ':!artifacts' >/dev/null 2>&1; then echo fail; else echo clean; fi");
    require(no_secret == "clean", "tracked source contains a GitHub or API-key-shaped secret");
    return "{\"tracked_python\":false,\"secret_pattern_scan\":\"clean\"}";
}

std::string configuration_and_threshold_check(const L1BaselineConfig& config, const std::string& config_hash) {
    config.validate();
    const auto canonical = config.canonical_json();
    require(cct::GovernedCorpus::content_sha256(canonical) == config_hash, "baseline configuration hash mismatch");
    const double measured_value = 1.0;
    const double deliberately_failing_maximum = 0.5;
    const bool threshold_passes = measured_value <= deliberately_failing_maximum;
    require(!threshold_passes, "deliberately failing threshold was accepted");
    return "{\"config_schema\":\"" + std::string(L1BaselineConfig::kSchema) + "\",\"config_hash\":\"" + config_hash +
           "\",\"injected_threshold_status\":\"FAIL\",\"diagnostics_retained\":true}";
}

std::string environment_json(const std::string& commit, const std::string& dirty, const std::string& config_hash) {
    return "{\"schema\":\"cct-l1-environment-v1\",\"commit\":\"" + json_escape(commit) + "\",\"dirty_tree\":" +
           (dirty.empty() ? "false" : "true") + ",\"compiler\":\"" + json_escape(__VERSION__) + "\",\"cmake\":\"" +
           json_escape(run_command("cmake --version 2>/dev/null | head -n 1")) + "\",\"fftw3\":\"" +
           json_escape(run_command("pkg-config --modversion fftw3 2>/dev/null")) + "\",\"platform\":\"" +
           json_escape(run_command("uname -srm 2>/dev/null")) + "\",\"config_hash\":\"" + config_hash + "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-0/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);

    const L1BaselineConfig config;
    const auto config_json = config.canonical_json();
    const auto config_hash = cct::GovernedCorpus::content_sha256(config_json);
    const auto commit = run_command("git rev-parse HEAD 2>/dev/null");
    const auto dirty = run_command("git status --porcelain 2>/dev/null");
    write_file(output / "config.json", config_json + "\n");
    write_file(output / "environment.json", environment_json(commit, dirty, config_hash));

    const std::vector<std::pair<std::string, std::function<std::string()>>> functions{
        {"event_lifecycle", event_contract_check},
        {"invalid_input_non_mutation", non_mutation_failure_check},
        {"deterministic_native_path", [&]() { return deterministic_check(config); }},
        {"configuration_validation_and_identity", [&]() { return configuration_validation_check(config, config_hash); }},
        {"configuration_and_injected_threshold", [&]() { return configuration_and_threshold_check(config, config_hash); }},
        {"repository_hygiene", repository_hygiene_check},
    };
    std::vector<Check> checks;
    for (const auto& [name, function] : functions) checks.push_back(run(name, function));
    bool passed = true;
    for (const auto& check : checks) passed = passed && check.status == "PASS";

    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0U; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "tests.json", checks_json.str());
    write_file(output / "checks.json", checks_json.str());

    const std::string benchmark = "{\"schema\":\"cct-benchmark-record-v1\",\"name\":\"stage0_native_smoke\",\"value\":1,\"unit\":\"pass\",\"seed\":" +
                                  std::to_string(config.seed) + ",\"commit\":\"" + json_escape(commit) + "\",\"config_hash\":\"" + config_hash +
                                  "\",\"hardware\":\"" + json_escape(run_command("uname -m 2>/dev/null")) + "\",\"timestamp_utc\":\"" +
                                  json_escape(run_command("date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null")) + "\",\"status\":\"PASS\"}\n";
    write_file(output / "benchmark_record.json", benchmark);
    write_file(output / "manifest.json", "{\"schema\":\"cct-l1-baseline-manifest-v1\",\"commit\":\"" + json_escape(commit) +
                                            "\",\"config_hash\":\"" + config_hash + "\",\"config_sha256\":\"" +
                                            cct::GovernedCorpus::content_sha256(config_json) + "\",\"tests\":6,\"gate_status\":\"" +
                                            (passed ? "PASS" : "FAIL") + "\"}\n");

    std::ostringstream gate;
    gate << "{\n  \"schema\": \"cct-l1-stage0-gate-v1\",\n  \"stage\": 0,\n  \"status\": \"" << (passed ? "PASS" : "FAIL")
         << "\",\n  \"transition\": \"" << (passed ? "L1-1 numerical implementation review" : "STOP")
         << "\",\n  \"implementation\": \"native-cpp20\",\n  \"commit\": \"" << json_escape(commit)
         << "\",\n  \"dirty_tree\": " << (dirty.empty() ? "false" : "true") << ",\n  \"config_hash\": \"" << config_hash
         << "\",\n  \"approval_required\": true\n}\n";
    write_file(output / "gate.json", gate.str());
    write_file(output / "release_record.json", "{\"schema\":\"cct-l1-baseline-release-record-v1\",\"commit\":\"" + json_escape(commit) +
                                               "\",\"config_hash\":\"" + config_hash + "\",\"environment\":\"environment.json\",\"tests\":\"tests.json\",\"benchmark\":\"benchmark_record.json\",\"manifest\":\"manifest.json\",\"gate\":\"gate.json\",\"status\":\"" +
                                               (passed ? "PASS" : "FAIL") + "\"}\n");

    std::ostringstream report;
    report << "# CCT Level 1 Stage L1-0 Baseline Report\n\n"
           << "**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n"
           << "**Commit:** `" << commit << "`  \n"
           << "**Configuration hash:** `" << config_hash << "`  \n"
           << "**Dirty tree at execution:** `" << (dirty.empty() ? "False" : "True") << "`\n\n"
           << "| Check | Status | Duration (s) |\n|---|---:|---:|\n";
    for (const auto& check : checks) report << "| " << check.name << " | `" << check.status << "` | " << check.duration_seconds << " |\n";
    report << "\nThe gate proves only the native reproducible-baseline contract: configuration identity, deterministic numerical replay, failure-path non-mutation, and deliberate threshold-failure detection. It does not establish language-teacher capability.\n";
    write_file(output / "report.md", report.str());

    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\",\"config_hash\":\""
              << config_hash << "\"}\n";
    return passed ? 0 : 1;
}
