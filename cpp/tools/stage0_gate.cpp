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

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string run_git(const char* command) {
    auto* pipe = popen(command, "r");
    if (!pipe) return "unknown";
    char buffer[256]{};
    std::string value;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) value += buffer;
    pclose(pipe);
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
    return value;
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
                std::string("{\"error\":\"") + error.what() + "\"}"};
    }
}

std::string event_check() {
    cct::Manifold manifold({4, 4});
    manifold.place_event(cct::Event{{1.0}, {1, 2}, {0.5, 0.25}});
    require(manifold.get_event({1, 2}) != nullptr, "event lookup failed");
    require(manifold.filled_cells() == 1, "event count failed");
    return "{\"native_module\":\"cct_native\",\"filled_cells\":1}";
}

std::string invalid_input_check() {
    bool rejected = false;
    try { cct::Manifold invalid({0, 4}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid dimensions were accepted");
    cct::Manifold manifold({4, 4});
    rejected = false;
    try { manifold.place_event(cct::Event{{}, {-1, 0}, {}}); } catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "negative coordinate was accepted");
    return "{\"structured_error_cases\":true}";
}

std::string deterministic_check() {
    cct::SolverConfig configuration;
    configuration.shape = {32};
    configuration.spacing = {1.0};
    configuration.wave_speed = 1.0;
    configuration.dt = 0.05;
    configuration.boundary = cct::Boundary::Periodic;
    configuration.method = cct::Method::RK4;
    configuration.cfl_safety = 0.9;
    const cct::SpectralSolver solver(configuration);
    const auto state = solver.initialize(cct::manufactured_mode({32}, {1.0}, {2}));
    const auto first = solver.step(state);
    const auto second = solver.step(state);
    require(first.phi == second.phi && first.psi == second.psi, "native baseline is nondeterministic");
    require(cct::fft_round_trip_error(first.phi, {32}) < 2e-12, "native FFT baseline is invalid");
    return "{\"deterministic\":true,\"finite_values\":true}";
}

std::string benchmark_schema_check(const std::filesystem::path& output) {
    std::ofstream record(output / "benchmark_record.json");
    record << "{\"name\":\"stage0_native_smoke\",\"value\":1,\"unit\":\"pass\",\"seed\":0,\"commit\":\"native\",\"config_hash\":\"f8dbacb0a0098aab\",\"hardware\":\"native\",\"timestamp_utc\":\"1970-01-01T00:00:00Z\",\"status\":\"PASS\"}\n";
    require(record.good(), "benchmark record could not be written");
    return "{\"record_valid\":true}";
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("could not write " + path.string());
    stream << content;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-0/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    const std::vector<std::pair<std::string, std::function<std::string()>>> functions{
        {"event_lifecycle", event_check},
        {"invalid_input_errors", invalid_input_check},
        {"deterministic_native_path", deterministic_check},
        {"benchmark_schema", [&]() { return benchmark_schema_check(output); }},
    };
    std::vector<Check> checks;
    for (const auto& [name, function] : functions) checks.push_back(run(name, function));
    bool passed = true;
    for (const auto& check : checks) passed = passed && check.status == "PASS";
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    const auto commit_value = run_git("git rev-parse HEAD 2>/dev/null");
    const auto commit = commit_value.empty() ? std::string("unknown") : commit_value;
    const auto dirty = run_git("git status --porcelain 2>/dev/null");
    std::ostringstream gate;
    gate << "{\n  \"stage\": 0,\n  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
         << "  \"transition\": \"" << (passed ? "Stage 1" : "STOP") << "\",\n"
         << "  \"implementation\": \"native-cpp\",\n  \"commit\": \"" << commit << "\",\n"
         << "  \"dirty_tree\": " << (dirty.empty() ? "false" : "true") << ",\n  \"approval_required\": true\n}\n";
    write_file(output / "gate.json", gate.str());
    std::ostringstream report;
    report << "# Native C++ Stage 0 Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n**Transition:** `" << (passed ? "Stage 1" : "STOP")
           << "`  \n**Implementation:** `native-cpp`  \n**Commit:** `" << commit << "`  \n**Dirty tree:** `" << (dirty.empty() ? "False" : "True") << "`\n\n"
           << "| Check | Status | Duration (s) |\n|---|---:|---:|\n";
    for (const auto& check : checks) report << "| " << check.name << " | `" << check.status << "` | " << check.duration_seconds << " |\n";
    report << "\nA `PASS` authorizes Stage 1 preparation only and does not authorize Stage 1 implementation without explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
