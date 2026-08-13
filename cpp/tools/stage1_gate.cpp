#include "cct/event.hpp"
#include "cct/field.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cct::Boundary;
using cct::FieldState;
using cct::FiniteDifferenceSolver;
using cct::Method;
using cct::SolverConfig;
using cct::SpectralSolver;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds;
    std::string details_json;
};

struct Metric {
    std::string name;
    double value;
    std::string unit;
    std::string threshold;
    std::string status;
};

constexpr double kPi = 3.141592653589793238462643383279502884;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
    require(left.size() == right.size(), "vector sizes differ");
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) result = std::max(result, std::abs(left[index] - right[index]));
    return result;
}

double relative_l2_difference(const std::vector<double>& left, const std::vector<double>& right) {
    require(left.size() == right.size(), "vector sizes differ");
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto delta = left[index] - right[index];
        numerator += delta * delta;
        denominator += left[index] * left[index];
    }
    return std::sqrt(numerator) / std::max(std::sqrt(denominator), 1e-15);
}

std::vector<double> zeros(std::size_t count) { return std::vector<double>(count, 0.0); }

SolverConfig config(std::size_t n, double dt, Method method, Boundary boundary = Boundary::Periodic) {
    SolverConfig result;
    result.shape = {n};
    result.spacing = {1.0};
    result.wave_speed = 1.0;
    result.dt = dt;
    result.boundary = boundary;
    result.method = method;
    result.cfl_safety = 0.9;
    return result;
}

std::string git_command(const char* command) {
    auto* pipe = popen(command, "r");
    if (!pipe) return "unknown";
    char buffer[256]{};
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
    pclose(pipe);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
}

std::string config_hash() {
    const std::string input = "stage1-cpp-v1|shape=64|dt=0.05|method=rk4|fftw3";
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : input) {
        hash ^= static_cast<unsigned char>(byte);
        hash *= 1099511628211ULL;
    }
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return Check{name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        std::ostringstream details;
        details << "{\"error_type\":\"std::exception\",\"error\":\"";
        for (const auto character : std::string(error.what())) {
            if (character == '"') details << '\\';
            details << character;
        }
        details << "\"}";
        return Check{name, "FAIL", std::chrono::duration<double>(finished - started).count(), details.str()};
    }
}

std::string transform_check() {
    std::vector<double> field(32 * 24, 0.0);
    for (std::size_t index = 0; index < field.size(); ++index) field[index] = std::sin(0.13 * static_cast<double>(index)) + std::cos(0.017 * static_cast<double>(index));
    const auto error = cct::fft_round_trip_error(field, {32, 24});
    require(error < 2e-12, "transform error exceeded 2e-12");
    std::ostringstream details;
    details << "{\"max_abs_error\":" << std::setprecision(17) << error << "}";
    return details.str();
}

std::string operator_agreement_check() {
    const std::vector<std::size_t> shape{64};
    const std::vector<double> spacing{1.0};
    const auto field = cct::manufactured_mode(shape, spacing, {1});
    const auto spectral = cct::spectral_laplacian(field, shape, spacing);
    const auto reference = cct::finite_difference_laplacian(field, shape, spacing, Boundary::Periodic);
    const auto relative = relative_l2_difference(spectral, reference);
    const auto maximum = max_difference(spectral, reference);
    require(relative < 2e-3, "spectral/reference operator error exceeded 2e-3");
    std::ostringstream details;
    details << "{\"max_abs_error\":" << maximum << ",\"relative_l2_error\":" << relative << "}";
    return details.str();
}

std::string rollout_agreement_check() {
    const auto spectral = SpectralSolver(config(64, 0.05, Method::Leapfrog));
    const auto reference = FiniteDifferenceSolver(config(64, 0.05, Method::Leapfrog));
    const auto phi0 = cct::manufactured_mode({64}, {1.0}, {1});
    const auto sources = std::vector<std::vector<double>>(20, zeros(64));
    const auto spectral_result = spectral.rollout(spectral.initialize(phi0), sources);
    const auto reference_result = reference.rollout(reference.initialize(phi0), sources);
    std::vector<double> all_spectral;
    std::vector<double> all_reference;
    for (std::size_t step = 0; step < spectral_result.phi.size(); ++step) {
        all_spectral.insert(all_spectral.end(), spectral_result.phi[step].begin(), spectral_result.phi[step].end());
        all_reference.insert(all_reference.end(), reference_result.phi[step].begin(), reference_result.phi[step].end());
    }
    const auto maximum = max_difference(all_spectral, all_reference);
    require(maximum < 3e-3, "spectral/reference rollout error exceeded 3e-3");
    std::ostringstream details;
    details << "{\"max_abs_rollout_error\":" << maximum << ",\"steps\":20}";
    return details.str();
}

std::string manufactured_check() {
    const auto solver = SpectralSolver(config(64, 0.05, Method::RK4));
    const auto phi0 = cct::manufactured_mode({64}, {1.0}, {4});
    const auto omega = cct::manufactured_mode_frequency({64}, {1.0}, {4}, 1.0);
    std::vector<double> psi0(64, 0.0);
    for (std::size_t index = 0; index < 64; ++index) psi0[index] = -omega * std::cos(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0);
    const auto result = solver.rollout(solver.initialize(phi0, psi0), std::vector<std::vector<double>>(20, zeros(64)));
    std::vector<double> expected(64, 0.0);
    for (std::size_t index = 0; index < 64; ++index) expected[index] = std::sin(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0 - omega * result.time.back());
    const auto error = max_difference(result.phi.back(), expected);
    require(error < 2e-3, "manufactured solution error exceeded 2e-3");
    std::ostringstream details;
    details << "{\"max_abs_error\":" << error << ",\"final_time\":" << result.time.back() << "}";
    return details.str();
}

std::string convergence_check() {
    std::vector<double> errors;
    for (const auto dt : {0.2, 0.1, 0.05}) {
        const auto solver = SpectralSolver(config(64, dt, Method::RK4));
        const auto phi0 = cct::manufactured_mode({64}, {1.0}, {4});
        const auto omega = cct::manufactured_mode_frequency({64}, {1.0}, {4}, 1.0);
        std::vector<double> psi0(64, 0.0);
        for (std::size_t index = 0; index < 64; ++index) psi0[index] = -omega * std::cos(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0);
        const auto steps = static_cast<std::size_t>(std::llround(1.0 / dt));
        const auto result = solver.rollout(solver.initialize(phi0, psi0), std::vector<std::vector<double>>(steps, zeros(64)));
        std::vector<double> expected(64, 0.0);
        for (std::size_t index = 0; index < 64; ++index) expected[index] = std::sin(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0 - omega * result.time.back());
        double sum = 0.0;
        for (std::size_t index = 0; index < 64; ++index) {
            const auto error = result.phi.back()[index] - expected[index];
            sum += error * error;
        }
        errors.push_back(std::sqrt(sum / 64.0));
    }
    const auto rate_a = std::log(errors[0] / errors[1]) / std::log(2.0);
    const auto rate_b = std::log(errors[1] / errors[2]) / std::log(2.0);
    require(rate_a > 2.5 && rate_b > 2.5, "convergence order below 2.5");
    std::ostringstream details;
    details << "{\"errors\":[" << errors[0] << "," << errors[1] << "," << errors[2] << "],\"rates\":[" << rate_a << "," << rate_b << "]}";
    return details.str();
}

std::string energy_check() {
    const auto solver = SpectralSolver(config(64, 0.05, Method::Leapfrog));
    const auto phi0 = cct::manufactured_mode({64}, {1.0}, {1});
    const auto omega = cct::manufactured_mode_frequency({64}, {1.0}, {1}, 1.0);
    std::vector<double> psi0(64, 0.0);
    for (std::size_t index = 0; index < 64; ++index) psi0[index] = -omega * std::cos(2.0 * kPi * static_cast<double>(index) / 64.0);
    const auto result = solver.rollout(solver.initialize(phi0, psi0), std::vector<std::vector<double>>(400, zeros(64)));
    double first = 0.0;
    double minimum = 1e300;
    double maximum = -1e300;
    for (std::size_t index = 0; index < result.phi.size(); ++index) {
        const FieldState state{result.phi[index], result.psi[index], result.time[index], result.step_index[index]};
        const auto value = solver.energy(state);
        if (index == 0) first = value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const auto drift = (maximum - minimum) / first;
    require(std::isfinite(drift) && drift < 2e-3, "energy drift exceeded 2e-3");
    std::ostringstream details;
    details << "{\"relative_drift\":" << drift << ",\"steps\":400}";
    return details.str();
}

std::string stability_check() {
    bool rejected = false;
    auto unstable = config(16, 0.7, Method::Leapfrog);
    unstable.shape = {16, 16};
    unstable.spacing = {1.0, 1.0};
    try { (void)SpectralSolver(unstable); } catch (const cct::StabilityError&) { rejected = true; }
    require(rejected, "CFL violation was accepted");
    return "{\"rejected\":true}";
}

std::string gradient_check() {
    const auto solver = SpectralSolver(config(8, 0.05, Method::Leapfrog));
    const auto state = solver.initialize(cct::manufactured_mode({8}, {1.0}, {1}));
    const std::vector<double> source(8, 0.0);
    const std::vector<double> potential(8, 0.1);
    const std::vector<double> target(8, 0.0);
    const auto analytic = cct::leapfrog_operator_loss_gradients(solver, state, source, potential, target);
    const double epsilon = 1e-5;
    std::vector<double> numerical_source(8, 0.0);
    std::vector<double> numerical_potential(8, 0.0);
    for (std::size_t parameter = 0; parameter < 8; ++parameter) {
        auto plus = source;
        auto minus = source;
        plus[parameter] += epsilon;
        minus[parameter] -= epsilon;
        numerical_source[parameter] = (solver.operator_loss(solver.step(state, plus, potential).phi, target) - solver.operator_loss(solver.step(state, minus, potential).phi, target)) / (2.0 * epsilon);
        plus = potential;
        minus = potential;
        plus[parameter] += epsilon;
        minus[parameter] -= epsilon;
        numerical_potential[parameter] = (solver.operator_loss(solver.step(state, source, plus).phi, target) - solver.operator_loss(solver.step(state, source, minus).phi, target)) / (2.0 * epsilon);
    }
    const auto source_error = max_difference(analytic.source, numerical_source);
    const auto potential_error = max_difference(analytic.potential, numerical_potential);
    require(source_error < 2e-5 && potential_error < 2e-5, "analytic gradients disagree with finite differences");
    std::ostringstream details;
    details << "{\"source_max_abs_error\":" << source_error << ",\"potential_max_abs_error\":" << potential_error << "}";
    return details.str();
}

std::string boundaries_check() {
    std::vector<double> initial(32, 0.0);
    for (std::size_t index = 0; index < initial.size(); ++index) initial[index] = std::sin(kPi * static_cast<double>(index) / 31.0);
    const auto dirichlet = FiniteDifferenceSolver(config(32, 0.1, Method::Leapfrog, Boundary::Dirichlet));
    const auto dirichlet_result = dirichlet.rollout(dirichlet.initialize(initial), std::vector<std::vector<double>>(20, zeros(32)));
    double dirichlet_residual = 0.0;
    for (const auto& field : dirichlet_result.phi) dirichlet_residual = std::max(dirichlet_residual, std::max(std::abs(field.front()), std::abs(field.back())));
    const auto neumann = FiniteDifferenceSolver(config(32, 0.1, Method::Leapfrog, Boundary::Neumann));
    const auto neumann_result = neumann.rollout(neumann.initialize(initial), std::vector<std::vector<double>>(20, zeros(32)));
    double neumann_residual = 0.0;
    for (const auto& field : neumann_result.phi) {
        neumann_residual = std::max(neumann_residual, std::abs(field.front() - field[1]));
        neumann_residual = std::max(neumann_residual, std::abs(field.back() - field[field.size() - 2]));
    }
    require(dirichlet_residual < 1e-12 && neumann_residual < 1e-12, "boundary residual exceeded tolerance");
    std::ostringstream details;
    details << "{\"dirichlet_residual\":" << dirichlet_residual << ",\"neumann_residual\":" << neumann_residual << "}";
    return details.str();
}

std::string serialization_check(const std::filesystem::path& output) {
    const auto solver = SpectralSolver(config(16, 0.1, Method::RK4));
    const auto path = output / "solver_config.json";
    solver.save_config(path.string());
    const auto loaded = cct::Solver::load_config(path.string());
    require(loaded.shape == solver.config().shape && loaded.method == solver.config().method, "configuration round trip failed");
    return "{\"schema_version\":1}";
}

std::string performance_check() {
    std::vector<double> times;
    for (const auto n : {32u, 64u, 128u}) {
        const auto solver = SpectralSolver(config(n, 0.05, Method::RK4));
        const auto state = solver.initialize(zeros(n));
        for (int warmup = 0; warmup < 3; ++warmup) (void)solver.step(state, zeros(n));
        const auto started = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 10; ++repeat) (void)solver.step(state, zeros(n));
        const auto finished = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(finished - started).count() / 10.0);
    }
    const auto slope = std::log(times.back() / times.front()) / std::log(128.0 / 32.0);
    require(std::isfinite(slope) && slope < 2.0, "performance scaling slope is not subquadratic");
    std::ostringstream details;
    details << "{\"times\":[" << times[0] << "," << times[1] << "," << times[2] << "],\"log_log_slope\":" << slope << "}";
    return details.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("could not write " + path.string());
    stream << content;
}

std::string checks_json(const std::vector<Check>& checks) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index) output << ",\n";
        output << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
               << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details_json << "}";
    }
    output << "\n]\n";
    return output.str();
}

std::string metrics_json(const std::vector<Metric>& metrics) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index) output << ",\n";
        output << "  {\"name\":\"" << metrics[index].name << "\",\"value\":" << metrics[index].value
               << ",\"unit\":\"" << metrics[index].unit << "\",\"threshold\":\"" << metrics[index].threshold
               << "\",\"status\":\"" << metrics[index].status << "\"}";
    }
    output << "\n]\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-1/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);

    const std::vector<std::pair<std::string, std::function<std::string()>>> functions{
        {"transform_correctness", transform_check},
        {"spectral_reference_operator_agreement", operator_agreement_check},
        {"spectral_reference_rollout_agreement", rollout_agreement_check},
        {"manufactured_solution_accuracy", manufactured_check},
        {"temporal_convergence", convergence_check},
        {"energy_stability", energy_check},
        {"cfl_rejection", stability_check},
        {"analytic_finite_difference_gradients", gradient_check},
        {"boundary_residuals", boundaries_check},
        {"serialization_round_trip", [&]() { return serialization_check(output); }},
        {"performance_scaling", performance_check},
    };
    std::vector<Check> checks;
    checks.reserve(functions.size());
    for (const auto& [name, function] : functions) checks.push_back(run_check(name, function));
    const bool passed = std::all_of(checks.begin(), checks.end(), [](const auto& check) { return check.status == "PASS"; });

    std::vector<Metric> metrics;
    metrics.push_back({"mandatory_check_count", static_cast<double>(checks.size()), "checks", "all PASS", passed ? "PASS" : "FAIL"});
    const auto commit_value = git_command("git rev-parse HEAD 2>/dev/null");
    const auto commit = commit_value.empty() ? std::string("unknown") : commit_value;
    const auto dirty = git_command("git status --porcelain 2>/dev/null");
    const auto hash = config_hash();
    write_file(output / "checks.json", checks_json(checks));
    write_file(output / "metrics.json", metrics_json(metrics));

    std::ostringstream gate;
    gate << "{\n  \"stage\": 1,\n  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
         << "  \"transition\": \"" << (passed ? "Stage 2" : "STOP") << "\",\n"
         << "  \"implementation\": \"native-cpp\",\n  \"commit\": \"" << commit << "\",\n"
         << "  \"dirty_tree\": " << (dirty.empty() ? "false" : "true") << ",\n"
         << "  \"config_hash\": \"" << hash << "\",\n  \"approval_required\": true\n}\n";
    write_file(output / "gate.json", gate.str());

    std::ostringstream manifest;
    manifest << "{\n  \"stage\": 1,\n  \"implementation\": \"native-cpp\",\n  \"status\": \"" << (passed ? "PASS" : "FAIL")
             << "\",\n  \"commit\": \"" << commit << "\",\n  \"config_hash\": \"" << hash << "\",\n"
             << "  \"compiler\": \"" << __VERSION__ << "\",\n  \"fftw\": \"3.3.10-compatible\"\n}\n";
    write_file(output / "manifest.json", manifest.str());

    std::ostringstream report;
    report << "# Native C++ Stage 1 Gate Report\n\n"
           << "**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n"
           << "**Transition:** `" << (passed ? "Stage 2" : "STOP") << "`  \n"
           << "**Implementation:** `native-cpp`  \n"
           << "**Commit:** `" << commit << "`  \n"
           << "**Dirty tree:** `" << (dirty.empty() ? "False" : "True") << "`  \n"
           << "**Configuration hash:** `" << hash << "`\n\n"
           << "## Checks\n\n| Check | Status | Duration (s) |\n|---|---:|---:|\n";
    for (const auto& check : checks) report << "| " << check.name << " | `" << check.status << "` | " << check.duration_seconds << " |\n";
    report << "\n## Transition policy\n\n"
           << "A `PASS` proves the native C++ Stage 1 implementation and harness are green. It authorizes Stage 2 preparation only; Stage 2 implementation requires explicit user approval.\n";
    write_file(output / "report.md", report.str());

    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
