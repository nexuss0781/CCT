#include "cct/event.hpp"
#include "cct/field.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cct::Boundary;
using cct::Event;
using cct::FieldState;
using cct::FiniteDifferenceSolver;
using cct::Method;
using cct::NumericalError;
using cct::Precision;
using cct::UnsupportedPrecisionError;
using cct::SolverConfig;
using cct::Solver;
using cct::SpectralSolver;

constexpr double kPi = 3.141592653589793238462643383279502884;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_close(double actual, double expected, double absolute, const std::string& message) {
    if (std::abs(actual - expected) > absolute) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
    check(left.size() == right.size(), "vectors have different sizes");
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

std::vector<double> zeros(std::size_t count) { return std::vector<double>(count, 0.0); }

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("could not read temporary configuration");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("could not write temporary configuration");
    stream << text;
}

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

void test_event_lifecycle() {
    cct::Manifold manifold({4, 4});
    manifold.place_event(Event{{1.0}, {1, 2}, {0.5, 0.25}});
    const auto* event = manifold.get_event({1, 2});
    check(event != nullptr, "event lookup failed");
    check(manifold.filled_cells() == 1, "event count mismatch");
    check(manifold.events().size() == 1, "event enumeration mismatch");
    check(manifold.repr() == "Manifold(dimensions: [4, 4], filled_cells: 1)", "repr mismatch");
    bool rejected = false;
    try { manifold.place_event(Event{{}, {1, 2}, {}}); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "duplicate event was accepted");
    rejected = false;
    try { manifold.place_event(Event{{}, {-1, 0}, {}}); } catch (const std::out_of_range&) { rejected = true; }
    check(rejected, "negative coordinate was accepted");
}

void test_fft_round_trip() {
    std::vector<double> field(32 * 24, 0.0);
    for (std::size_t index = 0; index < field.size(); ++index) {
        field[index] = std::sin(0.13 * static_cast<double>(index)) + std::cos(0.017 * static_cast<double>(index));
    }
    check(cct::fft_round_trip_error(field, {32, 24}) < 2e-12, "FFT round trip failed");
}

void test_operator_agreement() {
    const std::vector<std::size_t> shape{64};
    const std::vector<double> spacing{1.0};
    const auto field = cct::manufactured_mode(shape, spacing, {1});
    const auto spectral = cct::spectral_laplacian(field, shape, spacing);
    const auto reference = cct::finite_difference_laplacian(field, shape, spacing, Boundary::Periodic);
    check(max_difference(spectral, reference) < 2e-5, "spectral/reference Laplacians disagree");
}

void test_manufactured_accuracy() {
    const auto solver = SpectralSolver(config(64, 0.05, Method::RK4));
    const auto phi0 = cct::manufactured_mode({64}, {1.0}, {4});
    const auto omega = cct::manufactured_mode_frequency({64}, {1.0}, {4}, 1.0);
    std::vector<double> psi0(64, 0.0);
    for (std::size_t index = 0; index < 64; ++index) {
        psi0[index] = -omega * std::cos(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0);
    }
    const auto trajectory = solver.rollout(solver.initialize(phi0, psi0), std::vector<std::vector<double>>(20, zeros(64)));
    const auto final_time = trajectory.time.back();
    std::vector<double> expected(64, 0.0);
    for (std::size_t index = 0; index < 64; ++index) {
        expected[index] = std::sin(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0 - omega * final_time);
    }
    check(max_difference(trajectory.phi.back(), expected) < 2e-3, "manufactured solution error too large");
}

void test_forced_manufactured_solution() {
    const auto solver = SpectralSolver(config(32, 0.1, Method::Leapfrog));
    const std::vector<double> source(32, 0.3);
    const auto trajectory = solver.rollout(solver.initialize(zeros(32)), std::vector<std::vector<double>>(20, source));
    const auto final_time = trajectory.time.back();
    std::vector<double> expected_phi(32, 0.5 * source.front() * final_time * final_time);
    std::vector<double> expected_psi(32, source.front() * final_time);
    check(max_difference(trajectory.phi.back(), expected_phi) < 1e-12, "forced manufactured field solution failed");
    check(max_difference(trajectory.psi.back(), expected_psi) < 1e-12, "forced manufactured velocity solution failed");
}

void test_convergence() {
    std::vector<double> errors;
    for (const auto dt : {0.2, 0.1, 0.05}) {
        auto solver_config = config(64, dt, Method::RK4);
        const auto solver = SpectralSolver(solver_config);
        const auto phi0 = cct::manufactured_mode({64}, {1.0}, {4});
        const auto omega = cct::manufactured_mode_frequency({64}, {1.0}, {4}, 1.0);
        std::vector<double> psi0(64, 0.0);
        for (std::size_t index = 0; index < 64; ++index) {
            psi0[index] = -omega * std::cos(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0);
        }
        const auto steps = static_cast<std::size_t>(std::llround(1.0 / dt));
        const auto trajectory = solver.rollout(solver.initialize(phi0, psi0), std::vector<std::vector<double>>(steps, zeros(64)));
        std::vector<double> expected(64, 0.0);
        for (std::size_t index = 0; index < 64; ++index) {
            expected[index] = std::sin(2.0 * kPi * 4.0 * static_cast<double>(index) / 64.0 - omega * trajectory.time.back());
        }
        double sum = 0.0;
        for (std::size_t index = 0; index < 64; ++index) {
            const auto delta = trajectory.phi.back()[index] - expected[index];
            sum += delta * delta;
        }
        errors.push_back(std::sqrt(sum / 64.0));
    }
    const auto rate_a = std::log(errors[0] / errors[1]) / std::log(2.0);
    const auto rate_b = std::log(errors[1] / errors[2]) / std::log(2.0);
    check(rate_a > 2.5 && rate_b > 2.5, "RK4 convergence order failed");
}

void test_energy_stability() {
    const auto solver = SpectralSolver(config(64, 0.05, Method::Leapfrog));
    const auto phi0 = cct::manufactured_mode({64}, {1.0}, {1});
    const auto omega = cct::manufactured_mode_frequency({64}, {1.0}, {1}, 1.0);
    std::vector<double> psi0(64, 0.0);
    for (std::size_t index = 0; index < 64; ++index) {
        psi0[index] = -omega * std::cos(2.0 * kPi * static_cast<double>(index) / 64.0);
    }
    const auto trajectory = solver.rollout(solver.initialize(phi0, psi0), std::vector<std::vector<double>>(400, zeros(64)));
    double first = 0.0;
    double minimum = 1e300;
    double maximum = -1e300;
    for (std::size_t index = 0; index < trajectory.phi.size(); ++index) {
        const FieldState state{trajectory.phi[index], trajectory.psi[index], trajectory.time[index], trajectory.step_index[index]};
        const auto value = solver.energy(state);
        if (index == 0) first = value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    check((maximum - minimum) / first < 2e-3, "energy drift failed");
}

void test_stability_and_boundaries() {
    bool rejected = false;
    auto unstable = config(16, 0.7, Method::Leapfrog);
    unstable.shape = {16, 16};
    unstable.spacing = {1.0, 1.0};
    try { (void)SpectralSolver(unstable); } catch (const cct::StabilityError&) { rejected = true; }
    check(rejected, "CFL violation was accepted");

    const auto dirichlet = FiniteDifferenceSolver(config(32, 0.1, Method::Leapfrog, Boundary::Dirichlet));
    std::vector<double> initial(32, 0.0);
    for (std::size_t index = 0; index < initial.size(); ++index) initial[index] = std::sin(kPi * static_cast<double>(index) / 31.0);
    const auto dirichlet_trajectory = dirichlet.rollout(dirichlet.initialize(initial), std::vector<std::vector<double>>(20, zeros(32)));
    for (const auto& field : dirichlet_trajectory.phi) check(std::abs(field.front()) < 1e-12 && std::abs(field.back()) < 1e-12, "Dirichlet residual failed");

    const auto neumann = FiniteDifferenceSolver(config(32, 0.1, Method::Leapfrog, Boundary::Neumann));
    for (const auto& field : neumann.rollout(neumann.initialize(initial), std::vector<std::vector<double>>(20, zeros(32))).phi) {
        check(std::abs(field.front() - field[1]) < 1e-12, "Neumann left residual failed");
        check(std::abs(field.back() - field[field.size() - 2]) < 1e-12, "Neumann right residual failed");
    }
}

void test_global_stability_domain_and_semantics() {
    auto periodic = config(32, 0.05, Method::Leapfrog);
    periodic.maximum_abs_potential = 0.25;
    check(cct::global_stability_limit(periodic) < cct::cfl_limit(periodic),
          "global spectral stability bound did not tighten the local CFL estimate");
    for (const auto method : {Method::Leapfrog, Method::RK4}) {
        for (const auto boundary : {Boundary::Periodic, Boundary::Dirichlet, Boundary::Neumann}) {
            for (const auto& shape : {std::vector<std::size_t>{8U}, std::vector<std::size_t>{5U, 5U}}) {
                SolverConfig boundary_config;
                boundary_config.shape = shape;
                boundary_config.spacing.assign(shape.size(), 1.0);
                boundary_config.wave_speed = 1.0;
                boundary_config.boundary = boundary;
                boundary_config.method = method;
                boundary_config.maximum_abs_potential = 0.25;
                const auto limit = std::min(cct::cfl_limit(boundary_config), cct::global_stability_limit(boundary_config));
                boundary_config.dt = 0.8 * limit * boundary_config.cfl_safety;
                const FiniteDifferenceSolver solver(boundary_config);
                const auto cell_total = cct::cell_count(shape);
                const auto trajectory = solver.rollout(solver.initialize(zeros(cell_total)),
                                                       std::vector<std::vector<double>>(4, zeros(cell_total)), {0.25});
                for (const auto& field : trajectory.phi) {
                    for (const auto value : field) check(std::isfinite(value), "bounded stability-domain rollout produced a non-finite field");
                }
                boundary_config.dt = 1.01 * limit * boundary_config.cfl_safety;
                bool rejected_boundary = false;
                try { static_cast<void>(FiniteDifferenceSolver(boundary_config)); } catch (const cct::StabilityError&) { rejected_boundary = true; }
                check(rejected_boundary, "stability-domain boundary beyond the declared limit was accepted");
            }
        }
    }
    auto bounded = config(8, 0.05, Method::Leapfrog);
    bounded.maximum_abs_potential = 0.1;
    const FiniteDifferenceSolver bounded_solver(bounded);
    bool rejected = false;
    try {
        static_cast<void>(bounded_solver.step(bounded_solver.initialize(zeros(8)), {}, {0.2}));
    } catch (const cct::StabilityError&) {
        rejected = true;
    }
    check(rejected, "potential outside the declared global stability domain was accepted");
    const auto periodic_field = cct::manufactured_mode({16}, {1.0}, {2});
    const auto spectral = cct::spectral_laplacian(periodic_field, {16}, {1.0});
    const auto finite_periodic = cct::finite_difference_laplacian(periodic_field, {16}, {1.0}, Boundary::Periodic);
    check(max_difference(spectral, finite_periodic) < 0.2, "periodic operator implementations diverged on a resolved mode");
    const auto finite_dirichlet = cct::finite_difference_laplacian(periodic_field, {16}, {1.0}, Boundary::Dirichlet);
    check(max_difference(spectral, finite_dirichlet) > 1e-3, "non-periodic finite-difference semantics were mistaken for spectral semantics");
}

void test_temporal_rollout_gradients() {
    auto solver_config = config(8, 0.05, Method::Leapfrog);
    solver_config.maximum_abs_potential = 1.0;
    const SpectralSolver solver(solver_config);
    const auto initial = solver.initialize(cct::manufactured_mode({8}, {1.0}, {1}));
    std::vector<std::vector<double>> sources(3, std::vector<double>(8, 0.0));
    sources[0][1] = 0.1;
    sources[1][2] = -0.08;
    sources[2][3] = 0.06;
    const std::vector<double> potential(8, 0.1);
    const std::vector<std::vector<double>> targets(3, std::vector<double>(8, 0.0));
    const auto gradients = cct::temporal_rollout_loss_gradients(solver, initial, sources, potential, targets, 1e-6);
    auto objective = [&](const std::vector<std::vector<double>>& source_values, const std::vector<double>& potential_values) {
        const auto trajectory = solver.rollout(initial, source_values, potential_values, false);
        double total = 0.0;
        for (std::size_t step = 0; step < trajectory.phi.size(); ++step) total += solver.operator_loss(trajectory.phi[step], targets[step]);
        return total / static_cast<double>(trajectory.phi.size());
    };
    const auto epsilon = 1e-5;
    check_close(gradients.loss, objective(sources, potential), 1e-8, "temporal rollout loss disagreed with direct objective");
    for (std::size_t step = 0; step < sources.size(); ++step) {
        for (std::size_t index = 0; index < sources[step].size(); ++index) {
            auto plus = sources;
            auto minus = sources;
            plus[step][index] += epsilon;
            minus[step][index] -= epsilon;
            const auto numerical = (objective(plus, potential) - objective(minus, potential)) / (2.0 * epsilon);
            check_close(gradients.source[step][index], numerical, 2e-5, "temporal source gradient disagreed with full rollout objective");
        }
    }
    for (std::size_t index = 0; index < potential.size(); ++index) {
        auto plus = potential;
        auto minus = potential;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto numerical = (objective(sources, plus) - objective(sources, minus)) / (2.0 * epsilon);
        check_close(gradients.potential[index], numerical, 2e-5, "temporal potential gradient disagreed with full rollout objective");
    }
    const auto one_step = cct::leapfrog_operator_loss_gradients(solver, initial, sources.front(), potential, targets.front());
    check(max_difference(one_step.source, gradients.source.front()) > 1e-8,
          "one-step gradient helper was incorrectly presented as the full temporal gradient");
}

void test_gradients_and_bounded_potential() {
    const auto solver = SpectralSolver(config(8, 0.05, Method::Leapfrog));
    const auto phi0 = cct::manufactured_mode({8}, {1.0}, {1});
    const auto state = solver.initialize(phi0);
    std::vector<double> source(8, 0.0);
    std::vector<double> potential(8, 0.1);
    std::vector<double> target(8, 0.0);
    const auto analytic = cct::leapfrog_operator_loss_gradients(solver, state, source, potential, target);
    const auto epsilon = 1e-5;
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
    check(max_difference(analytic.source, numerical_source) < 2e-5, "source gradient failed");
    check(max_difference(analytic.potential, numerical_potential) < 2e-5, "potential gradient failed");

    std::vector<double> raw{-4.0, -0.2, 0.0, 0.2, 4.0};
    const auto bounded = cct::bounded_local_potential(raw);
    const auto derivative = cct::bounded_local_potential_gradient(raw);
    for (std::size_t index = 0; index < raw.size(); ++index) {
        check(bounded[index] > 0.0 && bounded[index] < 1.0, "bounded potential escaped range");
        const auto plus = raw[index] + epsilon;
        const auto minus = raw[index] - epsilon;
        const auto numerical = ((1.0 / (1.0 + std::exp(-plus))) - (1.0 / (1.0 + std::exp(-minus)))) / (2.0 * epsilon);
        check(std::abs(derivative[index] - numerical) < 2e-6, "bounded potential gradient failed");
    }
}

void test_precision_source_causality_and_invalid_input_rejection() {
    auto reduced = config(8, 0.05, Method::Leapfrog);
    reduced.precision = Precision::Float32;
    bool rejected = false;
    try {
        static_cast<void>(SpectralSolver(reduced));
    } catch (const UnsupportedPrecisionError&) {
        rejected = true;
    }
    check(rejected, "unsupported float32 solver configuration was accepted");

    const SpectralSolver solver(config(8, 0.05, Method::Leapfrog));
    const auto clean = solver.initialize(zeros(8));
    std::vector<double> source(8, 0.0);
    source[3] = 1.0;
    const auto forced = solver.step(clean, source);
    check(max_difference(clean.phi, zeros(8)) == 0.0 && max_difference(clean.psi, zeros(8)) == 0.0,
          "step mutated its input state");
    check(forced.time == clean.time + solver.config().dt && forced.step_index == clean.step_index + 1 && forced.psi[3] > 0.0,
          "causal source was not applied to the next state");

    auto nonfinite = zeros(8);
    nonfinite.front() = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        static_cast<void>(solver.initialize(nonfinite));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "non-finite initial field was accepted");
    rejected = false;
    try {
        static_cast<void>(solver.step(clean, nonfinite));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "non-finite source was accepted");
    auto invalid_state = clean;
    invalid_state.psi.front() = std::numeric_limits<double>::infinity();
    rejected = false;
    try {
        static_cast<void>(solver.step(invalid_state));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "non-finite state was accepted");

    rejected = false;
    try {
        static_cast<void>(solver.operator_loss({}, {}));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "empty operator-loss target was accepted");
    rejected = false;
    try {
        static_cast<void>(solver.operator_loss(zeros(8), zeros(8), std::vector<double>(8, 0.0)));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "zero-denominator operator-loss mask was accepted");
    auto negative_mask = std::vector<double>(8, 1.0);
    negative_mask.front() = -1.0;
    rejected = false;
    try {
        static_cast<void>(solver.operator_loss(zeros(8), zeros(8), negative_mask));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "negative operator-loss mask was accepted");

    rejected = false;
    try {
        static_cast<void>(cct::cell_count({std::numeric_limits<std::size_t>::max(), 2U}));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "overflowing shape cell count was accepted");
    rejected = false;
    try {
        static_cast<void>(cct::cell_count({1U}));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "invalid one-cell dimension was accepted");
    rejected = false;
    try {
        static_cast<void>(cct::spectral_laplacian(nonfinite, {8U}, {1.0}));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "non-finite public operator input was accepted");
}

void test_serialization_loss_and_determinism() {
    const auto solver = SpectralSolver(config(16, 0.1, Method::RK4));
    const auto path = std::filesystem::temp_directory_path() / "cct_cpp_solver_config.json";
    solver.save_config(path.string());
    const auto loaded = Solver::load_config(path.string());
    check(loaded.shape == solver.config().shape, "config shape round trip failed");
    check(loaded.method == solver.config().method, "config method round trip failed");
    check(loaded.precision == Precision::Float64, "config precision round trip failed");
    const auto canonical = read_text_file(path);
    auto unsupported_schema = canonical;
    const auto schema_position = unsupported_schema.find("\"schema_version\": 1");
    check(schema_position != std::string::npos, "saved schema version was not found");
    unsupported_schema.replace(schema_position, std::string("\"schema_version\": 1").size(), "\"schema_version\": 2");
    write_text_file(path, unsupported_schema);
    bool rejected = false;
    try {
        static_cast<void>(Solver::load_config(path.string()));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "unknown configuration schema version was accepted");
    auto unsupported_precision = canonical;
    const auto precision_position = unsupported_precision.find("\"precision\": \"float64\"");
    check(precision_position != std::string::npos, "saved precision was not found");
    unsupported_precision.replace(precision_position, std::string("\"precision\": \"float64\"").size(), "\"precision\": \"float16\"");
    write_text_file(path, unsupported_precision);
    rejected = false;
    try {
        static_cast<void>(Solver::load_config(path.string()));
    } catch (const UnsupportedPrecisionError&) {
        rejected = true;
    }
    check(rejected, "unknown configuration precision was accepted");
    auto unsupported_reduced_precision = canonical;
    unsupported_reduced_precision.replace(precision_position, std::string("\"precision\": \"float64\"").size(), "\"precision\": \"float32\"");
    write_text_file(path, unsupported_reduced_precision);
    rejected = false;
    try {
        static_cast<void>(Solver::load_config(path.string()));
    } catch (const UnsupportedPrecisionError&) {
        rejected = true;
    }
    check(rejected, "float32 configuration request was accepted during load");
    auto nonfinite_number = canonical;
    const auto speed_position = nonfinite_number.find("\"wave_speed\": 1");
    check(speed_position != std::string::npos, "saved wave speed was not found");
    nonfinite_number.replace(speed_position, std::string("\"wave_speed\": 1").size(), "\"wave_speed\": nan");
    write_text_file(path, nonfinite_number);
    rejected = false;
    try {
        static_cast<void>(Solver::load_config(path.string()));
    } catch (const NumericalError&) {
        rejected = true;
    }
    check(rejected, "non-finite configuration number was accepted");
    write_text_file(path, canonical);
    const std::vector<double> a{1.0, 2.0, 4.0, 8.0};
    const std::vector<double> b{1.0, 0.0, 0.0, 0.0};
    const std::vector<double> mask{1.0, 0.0, 0.0, 0.0};
    check_close(solver.operator_loss(a, b, mask), 0.0, 1e-12, "masked loss failed");
    const auto state = solver.initialize(zeros(16));
    const auto first = solver.step(state, zeros(16));
    const auto second = solver.step(state, zeros(16));
    check(max_difference(first.phi, second.phi) == 0.0, "native step is not deterministic");
    std::filesystem::remove(path);
}

void test_performance() {
    for (const auto n : {32u, 64u, 128u}) {
        const auto solver = SpectralSolver(config(n, 0.05, Method::RK4));
        const auto state = solver.initialize(zeros(n));
        const auto started = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 10; ++repeat) (void)solver.step(state, zeros(n));
        const auto finished = std::chrono::steady_clock::now();
        const auto seconds = std::chrono::duration<double>(finished - started).count() / 10.0;
        check(std::isfinite(seconds) && seconds > 0.0, "performance timer failed");
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"event_lifecycle", test_event_lifecycle},
        {"fft_round_trip", test_fft_round_trip},
        {"operator_agreement", test_operator_agreement},
        {"manufactured_accuracy", test_manufactured_accuracy},
        {"forced_manufactured_solution", test_forced_manufactured_solution},
        {"convergence", test_convergence},
        {"energy_stability", test_energy_stability},
        {"stability_and_boundaries", test_stability_and_boundaries},
        {"global_stability_domain_and_semantics", test_global_stability_domain_and_semantics},
        {"temporal_rollout_gradients", test_temporal_rollout_gradients},
        {"gradients_and_bounded_potential", test_gradients_and_bounded_potential},
        {"precision_source_causality_and_invalid_input_rejection", test_precision_source_causality_and_invalid_input_rejection},
        {"serialization_loss_determinism", test_serialization_loss_and_determinism},
        {"performance", test_performance},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "PASS " << name << "\n";
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << "\n";
            return 1;
        }
    }
    std::cout << "SUMMARY " << passed << "/" << tests.size() << " passed\n";
    return 0;
}
