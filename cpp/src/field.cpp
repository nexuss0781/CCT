#include "cct/field.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <fftw3.h>

namespace cct {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::vector<std::size_t> strides_for(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis-- > 1;) {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    return strides;
}

std::vector<std::size_t> unravel(std::size_t index, const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> coordinates(shape.size(), 0);
    const auto strides = strides_for(shape);
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        coordinates[axis] = index / strides[axis];
        index %= strides[axis];
    }
    return coordinates;
}

std::size_t flatten(const std::vector<std::size_t>& coordinates,
                    const std::vector<std::size_t>& shape) {
    const auto strides = strides_for(shape);
    std::size_t index = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        index += coordinates[axis] * strides[axis];
    }
    return index;
}

double product(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 1.0, std::multiplies<double>());
}

double sum_square(const std::vector<double>& values) {
    return std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
}

void require_size(const std::vector<double>& values, std::size_t expected, const char* name) {
    if (values.size() != expected) {
        throw NumericalError(std::string(name) + " has an unexpected size");
    }
}

void require_finite(const std::vector<double>& values, const char* name) {
    if (std::any_of(values.begin(), values.end(), [](const double value) { return !std::isfinite(value); })) {
        throw NumericalError(std::string(name) + " contains a non-finite value");
    }
}

void require_valid_state(const FieldState& state, std::size_t expected) {
    require_size(state.phi, expected, "state phi");
    require_size(state.psi, expected, "state psi");
    require_finite(state.phi, "state phi");
    require_finite(state.psi, "state psi");
    if (!std::isfinite(state.time) || state.step_index < 0) {
        throw NumericalError("state time or step index is invalid");
    }
}

std::string precision_name(const Precision precision) {
    switch (precision) {
        case Precision::Float64:
            return "float64";
        case Precision::Float32:
            return "float32";
    }
    throw UnsupportedPrecisionError("unknown solver precision");
}

std::vector<double> zeros(std::size_t size) {
    return std::vector<double>(size, 0.0);
}

std::vector<double> add_scaled(const std::vector<double>& left,
                               const std::vector<double>& right,
                               double scale) {
    std::vector<double> result(left.size(), 0.0);
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] + scale * right[index];
    }
    return result;
}

std::vector<double> linear_combination(const std::vector<double>& a,
                                       const std::vector<double>& b,
                                       const std::vector<double>& c,
                                       const std::vector<double>& d,
                                       double wa,
                                       double wb,
                                       double wc,
                                       double wd) {
    std::vector<double> result(a.size(), 0.0);
    for (std::size_t index = 0; index < a.size(); ++index) {
        result[index] = wa * a[index] + wb * b[index] + wc * c[index] + wd * d[index];
    }
    return result;
}

std::string trim_copy(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char character) { return std::isspace(character) != 0; });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char character) { return std::isspace(character) != 0; }).base();
    if (first >= last) return {};
    return {first, last};
}

double parse_double_token(const std::string& token, const std::string& key) {
    const auto trimmed = trim_copy(token);
    if (trimmed.empty()) throw NumericalError("empty numeric configuration value: " + key);
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(trimmed, &consumed);
    } catch (const std::exception&) {
        throw NumericalError("invalid numeric configuration value: " + key);
    }
    if (consumed != trimmed.size() || !std::isfinite(value)) {
        throw NumericalError("invalid numeric configuration value: " + key);
    }
    return value;
}

std::vector<double> parse_array(const std::string& text, const std::string& key) {
    const auto marker = text.find("\"" + key + "\"");
    if (marker == std::string::npos) {
        throw NumericalError("missing configuration key: " + key);
    }
    const auto begin = text.find('[', marker);
    const auto end = text.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos || end < begin) {
        throw NumericalError("invalid array configuration key: " + key);
    }
    const auto body = trim_copy(text.substr(begin + 1, end - begin - 1));
    if (body.empty()) return {};
    if (body.back() == ',') throw NumericalError("invalid trailing comma in configuration array: " + key);
    std::stringstream stream(body);
    std::vector<double> result;
    std::string token;
    while (std::getline(stream, token, ',')) result.push_back(parse_double_token(token, key));
    return result;
}

double parse_number(const std::string& text, const std::string& key) {
    const auto marker = text.find("\"" + key + "\"");
    if (marker == std::string::npos) {
        throw NumericalError("missing configuration key: " + key);
    }
    const auto colon = text.find(':', marker);
    const auto end = text.find_first_of(",\n}", colon);
    if (colon == std::string::npos || end == std::string::npos || end <= colon) {
        throw NumericalError("invalid numeric configuration key: " + key);
    }
    return parse_double_token(text.substr(colon + 1, end - colon - 1), key);
}

std::string parse_string(const std::string& text, const std::string& key) {
    const auto marker = text.find("\"" + key + "\"");
    if (marker == std::string::npos) {
        throw NumericalError("missing configuration key: " + key);
    }
    const auto colon = text.find(':', marker);
    const auto first = colon == std::string::npos ? std::string::npos : text.find('"', colon);
    const auto second = first == std::string::npos ? std::string::npos : text.find('"', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        throw NumericalError("invalid string configuration key: " + key);
    }
    return text.substr(first + 1, second - first - 1);
}

}  // namespace

std::string boundary_name(Boundary boundary) {
    switch (boundary) {
        case Boundary::Periodic:
            return "periodic";
        case Boundary::Dirichlet:
            return "dirichlet";
        case Boundary::Neumann:
            return "neumann";
    }
    throw NumericalError("unknown boundary");
}

std::string method_name(Method method) {
    switch (method) {
        case Method::Leapfrog:
            return "leapfrog";
        case Method::RK4:
            return "rk4";
    }
    throw NumericalError("unknown method");
}

std::size_t cell_count(const std::vector<std::size_t>& shape) {
    if (shape.empty() || std::any_of(shape.begin(), shape.end(), [](std::size_t value) { return value <= 1; })) {
        throw NumericalError("shape must contain dimensions greater than one");
    }
    std::size_t count = 1U;
    for (const auto dimension : shape) {
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw NumericalError("shape cell count overflows size_t");
        }
        count *= dimension;
    }
    return count;
}

double cfl_limit(const SolverConfig& config) {
    if (!std::isfinite(config.wave_speed) || config.wave_speed <= 0.0 || config.spacing.empty() || config.shape.size() != config.spacing.size() ||
        std::any_of(config.spacing.begin(), config.spacing.end(), [](double value) { return !std::isfinite(value) || value <= 0.0; })) {
        throw NumericalError("wave speed, shape, and spacing must be positive and finite");
    }
    const auto minimum = *std::min_element(config.spacing.begin(), config.spacing.end());
    return minimum / (config.wave_speed * std::sqrt(static_cast<double>(config.shape.size())));
}

double global_stability_limit(const SolverConfig& config) {
    if (!std::isfinite(config.maximum_abs_potential) || config.maximum_abs_potential < 0.0) {
        throw StabilityError("maximum absolute potential must be finite and non-negative");
    }
    const auto spatial_term = std::accumulate(config.spacing.begin(), config.spacing.end(), 0.0, [&](const double sum, const double spacing) {
        if (!std::isfinite(spacing) || spacing <= 0.0) throw StabilityError("spacing must be finite and positive");
        return sum + (kPi * kPi) / (spacing * spacing);
    });
    const auto angular_frequency_squared = config.wave_speed * config.wave_speed * spatial_term + config.maximum_abs_potential;
    if (!std::isfinite(angular_frequency_squared) || angular_frequency_squared <= 0.0) {
        throw StabilityError("global stability spectrum is invalid");
    }
    return 2.0 / std::sqrt(angular_frequency_squared);
}

void validate_stability(const SolverConfig& config) {
    if (!std::isfinite(config.dt) || config.dt <= 0.0) {
        throw StabilityError("dt must be positive and finite");
    }
    if (!std::isfinite(config.cfl_safety) || config.cfl_safety <= 0.0 || config.cfl_safety > 1.0) {
        throw StabilityError("cfl safety must be finite and in (0, 1]");
    }
    const auto limit = std::min(cfl_limit(config), global_stability_limit(config)) * config.cfl_safety;
    if (config.dt > limit + 1e-12) {
        std::ostringstream message;
        message << "dt=" << config.dt << " exceeds conservative global stability limit=" << limit;
        throw StabilityError(message.str());
    }
}

std::vector<double> frequency_axis(std::size_t n, double spacing) {
    if (n <= 1 || !std::isfinite(spacing) || spacing <= 0.0) {
        throw NumericalError("frequency axis requires n > 1 and finite positive spacing");
    }
    std::vector<double> result(n, 0.0);
    for (std::size_t index = 0; index < n; ++index) {
        const auto signed_index = index <= n / 2 ? static_cast<long long>(index)
                                                  : static_cast<long long>(index) - static_cast<long long>(n);
        result[index] = 2.0 * kPi * static_cast<double>(signed_index) / (static_cast<double>(n) * spacing);
    }
    return result;
}

std::vector<double> apply_boundary(std::vector<double> field,
                                   const std::vector<std::size_t>& shape,
                                   Boundary boundary,
                                   double value) {
    require_size(field, cell_count(shape), "field");
    if (boundary == Boundary::Periodic) {
        return field;
    }
    for (std::size_t flat = 0; flat < field.size(); ++flat) {
        const auto coordinates = unravel(flat, shape);
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            if (coordinates[axis] != 0 && coordinates[axis] != shape[axis] - 1) {
                continue;
            }
            auto source = coordinates;
            if (boundary == Boundary::Dirichlet) {
                field[flat] = value;
            } else {
                source[axis] = coordinates[axis] == 0 ? 1 : shape[axis] - 2;
                field[flat] = field[flatten(source, shape)];
            }
            break;
        }
    }
    return field;
}

std::vector<double> finite_difference_laplacian(const std::vector<double>& field,
                                                const std::vector<std::size_t>& shape,
                                                const std::vector<double>& spacing,
                                                Boundary boundary) {
    const auto count = cell_count(shape);
    require_size(field, count, "field");
    require_finite(field, "field");
    if (spacing.size() != shape.size() || std::any_of(spacing.begin(), spacing.end(), [](double value) { return !std::isfinite(value) || value <= 0.0; })) {
        throw NumericalError("spacing does not match shape");
    }
    std::vector<double> result(count, 0.0);
    for (std::size_t flat = 0; flat < count; ++flat) {
        const auto coordinates = unravel(flat, shape);
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            auto plus = coordinates;
            auto minus = coordinates;
            bool plus_outside = false;
            bool minus_outside = false;
            if (coordinates[axis] + 1 >= shape[axis]) {
                plus_outside = true;
                plus[axis] = shape[axis] - 1;
            } else {
                plus[axis] += 1;
            }
            if (coordinates[axis] == 0) {
                minus_outside = true;
                minus[axis] = 0;
            } else {
                minus[axis] -= 1;
            }
            if (boundary == Boundary::Periodic) {
                if (plus_outside) {
                    plus[axis] = 0;
                }
                if (minus_outside) {
                    minus[axis] = shape[axis] - 1;
                }
            }
            const double center = field[flat];
            const double plus_value = plus_outside && boundary == Boundary::Dirichlet
                                          ? 0.0
                                          : field[flatten(plus, shape)];
            const double minus_value = minus_outside && boundary == Boundary::Dirichlet
                                           ? 0.0
                                           : field[flatten(minus, shape)];
            result[flat] += (plus_value - 2.0 * center + minus_value) / (spacing[axis] * spacing[axis]);
        }
    }
    if (boundary != Boundary::Periodic) {
        result = apply_boundary(std::move(result), shape, boundary, 0.0);
    }
    return result;
}

std::vector<double> spectral_laplacian(const std::vector<double>& field,
                                       const std::vector<std::size_t>& shape,
                                       const std::vector<double>& spacing) {
    const auto count = cell_count(shape);
    require_size(field, count, "field");
    require_finite(field, "field");
    if (shape.empty() || shape.size() > 3 || spacing.size() != shape.size() ||
        std::any_of(spacing.begin(), spacing.end(), [](double value) { return !std::isfinite(value) || value <= 0.0; })) {
        throw NumericalError("FFTW spectral path supports one to three dimensions with finite positive spacing");
    }
    std::vector<int> dimensions;
    dimensions.reserve(shape.size());
    for (const auto value : shape) {
        dimensions.push_back(static_cast<int>(value));
    }
    const auto last = shape.back();
    const auto compact_count = (count / last) * (last / 2 + 1);
    auto* input = static_cast<double*>(fftw_malloc(sizeof(double) * count));
    auto* spectrum = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * compact_count));
    auto* output = static_cast<double*>(fftw_malloc(sizeof(double) * count));
    if (input == nullptr || spectrum == nullptr || output == nullptr) {
        if (input) fftw_free(input);
        if (spectrum) fftw_free(spectrum);
        if (output) fftw_free(output);
        throw NumericalError("FFTW allocation failed");
    }
    std::copy(field.begin(), field.end(), input);
    const auto forward = fftw_plan_dft_r2c(static_cast<int>(shape.size()), dimensions.data(), input, spectrum, FFTW_ESTIMATE);
    const auto inverse = fftw_plan_dft_c2r(static_cast<int>(shape.size()), dimensions.data(), spectrum, output, FFTW_ESTIMATE);
    if (forward == nullptr || inverse == nullptr) {
        if (forward) fftw_destroy_plan(forward);
        if (inverse) fftw_destroy_plan(inverse);
        fftw_free(input);
        fftw_free(spectrum);
        fftw_free(output);
        throw NumericalError("FFTW plan creation failed");
    }
    fftw_execute(forward);
    const auto axes_strides = strides_for(shape);
    for (std::size_t compact = 0; compact < compact_count; ++compact) {
        std::size_t remainder = compact;
        std::vector<std::size_t> coordinates(shape.size(), 0);
        for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis) {
            coordinates[axis] = remainder / axes_strides[axis];
            remainder %= axes_strides[axis];
        }
        coordinates.back() = remainder;
        double k_squared = 0.0;
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            const auto n = shape[axis];
            const auto signed_index = axis + 1 == shape.size()
                                          ? static_cast<long long>(coordinates[axis])
                                          : (coordinates[axis] <= n / 2
                                                 ? static_cast<long long>(coordinates[axis])
                                                 : static_cast<long long>(coordinates[axis]) - static_cast<long long>(n));
            const auto k = 2.0 * kPi * static_cast<double>(signed_index) / (static_cast<double>(n) * spacing[axis]);
            k_squared += k * k;
        }
        spectrum[compact][0] *= -k_squared;
        spectrum[compact][1] *= -k_squared;
    }
    fftw_execute(inverse);
    std::vector<double> result(count, 0.0);
    const double scale = 1.0 / static_cast<double>(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = output[index] * scale;
    }
    fftw_destroy_plan(forward);
    fftw_destroy_plan(inverse);
    fftw_free(input);
    fftw_free(spectrum);
    fftw_free(output);
    return result;
}

std::vector<double> manufactured_mode(const std::vector<std::size_t>& shape,
                                      const std::vector<double>& spacing,
                                      const std::vector<int>& mode,
                                      double phase) {
    const auto count = cell_count(shape);
    if (shape.size() != spacing.size() || shape.size() != mode.size()) {
        throw NumericalError("manufactured mode dimensions do not match");
    }
    std::vector<double> result(count, 0.0);
    for (std::size_t flat = 0; flat < count; ++flat) {
        const auto coordinates = unravel(flat, shape);
        double argument = phase;
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            argument += 2.0 * kPi * static_cast<double>(mode[axis]) * static_cast<double>(coordinates[axis]) /
                        static_cast<double>(shape[axis]);
        }
        result[flat] = std::sin(argument);
    }
    return result;
}

double manufactured_mode_frequency(const std::vector<std::size_t>& shape,
                                   const std::vector<double>& spacing,
                                   const std::vector<int>& mode,
                                   double wave_speed,
                                   double potential) {
    if (shape.size() != spacing.size() || shape.size() != mode.size()) {
        throw NumericalError("manufactured mode dimensions do not match");
    }
    double k_squared = 0.0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto k = 2.0 * kPi * static_cast<double>(mode[axis]) /
                       (static_cast<double>(shape[axis]) * spacing[axis]);
        k_squared += k * k;
    }
    return std::sqrt(wave_speed * wave_speed * k_squared + potential);
}

Solver::Solver(SolverConfig config) : config_(std::move(config)) {
    if (config_.schema_version != 1) {
        throw NumericalError("unsupported solver configuration schema version");
    }
    if (config_.precision != Precision::Float64) {
        throw UnsupportedPrecisionError("only IEEE-754 float64 is supported by the native reference solver");
    }
    if (config_.shape.empty() || config_.spacing.size() != config_.shape.size()) {
        throw NumericalError("solver shape and spacing must have the same nonzero rank");
    }
    (void)cell_count(config_.shape);
    validate_stability(config_);
}

std::vector<double> Solver::normalize_source(const std::vector<double>& source) const {
    const auto count = cell_count(config_.shape);
    std::vector<double> normalized;
    if (source.empty()) normalized = zeros(count);
    else if (source.size() == 1U) normalized = std::vector<double>(count, source[0]);
    else {
        require_size(source, count, "source");
        normalized = source;
    }
    require_finite(normalized, "source");
    return normalized;
}

std::vector<double> Solver::normalize_potential(const std::vector<double>& potential) const {
    const auto count = cell_count(config_.shape);
    std::vector<double> normalized;
    if (potential.empty()) normalized = zeros(count);
    else if (potential.size() == 1) normalized = std::vector<double>(count, potential[0]);
    else {
        require_size(potential, count, "potential");
        normalized = potential;
    }
    for (const auto value : normalized) {
        if (!std::isfinite(value) || std::abs(value) > config_.maximum_abs_potential + 1e-12) {
            throw StabilityError("potential exceeds the configured global stability domain");
        }
    }
    return normalized;
}

FieldState Solver::initialize(const std::vector<double>& phi0,
                              const std::vector<double>& psi0,
                              double time) const {
    const auto count = cell_count(config_.shape);
    require_size(phi0, count, "phi0");
    require_finite(phi0, "phi0");
    std::vector<double> psi = psi0.empty() ? zeros(count) : psi0;
    require_size(psi, count, "psi0");
    require_finite(psi, "psi0");
    if (!std::isfinite(time)) {
        throw NumericalError("initial time must be finite");
    }
    FieldState state{apply_boundary(phi0, config_.shape, config_.boundary),
                     apply_boundary(psi, config_.shape, config_.boundary, 0.0),
                     time,
                     0};
    require_valid_state(state, count);
    return state;
}

FieldState Solver::leapfrog_step(const FieldState& state,
                                 const std::vector<double>& source,
                                 const std::vector<double>& potential) const {
    const auto first = acceleration(state.phi, source, potential);
    std::vector<double> psi_half = add_scaled(state.psi, first, 0.5 * config_.dt);
    std::vector<double> phi_new = add_scaled(state.phi, psi_half, config_.dt);
    phi_new = apply_boundary(std::move(phi_new), config_.shape, config_.boundary);
    const auto second = acceleration(phi_new, source, potential);
    std::vector<double> psi_new = add_scaled(psi_half, second, 0.5 * config_.dt);
    psi_new = apply_boundary(std::move(psi_new), config_.shape, config_.boundary, 0.0);
    return FieldState{std::move(phi_new), std::move(psi_new), state.time + config_.dt, state.step_index + 1};
}

FieldState Solver::rk4_step(const FieldState& state,
                            const std::vector<double>& source,
                            const std::vector<double>& potential) const {
    const auto rhs_acceleration = [&](const std::vector<double>& phi) {
        return acceleration(phi, source, potential);
    };
    const auto k1_phi = state.psi;
    const auto k1_psi = rhs_acceleration(state.phi);
    const auto phi2 = add_scaled(state.phi, k1_phi, 0.5 * config_.dt);
    const auto psi2 = add_scaled(state.psi, k1_psi, 0.5 * config_.dt);
    const auto k2_phi = psi2;
    const auto k2_psi = rhs_acceleration(phi2);
    const auto phi3 = add_scaled(state.phi, k2_phi, 0.5 * config_.dt);
    const auto psi3 = add_scaled(state.psi, k2_psi, 0.5 * config_.dt);
    const auto k3_phi = psi3;
    const auto k3_psi = rhs_acceleration(phi3);
    const auto phi4 = add_scaled(state.phi, k3_phi, config_.dt);
    const auto psi4 = add_scaled(state.psi, k3_psi, config_.dt);
    const auto k4_phi = psi4;
    const auto k4_psi = rhs_acceleration(phi4);
    std::vector<double> phi_new = add_scaled(
        state.phi,
        linear_combination(k1_phi, k2_phi, k3_phi, k4_phi, 1.0, 2.0, 2.0, 1.0),
        config_.dt / 6.0);
    std::vector<double> psi_new = add_scaled(
        state.psi,
        linear_combination(k1_psi, k2_psi, k3_psi, k4_psi, 1.0, 2.0, 2.0, 1.0),
        config_.dt / 6.0);
    phi_new = apply_boundary(std::move(phi_new), config_.shape, config_.boundary);
    psi_new = apply_boundary(std::move(psi_new), config_.shape, config_.boundary, 0.0);
    return FieldState{std::move(phi_new), std::move(psi_new), state.time + config_.dt, state.step_index + 1};
}

FieldState Solver::step(const FieldState& state,
                        const std::vector<double>& source,
                        const std::vector<double>& potential) const {
    const auto count = cell_count(config_.shape);
    require_valid_state(state, count);
    const auto source_value = normalize_source(source);
    const auto potential_value = normalize_potential(potential);
    FieldState next = config_.method == Method::Leapfrog
                          ? leapfrog_step(state, source_value, potential_value)
                          : rk4_step(state, source_value, potential_value);
    require_valid_state(next, count);
    return next;
}

Trajectory Solver::rollout(const FieldState& state,
                           const std::vector<std::vector<double>>& source_sequence,
                           const std::vector<double>& potential,
                           bool include_initial) const {
    Trajectory trajectory;
    require_valid_state(state, cell_count(config_.shape));
    FieldState current = state;
    if (include_initial) {
        trajectory.phi.push_back(current.phi);
        trajectory.psi.push_back(current.psi);
        trajectory.time.push_back(current.time);
        trajectory.step_index.push_back(current.step_index);
    }
    for (const auto& source : source_sequence) {
        current = step(current, source, potential);
        trajectory.phi.push_back(current.phi);
        trajectory.psi.push_back(current.psi);
        trajectory.time.push_back(current.time);
        trajectory.step_index.push_back(current.step_index);
    }
    return trajectory;
}

double Solver::energy(const FieldState& state, const std::vector<double>& potential) const {
    require_valid_state(state, cell_count(config_.shape));
    const auto pot = normalize_potential(potential);
    const auto gradient = finite_difference_laplacian(state.phi, config_.shape, config_.spacing, config_.boundary);
    const auto volume = product(config_.spacing);
    double kinetic = 0.5 * sum_square(state.psi) * volume;
    double gradient_energy = 0.0;
    for (std::size_t index = 0; index < gradient.size(); ++index) {
        gradient_energy += -0.5 * state.phi[index] * gradient[index] * volume;
    }
    double potential_energy = 0.0;
    for (std::size_t index = 0; index < pot.size(); ++index) {
        potential_energy += 0.5 * pot[index] * state.phi[index] * state.phi[index] * volume;
    }
    const auto total = kinetic + config_.wave_speed * config_.wave_speed * gradient_energy + potential_energy;
    if (!std::isfinite(total)) {
        throw NumericalError("energy diagnostic is non-finite");
    }
    return total;
}

double Solver::operator_loss(const std::vector<double>& prediction,
                             const std::vector<double>& target,
                             const std::vector<double>& mask) const {
    if (target.empty()) {
        throw NumericalError("operator loss target must be non-empty");
    }
    require_size(prediction, target.size(), "prediction");
    require_finite(prediction, "prediction");
    require_finite(target, "target");
    if (!mask.empty()) {
        require_size(mask, target.size(), "mask");
        if (std::any_of(mask.begin(), mask.end(), [](const double value) { return !std::isfinite(value) || value < 0.0; })) {
            throw NumericalError("operator loss mask must be finite and non-negative");
        }
    }
    double numerator = 0.0;
    double denominator = mask.empty() ? static_cast<double>(target.size()) : 0.0;
    for (std::size_t index = 0; index < target.size(); ++index) {
        const auto weight = mask.empty() ? 1.0 : mask[index];
        numerator += weight * (prediction[index] - target[index]) * (prediction[index] - target[index]);
        if (!mask.empty()) denominator += weight;
    }
    if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator <= 0.0) {
        throw NumericalError("operator loss normalization is invalid");
    }
    const auto loss = numerator / denominator;
    if (!std::isfinite(loss)) {
        throw NumericalError("operator loss is non-finite");
    }
    return loss;
}

void Solver::save_config(const std::string& path) const {
    std::ofstream stream(path);
    if (!stream) throw NumericalError("could not open configuration for writing");
    stream << std::setprecision(17) << "{\n";
    stream << "  \"schema_version\": " << config_.schema_version << ",\n";
    stream << "  \"precision\": \"" << precision_name(config_.precision) << "\",\n";
    stream << "  \"shape\": [";
    for (std::size_t axis = 0; axis < config_.shape.size(); ++axis) {
        if (axis) stream << ", ";
        stream << config_.shape[axis];
    }
    stream << "],\n  \"spacing\": [";
    for (std::size_t axis = 0; axis < config_.spacing.size(); ++axis) {
        if (axis) stream << ", ";
        stream << config_.spacing[axis];
    }
    stream << "],\n";
    stream << "  \"wave_speed\": " << config_.wave_speed << ",\n";
    stream << "  \"dt\": " << config_.dt << ",\n";
    stream << "  \"boundary\": \"" << boundary_name(config_.boundary) << "\",\n";
    stream << "  \"method\": \"" << method_name(config_.method) << "\",\n";
    stream << "  \"cfl_safety\": " << config_.cfl_safety << ",\n";
    stream << "  \"maximum_abs_potential\": " << config_.maximum_abs_potential << "\n}\n";
}

SolverConfig Solver::load_config(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw NumericalError("could not open configuration for reading");
    std::stringstream buffer;
    buffer << stream.rdbuf();
    const auto text = buffer.str();
    SolverConfig config;
    const auto schema_version = parse_number(text, "schema_version");
    if (schema_version != 1.0) {
        throw NumericalError("unsupported solver configuration schema version");
    }
    config.schema_version = 1;
    if (text.find("\"precision\"") != std::string::npos) {
        const auto precision = parse_string(text, "precision");
        if (precision == "float64") config.precision = Precision::Float64;
        else throw UnsupportedPrecisionError("unsupported solver configuration precision");
    }
    const auto shape_values = parse_array(text, "shape");
    const auto spacing_values = parse_array(text, "spacing");
    config.shape.reserve(shape_values.size());
    for (const auto value : shape_values) {
        if (value < 2.0 || std::floor(value) != value || value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            throw NumericalError("shape contains an invalid dimension");
        }
        config.shape.push_back(static_cast<std::size_t>(value));
    }
    config.spacing = spacing_values;
    config.wave_speed = parse_number(text, "wave_speed");
    config.dt = parse_number(text, "dt");
    const auto boundary = parse_string(text, "boundary");
    const auto method = parse_string(text, "method");
    if (boundary == "periodic") config.boundary = Boundary::Periodic;
    else if (boundary == "dirichlet") config.boundary = Boundary::Dirichlet;
    else if (boundary == "neumann") config.boundary = Boundary::Neumann;
    else throw NumericalError("unsupported solver boundary");
    if (method == "leapfrog") config.method = Method::Leapfrog;
    else if (method == "rk4") config.method = Method::RK4;
    else throw NumericalError("unsupported solver method");
    config.cfl_safety = parse_number(text, "cfl_safety");
    if (text.find("\"maximum_abs_potential\"") != std::string::npos) {
        config.maximum_abs_potential = parse_number(text, "maximum_abs_potential");
    }
    return config;
}

SpectralSolver::SpectralSolver(SolverConfig config) : Solver(std::move(config)) {
    if (config_.boundary != Boundary::Periodic) {
        throw NumericalError("SpectralSolver supports only periodic boundaries");
    }
}

std::vector<double> SpectralSolver::acceleration(const std::vector<double>& phi,
                                                 const std::vector<double>& source,
                                                 const std::vector<double>& potential) const {
    const auto laplacian = spectral_laplacian(phi, config_.shape, config_.spacing);
    std::vector<double> result(phi.size(), 0.0);
    for (std::size_t index = 0; index < phi.size(); ++index) {
        result[index] = config_.wave_speed * config_.wave_speed * laplacian[index] - potential[index] * phi[index] + source[index];
    }
    return result;
}

FiniteDifferenceSolver::FiniteDifferenceSolver(SolverConfig config) : Solver(std::move(config)) {}

std::vector<double> FiniteDifferenceSolver::acceleration(const std::vector<double>& phi,
                                                         const std::vector<double>& source,
                                                         const std::vector<double>& potential) const {
    const auto laplacian = finite_difference_laplacian(phi, config_.shape, config_.spacing, config_.boundary);
    std::vector<double> result(phi.size(), 0.0);
    for (std::size_t index = 0; index < phi.size(); ++index) {
        result[index] = config_.wave_speed * config_.wave_speed * laplacian[index] - potential[index] * phi[index] + source[index];
    }
    return result;
}

}  // namespace cct

namespace cct {

OneStepLossGradients leapfrog_operator_loss_gradients(
    const Solver& solver,
    const FieldState& state,
    const std::vector<double>& source,
    const std::vector<double>& potential,
    const std::vector<double>& target) {
    const auto next = solver.step(state, source, potential);
    if (target.size() != next.phi.size()) {
        throw NumericalError("gradient target has an unexpected size");
    }
    require_finite(target, "gradient target");
    const auto& config = solver.config();
    const auto dt_factor = 0.5 * config.dt * config.dt;
    const auto count = static_cast<double>(next.phi.size());
    OneStepLossGradients gradients{std::vector<double>(next.phi.size(), 0.0),
                                   std::vector<double>(next.phi.size(), 0.0)};
    for (std::size_t index = 0; index < next.phi.size(); ++index) {
        const auto coordinates = [&]() {
            std::vector<std::size_t> result(config.shape.size(), 0);
            std::size_t remainder = index;
            for (std::size_t axis = config.shape.size(); axis-- > 0;) {
                result[axis] = remainder % config.shape[axis];
                remainder /= config.shape[axis];
            }
            return result;
        }();
        bool fixed_boundary = false;
        if (config.boundary == Boundary::Dirichlet) {
            for (std::size_t axis = 0; axis < config.shape.size(); ++axis) {
                if (coordinates[axis] == 0 || coordinates[axis] == config.shape[axis] - 1) {
                    fixed_boundary = true;
                }
            }
        }
        if (!fixed_boundary) {
            const auto dloss_dphi = 2.0 * (next.phi[index] - target[index]) / count;
            gradients.source[index] = dloss_dphi * dt_factor;
            gradients.potential[index] = dloss_dphi * (-dt_factor * state.phi[index]);
        }
    }
    require_finite(gradients.source, "one-step source gradients");
    require_finite(gradients.potential, "one-step potential gradients");
    return gradients;
}

TemporalRolloutGradients temporal_rollout_loss_gradients(
    const Solver& solver,
    const FieldState& initial,
    const std::vector<std::vector<double>>& source_sequence,
    const std::vector<double>& potential,
    const std::vector<std::vector<double>>& targets,
    const double finite_difference_epsilon) {
    if (finite_difference_epsilon <= 0.0 || !std::isfinite(finite_difference_epsilon)) {
        throw NumericalError("temporal gradient finite-difference epsilon must be positive and finite");
    }
    const auto count = cell_count(solver.config().shape);
    require_valid_state(initial, count);
    if (source_sequence.empty() || source_sequence.size() != targets.size()) {
        throw NumericalError("temporal gradient source and target rollout lengths do not match");
    }
    auto expand = [&](const std::vector<double>& values, const char* name) {
        std::vector<double> normalized;
        if (values.empty()) normalized = zeros(count);
        else if (values.size() == 1U) normalized = std::vector<double>(count, values.front());
        else {
            require_size(values, count, name);
            normalized = values;
        }
        require_finite(normalized, name);
        return normalized;
    };
    std::vector<std::vector<double>> normalized_sources;
    normalized_sources.reserve(source_sequence.size());
    for (const auto& source : source_sequence) normalized_sources.push_back(expand(source, "source"));
    const auto normalized_potential = expand(potential, "potential");
    const auto objective_denominator = static_cast<double>(source_sequence.size() * count);
    std::vector<FieldState> states{initial};
    states.reserve(source_sequence.size() + 1U);
    for (std::size_t step_index = 0; step_index < source_sequence.size(); ++step_index) {
        states.push_back(solver.step(states.back(), normalized_sources[step_index], normalized_potential));
        require_size(targets[step_index], count, "temporal target");
        require_finite(targets[step_index], "temporal target");
    }
    TemporalRolloutGradients gradients;
    gradients.source.assign(source_sequence.size(), std::vector<double>(count, 0.0));
    gradients.potential.assign(count, 0.0);
    std::vector<double> lambda_phi(count, 0.0);
    std::vector<double> lambda_psi(count, 0.0);
    for (std::size_t step_index = source_sequence.size(); step_index-- > 0U;) {
        const auto& next = states[step_index + 1U];
        for (std::size_t index = 0; index < count; ++index) {
            const auto difference = next.phi[index] - targets[step_index][index];
            gradients.loss += difference * difference / objective_denominator;
            lambda_phi[index] += 2.0 * difference / objective_denominator;
        }
        const auto& current = states[step_index];
        auto step_with = [&](const FieldState& state, const std::vector<double>& source, const std::vector<double>& potential_value) {
            return solver.step(state, source, potential_value);
        };
        std::vector<double> previous_lambda_phi(count, 0.0);
        std::vector<double> previous_lambda_psi(count, 0.0);
        auto accumulate_state_column = [&](const FieldState& plus, const FieldState& minus, const std::size_t index,
                                           std::vector<double>& target_lambda) {
            double value = 0.0;
            for (std::size_t output = 0; output < count; ++output) {
                value += lambda_phi[output] * (plus.phi[output] - minus.phi[output]) / (2.0 * finite_difference_epsilon);
                value += lambda_psi[output] * (plus.psi[output] - minus.psi[output]) / (2.0 * finite_difference_epsilon);
            }
            target_lambda[index] += value;
        };
        for (std::size_t index = 0; index < count; ++index) {
            auto plus = current;
            auto minus = current;
            plus.phi[index] += finite_difference_epsilon;
            minus.phi[index] -= finite_difference_epsilon;
            accumulate_state_column(step_with(plus, normalized_sources[step_index], normalized_potential),
                                    step_with(minus, normalized_sources[step_index], normalized_potential), index, previous_lambda_phi);
            plus = current;
            minus = current;
            plus.psi[index] += finite_difference_epsilon;
            minus.psi[index] -= finite_difference_epsilon;
            accumulate_state_column(step_with(plus, normalized_sources[step_index], normalized_potential),
                                    step_with(minus, normalized_sources[step_index], normalized_potential), index, previous_lambda_psi);
            auto source_plus = normalized_sources[step_index];
            auto source_minus = normalized_sources[step_index];
            source_plus[index] += finite_difference_epsilon;
            source_minus[index] -= finite_difference_epsilon;
            const auto source_plus_state = step_with(current, source_plus, normalized_potential);
            const auto source_minus_state = step_with(current, source_minus, normalized_potential);
            for (std::size_t output = 0; output < count; ++output) {
                gradients.source[step_index][index] += lambda_phi[output] * (source_plus_state.phi[output] - source_minus_state.phi[output]) /
                                                       (2.0 * finite_difference_epsilon);
                gradients.source[step_index][index] += lambda_psi[output] * (source_plus_state.psi[output] - source_minus_state.psi[output]) /
                                                       (2.0 * finite_difference_epsilon);
            }
            auto potential_plus = normalized_potential;
            auto potential_minus = normalized_potential;
            potential_plus[index] += finite_difference_epsilon;
            potential_minus[index] -= finite_difference_epsilon;
            const auto potential_plus_state = step_with(current, normalized_sources[step_index], potential_plus);
            const auto potential_minus_state = step_with(current, normalized_sources[step_index], potential_minus);
            for (std::size_t output = 0; output < count; ++output) {
                gradients.potential[index] += lambda_phi[output] * (potential_plus_state.phi[output] - potential_minus_state.phi[output]) /
                                              (2.0 * finite_difference_epsilon);
                gradients.potential[index] += lambda_psi[output] * (potential_plus_state.psi[output] - potential_minus_state.psi[output]) /
                                              (2.0 * finite_difference_epsilon);
            }
        }
        lambda_phi = std::move(previous_lambda_phi);
        lambda_psi = std::move(previous_lambda_psi);
    }
    if (!std::isfinite(gradients.loss)) throw NumericalError("temporal rollout loss is non-finite");
    for (const auto& source_gradient : gradients.source) require_finite(source_gradient, "temporal source gradients");
    require_finite(gradients.potential, "temporal potential gradients");
    return gradients;
}

std::vector<double> bounded_local_potential(const std::vector<double>& raw,
                                            double max_value) {
    if (!std::isfinite(max_value) || max_value <= 0.0) {
        throw NumericalError("max_value must be positive and finite");
    }
    require_finite(raw, "potential parameters");
    std::vector<double> result(raw.size(), 0.0);
    for (std::size_t index = 0; index < raw.size(); ++index) {
        result[index] = max_value / (1.0 + std::exp(-raw[index]));
    }
    return result;
}

std::vector<double> bounded_local_potential_gradient(const std::vector<double>& raw,
                                                     double max_value) {
    const auto bounded = bounded_local_potential(raw, max_value);
    std::vector<double> result(raw.size(), 0.0);
    for (std::size_t index = 0; index < raw.size(); ++index) {
        const auto normalized = bounded[index] / max_value;
        result[index] = max_value * normalized * (1.0 - normalized);
    }
    return result;
}

}  // namespace cct
