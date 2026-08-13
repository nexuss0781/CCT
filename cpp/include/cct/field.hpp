#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class Boundary { Periodic, Dirichlet, Neumann };
enum class Method { Leapfrog, RK4 };

struct SolverConfig {
    int schema_version = 1;
    std::vector<std::size_t> shape;
    std::vector<double> spacing;
    double wave_speed = 1.0;
    double dt = 0.05;
    Boundary boundary = Boundary::Periodic;
    Method method = Method::Leapfrog;
    double cfl_safety = 0.9;
    double maximum_abs_potential = 1.0;
};

struct FieldState {
    std::vector<double> phi;
    std::vector<double> psi;
    double time = 0.0;
    std::int64_t step_index = 0;
};

struct Trajectory {
    std::vector<std::vector<double>> phi;
    std::vector<std::vector<double>> psi;
    std::vector<double> time;
    std::vector<std::int64_t> step_index;
};

struct PerformanceSample {
    std::size_t cells = 0;
    double compile_seconds = 0.0;
    double steady_seconds = 0.0;
};

class NumericalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class StabilityError : public NumericalError {
public:
    using NumericalError::NumericalError;
};

class UnsupportedPrecisionError : public NumericalError {
public:
    using NumericalError::NumericalError;
};

std::string boundary_name(Boundary boundary);
std::string method_name(Method method);

double cfl_limit(const SolverConfig& config);
/** Returns a conservative global bound for the configured spatial operator and bounded potential. */
double global_stability_limit(const SolverConfig& config);
void validate_stability(const SolverConfig& config);
std::size_t cell_count(const std::vector<std::size_t>& shape);

std::vector<double> frequency_axis(std::size_t n, double spacing);
/** Finite-difference semantics apply the requested periodic, Dirichlet, or Neumann boundary closure. */
std::vector<double> finite_difference_laplacian(
    const std::vector<double>& field,
    const std::vector<std::size_t>& shape,
    const std::vector<double>& spacing,
    Boundary boundary);
/** Spectral semantics are periodic Fourier modes only; non-periodic boundaries are rejected by SpectralSolver. */
std::vector<double> spectral_laplacian(
    const std::vector<double>& field,
    const std::vector<std::size_t>& shape,
    const std::vector<double>& spacing);
double fft_round_trip_error(
    const std::vector<double>& field,
    const std::vector<std::size_t>& shape);

std::vector<double> apply_boundary(
    std::vector<double> field,
    const std::vector<std::size_t>& shape,
    Boundary boundary,
    double value = 0.0);

std::vector<double> manufactured_mode(
    const std::vector<std::size_t>& shape,
    const std::vector<double>& spacing,
    const std::vector<int>& mode,
    double phase = 0.0);
double manufactured_mode_frequency(
    const std::vector<std::size_t>& shape,
    const std::vector<double>& spacing,
    const std::vector<int>& mode,
    double wave_speed,
    double potential = 0.0);

class Solver {
public:
    explicit Solver(SolverConfig config);
    virtual ~Solver() = default;

    const SolverConfig& config() const noexcept { return config_; }
    FieldState initialize(const std::vector<double>& phi0,
                          const std::vector<double>& psi0 = {},
                          double time = 0.0) const;
    FieldState step(const FieldState& state,
                    const std::vector<double>& source = {},
                    const std::vector<double>& potential = {}) const;
    Trajectory rollout(const FieldState& state,
                       const std::vector<std::vector<double>>& source_sequence,
                       const std::vector<double>& potential = {},
                       bool include_initial = true) const;
    double energy(const FieldState& state,
                  const std::vector<double>& potential = {}) const;
    double operator_loss(const std::vector<double>& prediction,
                         const std::vector<double>& target,
                         const std::vector<double>& mask = {}) const;
    void save_config(const std::string& path) const;
    static SolverConfig load_config(const std::string& path);
    virtual std::string implementation_name() const = 0;

protected:
    virtual std::vector<double> acceleration(
        const std::vector<double>& phi,
        const std::vector<double>& source,
        const std::vector<double>& potential) const = 0;
    std::vector<double> normalize_source(const std::vector<double>& source) const;
    std::vector<double> normalize_potential(const std::vector<double>& potential) const;
    FieldState leapfrog_step(const FieldState& state,
                             const std::vector<double>& source,
                             const std::vector<double>& potential) const;
    FieldState rk4_step(const FieldState& state,
                        const std::vector<double>& source,
                        const std::vector<double>& potential) const;
    SolverConfig config_;
};

class SpectralSolver final : public Solver {
public:
    explicit SpectralSolver(SolverConfig config);
    std::string implementation_name() const override { return "spectral"; }

protected:
    std::vector<double> acceleration(
        const std::vector<double>& phi,
        const std::vector<double>& source,
        const std::vector<double>& potential) const override;
};

class FiniteDifferenceSolver final : public Solver {
public:
    explicit FiniteDifferenceSolver(SolverConfig config);
    std::string implementation_name() const override { return "finite_difference"; }

protected:
    std::vector<double> acceleration(
        const std::vector<double>& phi,
        const std::vector<double>& source,
        const std::vector<double>& potential) const override;
};

/** Derivatives of one leapfrog step only; this type is not a temporal rollout adjoint. */
struct OneStepLossGradients {
    std::vector<double> source;
    std::vector<double> potential;
};

OneStepLossGradients leapfrog_operator_loss_gradients(
    const Solver& solver,
    const FieldState& state,
    const std::vector<double>& source,
    const std::vector<double>& potential,
    const std::vector<double>& target);

/**
 * Full multi-step loss sensitivities. The implementation is a deterministic
 * central-difference local-Jacobian adjoint oracle across the complete rollout;
 * it is intentionally separate from the optimized one-step helper.
 */
struct TemporalRolloutGradients {
    std::vector<std::vector<double>> source;
    std::vector<double> potential;
    double loss = 0.0;
};

TemporalRolloutGradients temporal_rollout_loss_gradients(
    const Solver& solver,
    const FieldState& initial,
    const std::vector<std::vector<double>>& source_sequence,
    const std::vector<double>& potential,
    const std::vector<std::vector<double>>& targets,
    double finite_difference_epsilon = 1e-6);

std::vector<double> bounded_local_potential(
    const std::vector<double>& raw,
    double max_value = 1.0);
std::vector<double> bounded_local_potential_gradient(
    const std::vector<double>& raw,
    double max_value = 1.0);

}  // namespace cct
