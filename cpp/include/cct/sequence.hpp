#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

struct SequenceConfig {
    std::size_t input_dim = 1;
    std::size_t hidden_dim = 16;
    std::size_t output_dim = 1;
    double gate_epsilon = 1e-5;
    std::uint64_t seed = 0;
};

struct SequenceState {
    std::vector<double> hidden;
    std::vector<double> previous_input;
};

struct SequenceOutput {
    std::vector<std::vector<double>> outputs;
    SequenceState final_state;
};

struct SequenceGradients {
    double loss = 0.0;
    std::vector<double> d_input_projection;
    std::vector<double> d_previous_projection;
    std::vector<double> d_retain_projection;
    std::vector<double> d_write_projection;
    std::vector<double> d_output_projection;
    std::vector<double> d_skip_projection;
    std::vector<double> d_bias;
    std::vector<double> d_retain_bias;
    std::vector<double> d_write_bias;
    std::vector<double> d_output_bias;
};

class SequenceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class SelectiveSequenceCore {
public:
    explicit SelectiveSequenceCore(SequenceConfig config);

    const SequenceConfig& config() const noexcept { return config_; }
    SequenceState initial_state() const;
    SequenceState step(const std::vector<double>& input, const SequenceState& state,
                       std::vector<double>* output = nullptr) const;
    SequenceOutput forward(const std::vector<std::vector<double>>& inputs,
                           const std::vector<std::uint8_t>& mask = {},
                           const SequenceState* initial = nullptr) const;
    SequenceOutput forward_scan(const std::vector<std::vector<double>>& inputs,
                                const std::vector<std::uint8_t>& mask = {},
                                const SequenceState* initial = nullptr) const;
    SequenceGradients loss_and_gradients(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        const std::vector<std::uint8_t>& mask = {},
        const SequenceState* initial = nullptr) const;
    void apply_sgd(const SequenceGradients& gradients, double learning_rate,
                   double clip_norm = 1.0);
    double loss_only(const std::vector<std::vector<double>>& inputs,
                     const std::vector<std::vector<double>>& targets,
                     const std::vector<std::uint8_t>& mask = {},
                     const SequenceState* initial = nullptr) const;
    std::vector<double> parameter_vector() const;
    void set_parameter_vector(const std::vector<double>& values);

    std::size_t parameter_count() const noexcept;
    double transition_radius_bound() const;
    double state_norm(const SequenceState& state) const;
    double output_norm(const std::vector<double>& output) const;

    void save_checkpoint(const std::string& path, std::uint64_t optimizer_step = 0) const;
    static SelectiveSequenceCore load_checkpoint(const std::string& path,
                                                 std::uint64_t* optimizer_step = nullptr);

private:
    struct Parameters {
        std::vector<double> input_projection;
        std::vector<double> previous_projection;
        std::vector<double> retain_projection;
        std::vector<double> write_projection;
        std::vector<double> output_projection;
        std::vector<double> skip_projection;
        std::vector<double> bias;
        std::vector<double> retain_bias;
        std::vector<double> write_bias;
        std::vector<double> output_bias;
    };

    SequenceConfig config_;
    Parameters parameters_;

    void validate_input(const std::vector<double>& input) const;
    void validate_state(const SequenceState& state) const;
    std::vector<double> matvec(const std::vector<double>& matrix,
                               std::size_t rows, std::size_t columns,
                               const std::vector<double>& vector) const;
    std::vector<double> affine(const std::vector<double>& matrix,
                               const std::vector<double>& bias,
                               std::size_t rows, std::size_t columns,
                               const std::vector<double>& vector) const;
    std::vector<double> output_from_state(const std::vector<double>& input,
                                          const std::vector<double>& hidden) const;
    void add_outer(std::vector<double>& matrix, const std::vector<double>& left,
                   const std::vector<double>& right, double scale) const;
    void add_vector(std::vector<double>& target, const std::vector<double>& source,
                    double scale) const;
    void initialize_parameters();
};

}  // namespace cct
