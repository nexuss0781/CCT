#include "cct/sequence.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

std::size_t matrix_size(std::size_t rows, std::size_t columns) { return rows * columns; }

double sigmoid(double value) {
    if (value >= 0.0) {
        const auto z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const auto z = std::exp(value);
    return z / (1.0 + z);
}

double stable_gate(double raw) {
    return std::clamp(sigmoid(raw), 1e-4, 1.0 - 1e-4);
}

double norm(const std::vector<double>& values) {
    return std::sqrt(std::inner_product(values.begin(), values.end(), values.begin(), 0.0));
}

void check_same_size(const std::vector<double>& left, const std::vector<double>& right,
                     const char* name) {
    if (left.size() != right.size()) throw SequenceError(std::string(name) + " size mismatch");
}

std::vector<double> read_vector(std::istream& stream, std::size_t count) {
    std::vector<double> values(count, 0.0);
    for (auto& value : values) {
        if (!(stream >> value)) throw SequenceError("checkpoint ended before parameter vector");
    }
    return values;
}

void write_vector(std::ostream& stream, const std::vector<double>& values) {
    stream << values.size();
    for (const auto value : values) stream << ' ' << std::setprecision(17) << value;
    stream << '\n';
}

std::vector<double> read_counted_vector(std::istream& stream) {
    std::size_t count = 0;
    if (!(stream >> count)) throw SequenceError("checkpoint missing vector count");
    return read_vector(stream, count);
}

}  // namespace

SelectiveSequenceCore::SelectiveSequenceCore(SequenceConfig config)
    : config_(std::move(config)) {
    if (config_.input_dim == 0 || config_.hidden_dim == 0 || config_.output_dim == 0) {
        throw SequenceError("input, hidden, and output dimensions must be positive");
    }
    if (!(config_.gate_epsilon > 0.0 && config_.gate_epsilon < 0.5)) {
        throw SequenceError("gate_epsilon must be in (0, 0.5)");
    }
    if (!(config_.normalization_epsilon > 0.0) || !std::isfinite(config_.normalization_epsilon)) {
        throw SequenceError("normalization_epsilon must be finite and positive");
    }
    initialize_parameters();
}

void SelectiveSequenceCore::initialize_parameters() {
    std::mt19937_64 generator(config_.seed);
    const auto scale = 1.0 / std::sqrt(static_cast<double>(config_.input_dim + config_.hidden_dim));
    std::normal_distribution<double> distribution(0.0, scale);
    const auto initialize = [&](std::vector<double>& values) {
        for (auto& value : values) value = distribution(generator);
    };
    const auto hidden = config_.hidden_dim;
    const auto input = config_.input_dim;
    const auto output = config_.output_dim;
    parameters_.input_projection.resize(matrix_size(hidden, input));
    parameters_.previous_projection.resize(matrix_size(hidden, input));
    parameters_.retain_projection.resize(matrix_size(hidden, input));
    parameters_.write_projection.resize(matrix_size(hidden, input));
    parameters_.output_projection.resize(matrix_size(output, hidden));
    parameters_.skip_projection.resize(matrix_size(output, input));
    parameters_.bias.assign(hidden, 0.0);
    parameters_.retain_bias.assign(hidden, 4.5);
    parameters_.write_bias.assign(hidden, 0.0);
    parameters_.output_bias.assign(output, 0.0);
    initialize(parameters_.input_projection);
    initialize(parameters_.previous_projection);
    initialize(parameters_.retain_projection);
    initialize(parameters_.write_projection);
    initialize(parameters_.output_projection);
    initialize(parameters_.skip_projection);
}

SequenceState SelectiveSequenceCore::initial_state() const {
    return SequenceState{std::vector<double>(config_.hidden_dim, 0.0),
                          std::vector<double>(config_.hidden_dim, 0.0),
                          std::vector<double>(config_.input_dim, 0.0)};
}

void SelectiveSequenceCore::validate_input(const std::vector<double>& input) const {
    if (input.size() != config_.input_dim) throw SequenceError("input dimension mismatch");
    for (const auto value : input) {
        if (!std::isfinite(value)) throw SequenceError("input contains non-finite value");
    }
}

void SelectiveSequenceCore::validate_state(const SequenceState& state) const {
    if (state.hidden.size() != config_.hidden_dim || state.hidden_imag.size() != config_.hidden_dim ||
        state.previous_input.size() != config_.input_dim) {
        throw SequenceError("state dimension mismatch");
    }
    for (const auto value : state.hidden) {
        if (!std::isfinite(value)) throw SequenceError("state contains non-finite value");
    }
    for (const auto value : state.hidden_imag) {
        if (!std::isfinite(value)) throw SequenceError("imaginary state contains non-finite value");
    }
    for (const auto value : state.previous_input) {
        if (!std::isfinite(value)) throw SequenceError("previous input contains non-finite value");
    }
}

std::vector<double> SelectiveSequenceCore::matvec(const std::vector<double>& matrix,
                                                   std::size_t rows,
                                                   std::size_t columns,
                                                   const std::vector<double>& vector) const {
    if (matrix.size() != matrix_size(rows, columns) || vector.size() != columns) {
        throw SequenceError("matrix-vector dimensions are inconsistent");
    }
    std::vector<double> result(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row] += matrix[row * columns + column] * vector[column];
        }
    }
    return result;
}

std::vector<double> SelectiveSequenceCore::affine(const std::vector<double>& matrix,
                                                   const std::vector<double>& bias,
                                                   std::size_t rows,
                                                   std::size_t columns,
                                                   const std::vector<double>& vector) const {
    auto result = matvec(matrix, rows, columns, vector);
    check_same_size(result, bias, "bias");
    for (std::size_t index = 0; index < result.size(); ++index) result[index] += bias[index];
    return result;
}

std::vector<double> SelectiveSequenceCore::output_from_state(
    const std::vector<double>& input,
    const std::vector<double>& hidden) const {
    auto output = affine(parameters_.output_projection, parameters_.output_bias,
                         config_.output_dim, config_.hidden_dim, hidden);
    const auto skip = matvec(parameters_.skip_projection, config_.output_dim,
                             config_.input_dim, input);
    for (std::size_t index = 0; index < output.size(); ++index) output[index] += skip[index];
    if (config_.normalize_output) {
        double squared = 0.0;
        for (const auto value : output) squared += value * value;
        const auto rms = std::sqrt(squared / static_cast<double>(output.size()) + config_.normalization_epsilon);
        for (auto& value : output) value /= rms;
    }
    return output;
}

SequenceState SelectiveSequenceCore::step(const std::vector<double>& input,
                                          const SequenceState& state,
                                          std::vector<double>* output) const {
    validate_input(input);
    validate_state(state);
    const auto retain_raw = affine(parameters_.retain_projection, parameters_.retain_bias,
                                   config_.hidden_dim, config_.input_dim, input);
    const auto write_raw = affine(parameters_.write_projection, parameters_.write_bias,
                                  config_.hidden_dim, config_.input_dim, input);
    const auto candidate_raw = affine(parameters_.input_projection, parameters_.bias,
                                      config_.hidden_dim, config_.input_dim, input);
    const auto previous = matvec(parameters_.previous_projection, config_.hidden_dim,
                                 config_.input_dim, state.previous_input);
    std::vector<double> next_hidden(config_.hidden_dim, 0.0);
    std::vector<double> next_hidden_imag(config_.hidden_dim, 0.0);
    for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
        const auto retain = config_.selective_gates ? stable_gate(retain_raw[index]) : 0.95;
        const auto write = config_.selective_gates ? sigmoid(write_raw[index]) : 0.5;
        const auto candidate = std::tanh(candidate_raw[index] + previous[index]);
        next_hidden[index] = retain * state.hidden[index] + write * candidate;
        if (config_.complex_state) {
            const auto phase = 0.17 * std::sin(candidate_raw[index]);
            const auto imaginary_candidate = std::sin(candidate_raw[index] + previous[index]);
            next_hidden_imag[index] = retain * state.hidden_imag[index] + write * imaginary_candidate;
            const auto rotated_real = std::cos(phase) * next_hidden[index] - std::sin(phase) * next_hidden_imag[index];
            const auto rotated_imag = std::sin(phase) * next_hidden[index] + std::cos(phase) * next_hidden_imag[index];
            next_hidden[index] = rotated_real;
            next_hidden_imag[index] = rotated_imag;
        }
        if (!std::isfinite(next_hidden[index]) || !std::isfinite(next_hidden_imag[index])) {
            throw SequenceError("state update became non-finite");
        }
    }
    if (config_.normalize_state) {
        double squared = 0.0;
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            squared += next_hidden[index] * next_hidden[index] + next_hidden_imag[index] * next_hidden_imag[index];
        }
        const auto rms = std::sqrt(squared / static_cast<double>(config_.hidden_dim) + config_.normalization_epsilon);
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            next_hidden[index] /= rms;
            next_hidden_imag[index] /= rms;
        }
    }
    if (output != nullptr) *output = output_from_state(input, next_hidden);
    return SequenceState{std::move(next_hidden), std::move(next_hidden_imag), input};
}

SequenceOutput SelectiveSequenceCore::forward(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::uint8_t>& mask,
    const SequenceState* initial) const {
    if (!mask.empty() && mask.size() != inputs.size()) throw SequenceError("mask length mismatch");
    auto state = initial == nullptr ? initial_state() : *initial;
    validate_state(state);
    SequenceOutput result;
    result.outputs.reserve(inputs.size());
    for (std::size_t time = 0; time < inputs.size(); ++time) {
        std::vector<double> output;
        if (!mask.empty() && mask[time] == 0) {
            validate_input(inputs[time]);
            output = output_from_state(inputs[time], state.hidden);
        } else {
            state = step(inputs[time], state, &output);
        }
        result.outputs.push_back(std::move(output));
    }
    result.final_state = std::move(state);
    return result;
}

SequenceGradients SelectiveSequenceCore::loss_and_gradients(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    const std::vector<std::uint8_t>& mask,
    const SequenceState* initial) const {
    if (inputs.size() != targets.size()) throw SequenceError("input and target lengths differ");
    if (inputs.empty()) throw SequenceError("cannot train on an empty sequence");
    if (!mask.empty() && mask.size() != inputs.size()) throw SequenceError("mask length mismatch");
    for (const auto& target : targets) {
        if (target.size() != config_.output_dim) throw SequenceError("target dimension mismatch");
    }

    struct Cache {
        std::vector<double> input;
        std::vector<double> previous_input;
        std::vector<double> hidden_before;
        std::vector<double> hidden_after;
        std::vector<double> retain;
        std::vector<double> write;
        std::vector<double> candidate;
        std::vector<double> candidate_pre_tanh;
        std::vector<double> output;
    };
    std::vector<Cache> cache;
    cache.reserve(inputs.size());
    auto state = initial == nullptr ? initial_state() : *initial;
    SequenceGradients gradients;
    gradients.d_input_projection.assign(parameters_.input_projection.size(), 0.0);
    gradients.d_previous_projection.assign(parameters_.previous_projection.size(), 0.0);
    gradients.d_retain_projection.assign(parameters_.retain_projection.size(), 0.0);
    gradients.d_write_projection.assign(parameters_.write_projection.size(), 0.0);
    gradients.d_output_projection.assign(parameters_.output_projection.size(), 0.0);
    gradients.d_skip_projection.assign(parameters_.skip_projection.size(), 0.0);
    gradients.d_bias.assign(parameters_.bias.size(), 0.0);
    gradients.d_retain_bias.assign(parameters_.retain_bias.size(), 0.0);
    gradients.d_write_bias.assign(parameters_.write_bias.size(), 0.0);
    gradients.d_output_bias.assign(parameters_.output_bias.size(), 0.0);

    std::size_t active_count = 0;
    for (std::size_t time = 0; time < inputs.size(); ++time) {
        validate_input(inputs[time]);
        auto output = std::vector<double>{};
        const auto before = state;
        const auto after = step(inputs[time], state, &output);
        const auto retain_raw = affine(parameters_.retain_projection, parameters_.retain_bias,
                                       config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto write_raw = affine(parameters_.write_projection, parameters_.write_bias,
                                      config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto candidate_raw = affine(parameters_.input_projection, parameters_.bias,
                                          config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto previous = matvec(parameters_.previous_projection, config_.hidden_dim,
                                     config_.input_dim, before.previous_input);
        Cache item{inputs[time], before.previous_input, before.hidden, after.hidden, {}, {}, {}, {}, output};
        item.retain.resize(config_.hidden_dim);
        item.write.resize(config_.hidden_dim);
        item.candidate.resize(config_.hidden_dim);
        item.candidate_pre_tanh.resize(config_.hidden_dim);
        for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
            item.retain[hidden] = config_.selective_gates ? stable_gate(retain_raw[hidden]) : 0.95;
            item.write[hidden] = config_.selective_gates ? sigmoid(write_raw[hidden]) : 0.5;
            item.candidate_pre_tanh[hidden] = candidate_raw[hidden] + previous[hidden];
            item.candidate[hidden] = std::tanh(item.candidate_pre_tanh[hidden]);
        }
        cache.push_back(std::move(item));
        state = after;
        if (mask.empty() || mask[time] != 0) ++active_count;
    }
    if (active_count == 0) throw SequenceError("mask excludes every training position");
    const auto normalization = 1.0 / static_cast<double>(active_count * config_.output_dim);

    std::vector<double> d_hidden_next(config_.hidden_dim, 0.0);
    std::vector<double> d_previous_next(config_.input_dim, 0.0);
    for (std::size_t reverse = inputs.size(); reverse-- > 0;) {
        auto& item = cache[reverse];
        std::vector<double> d_output(config_.output_dim, 0.0);
        if (mask.empty() || mask[reverse] != 0) {
            for (std::size_t output = 0; output < config_.output_dim; ++output) {
                const auto error = item.output[output] - targets[reverse][output];
                gradients.loss += 0.5 * error * error * normalization;
                d_output[output] = error * normalization;
                gradients.d_output_bias[output] += d_output[output];
            }
        }
        for (std::size_t output = 0; output < config_.output_dim; ++output) {
            for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
                gradients.d_output_projection[output * config_.hidden_dim + hidden] +=
                    d_output[output] * item.hidden_after[hidden];
                d_hidden_next[hidden] += d_output[output] * parameters_.output_projection[output * config_.hidden_dim + hidden];
            }
            for (std::size_t input = 0; input < config_.input_dim; ++input) {
                gradients.d_skip_projection[output * config_.input_dim + input] += d_output[output] * item.input[input];
            }
        }
        std::vector<double> d_hidden_before(config_.hidden_dim, 0.0);
        std::vector<double> d_input(config_.input_dim, 0.0);
        std::vector<double> d_previous_input(config_.input_dim, 0.0);
        for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
            const auto retain_raw = item.retain[hidden];
            const auto write_raw = item.write[hidden];
            const auto candidate = item.candidate[hidden];
            const auto candidate_derivative = 1.0 - candidate * candidate;
            const auto retain_derivative = config_.selective_gates ? retain_raw * (1.0 - retain_raw) : 0.0;
            const auto write_derivative = config_.selective_gates ? write_raw * (1.0 - write_raw) : 0.0;
            const auto d_retain = d_hidden_next[hidden] * item.hidden_before[hidden];
            const auto d_write = d_hidden_next[hidden] * candidate;
            const auto d_candidate_pre = d_hidden_next[hidden] * write_raw * candidate_derivative;
            d_hidden_before[hidden] += d_hidden_next[hidden] * retain_raw;
            gradients.d_retain_bias[hidden] += d_retain * retain_derivative;
            gradients.d_write_bias[hidden] += d_write * write_derivative;
            gradients.d_bias[hidden] += d_candidate_pre;
            for (std::size_t input = 0; input < config_.input_dim; ++input) {
                gradients.d_retain_projection[hidden * config_.input_dim + input] += d_retain * retain_derivative * item.input[input];
                gradients.d_write_projection[hidden * config_.input_dim + input] += d_write * write_derivative * item.input[input];
                gradients.d_input_projection[hidden * config_.input_dim + input] += d_candidate_pre * item.input[input];
                d_input[input] += d_retain * retain_derivative * parameters_.retain_projection[hidden * config_.input_dim + input];
                d_input[input] += d_write * write_derivative * parameters_.write_projection[hidden * config_.input_dim + input];
                d_input[input] += d_candidate_pre * parameters_.input_projection[hidden * config_.input_dim + input];
                gradients.d_previous_projection[hidden * config_.input_dim + input] += d_candidate_pre * item.previous_input[input];
                d_previous_input[input] += d_candidate_pre * parameters_.previous_projection[hidden * config_.input_dim + input];
            }
        }
        (void)d_input;
        d_hidden_next = std::move(d_hidden_before);
        d_previous_next = std::move(d_previous_input);
    }
    (void)d_previous_next;
    return gradients;
}

void SelectiveSequenceCore::add_outer(std::vector<double>& matrix,
                                      const std::vector<double>& left,
                                      const std::vector<double>& right,
                                      double scale) const {
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t column = 0; column < right.size(); ++column) {
            matrix[row * right.size() + column] += scale * left[row] * right[column];
        }
    }
}

void SelectiveSequenceCore::add_vector(std::vector<double>& target,
                                       const std::vector<double>& source,
                                       double scale) const {
    check_same_size(target, source, "gradient");
    for (std::size_t index = 0; index < target.size(); ++index) target[index] += scale * source[index];
}

void SelectiveSequenceCore::apply_sgd(const SequenceGradients& gradients,
                                      double learning_rate,
                                      double clip_norm) {
    if (!(learning_rate > 0.0) || !(clip_norm > 0.0)) throw SequenceError("learning rate and clip norm must be positive");
    const auto squared = [&](const std::vector<double>& values) { return std::inner_product(values.begin(), values.end(), values.begin(), 0.0); };
    double total_squared = squared(gradients.d_input_projection) + squared(gradients.d_previous_projection) +
                           squared(gradients.d_retain_projection) + squared(gradients.d_write_projection) +
                           squared(gradients.d_output_projection) + squared(gradients.d_skip_projection) +
                           squared(gradients.d_bias) + squared(gradients.d_retain_bias) + squared(gradients.d_write_bias) +
                           squared(gradients.d_output_bias);
    const auto scale = std::min(1.0, clip_norm / std::max(std::sqrt(total_squared), 1e-12));
    const auto update = [&](std::vector<double>& parameter, const std::vector<double>& gradient) {
        check_same_size(parameter, gradient, "parameter gradient");
        for (std::size_t index = 0; index < parameter.size(); ++index) parameter[index] -= learning_rate * scale * gradient[index];
    };
    update(parameters_.input_projection, gradients.d_input_projection);
    update(parameters_.previous_projection, gradients.d_previous_projection);
    update(parameters_.retain_projection, gradients.d_retain_projection);
    update(parameters_.write_projection, gradients.d_write_projection);
    update(parameters_.output_projection, gradients.d_output_projection);
    update(parameters_.skip_projection, gradients.d_skip_projection);
    update(parameters_.bias, gradients.d_bias);
    update(parameters_.retain_bias, gradients.d_retain_bias);
    update(parameters_.write_bias, gradients.d_write_bias);
    update(parameters_.output_bias, gradients.d_output_bias);
}

std::size_t SelectiveSequenceCore::parameter_count() const noexcept {
    return parameters_.input_projection.size() + parameters_.previous_projection.size() +
           parameters_.retain_projection.size() + parameters_.write_projection.size() +
           parameters_.output_projection.size() + parameters_.skip_projection.size() +
           parameters_.bias.size() + parameters_.retain_bias.size() + parameters_.write_bias.size() +
           parameters_.output_bias.size();
}

double SelectiveSequenceCore::transition_radius_bound() const {
    double maximum = 0.0;
    for (std::size_t row = 0; row < config_.hidden_dim; ++row) {
        double row_sum = 0.0;
        for (std::size_t column = 0; column < config_.input_dim; ++column) {
            row_sum += std::abs(parameters_.retain_projection[row * config_.input_dim + column]);
        }
        maximum = std::max(maximum, stable_gate(parameters_.retain_bias[row]) + 0.25 * row_sum);
    }
    return maximum;
}

double SelectiveSequenceCore::state_norm(const SequenceState& state) const {
    double squared = 0.0;
    for (std::size_t index = 0; index < state.hidden.size(); ++index) {
        const auto imaginary = index < state.hidden_imag.size() ? state.hidden_imag[index] : 0.0;
        squared += state.hidden[index] * state.hidden[index] + imaginary * imaginary;
    }
    return std::sqrt(squared);
}

double SelectiveSequenceCore::output_norm(const std::vector<double>& output) const { return norm(output); }

double SelectiveSequenceCore::hidden_rms(const SequenceState& state) const {
    if (state.hidden.empty()) return 0.0;
    return state_norm(state) / std::sqrt(static_cast<double>(state.hidden.size()));
}

double SelectiveSequenceCore::output_rms(const std::vector<double>& output) const {
    if (output.empty()) return 0.0;
    return output_norm(output) / std::sqrt(static_cast<double>(output.size()));
}

void SelectiveSequenceCore::save_checkpoint(const std::string& path,
                                            std::uint64_t optimizer_step) const {
    std::ofstream stream(path);
    if (!stream) throw SequenceError("could not open checkpoint for writing");
    stream << "CCT_SEQUENCE_CHECKPOINT_V2\n";
    stream << config_.input_dim << ' ' << config_.hidden_dim << ' ' << config_.output_dim << ' '
           << std::setprecision(17) << config_.gate_epsilon << ' ' << config_.seed << ' '
           << static_cast<int>(config_.complex_state) << ' ' << static_cast<int>(config_.normalize_state) << ' '
           << static_cast<int>(config_.normalize_output) << ' ' << config_.normalization_epsilon << ' '
           << static_cast<int>(config_.selective_gates) << ' ' << optimizer_step << '\n';
    write_vector(stream, parameters_.input_projection);
    write_vector(stream, parameters_.previous_projection);
    write_vector(stream, parameters_.retain_projection);
    write_vector(stream, parameters_.write_projection);
    write_vector(stream, parameters_.output_projection);
    write_vector(stream, parameters_.skip_projection);
    write_vector(stream, parameters_.bias);
    write_vector(stream, parameters_.retain_bias);
    write_vector(stream, parameters_.write_bias);
    write_vector(stream, parameters_.output_bias);
}

SelectiveSequenceCore SelectiveSequenceCore::load_checkpoint(const std::string& path,
                                                              std::uint64_t* optimizer_step) {
    std::ifstream stream(path);
    if (!stream) throw SequenceError("could not open checkpoint for reading");
    std::string header;
    std::getline(stream, header);
    if (header != "CCT_SEQUENCE_CHECKPOINT_V1" && header != "CCT_SEQUENCE_CHECKPOINT_V2") {
        throw SequenceError("unsupported checkpoint version");
    }
    SequenceConfig config;
    std::uint64_t saved_step = 0;
    if (header == "CCT_SEQUENCE_CHECKPOINT_V2") {
        int complex_state = 0;
        int normalize_state = 0;
        int normalize_output = 0;
        int selective_gates = 1;
        if (!(stream >> config.input_dim >> config.hidden_dim >> config.output_dim >> config.gate_epsilon >> config.seed
              >> complex_state >> normalize_state >> normalize_output >> config.normalization_epsilon >> selective_gates >> saved_step)) {
            throw SequenceError("checkpoint configuration is incomplete");
        }
        config.complex_state = complex_state != 0;
        config.normalize_state = normalize_state != 0;
        config.normalize_output = normalize_output != 0;
        config.selective_gates = selective_gates != 0;
    } else if (!(stream >> config.input_dim >> config.hidden_dim >> config.output_dim >> config.gate_epsilon >> config.seed >> saved_step)) {
        throw SequenceError("checkpoint configuration is incomplete");
    }
    SelectiveSequenceCore core(config);
    core.parameters_.input_projection = read_counted_vector(stream);
    core.parameters_.previous_projection = read_counted_vector(stream);
    core.parameters_.retain_projection = read_counted_vector(stream);
    core.parameters_.write_projection = read_counted_vector(stream);
    core.parameters_.output_projection = read_counted_vector(stream);
    core.parameters_.skip_projection = read_counted_vector(stream);
    core.parameters_.bias = read_counted_vector(stream);
    core.parameters_.retain_bias = read_counted_vector(stream);
    core.parameters_.write_bias = read_counted_vector(stream);
    core.parameters_.output_bias = read_counted_vector(stream);
    if (optimizer_step != nullptr) *optimizer_step = saved_step;
    if (core.parameter_count() == 0) throw SequenceError("checkpoint contains no parameters");
    return core;
}

double SelectiveSequenceCore::loss_only(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    const std::vector<std::uint8_t>& mask,
    const SequenceState* initial) const {
    return loss_and_gradients(inputs, targets, mask, initial).loss;
}

std::vector<double> SelectiveSequenceCore::parameter_vector() const {
    std::vector<double> values;
    values.reserve(parameter_count());
    const auto append = [&](const std::vector<double>& source) {
        values.insert(values.end(), source.begin(), source.end());
    };
    append(parameters_.input_projection);
    append(parameters_.previous_projection);
    append(parameters_.retain_projection);
    append(parameters_.write_projection);
    append(parameters_.output_projection);
    append(parameters_.skip_projection);
    append(parameters_.bias);
    append(parameters_.retain_bias);
    append(parameters_.write_bias);
    append(parameters_.output_bias);
    return values;
}

void SelectiveSequenceCore::set_parameter_vector(const std::vector<double>& values) {
    if (values.size() != parameter_count()) throw SequenceError("parameter vector size mismatch");
    std::size_t offset = 0;
    const auto assign = [&](std::vector<double>& target) {
        std::copy(values.begin() + static_cast<std::ptrdiff_t>(offset),
                  values.begin() + static_cast<std::ptrdiff_t>(offset + target.size()), target.begin());
        offset += target.size();
    };
    assign(parameters_.input_projection);
    assign(parameters_.previous_projection);
    assign(parameters_.retain_projection);
    assign(parameters_.write_projection);
    assign(parameters_.output_projection);
    assign(parameters_.skip_projection);
    assign(parameters_.bias);
    assign(parameters_.retain_bias);
    assign(parameters_.write_bias);
    assign(parameters_.output_bias);
}

}  // namespace cct

