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

constexpr std::size_t kMaximumSequenceDimension = 8192U;
constexpr std::size_t kMaximumCheckpointValues = 64U * 1024U * 1024U;

std::size_t matrix_size(std::size_t rows, std::size_t columns) {
    if (rows != 0U && columns > std::numeric_limits<std::size_t>::max() / rows) {
        throw SequenceError("matrix dimensions overflow size_t");
    }
    return rows * columns;
}

double sigmoid(double value) {
    if (value >= 0.0) {
        const auto z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const auto z = std::exp(value);
    return z / (1.0 + z);
}

double stable_gate(double raw, double epsilon) {
    return std::clamp(sigmoid(raw), epsilon, 1.0 - epsilon);
}

void require_finite(const std::vector<double>& values, const char* name) {
    for (const auto value : values) {
        if (!std::isfinite(value)) throw SequenceError(std::string(name) + " contains non-finite value");
    }
}

double norm(const std::vector<double>& values) {
    return std::sqrt(std::inner_product(values.begin(), values.end(), values.begin(), 0.0));
}

void check_same_size(const std::vector<double>& left, const std::vector<double>& right,
                     const char* name) {
    if (left.size() != right.size()) throw SequenceError(std::string(name) + " size mismatch");
}

std::vector<double> read_vector(std::istream& stream, std::size_t count) {
    if (count > kMaximumCheckpointValues) throw SequenceError("checkpoint vector exceeds safety budget");
    std::vector<double> values(count, 0.0);
    for (auto& value : values) {
        if (!(stream >> value) || !std::isfinite(value)) throw SequenceError("checkpoint contains invalid parameter value");
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
    if (config_.input_dim == 0 || config_.hidden_dim == 0 || config_.output_dim == 0 ||
        config_.input_dim > kMaximumSequenceDimension || config_.hidden_dim > kMaximumSequenceDimension ||
        config_.output_dim > kMaximumSequenceDimension) {
        throw SequenceError("sequence dimensions are outside the supported safety budget");
    }
    if (!std::isfinite(config_.gate_epsilon) || !(config_.gate_epsilon > 0.0 && config_.gate_epsilon < 0.5)) {
        throw SequenceError("gate_epsilon must be finite and in (0, 0.5)");
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

SequenceState SelectiveSequenceCore::initial_state(std::uint64_t reset_epoch) const {
    return SequenceState{std::vector<double>(config_.hidden_dim, 0.0),
                          std::vector<double>(config_.hidden_dim, 0.0),
                          std::vector<double>(config_.input_dim, 0.0),
                          0U,
                          reset_epoch};
}

SequenceState SelectiveSequenceCore::reset_state(const SequenceState& state,
                                                 std::uint64_t expected_position) const {
    validate_state(state);
    if (state.position != expected_position) {
        throw SequenceError("reset position does not match the supplied state");
    }
    if (state.reset_epoch == std::numeric_limits<std::uint64_t>::max()) {
        throw SequenceError("reset epoch overflow");
    }
    return initial_state(state.reset_epoch + 1U);
}

void SelectiveSequenceCore::validate_input(const std::vector<double>& input) const {
    if (input.size() != config_.input_dim) throw SequenceError("input dimension mismatch");
    for (const auto value : input) {
        if (!std::isfinite(value)) throw SequenceError("input contains non-finite value");
    }
}

void SelectiveSequenceCore::validate_mask(const std::vector<std::uint8_t>& mask,
                                              std::size_t expected) const {
    if (!mask.empty() && mask.size() != expected) throw SequenceError("mask length mismatch");
    if (std::any_of(mask.begin(), mask.end(), [](const std::uint8_t value) { return value != 0U && value != 1U; })) {
        throw SequenceError("mask values must be binary");
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
        if (!std::isfinite(rms) || rms <= 0.0) throw SequenceError("output normalization became non-finite");
        for (auto& value : output) value /= rms;
    }
    require_finite(output, "sequence output");
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
        const auto retain = config_.selective_gates ? stable_gate(retain_raw[index], config_.gate_epsilon) : 0.95;
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
        if (!std::isfinite(rms) || rms <= 0.0) throw SequenceError("state normalization became non-finite");
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            next_hidden[index] /= rms;
            next_hidden_imag[index] /= rms;
        }
    }
    if (state.position == std::numeric_limits<std::uint64_t>::max()) {
        throw SequenceError("sequence position overflow");
    }
    if (output != nullptr) *output = output_from_state(input, next_hidden);
    return SequenceState{std::move(next_hidden), std::move(next_hidden_imag), input, state.position + 1U, state.reset_epoch};
}

SequenceOutput SelectiveSequenceCore::forward(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::uint8_t>& mask,
    const SequenceState* initial) const {
    validate_mask(mask, inputs.size());
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
    validate_mask(mask, inputs.size());
    for (const auto& target : targets) {
        if (target.size() != config_.output_dim) throw SequenceError("target dimension mismatch");
        for (const auto value : target) {
            if (!std::isfinite(value)) throw SequenceError("target contains a non-finite value");
        }
    }

    struct Cache {
        std::vector<double> input;
        std::vector<double> previous_input;
        std::vector<double> hidden_before;
        std::vector<double> hidden_imag_before;
        std::vector<double> hidden_after;
        std::vector<double> hidden_imag_after;
        std::vector<double> retain;
        std::vector<double> write;
        std::vector<double> retain_derivative;
        std::vector<double> write_derivative;
        std::vector<double> candidate;
        std::vector<double> imaginary_candidate;
        std::vector<double> candidate_pre_activation;
        std::vector<double> candidate_raw;
        std::vector<double> phase;
        std::vector<double> base_real;
        std::vector<double> base_imag;
        std::vector<double> rotated_real;
        std::vector<double> rotated_imag;
        std::vector<double> output_pre_normalized;
        std::vector<double> output;
        double state_rms = 1.0;
        double output_rms = 1.0;
        bool active = false;
    };
    std::vector<Cache> cache;
    cache.reserve(inputs.size());
    auto state = initial == nullptr ? initial_state() : *initial;
    validate_state(state);
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

    std::size_t active_count = 0U;
    for (std::size_t time = 0; time < inputs.size(); ++time) {
        validate_input(inputs[time]);
        const bool active = mask.empty() || mask[time] != 0U;
        Cache item;
        item.input = inputs[time];
        item.previous_input = state.previous_input;
        item.hidden_before = state.hidden;
        item.hidden_imag_before = state.hidden_imag;
        item.active = active;
        if (!active) {
            item.hidden_after = state.hidden;
            item.hidden_imag_after = state.hidden_imag;
            item.output_pre_normalized = affine(parameters_.output_projection, parameters_.output_bias,
                                                config_.output_dim, config_.hidden_dim, state.hidden);
            const auto skip = matvec(parameters_.skip_projection, config_.output_dim,
                                     config_.input_dim, inputs[time]);
            for (std::size_t output = 0; output < config_.output_dim; ++output) item.output_pre_normalized[output] += skip[output];
            item.output = item.output_pre_normalized;
            if (config_.normalize_output) {
                double squared = 0.0;
                for (const auto value : item.output_pre_normalized) squared += value * value;
                item.output_rms = std::sqrt(squared / static_cast<double>(config_.output_dim) + config_.normalization_epsilon);
                for (auto& value : item.output) value /= item.output_rms;
            }
            cache.push_back(std::move(item));
            continue;
        }
        ++active_count;
        const auto retain_raw = affine(parameters_.retain_projection, parameters_.retain_bias,
                                       config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto write_raw = affine(parameters_.write_projection, parameters_.write_bias,
                                      config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto candidate_raw = affine(parameters_.input_projection, parameters_.bias,
                                          config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto previous = matvec(parameters_.previous_projection, config_.hidden_dim,
                                     config_.input_dim, state.previous_input);
        item.retain.resize(config_.hidden_dim);
        item.write.resize(config_.hidden_dim);
        item.retain_derivative.resize(config_.hidden_dim);
        item.write_derivative.resize(config_.hidden_dim);
        item.candidate.resize(config_.hidden_dim);
        item.imaginary_candidate.resize(config_.hidden_dim);
        item.candidate_pre_activation.resize(config_.hidden_dim);
        item.candidate_raw = candidate_raw;
        item.phase.resize(config_.hidden_dim);
        item.base_real.resize(config_.hidden_dim);
        item.base_imag.resize(config_.hidden_dim);
        item.rotated_real.resize(config_.hidden_dim);
        item.rotated_imag.resize(config_.hidden_dim);
        std::vector<double> next_hidden(config_.hidden_dim, 0.0);
        std::vector<double> next_hidden_imag(config_.hidden_dim, 0.0);
        for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
            const auto retain_sigmoid = sigmoid(retain_raw[hidden]);
            item.retain[hidden] = config_.selective_gates ? stable_gate(retain_raw[hidden], config_.gate_epsilon) : 0.95;
            item.retain_derivative[hidden] = config_.selective_gates && retain_sigmoid > config_.gate_epsilon && retain_sigmoid < 1.0 - config_.gate_epsilon
                                                  ? retain_sigmoid * (1.0 - retain_sigmoid)
                                                  : 0.0;
            item.write[hidden] = config_.selective_gates ? sigmoid(write_raw[hidden]) : 0.5;
            item.write_derivative[hidden] = config_.selective_gates ? item.write[hidden] * (1.0 - item.write[hidden]) : 0.0;
            item.candidate_pre_activation[hidden] = candidate_raw[hidden] + previous[hidden];
            item.candidate[hidden] = std::tanh(item.candidate_pre_activation[hidden]);
            item.imaginary_candidate[hidden] = std::sin(item.candidate_pre_activation[hidden]);
            item.base_real[hidden] = item.retain[hidden] * state.hidden[hidden] + item.write[hidden] * item.candidate[hidden];
            item.base_imag[hidden] = item.retain[hidden] * state.hidden_imag[hidden] + item.write[hidden] * item.imaginary_candidate[hidden];
            if (config_.complex_state) {
                item.phase[hidden] = 0.17 * std::sin(candidate_raw[hidden]);
                const auto cosine = std::cos(item.phase[hidden]);
                const auto sine = std::sin(item.phase[hidden]);
                item.rotated_real[hidden] = cosine * item.base_real[hidden] - sine * item.base_imag[hidden];
                item.rotated_imag[hidden] = sine * item.base_real[hidden] + cosine * item.base_imag[hidden];
            } else {
                item.phase[hidden] = 0.0;
                item.rotated_real[hidden] = item.base_real[hidden];
                item.rotated_imag[hidden] = 0.0;
            }
            next_hidden[hidden] = item.rotated_real[hidden];
            next_hidden_imag[hidden] = item.rotated_imag[hidden];
        }
        if (config_.normalize_state) {
            double squared = 0.0;
            for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
                squared += next_hidden[hidden] * next_hidden[hidden] + next_hidden_imag[hidden] * next_hidden_imag[hidden];
            }
            item.state_rms = std::sqrt(squared / static_cast<double>(config_.hidden_dim) + config_.normalization_epsilon);
            for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
                next_hidden[hidden] /= item.state_rms;
                next_hidden_imag[hidden] /= item.state_rms;
            }
        }
        item.hidden_after = next_hidden;
        item.hidden_imag_after = next_hidden_imag;
        item.output_pre_normalized = affine(parameters_.output_projection, parameters_.output_bias,
                                            config_.output_dim, config_.hidden_dim, next_hidden);
        const auto skip = matvec(parameters_.skip_projection, config_.output_dim,
                                 config_.input_dim, inputs[time]);
        for (std::size_t output = 0; output < config_.output_dim; ++output) item.output_pre_normalized[output] += skip[output];
        item.output = item.output_pre_normalized;
        if (config_.normalize_output) {
            double squared = 0.0;
            for (const auto value : item.output_pre_normalized) squared += value * value;
            item.output_rms = std::sqrt(squared / static_cast<double>(config_.output_dim) + config_.normalization_epsilon);
            for (auto& value : item.output) value /= item.output_rms;
        }
        if (state.position == std::numeric_limits<std::uint64_t>::max()) {
            throw SequenceError("sequence position overflow");
        }
        state = SequenceState{std::move(next_hidden), std::move(next_hidden_imag), inputs[time], state.position + 1U, state.reset_epoch};
        cache.push_back(std::move(item));
    }
    if (active_count == 0) throw SequenceError("mask excludes every training position");
    const auto normalization = 1.0 / static_cast<double>(active_count * config_.output_dim);

    std::vector<double> d_hidden_next(config_.hidden_dim, 0.0);
    std::vector<double> d_hidden_imag_next(config_.hidden_dim, 0.0);
    for (std::size_t reverse = inputs.size(); reverse-- > 0;) {
        auto& item = cache[reverse];
        if (!item.active) continue;
        std::vector<double> d_output(config_.output_dim, 0.0);
        for (std::size_t output = 0; output < config_.output_dim; ++output) {
            const auto error = item.output[output] - targets[reverse][output];
            gradients.loss += 0.5 * error * error * normalization;
            d_output[output] = error * normalization;
        }
        if (config_.normalize_output) {
            double dot = 0.0;
            for (std::size_t output = 0; output < config_.output_dim; ++output) dot += d_output[output] * item.output[output];
            const auto scale = static_cast<double>(config_.output_dim);
            for (std::size_t output = 0; output < config_.output_dim; ++output) {
                d_output[output] = d_output[output] / item.output_rms -
                                   item.output[output] * dot / (scale * item.output_rms);
            }
        }
        for (std::size_t output = 0; output < config_.output_dim; ++output) {
            gradients.d_output_bias[output] += d_output[output];
            for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
                gradients.d_output_projection[output * config_.hidden_dim + hidden] += d_output[output] * item.hidden_after[hidden];
                d_hidden_next[hidden] += d_output[output] * parameters_.output_projection[output * config_.hidden_dim + hidden];
            }
            for (std::size_t input = 0; input < config_.input_dim; ++input) {
                gradients.d_skip_projection[output * config_.input_dim + input] += d_output[output] * item.input[input];
            }
        }
        std::vector<double> d_rotated_real = d_hidden_next;
        std::vector<double> d_rotated_imag = d_hidden_imag_next;
        if (config_.normalize_state) {
            double dot = 0.0;
            for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
                dot += d_rotated_real[hidden] * item.hidden_after[hidden] + d_rotated_imag[hidden] * item.hidden_imag_after[hidden];
            }
            const auto scale = static_cast<double>(config_.hidden_dim);
            for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
                d_rotated_real[hidden] = d_rotated_real[hidden] / item.state_rms -
                                         item.hidden_after[hidden] * dot / (scale * item.state_rms);
                d_rotated_imag[hidden] = d_rotated_imag[hidden] / item.state_rms -
                                         item.hidden_imag_after[hidden] * dot / (scale * item.state_rms);
            }
        }
        std::vector<double> d_hidden_before(config_.hidden_dim, 0.0);
        std::vector<double> d_hidden_imag_before(config_.hidden_dim, 0.0);
        for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
            double d_base_real = d_rotated_real[hidden];
            double d_base_imag = d_rotated_imag[hidden];
            double d_candidate_raw = 0.0;
            if (config_.complex_state) {
                const auto cosine = std::cos(item.phase[hidden]);
                const auto sine = std::sin(item.phase[hidden]);
                d_base_real = cosine * d_rotated_real[hidden] + sine * d_rotated_imag[hidden];
                d_base_imag = -sine * d_rotated_real[hidden] + cosine * d_rotated_imag[hidden];
                const auto d_phase = -item.rotated_imag[hidden] * d_rotated_real[hidden] +
                                     item.rotated_real[hidden] * d_rotated_imag[hidden];
                d_candidate_raw += d_phase * 0.17 * std::cos(item.candidate_raw[hidden]);
            } else {
                d_base_imag = 0.0;
            }
            const auto d_retain = d_base_real * item.hidden_before[hidden] + d_base_imag * item.hidden_imag_before[hidden];
            const auto d_write = d_base_real * item.candidate[hidden] + d_base_imag * item.imaginary_candidate[hidden];
            d_hidden_before[hidden] = d_base_real * item.retain[hidden];
            d_hidden_imag_before[hidden] = d_base_imag * item.retain[hidden];
            const auto d_candidate_pre = d_base_real * item.write[hidden] *
                                             (1.0 - item.candidate[hidden] * item.candidate[hidden]) +
                                         d_base_imag * item.write[hidden] * std::cos(item.candidate_pre_activation[hidden]);
            d_candidate_raw += d_candidate_pre;
            const auto d_retain_raw = d_retain * item.retain_derivative[hidden];
            const auto d_write_raw = d_write * item.write_derivative[hidden];
            gradients.d_retain_bias[hidden] += d_retain_raw;
            gradients.d_write_bias[hidden] += d_write_raw;
            gradients.d_bias[hidden] += d_candidate_raw;
            for (std::size_t input = 0; input < config_.input_dim; ++input) {
                gradients.d_retain_projection[hidden * config_.input_dim + input] += d_retain_raw * item.input[input];
                gradients.d_write_projection[hidden * config_.input_dim + input] += d_write_raw * item.input[input];
                gradients.d_input_projection[hidden * config_.input_dim + input] += d_candidate_raw * item.input[input];
                gradients.d_previous_projection[hidden * config_.input_dim + input] += d_candidate_pre * item.previous_input[input];
            }
        }
        d_hidden_next = std::move(d_hidden_before);
        d_hidden_imag_next = std::move(d_hidden_imag_before);
    }
    if (!std::isfinite(gradients.loss)) throw SequenceError("sequence loss became non-finite");
    require_finite(gradients.d_input_projection, "input projection gradient became non-finite");
    require_finite(gradients.d_previous_projection, "previous projection gradient became non-finite");
    require_finite(gradients.d_retain_projection, "retain projection gradient became non-finite");
    require_finite(gradients.d_write_projection, "write projection gradient became non-finite");
    require_finite(gradients.d_output_projection, "output projection gradient became non-finite");
    require_finite(gradients.d_skip_projection, "skip projection gradient became non-finite");
    require_finite(gradients.d_bias, "candidate bias gradient became non-finite");
    require_finite(gradients.d_retain_bias, "retain bias gradient became non-finite");
    require_finite(gradients.d_write_bias, "write bias gradient became non-finite");
    require_finite(gradients.d_output_bias, "output bias gradient became non-finite");
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
    if (!std::isfinite(learning_rate) || !std::isfinite(clip_norm) || !(learning_rate > 0.0) || !(clip_norm > 0.0)) {
        throw SequenceError("learning rate and clip norm must be finite and positive");
    }
    const auto validate_gradient = [](const std::vector<double>& parameter, const std::vector<double>& gradient, const char* name) {
        check_same_size(parameter, gradient, name);
        require_finite(gradient, name);
    };
    validate_gradient(parameters_.input_projection, gradients.d_input_projection, "input projection gradient");
    validate_gradient(parameters_.previous_projection, gradients.d_previous_projection, "previous projection gradient");
    validate_gradient(parameters_.retain_projection, gradients.d_retain_projection, "retain projection gradient");
    validate_gradient(parameters_.write_projection, gradients.d_write_projection, "write projection gradient");
    validate_gradient(parameters_.output_projection, gradients.d_output_projection, "output projection gradient");
    validate_gradient(parameters_.skip_projection, gradients.d_skip_projection, "skip projection gradient");
    validate_gradient(parameters_.bias, gradients.d_bias, "candidate bias gradient");
    validate_gradient(parameters_.retain_bias, gradients.d_retain_bias, "retain bias gradient");
    validate_gradient(parameters_.write_bias, gradients.d_write_bias, "write bias gradient");
    validate_gradient(parameters_.output_bias, gradients.d_output_bias, "output bias gradient");
    const auto squared = [&](const std::vector<double>& values) { return std::inner_product(values.begin(), values.end(), values.begin(), 0.0); };
    const double total_squared = squared(gradients.d_input_projection) + squared(gradients.d_previous_projection) +
                                 squared(gradients.d_retain_projection) + squared(gradients.d_write_projection) +
                                 squared(gradients.d_output_projection) + squared(gradients.d_skip_projection) +
                                 squared(gradients.d_bias) + squared(gradients.d_retain_bias) + squared(gradients.d_write_bias) +
                                 squared(gradients.d_output_bias);
    if (!std::isfinite(total_squared) || total_squared < 0.0) throw SequenceError("gradient norm is non-finite");
    const auto scale = std::min(1.0, clip_norm / std::max(std::sqrt(total_squared), 1e-12));
    const auto updated = [&](const std::vector<double>& parameter, const std::vector<double>& gradient) {
        auto candidate = parameter;
        for (std::size_t index = 0; index < candidate.size(); ++index) candidate[index] -= learning_rate * scale * gradient[index];
        require_finite(candidate, "updated parameter");
        return candidate;
    };
    auto input_projection = updated(parameters_.input_projection, gradients.d_input_projection);
    auto previous_projection = updated(parameters_.previous_projection, gradients.d_previous_projection);
    auto retain_projection = updated(parameters_.retain_projection, gradients.d_retain_projection);
    auto write_projection = updated(parameters_.write_projection, gradients.d_write_projection);
    auto output_projection = updated(parameters_.output_projection, gradients.d_output_projection);
    auto skip_projection = updated(parameters_.skip_projection, gradients.d_skip_projection);
    auto bias = updated(parameters_.bias, gradients.d_bias);
    auto retain_bias = updated(parameters_.retain_bias, gradients.d_retain_bias);
    auto write_bias = updated(parameters_.write_bias, gradients.d_write_bias);
    auto output_bias = updated(parameters_.output_bias, gradients.d_output_bias);
    parameters_ = Parameters{std::move(input_projection), std::move(previous_projection), std::move(retain_projection),
                             std::move(write_projection), std::move(output_projection), std::move(skip_projection),
                             std::move(bias), std::move(retain_bias), std::move(write_bias), std::move(output_bias)};
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
        maximum = std::max(maximum, stable_gate(parameters_.retain_bias[row], config_.gate_epsilon) + 0.25 * row_sum);
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
                                            std::uint64_t optimizer_step,
                                            const SequenceState* recurrent_state) const {
    if (recurrent_state != nullptr) validate_state(*recurrent_state);
    std::ofstream stream(path);
    if (!stream) throw SequenceError("could not open checkpoint for writing");
    stream << "CCT_SEQUENCE_CHECKPOINT_V3\n";
    stream << config_.input_dim << ' ' << config_.hidden_dim << ' ' << config_.output_dim << ' '
           << std::setprecision(17) << config_.gate_epsilon << ' ' << config_.seed << ' '
           << static_cast<int>(config_.complex_state) << ' ' << static_cast<int>(config_.normalize_state) << ' '
           << static_cast<int>(config_.normalize_output) << ' ' << config_.normalization_epsilon << ' '
           << static_cast<int>(config_.selective_gates) << ' ' << optimizer_step << ' '
           << static_cast<int>(recurrent_state != nullptr) << '\n';
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
    if (recurrent_state != nullptr) {
        write_vector(stream, recurrent_state->hidden);
        write_vector(stream, recurrent_state->hidden_imag);
        write_vector(stream, recurrent_state->previous_input);
        stream << recurrent_state->position << ' ' << recurrent_state->reset_epoch << '\n';
    }
    if (!stream) throw SequenceError("could not write complete checkpoint");
}

SelectiveSequenceCore SelectiveSequenceCore::load_checkpoint(const std::string& path,
                                                              std::uint64_t* optimizer_step,
                                                              SequenceState* recurrent_state) {
    std::ifstream stream(path);
    if (!stream) throw SequenceError("could not open checkpoint for reading");
    std::string header;
    std::getline(stream, header);
    if (header != "CCT_SEQUENCE_CHECKPOINT_V1" && header != "CCT_SEQUENCE_CHECKPOINT_V2" &&
        header != "CCT_SEQUENCE_CHECKPOINT_V3") {
        throw SequenceError("unsupported checkpoint version");
    }
    SequenceConfig config;
    std::uint64_t saved_step = 0;
    bool has_recurrent_state = false;
    if (header == "CCT_SEQUENCE_CHECKPOINT_V2" || header == "CCT_SEQUENCE_CHECKPOINT_V3") {
        int complex_state = 0;
        int normalize_state = 0;
        int normalize_output = 0;
        int selective_gates = 1;
        int serialized_state = 0;
        if (!(stream >> config.input_dim >> config.hidden_dim >> config.output_dim >> config.gate_epsilon >> config.seed
              >> complex_state >> normalize_state >> normalize_output >> config.normalization_epsilon >> selective_gates >> saved_step) ||
            (header == "CCT_SEQUENCE_CHECKPOINT_V3" && !(stream >> serialized_state))) {
            throw SequenceError("checkpoint configuration is incomplete");
        }
        if ((complex_state != 0 && complex_state != 1) || (normalize_state != 0 && normalize_state != 1) ||
            (normalize_output != 0 && normalize_output != 1) || (selective_gates != 0 && selective_gates != 1) ||
            (header == "CCT_SEQUENCE_CHECKPOINT_V3" && serialized_state != 0 && serialized_state != 1)) {
            throw SequenceError("checkpoint contains invalid boolean configuration");
        }
        config.complex_state = complex_state != 0;
        config.normalize_state = normalize_state != 0;
        config.normalize_output = normalize_output != 0;
        config.selective_gates = selective_gates != 0;
        has_recurrent_state = header == "CCT_SEQUENCE_CHECKPOINT_V3" && serialized_state != 0;
    } else if (!(stream >> config.input_dim >> config.hidden_dim >> config.output_dim >> config.gate_epsilon >> config.seed >> saved_step)) {
        throw SequenceError("checkpoint configuration is incomplete");
    }
    SelectiveSequenceCore core(config);
    const auto read_parameter = [&](std::vector<double>& target, const char* name) {
        auto values = read_counted_vector(stream);
        if (values.size() != target.size()) throw SequenceError(std::string("checkpoint ") + name + " size mismatch");
        target = std::move(values);
    };
    read_parameter(core.parameters_.input_projection, "input projection");
    read_parameter(core.parameters_.previous_projection, "previous projection");
    read_parameter(core.parameters_.retain_projection, "retain projection");
    read_parameter(core.parameters_.write_projection, "write projection");
    read_parameter(core.parameters_.output_projection, "output projection");
    read_parameter(core.parameters_.skip_projection, "skip projection");
    read_parameter(core.parameters_.bias, "candidate bias");
    read_parameter(core.parameters_.retain_bias, "retain bias");
    read_parameter(core.parameters_.write_bias, "write bias");
    read_parameter(core.parameters_.output_bias, "output bias");
    SequenceState restored_state = core.initial_state();
    if (has_recurrent_state) {
        restored_state.hidden = read_counted_vector(stream);
        restored_state.hidden_imag = read_counted_vector(stream);
        restored_state.previous_input = read_counted_vector(stream);
        if (!(stream >> restored_state.position >> restored_state.reset_epoch)) {
            throw SequenceError("checkpoint recurrent state is incomplete");
        }
        core.validate_state(restored_state);
    }
    std::string trailing;
    if (stream >> trailing) throw SequenceError("checkpoint contains trailing data");
    if (optimizer_step != nullptr) *optimizer_step = saved_step;
    if (recurrent_state != nullptr) *recurrent_state = std::move(restored_state);
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
    require_finite(values, "parameter vector");
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

