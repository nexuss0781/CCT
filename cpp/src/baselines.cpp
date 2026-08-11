#include "cct/baselines.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

double sigmoid(double value) {
    if (value >= 0.0) {
        const auto z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const auto z = std::exp(value);
    return z / (1.0 + z);
}

double normal_norm(const std::vector<double>& values) {
    return std::sqrt(std::inner_product(values.begin(), values.end(), values.begin(), 0.0));
}

void require_dimension(std::size_t actual, std::size_t expected, const char* name) {
    if (actual != expected) throw SequenceError(std::string(name) + " dimension mismatch");
}

}  // namespace

MatchedBaseline::MatchedBaseline(BaselineKind kind, BaselineConfig config)
    : kind_(kind), config_(std::move(config)) {
    if (config_.input_dim == 0 || config_.hidden_dim == 0 || config_.output_dim == 0) {
        throw SequenceError("baseline dimensions must be positive");
    }
    initialize();
}

void MatchedBaseline::initialize() {
    const auto input = config_.input_dim;
    const auto hidden = config_.hidden_dim;
    const auto output = config_.output_dim;
    std::size_t count = 0;
    if (kind_ == BaselineKind::DenseCausalAttention) {
        count = 3 * hidden * input + output * hidden + output;
    } else if (kind_ == BaselineKind::GRU) {
        count = 3 * (hidden * input + hidden * hidden + hidden) + output * hidden + output;
    } else {
        count = hidden + hidden * input + output * hidden + output;
    }
    parameters_.assign(count, 0.0);
    std::mt19937_64 generator(config_.seed);
    const auto scale = 1.0 / std::sqrt(static_cast<double>(input + hidden));
    std::normal_distribution<double> distribution(0.0, scale);
    for (auto& value : parameters_) value = distribution(generator);
    if (kind_ == BaselineKind::GRU) {
        const auto gate_bias_offset = 2 * (hidden * input + hidden * hidden);
        for (std::size_t index = 0; index < hidden; ++index) parameters_[gate_bias_offset + index] = 1.0;
    }
    if (kind_ == BaselineKind::DiagonalSSM) {
        for (std::size_t index = 0; index < hidden; ++index) parameters_[index] = 2.0;
    }
}

std::vector<double> MatchedBaseline::matvec(const std::vector<double>& matrix,
                                            std::size_t rows,
                                            std::size_t columns,
                                            const std::vector<double>& vector) const {
    require_dimension(matrix.size(), rows * columns, "matrix");
    require_dimension(vector.size(), columns, "vector");
    std::vector<double> result(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row] += matrix[row * columns + column] * vector[column];
        }
    }
    return result;
}

std::vector<double> MatchedBaseline::dense_forward_step(
    const std::vector<std::vector<double>>& inputs, std::size_t time) const {
    const auto input = config_.input_dim;
    const auto hidden = config_.hidden_dim;
    const auto output = config_.output_dim;
    const auto q_offset = 0;
    const auto k_offset = hidden * input;
    const auto v_offset = 2 * hidden * input;
    const auto output_offset = v_offset + hidden * input;
    const auto bias_offset = output_offset + output * hidden;
    const std::vector<double> query = matvec(std::vector<double>(parameters_.begin() + q_offset,
                                                                  parameters_.begin() + k_offset),
                                             hidden, input, inputs[time]);
    std::vector<std::vector<double>> keys;
    std::vector<std::vector<double>> values;
    keys.reserve(time + 1);
    values.reserve(time + 1);
    for (std::size_t position = 0; position <= time; ++position) {
        keys.push_back(matvec(std::vector<double>(parameters_.begin() + k_offset,
                                                  parameters_.begin() + v_offset),
                              hidden, input, inputs[position]));
        values.push_back(matvec(std::vector<double>(parameters_.begin() + v_offset,
                                                   parameters_.begin() + output_offset),
                                hidden, input, inputs[position]));
    }
    std::vector<double> scores(time + 1, 0.0);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t position = 0; position <= time; ++position) {
        scores[position] = std::inner_product(query.begin(), query.end(), keys[position].begin(), 0.0) /
                           std::sqrt(static_cast<double>(hidden));
        maximum = std::max(maximum, scores[position]);
    }
    double denominator = 0.0;
    for (auto& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }
    std::vector<double> context(hidden, 0.0);
    for (std::size_t position = 0; position <= time; ++position) {
        for (std::size_t index = 0; index < hidden; ++index) context[index] += scores[position] / denominator * values[position][index];
    }
    auto result = matvec(std::vector<double>(parameters_.begin() + output_offset,
                                              parameters_.begin() + bias_offset),
                         output, hidden, context);
    for (std::size_t index = 0; index < output; ++index) result[index] += parameters_[bias_offset + index];
    return result;
}

std::vector<double> MatchedBaseline::gru_forward_step(const std::vector<double>& input,
                                                      std::vector<double>& hidden_state) const {
    const auto input_dim = config_.input_dim;
    const auto hidden = config_.hidden_dim;
    const auto output = config_.output_dim;
    const auto gate_matrix = hidden * input_dim + hidden * hidden + hidden;
    const auto z_input = 0;
    const auto z_hidden = hidden * input_dim;
    const auto z_bias = z_hidden + hidden * hidden;
    const auto r_input = gate_matrix;
    const auto r_hidden = r_input + hidden * input_dim;
    const auto r_bias = r_hidden + hidden * hidden;
    const auto n_input = r_bias + hidden;
    const auto n_hidden = n_input + hidden * input_dim;
    const auto n_bias = n_hidden + hidden * hidden;
    const auto output_offset = n_bias + hidden;
    const auto output_bias = output_offset + output * hidden;
    const auto z = [&]() {
        auto result = matvec(std::vector<double>(parameters_.begin() + z_input,
                                                 parameters_.begin() + z_hidden), hidden, input_dim, input);
        const auto recurrent = matvec(std::vector<double>(parameters_.begin() + z_hidden,
                                                          parameters_.begin() + z_bias), hidden, hidden, hidden_state);
        for (std::size_t index = 0; index < hidden; ++index) result[index] = sigmoid(result[index] + recurrent[index] + parameters_[z_bias + index]);
        return result;
    }();
    const auto r = [&]() {
        auto result = matvec(std::vector<double>(parameters_.begin() + r_input,
                                                 parameters_.begin() + r_hidden), hidden, input_dim, input);
        const auto recurrent = matvec(std::vector<double>(parameters_.begin() + r_hidden,
                                                          parameters_.begin() + r_bias), hidden, hidden, hidden_state);
        for (std::size_t index = 0; index < hidden; ++index) result[index] = sigmoid(result[index] + recurrent[index] + parameters_[r_bias + index]);
        return result;
    }();
    auto candidate = matvec(std::vector<double>(parameters_.begin() + n_input,
                                                parameters_.begin() + n_hidden), hidden, input_dim, input);
    const auto recurrent = matvec(std::vector<double>(parameters_.begin() + n_hidden,
                                                      parameters_.begin() + n_bias), hidden, hidden, hidden_state);
    for (std::size_t index = 0; index < hidden; ++index) candidate[index] = std::tanh(candidate[index] + r[index] * recurrent[index] + parameters_[n_bias + index]);
    for (std::size_t index = 0; index < hidden; ++index) hidden_state[index] = z[index] * hidden_state[index] + (1.0 - z[index]) * candidate[index];
    auto result = matvec(std::vector<double>(parameters_.begin() + output_offset,
                                              parameters_.begin() + output_bias), output, hidden, hidden_state);
    for (std::size_t index = 0; index < output; ++index) result[index] += parameters_[output_bias + index];
    return result;
}

std::vector<double> MatchedBaseline::ssm_forward_step(const std::vector<double>& input,
                                                      std::vector<double>& hidden_state) const {
    const auto input_dim = config_.input_dim;
    const auto hidden = config_.hidden_dim;
    const auto output = config_.output_dim;
    const auto input_offset = hidden;
    const auto output_offset = input_offset + hidden * input_dim;
    const auto bias_offset = output_offset + output * hidden;
    const auto input_effect = matvec(std::vector<double>(parameters_.begin() + input_offset,
                                                         parameters_.begin() + output_offset), hidden, input_dim, input);
    for (std::size_t index = 0; index < hidden; ++index) hidden_state[index] = (0.999 * sigmoid(parameters_[index])) * hidden_state[index] + input_effect[index];
    auto result = matvec(std::vector<double>(parameters_.begin() + output_offset,
                                              parameters_.begin() + bias_offset), output, hidden, hidden_state);
    for (std::size_t index = 0; index < output; ++index) result[index] += parameters_[bias_offset + index];
    return result;
}

std::vector<std::vector<double>> MatchedBaseline::forward(
    const std::vector<std::vector<double>>& inputs) const {
    std::vector<std::vector<double>> outputs;
    outputs.reserve(inputs.size());
    if (kind_ == BaselineKind::DenseCausalAttention) {
        for (std::size_t time = 0; time < inputs.size(); ++time) {
            require_dimension(inputs[time].size(), config_.input_dim, "input");
            outputs.push_back(dense_forward_step(inputs, time));
        }
        return outputs;
    }
    std::vector<double> hidden(config_.hidden_dim, 0.0);
    for (const auto& input : inputs) {
        require_dimension(input.size(), config_.input_dim, "input");
        outputs.push_back(kind_ == BaselineKind::GRU ? gru_forward_step(input, hidden) : ssm_forward_step(input, hidden));
    }
    return outputs;
}

double MatchedBaseline::loss(const std::vector<std::vector<double>>& inputs,
                              const std::vector<std::vector<double>>& targets,
                              const std::vector<std::uint8_t>& mask) const {
    if (inputs.size() != targets.size() || (!mask.empty() && mask.size() != inputs.size())) throw SequenceError("baseline sequence shape mismatch");
    const auto outputs = forward(inputs);
    std::size_t active = 0;
    double total = 0.0;
    for (std::size_t time = 0; time < inputs.size(); ++time) {
        if (!mask.empty() && mask[time] == 0) continue;
        require_dimension(targets[time].size(), config_.output_dim, "target");
        ++active;
        for (std::size_t index = 0; index < config_.output_dim; ++index) {
            const auto error = outputs[time][index] - targets[time][index];
            total += 0.5 * error * error;
        }
    }
    return active == 0 ? 0.0 : total / static_cast<double>(active * config_.output_dim);
}

double MatchedBaseline::batch_loss(
    const std::vector<std::vector<std::vector<double>>>& input_batch,
    const std::vector<std::vector<std::vector<double>>>& target_batch,
    const std::vector<std::vector<std::uint8_t>>& masks) const {
    if (input_batch.size() != target_batch.size() || input_batch.size() != masks.size()) throw SequenceError("baseline batch shape mismatch");
    double total = 0.0;
    for (std::size_t batch = 0; batch < input_batch.size(); ++batch) total += loss(input_batch[batch], target_batch[batch], masks[batch]);
    return input_batch.empty() ? 0.0 : total / static_cast<double>(input_batch.size());
}

void MatchedBaseline::train_finite_difference(
    const std::vector<std::vector<std::vector<double>>>& input_batch,
    const std::vector<std::vector<std::vector<double>>>& target_batch,
    const std::vector<std::vector<std::uint8_t>>& masks,
    std::size_t epochs, double learning_rate, double clip_norm) {
    if (!(learning_rate > 0.0) || !(clip_norm > 0.0)) throw SequenceError("baseline optimizer settings must be positive");
    const auto epsilon = 1e-5;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        auto original = parameter_vector();
        std::vector<double> gradient(original.size(), 0.0);
        for (std::size_t index = 0; index < original.size(); ++index) {
            auto plus = original;
            auto minus = original;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            set_parameter_vector(plus);
            const auto plus_loss = batch_loss(input_batch, target_batch, masks);
            set_parameter_vector(minus);
            const auto minus_loss = batch_loss(input_batch, target_batch, masks);
            gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon);
        }
        set_parameter_vector(original);
        const auto gradient_norm = std::max(normal_norm(gradient), 1e-12);
        const auto scale = std::min(1.0, clip_norm / gradient_norm);
        for (std::size_t index = 0; index < original.size(); ++index) original[index] -= learning_rate * scale * gradient[index];
        set_parameter_vector(original);
    }
}

std::size_t MatchedBaseline::parameter_count() const noexcept { return parameters_.size(); }

std::size_t MatchedBaseline::state_memory_bytes(std::size_t sequence_length) const noexcept {
    const auto state_bytes = config_.hidden_dim * sizeof(double);
    if (kind_ == BaselineKind::DenseCausalAttention) return 2 * sequence_length * config_.hidden_dim * sizeof(double) + state_bytes;
    return state_bytes;
}

std::vector<double> MatchedBaseline::parameter_vector() const { return parameters_; }

void MatchedBaseline::set_parameter_vector(const std::vector<double>& values) {
    if (values.size() != parameters_.size()) throw SequenceError("baseline parameter vector size mismatch");
    parameters_ = values;
}

std::string MatchedBaseline::name() const {
    if (kind_ == BaselineKind::DenseCausalAttention) return "dense_causal_attention";
    if (kind_ == BaselineKind::GRU) return "gru";
    return "diagonal_ssm";
}

}  // namespace cct
