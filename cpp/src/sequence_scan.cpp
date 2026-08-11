#include "cct/sequence.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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

double stable_gate(double raw) { return std::clamp(sigmoid(raw), 1e-4, 1.0 - 1e-4); }

}  // namespace

SequenceOutput SelectiveSequenceCore::forward_scan(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::uint8_t>& mask,
    const SequenceState* initial) const {
    if (!mask.empty()) {
        // Masked positions are an explicit segmented-state boundary. The reference
        // loop is the correctness path until a segmented scan kernel is introduced.
        return forward(inputs, mask, initial);
    }
    auto state = initial == nullptr ? initial_state() : *initial;
    validate_state(state);
    SequenceOutput result;
    result.outputs.resize(inputs.size());
    std::vector<double> prefix_a(config_.hidden_dim, 1.0);
    std::vector<double> prefix_b(config_.hidden_dim, 0.0);
    for (std::size_t time = 0; time < inputs.size(); ++time) {
        validate_input(inputs[time]);
        const auto retain_raw = affine(parameters_.retain_projection, parameters_.retain_bias,
                                       config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto write_raw = affine(parameters_.write_projection, parameters_.write_bias,
                                      config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto candidate_raw = affine(parameters_.input_projection, parameters_.bias,
                                          config_.hidden_dim, config_.input_dim, inputs[time]);
        const auto previous = time == 0
                                  ? state.previous_input
                                  : inputs[time - 1];
        const auto previous_projection = matvec(parameters_.previous_projection,
                                                config_.hidden_dim, config_.input_dim, previous);
        for (std::size_t hidden = 0; hidden < config_.hidden_dim; ++hidden) {
            const auto a = stable_gate(retain_raw[hidden]);
            const auto b = sigmoid(write_raw[hidden]) *
                           std::tanh(candidate_raw[hidden] + previous_projection[hidden]);
            prefix_b[hidden] = a * prefix_b[hidden] + b;
            prefix_a[hidden] *= a;
        }
        std::vector<double> hidden(config_.hidden_dim, 0.0);
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            hidden[index] = prefix_a[index] * state.hidden[index] + prefix_b[index];
        }
        result.outputs[time] = output_from_state(inputs[time], hidden);
        if (!std::all_of(hidden.begin(), hidden.end(), [](double value) { return std::isfinite(value); })) {
            throw SequenceError("scan state became non-finite");
        }
    }
    if (!inputs.empty()) state.previous_input = inputs.back();
    if (!inputs.empty()) {
        for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
            state.hidden[index] = prefix_a[index] * state.hidden[index] + prefix_b[index];
        }
    }
    result.final_state = std::move(state);
    return result;
}

}  // namespace cct
