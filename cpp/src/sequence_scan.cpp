#include "cct/sequence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

double stable_gate(double raw, double epsilon) { return std::clamp(sigmoid(raw), epsilon, 1.0 - epsilon); }

}  // namespace

SequenceOutput SelectiveSequenceCore::forward_scan(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::uint8_t>& mask,
    const SequenceState* initial) const {
    validate_mask(mask, inputs.size());
    if (config_.normalize_state) return forward(inputs, mask, initial);
    auto state = initial == nullptr ? initial_state() : *initial;
    validate_state(state);
    SequenceOutput result;
    result.outputs.resize(inputs.size());
    std::size_t time = 0;
    while (time < inputs.size()) {
        if (!mask.empty() && mask[time] == 0) {
            validate_input(inputs[time]);
            result.outputs[time] = output_from_state(inputs[time], state.hidden);
            ++time;
            continue;
        }
        const auto segment_start = time;
        const auto initial_hidden = state.hidden;
        const auto initial_imag = state.hidden_imag;
        std::vector<double> prefix_a(config_.hidden_dim, 1.0);
        std::vector<double> prefix_b(config_.hidden_dim, 0.0);
        std::vector<double> prefix_m00(config_.hidden_dim, 1.0);
        std::vector<double> prefix_m01(config_.hidden_dim, 0.0);
        std::vector<double> prefix_m10(config_.hidden_dim, 0.0);
        std::vector<double> prefix_m11(config_.hidden_dim, 1.0);
        std::vector<double> prefix_br(config_.hidden_dim, 0.0);
        std::vector<double> prefix_bi(config_.hidden_dim, 0.0);
        while (time < inputs.size() && (mask.empty() || mask[time] != 0)) {
            validate_input(inputs[time]);
            const auto retain_raw = affine(parameters_.retain_projection, parameters_.retain_bias,
                                           config_.hidden_dim, config_.input_dim, inputs[time]);
            const auto write_raw = affine(parameters_.write_projection, parameters_.write_bias,
                                          config_.hidden_dim, config_.input_dim, inputs[time]);
            const auto candidate_raw = affine(parameters_.input_projection, parameters_.bias,
                                              config_.hidden_dim, config_.input_dim, inputs[time]);
            const auto previous = time == segment_start ? state.previous_input : inputs[time - 1];
            const auto previous_projection = matvec(parameters_.previous_projection,
                                                    config_.hidden_dim, config_.input_dim, previous);
            std::vector<double> hidden(config_.hidden_dim, 0.0);
            std::vector<double> hidden_imag(config_.hidden_dim, 0.0);
            for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
                const auto retain = config_.selective_gates ? stable_gate(retain_raw[index], config_.gate_epsilon) : 0.95;
                const auto write = config_.selective_gates ? sigmoid(write_raw[index]) : 0.5;
                const auto candidate = std::tanh(candidate_raw[index] + previous_projection[index]);
                if (!config_.complex_state) {
                    prefix_b[index] = retain * prefix_b[index] + write * candidate;
                    prefix_a[index] *= retain;
                    hidden[index] = prefix_a[index] * initial_hidden[index] + prefix_b[index];
                } else {
                    const auto phase = 0.17 * std::sin(candidate_raw[index]);
                    const auto cosine = std::cos(phase);
                    const auto sine = std::sin(phase);
                    const auto imag_candidate = std::sin(candidate_raw[index] + previous_projection[index]);
                    const auto m00 = retain * cosine;
                    const auto m01 = -retain * sine;
                    const auto m10 = retain * sine;
                    const auto m11 = retain * cosine;
                    const auto br = cosine * write * candidate - sine * write * imag_candidate;
                    const auto bi = sine * write * candidate + cosine * write * imag_candidate;
                    const auto old_m00 = prefix_m00[index];
                    const auto old_m01 = prefix_m01[index];
                    const auto old_m10 = prefix_m10[index];
                    const auto old_m11 = prefix_m11[index];
                    const auto old_br = prefix_br[index];
                    const auto old_bi = prefix_bi[index];
                    prefix_m00[index] = m00 * old_m00 + m01 * old_m10;
                    prefix_m01[index] = m00 * old_m01 + m01 * old_m11;
                    prefix_m10[index] = m10 * old_m00 + m11 * old_m10;
                    prefix_m11[index] = m10 * old_m01 + m11 * old_m11;
                    prefix_br[index] = m00 * old_br + m01 * old_bi + br;
                    prefix_bi[index] = m10 * old_br + m11 * old_bi + bi;
                    hidden[index] = prefix_m00[index] * initial_hidden[index] + prefix_m01[index] * initial_imag[index] + prefix_br[index];
                    hidden_imag[index] = prefix_m10[index] * initial_hidden[index] + prefix_m11[index] * initial_imag[index] + prefix_bi[index];
                }
            }
            result.outputs[time] = output_from_state(inputs[time], hidden);
            if (!std::all_of(hidden.begin(), hidden.end(), [](double value) { return std::isfinite(value); }) ||
                !std::all_of(hidden_imag.begin(), hidden_imag.end(), [](double value) { return std::isfinite(value); })) {
                throw SequenceError("scan state became non-finite");
            }
            ++time;
        }
        if (time > segment_start) {
            for (std::size_t index = 0; index < config_.hidden_dim; ++index) {
                if (!config_.complex_state) {
                    state.hidden[index] = prefix_a[index] * initial_hidden[index] + prefix_b[index];
                    state.hidden_imag[index] = 0.0;
                } else {
                    state.hidden[index] = prefix_m00[index] * initial_hidden[index] + prefix_m01[index] * initial_imag[index] + prefix_br[index];
                    state.hidden_imag[index] = prefix_m10[index] * initial_hidden[index] + prefix_m11[index] * initial_imag[index] + prefix_bi[index];
                }
            }
            const auto active_count = static_cast<std::uint64_t>(time - segment_start);
            if (active_count > std::numeric_limits<std::uint64_t>::max() - state.position) {
                throw SequenceError("sequence position overflow");
            }
            state.previous_input = inputs[time - 1];
            state.position += active_count;
        }
    }
    result.final_state = std::move(state);
    return result;
}

}  // namespace cct
