#include "cct/sequence.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cct::SelectiveSequenceCore;
using cct::SequenceConfig;
using cct::SequenceError;
using cct::SequenceState;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
    require(left.size() == right.size(), "vector size mismatch");
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) result = std::max(result, std::abs(left[index] - right[index]));
    return result;
}

double max_output_difference(const std::vector<std::vector<double>>& left,
                             const std::vector<std::vector<double>>& right) {
    require(left.size() == right.size(), "sequence length mismatch");
    double result = 0.0;
    for (std::size_t time = 0; time < left.size(); ++time) result = std::max(result, max_difference(left[time], right[time]));
    return result;
}

double masked_loss(const SelectiveSequenceCore& core,
                   const std::vector<std::vector<double>>& sequence,
                   const std::vector<std::vector<double>>& target,
                   const std::vector<std::uint8_t>& mask) {
    const auto result = core.forward(sequence, mask);
    double total = 0.0;
    std::size_t active = 0U;
    for (std::size_t time = 0U; time < sequence.size(); ++time) {
        if (!mask.empty() && mask[time] == 0U) continue;
        require(target[time].size() == result.outputs[time].size(), "masked loss target shape mismatch");
        for (std::size_t output = 0U; output < target[time].size(); ++output) {
            const auto error = result.outputs[time][output] - target[time][output];
            total += 0.5 * error * error;
        }
        ++active;
    }
    const auto output_dim = result.outputs.empty() ? 0U : result.outputs.front().size();
    return active == 0U || output_dim == 0U ? 0.0 : total / static_cast<double>(active * output_dim);
}

std::vector<double> flatten_gradients(const cct::SequenceGradients& gradients) {
    std::vector<double> result;
    const auto append = [&](const std::vector<double>& values) { result.insert(result.end(), values.begin(), values.end()); };
    append(gradients.d_input_projection);
    append(gradients.d_previous_projection);
    append(gradients.d_retain_projection);
    append(gradients.d_write_projection);
    append(gradients.d_output_projection);
    append(gradients.d_skip_projection);
    append(gradients.d_bias);
    append(gradients.d_retain_bias);
    append(gradients.d_write_bias);
    append(gradients.d_output_bias);
    return result;
}

std::vector<std::vector<double>> inputs(std::size_t length, std::size_t dimension) {
    std::vector<std::vector<double>> result(length, std::vector<double>(dimension, 0.0));
    for (std::size_t time = 0; time < length; ++time) {
        for (std::size_t feature = 0; feature < dimension; ++feature) {
            result[time][feature] = std::sin(0.17 * static_cast<double>((time + 1) * (feature + 2))) +
                                    0.1 * std::cos(0.07 * static_cast<double>(time + feature));
        }
    }
    return result;
}

std::vector<std::vector<double>> targets(const std::vector<std::vector<double>>& sequence) {
    std::vector<std::vector<double>> result(sequence.size(), std::vector<double>(1, 0.0));
    for (std::size_t time = 0; time < sequence.size(); ++time) result[time][0] = std::tanh(sequence[time][0]);
    return result;
}

SequenceConfig config() {
    return SequenceConfig{3, 12, 2, 1e-5, 17};
}

void test_reference_scan_equivalence() {
    SelectiveSequenceCore core(config());
    const auto sequence = inputs(64, 3);
    const auto loop = core.forward(sequence);
    const auto scan = core.forward_scan(sequence);
    require(max_output_difference(loop.outputs, scan.outputs) < 1e-12, "reference and scan outputs differ");
    require(max_difference(loop.final_state.hidden, scan.final_state.hidden) < 1e-12, "reference and scan state differs");
    require(loop.final_state.previous_input == scan.final_state.previous_input, "previous-input state differs");
}

void test_streaming_equivalence() {
    SelectiveSequenceCore core(config());
    const auto sequence = inputs(37, 3);
    const auto full = core.forward(sequence);
    auto state = core.initial_state();
    std::vector<std::vector<double>> streaming;
    for (const auto& input : sequence) {
        std::vector<double> output;
        state = core.step(input, state, &output);
        streaming.push_back(std::move(output));
    }
    require(max_output_difference(full.outputs, streaming) < 1e-12, "streaming output differs from batched output");
    require(max_difference(full.final_state.hidden, state.hidden) < 1e-12, "streaming state differs from batched state");
}

void test_chunked_equivalence() {
    SelectiveSequenceCore core(config());
    const auto sequence = inputs(41, 3);
    const auto full = core.forward(sequence);
    auto state = core.initial_state();
    std::vector<std::vector<double>> chunked;
    for (std::size_t start = 0; start < sequence.size(); start += 7) {
        const auto end = std::min(sequence.size(), start + 7);
        std::vector<std::vector<double>> chunk(sequence.begin() + static_cast<std::ptrdiff_t>(start),
                                               sequence.begin() + static_cast<std::ptrdiff_t>(end));
        const auto result = core.forward(chunk, {}, &state);
        chunked.insert(chunked.end(), result.outputs.begin(), result.outputs.end());
        state = result.final_state;
    }
    require(max_output_difference(full.outputs, chunked) < 1e-12, "chunked output differs from full output");
    require(max_difference(full.final_state.hidden, state.hidden) < 1e-12, "chunked state differs from full state");
}

void test_mask_semantics() {
    SelectiveSequenceCore core(config());
    const auto sequence = inputs(12, 3);
    std::vector<std::uint8_t> mask(sequence.size(), 1);
    mask[4] = 0;
    const auto masked = core.forward(sequence, mask);
    auto state = core.initial_state();
    for (std::size_t time = 0; time < sequence.size(); ++time) {
        if (mask[time] != 0) state = core.step(sequence[time], state);
    }
    require(max_difference(masked.final_state.hidden, state.hidden) < 1e-12, "masked state semantics differ");
}

void test_gradient_finite_difference() {
    SequenceConfig small{2, 5, 1, 1e-5, 3};
    SelectiveSequenceCore core(small);
    const auto sequence = inputs(6, 2);
    const auto target = targets(sequence);
    const auto analytic = core.loss_and_gradients(sequence, target);
    const auto original = core.parameter_vector();
    const auto epsilon = 1e-6;
    std::vector<double> analytic_selected{
        analytic.d_input_projection[0],
        analytic.d_previous_projection[3],
        analytic.d_output_bias.front(),
        analytic.d_output_projection.front(),
    };
    const std::vector<std::size_t> selected{0, small.hidden_dim * small.input_dim + 3, original.size() - 1, 4 * small.input_dim + 4 * small.hidden_dim + small.output_dim + 2 * small.hidden_dim + 1};
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto parameter = selected[index];
        require(parameter < original.size(), "selected parameter out of range");
        auto plus = original;
        auto minus = original;
        plus[parameter] += epsilon;
        minus[parameter] -= epsilon;
        core.set_parameter_vector(plus);
        const auto plus_loss = core.loss_only(sequence, target);
        core.set_parameter_vector(minus);
        const auto minus_loss = core.loss_only(sequence, target);
        const auto finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon);
        core.set_parameter_vector(original);
        const auto error = std::abs(finite_difference - analytic_selected[index]);
        require(error < 2e-5, "analytic gradient disagrees with finite difference");
    }
}

void test_gradient_all_modes_finite_difference() {
    struct Mode {
        bool complex_state;
        bool normalize_state;
        bool normalize_output;
    };
    const std::vector<Mode> modes{
        {false, false, false}, {true, false, false}, {false, true, false}, {false, false, true},
        {true, true, false}, {true, false, true}, {false, true, true}, {true, true, true},
    };
    const auto sequence = inputs(5, 2);
    std::vector<std::vector<double>> target(sequence.size(), std::vector<double>(2, 0.0));
    for (std::size_t time = 0U; time < target.size(); ++time) {
        target[time][0] = 0.2 * std::sin(static_cast<double>(time + 1U));
        target[time][1] = -0.15 * std::cos(static_cast<double>(time + 2U));
    }
    const std::vector<std::uint8_t> mask{1U, 0U, 1U, 0U, 1U};
    std::size_t mode_index = 0U;
    for (const auto& mode : modes) {
        SequenceConfig test_config;
        test_config.input_dim = 2U;
        test_config.hidden_dim = 3U;
        test_config.output_dim = 2U;
        test_config.gate_epsilon = 1e-5;
        test_config.seed = 71U;
        test_config.complex_state = mode.complex_state;
        test_config.normalize_state = mode.normalize_state;
        test_config.normalize_output = mode.normalize_output;
        test_config.normalization_epsilon = 1e-6;
        test_config.selective_gates = true;
        SelectiveSequenceCore core(test_config);
        const auto analytic = flatten_gradients(core.loss_and_gradients(sequence, target, mask));
        const auto original = core.parameter_vector();
        require(analytic.size() == original.size(), "flattened sequence gradient size mismatch");
        double maximum_error = 0.0;
        std::size_t maximum_index = 0U;
        constexpr double epsilon = 1e-6;
        for (std::size_t parameter = 0U; parameter < original.size(); ++parameter) {
            auto plus = original;
            auto minus = original;
            plus[parameter] += epsilon;
            minus[parameter] -= epsilon;
            SelectiveSequenceCore plus_core(test_config);
            SelectiveSequenceCore minus_core(test_config);
            plus_core.set_parameter_vector(plus);
            minus_core.set_parameter_vector(minus);
            const auto numerical = (masked_loss(plus_core, sequence, target, mask) -
                                    masked_loss(minus_core, sequence, target, mask)) / (2.0 * epsilon);
            const auto error = std::abs(analytic[parameter] - numerical);
            if (error > maximum_error) {
                maximum_error = error;
                maximum_index = parameter;
            }
        }
        require(maximum_error < 1e-8, "all-mode sequence gradient disagrees with finite difference in mode " +
                                             std::to_string(mode_index) + " (max error=" + std::to_string(maximum_error) + ", parameter=" +
                                             std::to_string(maximum_index) + ")");
        ++mode_index;
    }
}

void test_masked_positions_and_target_validation() {
    SequenceConfig test_config;
    test_config.input_dim = 2U;
    test_config.hidden_dim = 3U;
    test_config.output_dim = 2U;
    test_config.seed = 79U;
    test_config.complex_state = true;
    test_config.normalize_state = true;
    test_config.normalize_output = true;
    SelectiveSequenceCore core(test_config);
    const auto sequence = inputs(5, 2);
    auto target = std::vector<std::vector<double>>(sequence.size(), std::vector<double>(2, 0.0));
    const std::vector<std::uint8_t> mask{1U, 0U, 1U, 0U, 1U};
    const auto masked = core.forward(sequence, mask);
    auto expected_state = core.initial_state();
    for (std::size_t time = 0U; time < sequence.size(); ++time) {
        if (mask[time] != 0U) expected_state = core.step(sequence[time], expected_state);
    }
    require(max_difference(masked.final_state.hidden, expected_state.hidden) < 1e-12,
            "masked positions changed the real recurrent state");
    require(max_difference(masked.final_state.hidden_imag, expected_state.hidden_imag) < 1e-12,
            "masked positions changed the imaginary recurrent state");
    const auto gradients = core.loss_and_gradients(sequence, target, mask);
    for (const auto value : flatten_gradients(gradients)) require(std::isfinite(value), "masked gradient is non-finite");
    bool rejected = false;
    target[1][0] = std::numeric_limits<double>::quiet_NaN();
    try {
        (void)core.loss_and_gradients(sequence, target, mask);
    } catch (const SequenceError&) {
        rejected = true;
    }
    require(rejected, "non-finite target was accepted");
}

void test_checkpoint_recovery() {
    SelectiveSequenceCore core(config());
    const auto sequence = inputs(19, 3);
    const auto before = core.forward(sequence);
    const auto path = std::filesystem::temp_directory_path() / "cct_sequence_test.chk";
    core.save_checkpoint(path.string(), 23);
    std::uint64_t optimizer_step = 0;
    auto restored = SelectiveSequenceCore::load_checkpoint(path.string(), &optimizer_step);
    const auto after = restored.forward(sequence);
    require(optimizer_step == 23, "optimizer step was not restored");
    require(max_output_difference(before.outputs, after.outputs) < 1e-15, "checkpoint outputs differ");
    require(restored.parameter_count() == core.parameter_count(), "checkpoint parameter count differs");
    std::filesystem::remove(path);
}

void test_stability_and_updates() {
    SequenceConfig stability_config{3, 12, 1, 1e-5, 17};
    SelectiveSequenceCore core(stability_config);
    const auto sequence = inputs(512, 3);
    const auto result = core.forward(sequence);
    require(core.transition_radius_bound() < 4.0, "transition bound is unexplained or unstable");
    require(core.state_norm(result.final_state) < 100.0, "long-horizon state exploded");
    const auto target = targets(sequence);
    const auto before = core.loss_only(sequence, target);
    const auto gradients = core.loss_and_gradients(sequence, target);
    core.apply_sgd(gradients, 0.01, 5.0);
    const auto after = core.loss_only(sequence, target);
    require(std::isfinite(before) && std::isfinite(after), "loss became non-finite");
    require(after <= before + 1e-9, "one clipped SGD step increased the deterministic loss");
}

void test_invalid_inputs() {
    SelectiveSequenceCore core(config());
    bool rejected = false;
    try { (void)core.step({1.0}, core.initial_state()); } catch (const SequenceError&) { rejected = true; }
    require(rejected, "wrong input dimension was accepted");
    rejected = false;
    try { (void)core.forward(inputs(4, 3), {1, 1}); } catch (const SequenceError&) { rejected = true; }
    require(rejected, "wrong mask length was accepted");
}

void test_complex_state_equivalence() {
    SequenceConfig complex_config{3, 8, 2, 1e-5, 23, true, false, false, 1e-6};
    SelectiveSequenceCore core(complex_config);
    const auto sequence = inputs(96, 3);
    const auto loop = core.forward(sequence);
    const auto scan = core.forward_scan(sequence);
    require(max_output_difference(loop.outputs, scan.outputs) < 1e-12, "complex scan output differs");
    require(max_difference(loop.final_state.hidden, scan.final_state.hidden) < 1e-12, "complex real state differs");
    require(max_difference(loop.final_state.hidden_imag, scan.final_state.hidden_imag) < 1e-12, "complex imaginary state differs");
    require(core.state_norm(loop.final_state) < 100.0, "complex state became unstable");
}

void test_normalization_ablation() {
    SequenceConfig normalized_config{3, 8, 2, 1e-5, 29, false, true, true, 1e-6};
    SelectiveSequenceCore normalized(normalized_config);
    const auto sequence = inputs(64, 3);
    const auto result = normalized.forward(sequence);
    require(std::abs(normalized.hidden_rms(result.final_state) - 1.0) < 1e-3, "state RMS normalization is not active");
    require(std::abs(normalized.output_rms(result.outputs.back()) - 1.0) < 1e-3, "output RMS normalization is not active");
    const auto path = std::filesystem::temp_directory_path() / "cct_normalized_sequence.chk";
    normalized.save_checkpoint(path.string(), 31);
    std::uint64_t optimizer_step = 0;
    const auto restored = SelectiveSequenceCore::load_checkpoint(path.string(), &optimizer_step);
    require(optimizer_step == 31, "normalized checkpoint optimizer step mismatch");
    require(restored.config().normalize_state && restored.config().normalize_output, "normalization flags were not checkpointed");
    require(restored.config().complex_state == normalized.config().complex_state, "complex ablation flag changed");
    require(max_output_difference(result.outputs, restored.forward(sequence).outputs) < 1e-15, "normalized checkpoint output differs");
    std::filesystem::remove(path);
}

void test_segmented_mask_scan() {
    SelectiveSequenceCore core(config());
    const auto sequence = inputs(40, 3);
    std::vector<std::uint8_t> mask(sequence.size(), 1);
    mask[2] = 0;
    mask[3] = 0;
    mask[11] = 0;
    mask[12] = 0;
    mask[27] = 0;
    const auto loop = core.forward(sequence, mask);
    const auto scan = core.forward_scan(sequence, mask);
    require(max_output_difference(loop.outputs, scan.outputs) < 1e-12, "segmented mask output differs");
    require(max_difference(loop.final_state.hidden, scan.final_state.hidden) < 1e-12, "segmented mask real state differs");
    require(max_difference(loop.final_state.hidden_imag, scan.final_state.hidden_imag) < 1e-12, "segmented mask imaginary state differs");
    require(loop.final_state.previous_input == scan.final_state.previous_input, "segmented mask previous input differs");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"reference_scan_equivalence", test_reference_scan_equivalence},
        {"streaming_equivalence", test_streaming_equivalence},
        {"chunked_equivalence", test_chunked_equivalence},
        {"mask_semantics", test_mask_semantics},
        {"gradient_finite_difference", test_gradient_finite_difference},
        {"gradient_all_modes_finite_difference", test_gradient_all_modes_finite_difference},
        {"masked_positions_and_target_validation", test_masked_positions_and_target_validation},
        {"checkpoint_recovery", test_checkpoint_recovery},
        {"stability_and_updates", test_stability_and_updates},
        {"invalid_input_safety", test_invalid_inputs},
        {"complex_state_equivalence", test_complex_state_equivalence},
        {"normalization_ablation_and_checkpoint", test_normalization_ablation},
        {"segmented_mask_scan", test_segmented_mask_scan},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "SUMMARY " << passed << "/" << tests.size() << " passed\n";
    return 0;
}
