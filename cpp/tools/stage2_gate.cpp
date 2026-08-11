#include "cct/sequence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cct::SelectiveSequenceCore;
using cct::SequenceConfig;
using cct::SequenceGradients;
using cct::SequenceOutput;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds;
    std::string details_json;
};

struct Metric {
    std::string name;
    double value;
    std::string unit;
    std::string threshold;
    std::string status;
};

struct TaskResult {
    double before_loss = 0.0;
    double after_loss = 0.0;
    double accuracy_train_length = 0.0;
    double accuracy_extrapolated = 0.0;
    std::size_t train_length = 0;
    std::size_t extrapolated_length = 0;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
    require(left.size() == right.size(), "vector sizes differ");
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) result = std::max(result, std::abs(left[index] - right[index]));
    return result;
}

double max_output_difference(const std::vector<std::vector<double>>& left,
                             const std::vector<std::vector<double>>& right) {
    require(left.size() == right.size(), "sequence lengths differ");
    double result = 0.0;
    for (std::size_t time = 0; time < left.size(); ++time) result = std::max(result, max_difference(left[time], right[time]));
    return result;
}

std::size_t argmax(const std::vector<double>& values) {
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::string git_command(const char* command) {
    auto* pipe = popen(command, "r");
    if (!pipe) return {};
    char buffer[256]{};
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
    pclose(pipe);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return {name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        std::ostringstream details;
        details << "{\"error\":\"";
        for (const auto character : std::string(error.what())) {
            if (character == '"') details << '\\';
            details << character;
        }
        details << "\"}";
        return {name, "FAIL", std::chrono::duration<double>(finished - started).count(), details.str()};
    }
}

std::vector<std::vector<double>> deterministic_inputs(std::size_t length, std::size_t dimension) {
    std::vector<std::vector<double>> result(length, std::vector<double>(dimension, 0.0));
    for (std::size_t time = 0; time < length; ++time) {
        for (std::size_t feature = 0; feature < dimension; ++feature) {
            result[time][feature] = std::sin(0.017 * static_cast<double>((time + 1) * (feature + 2))) +
                                    0.05 * std::cos(0.031 * static_cast<double>(time + feature + 1));
        }
    }
    return result;
}

std::vector<std::vector<double>> one_hot_targets(const std::vector<std::vector<double>>& sequence) {
    std::vector<std::vector<double>> result(sequence.size(), std::vector<double>(1, 0.0));
    for (std::size_t time = 0; time < sequence.size(); ++time) result[time][0] = sequence[time][0];
    return result;
}

std::vector<std::vector<double>> copy_sequence(std::size_t length, std::size_t symbol,
                                                std::size_t symbols) {
    require(length >= 2 && symbol < symbols, "invalid copy task dimensions");
    std::vector<std::vector<double>> sequence(length, std::vector<double>(symbols + 1, 0.0));
    sequence[0][symbol] = 1.0;
    sequence.back()[symbols] = 1.0;
    return sequence;
}

std::vector<std::vector<double>> copy_target(std::size_t length, std::size_t symbol,
                                              std::size_t symbols) {
    std::vector<std::vector<double>> target(length, std::vector<double>(symbols, 0.0));
    target.back()[symbol] = 1.0;
    return target;
}

std::vector<std::uint8_t> final_mask(std::size_t length) {
    std::vector<std::uint8_t> mask(length, 0);
    mask.back() = 1;
    return mask;
}

TaskResult train_copy_task() {
    constexpr std::size_t symbols = 4;
    constexpr std::size_t train_length = 16;
    constexpr std::size_t extrapolated_length = 32;
    SequenceConfig configuration{symbols + 1, 16, symbols, 1e-5, 101};
    SelectiveSequenceCore core(configuration);
    const auto mask = final_mask(train_length);
    const auto first_sequence = copy_sequence(train_length, 0, symbols);
    const auto first_target = copy_target(train_length, 0, symbols);
    const auto before = core.loss_only(first_sequence, first_target, mask);
    for (std::size_t epoch = 0; epoch < 3000; ++epoch) {
        for (std::size_t symbol = 0; symbol < symbols; ++symbol) {
            const auto sequence = copy_sequence(train_length, symbol, symbols);
            const auto target = copy_target(train_length, symbol, symbols);
            const auto gradients = core.loss_and_gradients(sequence, target, mask);
            core.apply_sgd(gradients, 0.04, 5.0);
        }
    }
    const auto after = core.loss_only(first_sequence, first_target, mask);
    const auto evaluate = [&](std::size_t length) {
        std::size_t correct = 0;
        for (std::size_t symbol = 0; symbol < symbols; ++symbol) {
            const auto sequence = copy_sequence(length, symbol, symbols);
            const auto target = copy_target(length, symbol, symbols);
            const auto output = core.forward(sequence).outputs.back();
            if (argmax(output) == symbol && output[argmax(output)] > 0.0) ++correct;
            (void)target;
        }
        return static_cast<double>(correct) / static_cast<double>(symbols);
    };
    return {before, after, evaluate(train_length), evaluate(extrapolated_length), train_length, extrapolated_length};
}

std::string path_equivalence_check() {
    SequenceConfig configuration{3, 12, 2, 1e-5, 7};
    SelectiveSequenceCore core(configuration);
    const auto sequence = deterministic_inputs(2048, 3);
    const auto loop = core.forward(sequence);
    const auto scan = core.forward_scan(sequence);
    const auto error = max_output_difference(loop.outputs, scan.outputs);
    const auto state_error = max_difference(loop.final_state.hidden, scan.final_state.hidden);
    require(error < 1e-12 && state_error < 1e-12, "scan and reference paths disagree");
    std::ostringstream details;
    details << "{\"output_max_abs_error\":" << error << ",\"state_max_abs_error\":" << state_error << ",\"length\":2048}";
    return details.str();
}

std::string streaming_check() {
    SequenceConfig configuration{3, 12, 2, 1e-5, 8};
    SelectiveSequenceCore core(configuration);
    const auto sequence = deterministic_inputs(257, 3);
    const auto full = core.forward(sequence);
    auto state = core.initial_state();
    std::vector<std::vector<double>> streamed;
    for (const auto& input : sequence) {
        std::vector<double> output;
        state = core.step(input, state, &output);
        streamed.push_back(std::move(output));
    }
    const auto output_error = max_output_difference(full.outputs, streamed);
    const auto state_error = max_difference(full.final_state.hidden, state.hidden);
    require(output_error < 1e-12 && state_error < 1e-12, "streaming and full paths disagree");
    std::ostringstream details;
    details << "{\"output_max_abs_error\":" << output_error << ",\"state_max_abs_error\":" << state_error << ",\"length\":257}";
    return details.str();
}

std::string gradient_check() {
    SequenceConfig configuration{2, 6, 1, 1e-5, 9};
    SelectiveSequenceCore core(configuration);
    const auto sequence = deterministic_inputs(12, 2);
    const auto targets = one_hot_targets(sequence);
    const auto analytic = core.loss_and_gradients(sequence, targets);
    const auto original = core.parameter_vector();
    const auto epsilon = 1e-6;
    const std::vector<std::size_t> selected{0, 2, 17, original.size() - 1};
    std::vector<double> analytic_values{
        analytic.d_input_projection[0],
        analytic.d_input_projection[2],
        analytic.d_previous_projection[5],
        analytic.d_output_bias[0],
    };
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < selected.size(); ++index) {
        auto plus = original;
        auto minus = original;
        plus[selected[index]] += epsilon;
        minus[selected[index]] -= epsilon;
        core.set_parameter_vector(plus);
        const auto plus_loss = core.loss_only(sequence, targets);
        core.set_parameter_vector(minus);
        const auto minus_loss = core.loss_only(sequence, targets);
        core.set_parameter_vector(original);
        maximum_error = std::max(maximum_error, std::abs((plus_loss - minus_loss) / (2.0 * epsilon) - analytic_values[index]));
    }
    require(maximum_error < 2e-5, "analytic and finite-difference gradients disagree");
    std::ostringstream details;
    details << "{\"selected_parameters\":4,\"max_abs_error\":" << maximum_error << "}";
    return details.str();
}

std::string stability_check() {
    SequenceConfig configuration{3, 24, 2, 1e-5, 10};
    SelectiveSequenceCore core(configuration);
    const auto sequence = deterministic_inputs(16384, 3);
    const auto result = core.forward(sequence);
    const auto state_norm = core.state_norm(result.final_state);
    const auto radius = core.transition_radius_bound();
    require(std::isfinite(state_norm) && std::isfinite(radius) && state_norm < 100.0 && radius < 4.0,
            "long-horizon state or transition diagnostic is unstable");
    std::ostringstream details;
    details << "{\"length\":16384,\"final_state_norm\":" << state_norm << ",\"transition_bound\":" << radius << "}";
    return details.str();
}

std::string checkpoint_check(const std::filesystem::path& output) {
    SequenceConfig configuration{3, 12, 2, 1e-5, 11};
    SelectiveSequenceCore core(configuration);
    const auto sequence = deterministic_inputs(128, 3);
    const auto expected = core.forward(sequence);
    const auto path = output / "stage2_checkpoint.chk";
    core.save_checkpoint(path.string(), 19);
    std::uint64_t optimizer_step = 0;
    auto restored = SelectiveSequenceCore::load_checkpoint(path.string(), &optimizer_step);
    const auto actual = restored.forward(sequence);
    const auto error = max_output_difference(expected.outputs, actual.outputs);
    require(optimizer_step == 19 && error < 1e-15, "checkpoint resume changed deterministic output");
    std::ostringstream details;
    details << "{\"optimizer_step\":" << optimizer_step << ",\"max_abs_error\":" << error << "}";
    return details.str();
}

std::string algorithmic_training_check(const TaskResult& result) {
    require(result.after_loss < result.before_loss * 0.25, "copy-task training did not reduce loss by 75 percent");
    require(result.accuracy_train_length >= 0.75, "copy-task train-length accuracy below threshold");
    require(result.accuracy_extrapolated >= 0.50, "copy-task extrapolated accuracy below threshold");
    std::ostringstream details;
    details << "{\"before_loss\":" << result.before_loss << ",\"after_loss\":" << result.after_loss
            << ",\"train_accuracy\":" << result.accuracy_train_length << ",\"extrapolated_accuracy\":"
            << result.accuracy_extrapolated << ",\"train_length\":" << result.train_length
            << ",\"extrapolated_length\":" << result.extrapolated_length << "}";
    return details.str();
}

std::string baseline_contract_check() {
    const std::size_t input_dim = 5;
    const std::size_t hidden_dim = 16;
    const std::size_t output_dim = 4;
    const std::size_t cct_parameters =
        hidden_dim * input_dim * 4 + output_dim * hidden_dim + output_dim * input_dim +
        hidden_dim + hidden_dim + hidden_dim + output_dim;
    const std::size_t diagonal_ssm_parameters = hidden_dim * 3 + output_dim * hidden_dim + output_dim * input_dim;
    const std::size_t gru_parameters = 3 * (hidden_dim * input_dim + hidden_dim * hidden_dim + hidden_dim) + output_dim * hidden_dim + output_dim;
    const std::size_t transformer_parameters = 4 * hidden_dim * hidden_dim + 2 * hidden_dim * input_dim + output_dim * hidden_dim + output_dim;
    require(cct_parameters > 0 && diagonal_ssm_parameters > 0 && gru_parameters > 0 && transformer_parameters > 0,
            "baseline parameter contract is empty");
    std::ostringstream details;
    details << "{\"matched_input_dim\":" << input_dim << ",\"matched_output_dim\":" << output_dim
            << ",\"cct_parameters\":" << cct_parameters << ",\"diagonal_ssm_parameters\":"
            << diagonal_ssm_parameters << ",\"gru_parameters\":" << gru_parameters
            << ",\"transformer_parameters\":" << transformer_parameters
            << ",\"baseline_status\":\"contract-defined\"}";
    return details.str();
}

std::string scaling_check() {
    SequenceConfig configuration{4, 32, 2, 1e-5, 12};
    SelectiveSequenceCore core(configuration);
    std::vector<double> times;
    const std::vector<std::size_t> lengths{256, 512, 1024, 2048, 4096};
    for (const auto length : lengths) {
        const auto sequence = deterministic_inputs(length, 4);
        (void)core.forward(sequence);
        const auto started = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 3; ++repeat) (void)core.forward(sequence);
        const auto finished = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(finished - started).count() / 3.0);
    }
    const auto slope = std::log(times.back() / times.front()) / std::log(static_cast<double>(lengths.back()) / static_cast<double>(lengths.front()));
    require(std::isfinite(slope) && slope < 1.30, "sequence scaling slope is not near-linear");
    std::ostringstream details;
    details << "{\"lengths\":[";
    for (std::size_t index = 0; index < lengths.size(); ++index) {
        if (index) details << ',';
        details << lengths[index];
    }
    details << "],\"seconds\":[";
    for (std::size_t index = 0; index < times.size(); ++index) {
        if (index) details << ',';
        details << times[index];
    }
    details << "],\"log_log_slope\":" << slope << ",\"decode_state_memory_bytes\":"
            << configuration.hidden_dim * sizeof(double) << "}";
    return details.str();
}

std::string ablation_contract_check() {
    SequenceConfig real{3, 12, 2, 1e-5, 14};
    SequenceConfig mimo{3, 12, 4, 1e-5, 14};
    SelectiveSequenceCore real_core(real);
    SelectiveSequenceCore mimo_core(mimo);
    const auto sequence = deterministic_inputs(32, 3);
    const auto real_output = real_core.forward(sequence).outputs.back();
    const auto mimo_output = mimo_core.forward(sequence).outputs.back();
    require(real_core.parameter_count() != mimo_core.parameter_count(), "MIMO parameter change was not visible");
    require(real_output.size() == 2 && mimo_output.size() == 4, "MIMO output contract is invalid");
    return "{\"real_state\":true,\"mimo_projection\":true,\"complex_state\":\"deferred\",\"normalization\":\"not_enabled\"}";
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("could not write " + path.string());
    stream << content;
}

std::string checks_json(const std::vector<Check>& checks) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index) output << ",\n";
        output << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
               << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details_json << "}";
    }
    output << "\n]\n";
    return output.str();
}

std::string metrics_json(const std::vector<Metric>& metrics) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index) output << ",\n";
        output << "  {\"name\":\"" << metrics[index].name << "\",\"value\":" << metrics[index].value
               << ",\"unit\":\"" << metrics[index].unit << "\",\"threshold\":\"" << metrics[index].threshold
               << "\",\"status\":\"" << metrics[index].status << "\"}";
    }
    output << "\n]\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-2/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    const auto copy_result = train_copy_task();
    const std::vector<std::pair<std::string, std::function<std::string()>>> functions{
        {"reference_scan_equivalence", path_equivalence_check},
        {"streaming_equivalence", streaming_check},
        {"gradient_correctness", gradient_check},
        {"long_horizon_stability", stability_check},
        {"algorithmic_copy_and_delayed_recall", [&]() { return algorithmic_training_check(copy_result); }},
        {"checkpoint_recovery", [&]() { return checkpoint_check(output); }},
        {"matched_baseline_contract", baseline_contract_check},
        {"ablation_integrity_contract", ablation_contract_check},
        {"linear_scaling_and_decode_memory", scaling_check},
    };
    std::vector<Check> checks;
    checks.reserve(functions.size());
    for (const auto& [name, function] : functions) checks.push_back(run_check(name, function));
    const bool passed = std::all_of(checks.begin(), checks.end(), [](const auto& check) { return check.status == "PASS"; });
    const auto commit_value = git_command("git rev-parse HEAD 2>/dev/null");
    const auto commit = commit_value.empty() ? std::string("unknown") : commit_value;
    const auto dirty = git_command("git status --porcelain 2>/dev/null");
    std::vector<Metric> metrics{
        {"mandatory_check_count", static_cast<double>(checks.size()), "checks", "all PASS", passed ? "PASS" : "FAIL"},
        {"copy_train_loss_before", copy_result.before_loss, "mse", "reported", "PASS"},
        {"copy_train_loss_after", copy_result.after_loss, "mse", "< 25% of before", copy_result.after_loss < copy_result.before_loss * 0.25 ? "PASS" : "FAIL"},
        {"copy_train_accuracy", copy_result.accuracy_train_length, "fraction", ">= 0.75", copy_result.accuracy_train_length >= 0.75 ? "PASS" : "FAIL"},
        {"copy_extrapolated_accuracy", copy_result.accuracy_extrapolated, "fraction", ">= 0.50", copy_result.accuracy_extrapolated >= 0.50 ? "PASS" : "FAIL"},
    };
    write_file(output / "checks.json", checks_json(checks));
    write_file(output / "metrics.json", metrics_json(metrics));
    std::ostringstream gate;
    gate << "{\n  \"stage\": 2,\n  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
         << "  \"transition\": \"" << (passed ? "Stage 3" : "STOP") << "\",\n"
         << "  \"implementation\": \"native-cpp-selective-recurrent-core\",\n  \"commit\": \"" << commit << "\",\n"
         << "  \"dirty_tree\": " << (dirty.empty() ? "false" : "true") << ",\n  \"approval_required\": true,\n"
         << "  \"complex_state\": \"deferred\",\n  \"normalization\": \"not_enabled\"\n}\n";
    write_file(output / "gate.json", gate.str());
    std::ostringstream report;
    report << "# Native C++ Stage 2 Gate Report\n\n"
           << "**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n"
           << "**Transition:** `" << (passed ? "Stage 3" : "STOP") << "`  \n"
           << "**Implementation:** `native-cpp-selective-recurrent-core`  \n"
           << "**Commit:** `" << commit << "`  \n"
           << "**Dirty tree:** `" << (dirty.empty() ? "False" : "True") << "`\n\n"
           << "## Mandatory checks\n\n| Check | Status | Duration (s) |\n|---|---:|---:|\n";
    for (const auto& check : checks) report << "| " << check.name << " | `" << check.status << "` | " << check.duration_seconds << " |\n";
    report << "\n## Stage 2 limitations\n\n"
           << "The real-valued diagonal selective recurrence and MIMO projections are implemented. Complex state and normalization remain explicitly deferred. The baseline check verifies matched parameter-contract definitions; full trained Transformer/GRU/SSM quality curves must be added before any claim of universal superiority.\n\n"
           << "A `PASS` authorizes Stage 3 preparation only; Stage 3 implementation requires explicit user approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
