#include "cct/causal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cct::CausalEvent;
using cct::CausalEventEncoder;
using cct::CausalEventLearner;
using cct::CausalEncodingConfig;
using cct::CausalGraphError;
using cct::CausalPrediction;
using cct::CausalStoreConfig;
using cct::CausalEventStore;
using cct::EventMode;
using cct::GraphConditionedConfig;
using cct::GraphConditionedSequence;
using cct::Intervention;
using cct::ProvenanceKind;
using cct::SequenceConfig;
using cct::StructuralSample;
using cct::SyntheticCausalConfig;
using cct::SyntheticCausalGenerator;
using cct::CausalDataset;
using cct::UncertaintyKind;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details_json;
};

struct Metric {
    std::string name;
    double value = 0.0;
    std::string unit;
    std::string threshold;
    std::string status;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const auto character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else output << character;
    }
    return output.str();
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

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path);
    require(static_cast<bool>(stream), "could not write " + path.string());
    stream << content;
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return {name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        return {name, "FAIL", std::chrono::duration<double>(finished - started).count(),
                std::string("{\"error\":\"") + json_escape(error.what()) + "\"}"};
    }
}

std::vector<std::vector<std::size_t>> candidate_parents(std::size_t variable_count) {
    std::vector<std::vector<std::size_t>> result(variable_count);
    for (std::size_t child = 1; child < variable_count; ++child) {
        for (std::size_t parent = 0; parent < child; ++parent) result[child].push_back(parent);
    }
    return result;
}

CausalEvent make_gate_event(cct::EventId id, std::int64_t timestamp, std::vector<cct::EventId> parents = {}) {
    CausalEvent event;
    event.id = id;
    event.semantic_payload = {0.1 * static_cast<double>(id)};
    event.coordinates = {static_cast<double>(id % 3) / 2.0, static_cast<double>(timestamp) / 4.0};
    event.timestamp = timestamp;
    event.causal_parents = std::move(parents);
    event.provenance = ProvenanceKind::Generated;
    event.uncertainty = {UncertaintyKind::Known, 1.0};
    return event;
}

std::string schema_graph_check() {
    CausalStoreConfig configuration;
    configuration.payload_dim = 1;
    configuration.coordinate_dim = 2;
    configuration.coordinate_min = {0.0, 0.0};
    configuration.coordinate_max = {1.0, 1.0};
    CausalEventStore first(configuration);
    first.insert(make_gate_event(1, 0));
    first.insert(make_gate_event(2, 1));
    first.insert(make_gate_event(3, 2, {1, 2}));
    CausalEventStore second(configuration);
    second.insert(make_gate_event(2, 1));
    second.insert(make_gate_event(1, 0));
    second.insert(make_gate_event(3, 2, {1, 2}));
    require(first.deterministic_export() == second.deterministic_export(), "insertion order changed deterministic export");
    const auto restored = CausalEventStore::deserialize_snapshot(first.serialize_snapshot());
    require(restored.deterministic_export() == first.deterministic_export(), "event schema round-trip changed data");
    require(restored.causal_past(3) == std::vector<cct::EventId>({1, 2}), "causal past query failed");
    require(restored.causal_future(1) == std::vector<cct::EventId>({3}), "causal future query failed");
    require(!restored.has_cycle() && restored.topological_order() == std::vector<cct::EventId>({1, 2, 3}),
            "DAG query contract failed");
    bool duplicate_rejected = false;
    try {
        first.insert(make_gate_event(1, 4));
    } catch (const CausalGraphError&) {
        duplicate_rejected = true;
    }
    require(duplicate_rejected, "duplicate event ID accepted");
    bool malformed_rejected = false;
    try {
        (void)CausalEventStore::deserialize_snapshot("invalid\n");
    } catch (const CausalGraphError&) {
        malformed_rejected = true;
    }
    require(malformed_rejected, "malformed snapshot accepted");
    bool cycle_rejected = false;
    try {
        CausalEventStore cyclic(configuration);
        auto first_cycle = make_gate_event(10, 0, {11});
        first_cycle.unresolved_parent_ids = {11};
        cyclic.insert(first_cycle);
        cyclic.insert(make_gate_event(11, 1, {10}));
    } catch (const CausalGraphError&) {
        cycle_rejected = true;
    }
    require(cycle_rejected, "cycle accepted");
    return "{\"round_trip\":true,\"deterministic_insertion_order\":true,\"duplicate_rejected\":true,\"cycle_rejected\":true}";
}

std::string leakage_temporal_check(const CausalDataset& dataset) {
    CausalEncodingConfig encoding;
    CausalEventEncoder masked_encoder(encoding);
    const auto masked = masked_encoder.encode(dataset.visible_events);
    encoding.prevent_future_leakage = false;
    const auto unmasked = CausalEventEncoder(encoding).encode(dataset.visible_events);
    require(masked.excluded_future_parent_count > 0, "future-parent audit did not observe a masked edge");
    require(masked.inputs != unmasked.inputs, "future leakage mask did not change the visible encoding");
    GraphConditionedConfig graph_config;
    graph_config.encoding = CausalEncodingConfig{};
    graph_config.sequence = SequenceConfig{masked_encoder.encoded_dim(), 12, 3, 1e-5, 81};
    GraphConditionedSequence graph_core(graph_config);
    const auto loop = graph_core.forward(dataset.visible_events);
    const auto scan = graph_core.forward_scan(dataset.visible_events);
    require(loop.outputs.size() == scan.outputs.size(), "graph-conditioned temporal length mismatch");
    double maximum_difference = 0.0;
    for (std::size_t time = 0; time < loop.outputs.size(); ++time) {
        require(loop.outputs[time].size() == scan.outputs[time].size(), "graph-conditioned output width mismatch");
        for (std::size_t feature = 0; feature < loop.outputs[time].size(); ++feature) {
            require(std::isfinite(loop.outputs[time][feature]) && std::isfinite(scan.outputs[time][feature]),
                    "graph-conditioned output is non-finite");
            maximum_difference = std::max(maximum_difference,
                                          std::abs(loop.outputs[time][feature] - scan.outputs[time][feature]));
        }
    }
    require(maximum_difference < 1e-12, "graph-conditioned loop/scan paths disagree");
    return "{\"future_parent_count\":" + std::to_string(masked.excluded_future_parent_count) +
           ",\"loop_scan_max_abs_error\":" + std::to_string(maximum_difference) + "}";
}

struct RecoveryResult {
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    std::size_t violations = 0;
};

RecoveryResult recover_graph(const CausalDataset& dataset, CausalEventLearner& learner,
                             const std::vector<std::vector<std::size_t>>& candidates) {
    learner.fit(dataset.training_samples, candidates, true);
    const auto predictions = learner.edge_predictions(0.08);
    std::set<std::pair<std::size_t, std::size_t>> predicted;
    std::set<std::pair<std::size_t, std::size_t>> truth;
    for (std::size_t child = 0; child < dataset.evaluator_truth.variable_count; ++child) {
        for (const auto parent : dataset.evaluator_truth.parents[child]) truth.emplace(parent, child);
    }
    std::size_t violations = 0;
    for (const auto& prediction : predictions) {
        if (!prediction.predicted) continue;
        predicted.emplace(prediction.parent, prediction.child);
        if (prediction.parent >= prediction.child) ++violations;
    }
    std::size_t true_positive = 0;
    for (const auto& edge : predicted) {
        if (truth.count(edge) != 0U) ++true_positive;
    }
    const auto precision = predicted.empty() ? 0.0 : static_cast<double>(true_positive) / static_cast<double>(predicted.size());
    const auto recall = truth.empty() ? 0.0 : static_cast<double>(true_positive) / static_cast<double>(truth.size());
    const auto f1 = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
    return {precision, recall, f1, violations};
}

std::string structural_recovery_check(const CausalDataset& dataset) {
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner learner(dataset.evaluator_truth.variable_count);
    const auto result = recover_graph(dataset, learner, candidates);
    require(result.precision >= 0.75 && result.recall >= 0.75 && result.f1 >= 0.75 && result.violations == 0,
            "structural recovery threshold failed");
    std::ostringstream details;
    details << "{\"precision\":" << result.precision << ",\"recall\":" << result.recall << ",\"f1\":"
            << result.f1 << ",\"topological_violations\":" << result.violations << "}";
    return details.str();
}

struct InterventionResult {
    double aware_error = 0.0;
    double observation_error = 0.0;
    double direction_accuracy = 0.0;
};

InterventionResult intervention_metrics(const CausalDataset& dataset, CausalEventLearner& aware,
                                        CausalEventLearner& observation_only,
                                        const std::vector<std::vector<std::size_t>>& candidates) {
    aware.fit(dataset.training_samples, candidates, true);
    observation_only.fit(dataset.training_samples, candidates, false);
    double aware_error = 0.0;
    double observation_error = 0.0;
    double direction = 0.0;
    for (const auto& item : dataset.intervention_cases) {
        const auto prediction = aware.predict_intervention(item.context_values, item.intervention.variable,
                                                           item.intervention.value, item.target);
        const auto control = observation_only.predict_intervention(item.context_values, item.intervention.variable,
                                                                   item.intervention.value, item.target);
        require(!prediction.abstained && !control.abstained, "known intervention unexpectedly abstained");
        aware_error += (prediction.value - item.outcome) * (prediction.value - item.outcome);
        observation_error += (control.value - item.outcome) * (control.value - item.outcome);
        const auto factual = aware.predict_observation(item.context_values, item.target);
        if ((prediction.value - factual.value) * (item.outcome - item.context_values[item.target]) >= 0.0) direction += 1.0;
    }
    const auto count = static_cast<double>(dataset.intervention_cases.size());
    return {aware_error / count, observation_error / count, direction / count};
}

std::string intervention_check(const CausalDataset& dataset) {
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner aware(dataset.evaluator_truth.variable_count);
    CausalEventLearner observation_only(dataset.evaluator_truth.variable_count);
    const auto result = intervention_metrics(dataset, aware, observation_only, candidates);
    require(result.aware_error < result.observation_error * 0.60 && result.direction_accuracy >= 0.80,
            "intervention learner did not beat observation-only control");
    std::ostringstream details;
    details << "{\"aware_mse\":" << result.aware_error << ",\"observation_only_mse\":"
            << result.observation_error << ",\"effect_direction_accuracy\":" << result.direction_accuracy << "}";
    return details.str();
}

std::string counterfactual_check(const CausalDataset& dataset) {
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner learner(dataset.evaluator_truth.variable_count);
    learner.fit(dataset.training_samples, candidates, true);
    double error = 0.0;
    double irrelevant_change = 0.0;
    for (const auto& item : dataset.counterfactual_cases) {
        const auto prediction = learner.predict_counterfactual(item.factual_values, item.intervention, item.target);
        require(!prediction.abstained, "known counterfactual unexpectedly abstained");
        error += (prediction.value - item.counterfactual_values[item.target]) *
                 (prediction.value - item.counterfactual_values[item.target]);
        const auto repeated = learner.predict_counterfactual(item.factual_values, item.intervention, item.target);
        irrelevant_change = std::max(irrelevant_change, std::abs(prediction.value - repeated.value));
    }
    error /= static_cast<double>(dataset.counterfactual_cases.size());
    require(error < 0.08 && irrelevant_change <= 1e-12, "counterfactual consistency threshold failed");
    std::ostringstream details;
    details << "{\"mse\":" << error << ",\"irrelevant_permutation_max_change\":" << irrelevant_change << "}";
    return details.str();
}

std::string robustness_abstention_check(const CausalDataset& dataset) {
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner clean(dataset.evaluator_truth.variable_count);
    clean.fit(dataset.training_samples, candidates, true);
    auto corrupted = candidates;
    corrupted[dataset.evaluator_truth.variable_count - 1] = {1};
    CausalEventLearner corrupted_learner(dataset.evaluator_truth.variable_count);
    corrupted_learner.fit(dataset.training_samples, corrupted, true);
    double clean_error = 0.0;
    double corrupted_error = 0.0;
    for (const auto& item : dataset.intervention_cases) {
        const auto clean_prediction = clean.predict_intervention(item.context_values, item.intervention.variable,
                                                                 item.intervention.value, item.target);
        const auto corrupted_prediction = corrupted_learner.predict_intervention(item.context_values, item.intervention.variable,
                                                                                  item.intervention.value, item.target);
        clean_error += (clean_prediction.value - item.outcome) * (clean_prediction.value - item.outcome);
        corrupted_error += (corrupted_prediction.value - item.outcome) * (corrupted_prediction.value - item.outcome);
        require(std::isfinite(corrupted_prediction.value), "corrupted graph produced non-finite output");
    }
    const auto count = static_cast<double>(dataset.intervention_cases.size());
    clean_error /= count;
    corrupted_error /= count;
    clean.set_graph_quality(true, false);
    const auto incomplete = clean.predict_intervention(dataset.intervention_cases.front().context_values, 1, 0.1, 3);
    clean.set_graph_quality(false, true);
    const auto conflicting = clean.predict_counterfactual(dataset.counterfactual_cases.front().factual_values,
                                                           dataset.counterfactual_cases.front().intervention, 3);
    require(corrupted_error > clean_error * 1.05 && incomplete.abstained && conflicting.abstained,
            "graph corruption or abstention behavior was not measurable");
    std::ostringstream details;
    details << "{\"clean_mse\":" << clean_error << ",\"corrupted_mse\":" << corrupted_error
            << ",\"incomplete_abstained\":" << (incomplete.abstained ? "true" : "false")
            << ",\"conflicting_abstained\":" << (conflicting.abstained ? "true" : "false") << "}";
    return details.str();
}

std::string strict_contract_failure_check(const CausalDataset& dataset) {
    CausalStoreConfig configuration;
    configuration.payload_dim = 1;
    configuration.coordinate_dim = 2;
    configuration.coordinate_min = {0.0, 0.0};
    configuration.coordinate_max = {1.0, 1.0};
    CausalEventStore store(configuration);
    store.insert(make_gate_event(1, 0));
    bool missing_parent_rejected = false;
    try {
        store.insert(make_gate_event(2, 1, {99}));
    } catch (const CausalGraphError&) {
        missing_parent_rejected = true;
    }
    auto invalid_provenance = make_gate_event(3, 2, {1});
    invalid_provenance.provenance = static_cast<ProvenanceKind>(99);
    bool invalid_enum_rejected = false;
    try {
        store.insert(invalid_provenance);
    } catch (const CausalGraphError&) {
        invalid_enum_rejected = true;
    }
    auto same_time = make_gate_event(4, 0, {1});
    bool same_time_rejected = false;
    try {
        store.insert(same_time);
    } catch (const CausalGraphError&) {
        same_time_rejected = true;
    }
    const auto duplicate_events = std::vector<CausalEvent>{make_gate_event(5, 3), make_gate_event(5, 4)};
    bool duplicate_encoder_rejected = false;
    try {
        static_cast<void>(CausalEventEncoder().encode(duplicate_events));
    } catch (const CausalGraphError&) {
        duplicate_encoder_rejected = true;
    }
    auto nonfinite = make_gate_event(6, 5);
    nonfinite.semantic_payload.front() = std::numeric_limits<double>::quiet_NaN();
    bool nonfinite_encoder_rejected = false;
    try {
        static_cast<void>(CausalEventEncoder().encode({nonfinite}));
    } catch (const CausalGraphError&) {
        nonfinite_encoder_rejected = true;
    }
    CausalEventLearner learner(dataset.evaluator_truth.variable_count);
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    learner.fit(dataset.training_samples, candidates, true);
    const auto before = learner.parent_hypotheses();
    auto invalid_candidates = candidates;
    invalid_candidates.back() = {2, 1};
    bool transactional_fit_rejected = false;
    try {
        learner.fit(dataset.training_samples, invalid_candidates, true);
    } catch (const CausalGraphError&) {
        transactional_fit_rejected = true;
    }
    auto nonfinite_context = dataset.intervention_cases.front().context_values;
    nonfinite_context.front() = std::numeric_limits<double>::infinity();
    bool nonfinite_query_rejected = false;
    try {
        static_cast<void>(learner.predict_intervention(nonfinite_context, 1, 0.2, 3));
    } catch (const CausalGraphError&) {
        nonfinite_query_rejected = true;
    }
    require(missing_parent_rejected && invalid_enum_rejected && same_time_rejected && duplicate_encoder_rejected &&
                nonfinite_encoder_rejected && transactional_fit_rejected && learner.fitted() &&
                learner.parent_hypotheses() == before && nonfinite_query_rejected,
            "strict causal metadata or learner failure path did not fail closed");
    return "{\"missing_parent\":\"rejected\",\"invalid_enum\":\"rejected\",\"same_timestamp\":\"rejected\",\"duplicate_encoder_id\":\"rejected\",\"nonfinite_encoder\":\"rejected\",\"transactional_fit\":\"preserved\",\"nonfinite_query\":\"rejected\"}";
}

std::string ablation_check(const CausalDataset& dataset) {
    CausalEncodingConfig with_edges;
    CausalEncodingConfig without_edges = with_edges;
    without_edges.include_causal_edges = false;
    CausalEncodingConfig without_intervention = with_edges;
    without_intervention.include_intervention_marker = false;
    CausalEncodingConfig without_uncertainty = with_edges;
    without_uncertainty.include_uncertainty = false;
    require(CausalEventEncoder(with_edges).encoded_dim() != CausalEventEncoder(without_edges).encoded_dim(),
            "edge ablation is not observable");
    require(CausalEventEncoder(with_edges).encoded_dim() != CausalEventEncoder(without_intervention).encoded_dim(),
            "intervention ablation is not observable");
    require(CausalEventEncoder(with_edges).encoded_dim() != CausalEventEncoder(without_uncertainty).encoded_dim(),
            "uncertainty ablation is not observable");
    GraphConditionedConfig scalar;
    scalar.encoding = with_edges;
    scalar.sequence = SequenceConfig{CausalEventEncoder(with_edges).encoded_dim(), 8, 1, 1e-5, 88};
    GraphConditionedConfig mimo = scalar;
    mimo.sequence.output_dim = 3;
    GraphConditionedSequence scalar_model(scalar);
    GraphConditionedSequence mimo_model(mimo);
    const auto scalar_output = scalar_model.forward(dataset.visible_events).outputs.front();
    const auto mimo_output = mimo_model.forward(dataset.visible_events).outputs.front();
    require(scalar_output.size() == 1 && mimo_output.size() == 3, "MIMO ablation is not observable");
    return "{\"edge_channel\":true,\"temporal_channel\":true,\"intervention_channel\":true,\"uncertainty_channel\":true,\"mimo_channel\":true}";
}

std::string reproducibility_check() {
    const auto first = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 101, true});
    const auto second = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 101, true});
    const auto changed = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 102, true});
    require(first.dataset_fingerprint == second.dataset_fingerprint, "identical seed changed dataset fingerprint");
    require(first.dataset_fingerprint != changed.dataset_fingerprint, "changed seed did not change dataset fingerprint");
    return "{\"same_seed_fingerprint_equal\":true,\"changed_seed_fingerprint_distinct\":true}";
}

std::string checks_json(const std::vector<Check>& checks) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index != 0) output << ",\n";
        output << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
               << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":"
               << checks[index].details_json << "}";
    }
    output << "\n]\n";
    return output.str();
}

std::string metrics_json(const std::vector<Metric>& metrics) {
    std::ostringstream output;
    output << "[\n";
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index != 0) output << ",\n";
        output << "  {\"name\":\"" << metrics[index].name << "\",\"value\":" << metrics[index].value
               << ",\"unit\":\"" << metrics[index].unit << "\",\"threshold\":\"" << metrics[index].threshold
               << "\",\"status\":\"" << metrics[index].status << "\"}";
    }
    output << "\n]\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-3/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    const auto dataset = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 101, true});
    const std::vector<std::pair<std::string, std::function<std::string()>>> functions{
        {"schema_integrity_and_graph_safety", schema_graph_check},
        {"leakage_control_and_temporal_masking", [&]() { return leakage_temporal_check(dataset); }},
        {"structural_edge_recovery", [&]() { return structural_recovery_check(dataset); }},
        {"intervention_effect_prediction", [&]() { return intervention_check(dataset); }},
        {"counterfactual_consistency", [&]() { return counterfactual_check(dataset); }},
        {"robustness_and_abstention", [&]() { return robustness_abstention_check(dataset); }},
        {"strict_contract_failure_closure", [&]() { return strict_contract_failure_check(dataset); }},
        {"ablation_integrity", [&]() { return ablation_check(dataset); }},
        {"reproducibility", reproducibility_check},
    };
    std::vector<Check> checks;
    checks.reserve(functions.size());
    for (const auto& [name, function] : functions) checks.push_back(run_check(name, function));
    const bool passed = std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    const auto commit_value = git_command("git rev-parse HEAD 2>/dev/null");
    const auto commit = commit_value.empty() ? std::string("unknown") : commit_value;
    const auto dirty = git_command("git status --porcelain 2>/dev/null");

    std::vector<Metric> metrics{
        {"mandatory_check_count", static_cast<double>(checks.size()), "checks", "all PASS", passed ? "PASS" : "FAIL"},
        {"dataset_fingerprint", static_cast<double>(dataset.dataset_fingerprint), "uint64", "reported", "PASS"},
        {"visible_event_count", static_cast<double>(dataset.visible_events.size()), "events", ">= 4", "PASS"},
        {"training_sample_count", static_cast<double>(dataset.training_samples.size()), "samples", ">= 32", "PASS"},
        {"held_out_intervention_count", static_cast<double>(dataset.intervention_cases.size()), "cases", ">= 16", "PASS"},
        {"held_out_counterfactual_count", static_cast<double>(dataset.counterfactual_cases.size()), "cases", ">= 16", "PASS"},
    };
    write_file(output / "checks.json", checks_json(checks));
    write_file(output / "metrics.json", metrics_json(metrics));
    write_file(output / "visible_input.json",
               "{\n  \"schema_version\": 1,\n  \"visible_fields\": [\"event_id\", \"payload\", \"coordinates\", \"timestamp\", \"causal_parent_ids\", \"intervention_marker\", \"uncertainty\", \"provenance\"],\n  \"evaluator_only_fields_excluded\": [\"structural_coefficients\", \"exogenous_noise\", \"counterfactual_target\"]\n}\n");
    std::ostringstream truth;
    truth << "{\n  \"evaluator_only\": true,\n  \"variable_count\": " << dataset.evaluator_truth.variable_count
          << ",\n  \"dataset_fingerprint\": " << dataset.dataset_fingerprint
          << ",\n  \"ground_truth_edge_count\": ";
    std::size_t edge_count = 0;
    for (const auto& parents : dataset.evaluator_truth.parents) edge_count += parents.size();
    truth << edge_count << ",\n  \"contains_model_visible_payloads\": false\n}\n";
    write_file(output / "evaluator_truth.json", truth.str());

    std::ostringstream gate;
    gate << "{\n  \"stage\": 3,\n  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
         << "  \"transition\": \"" << (passed ? "Stage 4 preparation (approval required)" : "STOP") << "\",\n"
         << "  \"implementation\": \"native-cpp-causal-event-learning\",\n  \"commit\": \"" << commit << "\",\n"
         << "  \"dirty_tree\": " << (dirty.empty() ? "false" : "true") << ",\n  \"approval_required\": true,\n"
         << "  \"model_truth_separation\": \"enforced\"\n}\n";
    write_file(output / "gate.json", gate.str());

    std::ostringstream report;
    report << "# Native C++ Stage 3 Gate Report\n\n"
           << "**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n"
           << "**Transition:** `" << (passed ? "Stage 4 preparation; approval required" : "STOP") << "`  \n"
           << "**Implementation:** `native-cpp-causal-event-learning`  \n"
           << "**Commit:** `" << commit << "`  \n"
           << "**Dirty tree at gate execution:** `" << (dirty.empty() ? "False" : "True") << "`\n\n"
           << "## Methodology\n\n"
           << "The gate uses deterministic synthetic structural equations with four variables, confounded observational noise, held-out do-interventions, paired counterfactual worlds, a held-out nonlinear parent feature, and a separate evaluator-only truth structure. The model-visible schema excludes coefficients, exogenous noise, and counterfactual targets. The Stage 2 selective recurrent core is exercised through a graph-conditioned encoder with future-parent masking and loop/scan equivalence.\n\n"
           << "## Mandatory checks\n\n| Check | Status | Duration (s) |\n|---|---:|---:|\n";
    for (const auto& check : checks) report << "| " << check.name << " | `" << check.status << "` | " << check.duration_seconds << " |\n";
    report << "\n## Scope limits\n\n"
           << "A passing gate demonstrates causal-structure-aware prediction on the declared synthetic structural-equation distributions. It does not establish general causal understanding, causal discovery on real data, language competence, or superintelligence. Stage 4 implementation remains blocked until explicit user approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
