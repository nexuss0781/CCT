#include "cct/causal.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cct::CausalEvent;
using cct::CausalEventEncoder;
using cct::CausalEventStore;
using cct::CausalEncodingConfig;
using cct::CausalGraphError;
using cct::CausalPrediction;
using cct::CausalStoreConfig;
using cct::EventMode;
using cct::GraphConditionedConfig;
using cct::GraphConditionedSequence;
using cct::Intervention;
using cct::ProvenanceKind;
using cct::SequenceConfig;
using cct::StructuralSample;
using cct::SyntheticCausalConfig;
using cct::SyntheticCausalGenerator;
using cct::CausalEventLearner;
using cct::UncertaintyKind;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

CausalEvent make_event(cct::EventId id, std::int64_t timestamp, std::vector<cct::EventId> parents = {}) {
    CausalEvent event;
    event.id = id;
    event.semantic_payload = {static_cast<double>(id), static_cast<double>(id) * 0.5};
    event.coordinates = {static_cast<double>(id % 3) / 2.0, static_cast<double>(timestamp) / 3.0};
    event.timestamp = timestamp;
    event.causal_parents = std::move(parents);
    event.provenance = ProvenanceKind::Generated;
    event.uncertainty = {UncertaintyKind::Known, 1.0};
    return event;
}

void test_schema_graph_and_queries() {
    CausalStoreConfig configuration;
    configuration.payload_dim = 2;
    configuration.coordinate_dim = 2;
    configuration.coordinate_min = {0.0, 0.0};
    configuration.coordinate_max = {1.0, 1.0};
    CausalEventStore first(configuration);
    first.insert(make_event(1, 0, {}));
    first.insert(make_event(2, 1, {}));
    first.insert(make_event(3, 2, {1, 2}));
    require(first.parents_of(3) == std::vector<cct::EventId>({1, 2}), "parent query failed");
    require(first.children_of(1) == std::vector<cct::EventId>({3}), "child query failed");
    require(first.causal_past(3) == std::vector<cct::EventId>({1, 2}), "causal past query failed");
    require(first.causal_future(1) == std::vector<cct::EventId>({3}), "causal future query failed");
    require(first.topological_order() == std::vector<cct::EventId>({1, 2, 3}), "topological order failed");
    require(!first.has_cycle(), "valid graph reported a cycle");

    CausalEventStore second(configuration);
    second.insert(make_event(2, 1, {}));
    second.insert(make_event(1, 0, {}));
    second.insert(make_event(3, 2, {1, 2}));
    require(first.deterministic_export() == second.deterministic_export(), "insertion order changed snapshot");
    const auto restored = CausalEventStore::deserialize_snapshot(first.serialize_snapshot());
    require(restored.deterministic_export() == first.deterministic_export(), "schema round-trip changed event data");
    require(restored.event(3).provenance == ProvenanceKind::Generated, "provenance was not preserved");

    bool duplicate_rejected = false;
    try {
        first.insert(make_event(1, 3, {}));
    } catch (const CausalGraphError&) {
        duplicate_rejected = true;
    }
    require(duplicate_rejected, "duplicate event ID was accepted");

    bool coordinate_rejected = false;
    try {
        auto invalid = make_event(4, 3, {});
        invalid.coordinates[0] = 2.0;
        first.insert(invalid);
    } catch (const CausalGraphError&) {
        coordinate_rejected = true;
    }
    require(coordinate_rejected, "invalid coordinate was accepted");

    CausalEventStore cyclic(configuration);
    auto first_cycle = make_event(10, 0, {11});
    first_cycle.unresolved_parent_ids = {11};
    cyclic.insert(first_cycle);
    bool cycle_rejected = false;
    try {
        cyclic.insert(make_event(11, 1, {10}));
    } catch (const CausalGraphError&) {
        cycle_rejected = true;
    }
    require(cycle_rejected, "cycle was accepted");
}

void test_encoder_and_graph_conditioned_core() {
    std::vector<CausalEvent> events;
    events.push_back(make_event(1, 0, {}));
    events.push_back(make_event(2, 1, {1, 3}));
    events.push_back(make_event(3, 2, {1}));
    events[1].intervention = Intervention{0, 0.7, EventMode::DoIntervention};
    events[1].provenance = ProvenanceKind::Intervened;
    events[1].uncertainty = {UncertaintyKind::Estimated, 0.8};
    CausalEncodingConfig encoding;
    encoding.payload_dim = 2;
    CausalEventEncoder encoder(encoding);
    const auto encoded = encoder.encode(events);
    require(encoded.inputs.size() == events.size() && encoded.inputs.front().size() == encoder.encoded_dim(),
            "causal encoder shape failed");
    require(encoded.excluded_future_parent_count == 1, "future-parent masking did not count leakage");
    CausalEncodingConfig unmasked_config;
    unmasked_config.prevent_future_leakage = false;
    unmasked_config.payload_dim = 2;
    const auto unmasked = CausalEventEncoder(unmasked_config).encode(events);
    require(encoded.inputs[1] != unmasked.inputs[1], "future masking did not alter the encoded input");

    GraphConditionedConfig configuration;
    configuration.encoding = encoding;
    configuration.sequence = SequenceConfig{encoder.encoded_dim(), 10, 2, 1e-5, 77};
    GraphConditionedSequence model(configuration);
    const auto loop = model.forward(events);
    const auto scan = model.forward_scan(events);
    require(loop.outputs.size() == events.size() && scan.outputs.size() == events.size(),
            "graph-conditioned output length failed");
    for (std::size_t index = 0; index < events.size(); ++index) {
        require(loop.outputs[index].size() == 2 && scan.outputs[index].size() == 2, "graph-conditioned output width failed");
        for (const auto value : loop.outputs[index]) require(std::isfinite(value), "graph-conditioned output is non-finite");
        for (std::size_t feature = 0; feature < 2; ++feature) {
            require(std::abs(loop.outputs[index][feature] - scan.outputs[index][feature]) < 1e-12,
                    "graph-conditioned loop/scan mismatch");
        }
    }
}

std::vector<std::vector<std::size_t>> candidate_parents(std::size_t variable_count) {
    std::vector<std::vector<std::size_t>> result(variable_count);
    for (std::size_t child = 1; child < variable_count; ++child) {
        for (std::size_t parent = 0; parent < child; ++parent) result[child].push_back(parent);
    }
    return result;
}

void test_causal_learning_and_abstention() {
    const auto dataset = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 101, true});
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner learner(dataset.evaluator_truth.variable_count);
    learner.fit(dataset.training_samples, candidates, true);
    require(learner.fitted(), "causal learner did not fit");
    const auto edges = learner.edge_predictions(0.08);
    require(!edges.empty(), "causal edge prediction was empty");
    for (const auto& edge : edges) require(edge.parent < edge.child, "learner emitted invalid edge direction");

    double intervention_error = 0.0;
    double direction_correct = 0.0;
    for (const auto& item : dataset.intervention_cases) {
        const auto prediction = learner.predict_intervention(item.context_values, item.intervention.variable,
                                                              item.intervention.value, item.target);
        require(!prediction.abstained && std::isfinite(prediction.value), "known intervention was abstained");
        intervention_error += (prediction.value - item.outcome) * (prediction.value - item.outcome);
        const auto factual = learner.predict_observation(item.context_values, item.target);
        const auto predicted_effect = prediction.value - factual.value;
        const auto true_effect = item.outcome - item.context_values[item.target];
        if (predicted_effect * true_effect >= 0.0) direction_correct += 1.0;
    }
    intervention_error /= static_cast<double>(dataset.intervention_cases.size());
    direction_correct /= static_cast<double>(dataset.intervention_cases.size());
    require(intervention_error < 0.08 && direction_correct >= 0.80, "intervention learner threshold failed");

    double counterfactual_error = 0.0;
    for (const auto& item : dataset.counterfactual_cases) {
        const auto prediction = learner.predict_counterfactual(item.factual_values, item.intervention, item.target);
        require(!prediction.abstained, "known counterfactual was abstained");
        counterfactual_error += (prediction.value - item.counterfactual_values[item.target]) *
                                (prediction.value - item.counterfactual_values[item.target]);
    }
    counterfactual_error /= static_cast<double>(dataset.counterfactual_cases.size());
    require(counterfactual_error < 0.08, "counterfactual threshold failed");

    learner.set_graph_quality(true, false);
    const auto incomplete = learner.predict_intervention(dataset.intervention_cases.front().context_values, 1, 0.2, 3);
    require(incomplete.abstained && incomplete.confidence == 0.0, "incomplete graph did not abstain");
    learner.set_graph_quality(false, true);
    const auto conflicting = learner.predict_counterfactual(dataset.counterfactual_cases.front().factual_values,
                                                             dataset.counterfactual_cases.front().intervention, 3);
    require(conflicting.abstained, "conflicting graph did not abstain");
}

void test_strict_metadata_and_learner_failure_paths() {
    CausalStoreConfig configuration;
    configuration.payload_dim = 2;
    configuration.coordinate_dim = 2;
    configuration.coordinate_min = {0.0, 0.0};
    configuration.coordinate_max = {1.0, 1.0};
    CausalEventStore store(configuration);
    store.insert(make_event(1, 0, {}));
    auto unresolved_mismatch = make_event(2, 1, {99});
    bool rejected = false;
    try {
        store.insert(unresolved_mismatch);
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "parent missing without explicit unresolved marker was accepted");
    auto unresolved = make_event(2, 1, {99});
    unresolved.unresolved_parent_ids = {99};
    store.insert(unresolved);
    auto invalid_enum = make_event(3, 2, {1});
    invalid_enum.provenance = static_cast<ProvenanceKind>(99);
    rejected = false;
    try {
        store.insert(invalid_enum);
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "invalid provenance enum was accepted");

    auto same_timestamp = make_event(4, 0, {1});
    rejected = false;
    try {
        store.insert(same_timestamp);
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "strict timestamp policy accepted a same-time parent");
    configuration.temporal_policy = cct::TemporalCausalityPolicy::AllowSameTimestamp;
    CausalEventStore same_time_store(configuration);
    same_time_store.insert(make_event(1, 0, {}));
    same_time_store.insert(make_event(4, 0, {1}));

    auto duplicate_events = std::vector<CausalEvent>{make_event(1, 0, {}), make_event(1, 1, {})};
    rejected = false;
    try {
        static_cast<void>(CausalEventEncoder().encode(duplicate_events));
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "encoder accepted duplicate event IDs");
    auto nonfinite_event = make_event(5, 3, {});
    nonfinite_event.semantic_payload[0] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        static_cast<void>(CausalEventEncoder().encode({nonfinite_event}));
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "encoder accepted non-finite payload");

    const auto valid_snapshot = same_time_store.serialize_snapshot();
    auto malformed_snapshot = valid_snapshot;
    const auto event_marker = malformed_snapshot.find("EVENT 1 1 0 0");
    require(event_marker != std::string::npos, "snapshot fixture marker was not found");
    malformed_snapshot.replace(event_marker, std::string("EVENT 1 1 0 0").size(), "EVENT 1 1 0 99");
    rejected = false;
    try {
        static_cast<void>(CausalEventStore::deserialize_snapshot(malformed_snapshot));
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "malformed snapshot enum was accepted");

    const auto dataset = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 101, true});
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner learner(dataset.evaluator_truth.variable_count);
    learner.fit(dataset.training_samples, candidates, true);
    const auto before = learner.parent_hypotheses();
    auto invalid_candidates = candidates;
    invalid_candidates[3] = {2, 1};
    rejected = false;
    try {
        learner.fit(dataset.training_samples, invalid_candidates, true);
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected && learner.fitted() && learner.parent_hypotheses() == before,
            "failed learner fit corrupted the previously fitted model");
    auto nonfinite_context = dataset.intervention_cases.front().context_values;
    nonfinite_context.front() = std::numeric_limits<double>::infinity();
    rejected = false;
    try {
        static_cast<void>(learner.predict_intervention(nonfinite_context, 1, 0.2, 3));
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "non-finite intervention context was accepted");
    auto invalid_counterfactual = dataset.counterfactual_cases.front().intervention;
    invalid_counterfactual.mode = EventMode::Observed;
    rejected = false;
    try {
        static_cast<void>(learner.predict_counterfactual(dataset.counterfactual_cases.front().factual_values,
                                                         invalid_counterfactual, 3));
    } catch (const CausalGraphError&) {
        rejected = true;
    }
    require(rejected, "observational mode was accepted as a counterfactual query");
}

void test_observation_control() {
    const auto dataset = SyntheticCausalGenerator::generate(SyntheticCausalConfig{4, 128, 32, 101, true});
    const auto candidates = candidate_parents(dataset.evaluator_truth.variable_count);
    CausalEventLearner intervention_aware(dataset.evaluator_truth.variable_count);
    intervention_aware.fit(dataset.training_samples, candidates, true);
    CausalEventLearner observation_only(dataset.evaluator_truth.variable_count);
    observation_only.fit(dataset.training_samples, candidates, false);
    double aware_error = 0.0;
    double observation_error = 0.0;
    for (const auto& item : dataset.intervention_cases) {
        const auto aware = intervention_aware.predict_intervention(item.context_values, item.intervention.variable,
                                                                   item.intervention.value, item.target);
        const auto control = observation_only.predict_intervention(item.context_values, item.intervention.variable,
                                                                   item.intervention.value, item.target);
        aware_error += (aware.value - item.outcome) * (aware.value - item.outcome);
        observation_error += (control.value - item.outcome) * (control.value - item.outcome);
    }
    aware_error /= static_cast<double>(dataset.intervention_cases.size());
    observation_error /= static_cast<double>(dataset.intervention_cases.size());
    require(aware_error < observation_error * 0.60, "observation-only control was not worse on interventions");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"schema_graph_and_queries", test_schema_graph_and_queries},
        {"encoder_and_graph_conditioned_core", test_encoder_and_graph_conditioned_core},
        {"causal_learning_and_abstention", test_causal_learning_and_abstention},
        {"observation_control", test_observation_control},
        {"strict_metadata_and_learner_failure_paths", test_strict_metadata_and_learner_failure_paths},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
            ++passed;
        } catch (const std::exception& error) {
            std::cout << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << "SUMMARY " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
