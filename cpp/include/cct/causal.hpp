#pragma once

#include "cct/sequence.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class EventMode : std::uint8_t { Observed = 0, DoIntervention = 1, Counterfactual = 2 };
enum class ProvenanceKind : std::uint8_t { Generated = 0, Observed = 1, Intervened = 2, Retrieved = 3, Corrected = 4 };
enum class UncertaintyKind : std::uint8_t { Known = 0, Estimated = 1, Unknown = 2, Conflicting = 3 };

using EventId = std::uint64_t;

struct UncertaintyRecord {
    UncertaintyKind kind = UncertaintyKind::Known;
    double confidence = 1.0;
};

struct Intervention {
    std::size_t variable = 0;
    double value = 0.0;
    EventMode mode = EventMode::Observed;
};

struct CausalEvent {
    static constexpr std::uint32_t kSchemaVersion = 1;

    std::uint32_t schema_version = kSchemaVersion;
    EventId id = 0;
    std::vector<double> semantic_payload;
    std::vector<double> coordinates;
    std::int64_t timestamp = 0;
    std::vector<EventId> causal_parents;
    std::optional<Intervention> intervention;
    UncertaintyRecord uncertainty;
    ProvenanceKind provenance = ProvenanceKind::Generated;
    std::vector<EventId> unresolved_parent_ids;
    std::vector<EventId> provenance_links;
};

struct CausalStoreConfig {
    std::size_t payload_dim = 1;
    std::size_t coordinate_dim = 2;
    std::vector<double> coordinate_min{-1.0, -1.0};
    std::vector<double> coordinate_max{1.0, 1.0};
    bool allow_unresolved_parents = true;
};

class CausalGraphError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class CausalEventStore {
public:
    explicit CausalEventStore(CausalStoreConfig config = {});

    const CausalStoreConfig& config() const noexcept { return config_; }
    void insert(const CausalEvent& event);
    bool contains(EventId id) const noexcept;
    const CausalEvent& event(EventId id) const;
    std::size_t size() const noexcept;

    std::vector<EventId> parents_of(EventId id) const;
    std::vector<EventId> children_of(EventId id) const;
    std::vector<EventId> causal_past(EventId id) const;
    std::vector<EventId> causal_future(EventId id) const;
    std::vector<EventId> topological_order() const;
    bool has_cycle() const;

    std::string deterministic_export() const;
    std::string serialize_snapshot() const;
    static CausalEventStore deserialize_snapshot(const std::string& snapshot);
    void save_snapshot(const std::string& path) const;
    static CausalEventStore load_snapshot(const std::string& path);

private:
    CausalStoreConfig config_;
    std::vector<CausalEvent> ordered_events_;

    void validate_event(const CausalEvent& event) const;
    std::vector<EventId> existing_parent_ids(const CausalEvent& event) const;
    std::vector<EventId> sorted_ids() const;
    void validate_acyclic() const;
};

struct CausalEncodingConfig {
    std::size_t payload_dim = 1;
    std::size_t coordinate_dim = 2;
    bool include_causal_edges = true;
    bool include_intervention_marker = true;
    bool include_uncertainty = true;
    bool include_provenance = true;
    bool include_coordinates = true;
    bool prevent_future_leakage = true;
};

struct EncodedCausalSequence {
    std::vector<std::vector<double>> inputs;
    std::vector<std::uint8_t> mask;
    std::vector<EventId> event_ids;
    std::size_t excluded_future_parent_count = 0;
};

class CausalEventEncoder {
public:
    explicit CausalEventEncoder(CausalEncodingConfig config = {});

    const CausalEncodingConfig& config() const noexcept { return config_; }
    std::size_t encoded_dim() const noexcept;
    EncodedCausalSequence encode(const std::vector<CausalEvent>& events) const;

private:
    CausalEncodingConfig config_;
};

struct GraphConditionedConfig {
    SequenceConfig sequence;
    CausalEncodingConfig encoding;
};

class GraphConditionedSequence {
public:
    explicit GraphConditionedSequence(GraphConditionedConfig config);

    const GraphConditionedConfig& config() const noexcept { return config_; }
    const SelectiveSequenceCore& temporal_core() const noexcept { return core_; }
    SelectiveSequenceCore& temporal_core() noexcept { return core_; }
    EncodedCausalSequence encode(const std::vector<CausalEvent>& events) const;
    SequenceOutput forward(const std::vector<CausalEvent>& events) const;
    SequenceOutput forward_scan(const std::vector<CausalEvent>& events) const;

private:
    GraphConditionedConfig config_;
    CausalEventEncoder encoder_;
    SelectiveSequenceCore core_;
};

struct StructuralSample {
    std::vector<double> values;
    std::vector<Intervention> interventions;
};

struct StructuralModelTruth {
    std::size_t variable_count = 0;
    std::vector<double> intercepts;
    std::vector<std::vector<double>> coefficients;
    std::vector<std::vector<double>> nonlinear_coefficients;
    std::vector<std::vector<std::size_t>> parents;
};

struct CounterfactualCase {
    std::vector<double> factual_values;
    Intervention intervention;
    std::size_t target = 0;
    std::vector<double> counterfactual_values;
};

struct InterventionCase {
    std::vector<double> context_values;
    Intervention intervention;
    std::size_t target = 0;
    double outcome = 0.0;
};

struct CausalDataset {
    std::vector<CausalEvent> visible_events;
    std::vector<StructuralSample> training_samples;
    std::vector<StructuralSample> test_samples;
    std::vector<InterventionCase> intervention_cases;
    std::vector<CounterfactualCase> counterfactual_cases;
    StructuralModelTruth evaluator_truth;
    std::uint64_t dataset_fingerprint = 0;
};

struct SyntheticCausalConfig {
    std::size_t variable_count = 4;
    std::size_t training_samples = 96;
    std::size_t test_samples = 32;
    std::uint64_t seed = 101;
    bool confounded_observations = true;
};

class SyntheticCausalGenerator {
public:
    static CausalDataset generate(const SyntheticCausalConfig& config = {});
};

struct EdgePrediction {
    std::size_t parent = 0;
    std::size_t child = 0;
    double confidence = 0.0;
    bool predicted = false;
};

struct CausalPrediction {
    double value = 0.0;
    double confidence = 0.0;
    bool abstained = false;
};

class CausalEventLearner {
public:
    explicit CausalEventLearner(std::size_t variable_count);

    void fit(const std::vector<StructuralSample>& samples,
             const std::vector<std::vector<std::size_t>>& parent_hypotheses,
             bool intervention_aware = true);
    bool fitted() const noexcept { return fitted_; }
    std::size_t variable_count() const noexcept { return variable_count_; }
    const std::vector<std::vector<std::size_t>>& parent_hypotheses() const noexcept { return parents_; }
    const std::vector<std::vector<double>>& coefficients() const noexcept { return coefficients_; }
    const std::vector<std::vector<double>>& nonlinear_coefficients() const noexcept { return nonlinear_coefficients_; }
    const std::vector<double>& intercepts() const noexcept { return intercepts_; }

    std::vector<EdgePrediction> edge_predictions(double threshold = 0.05) const;
    CausalPrediction predict_intervention(const std::vector<double>& context,
                                          std::size_t variable,
                                          double value,
                                          std::size_t target) const;
    CausalPrediction predict_counterfactual(const std::vector<double>& factual,
                                            const Intervention& intervention,
                                            std::size_t target) const;
    CausalPrediction predict_observation(const std::vector<double>& context,
                                         std::size_t target) const;

    void set_graph_quality(bool incomplete, bool conflicting) noexcept;
    bool graph_incomplete() const noexcept { return graph_incomplete_; }
    bool graph_conflicting() const noexcept { return graph_conflicting_; }

private:
    std::size_t variable_count_;
    std::vector<std::vector<std::size_t>> parents_;
    std::vector<std::vector<double>> coefficients_;
    std::vector<std::vector<double>> nonlinear_coefficients_;
    std::vector<double> intercepts_;
    std::vector<double> confidences_;
    bool fitted_ = false;
    bool graph_incomplete_ = false;
    bool graph_conflicting_ = false;

    std::vector<double> solve_ridge(const std::vector<std::vector<double>>& matrix,
                                    const std::vector<double>& vector) const;
    std::vector<double> evaluate_world(const std::vector<double>& residuals,
                                        std::optional<Intervention> intervention) const;
    void validate_sample(const StructuralSample& sample) const;
};

}  // namespace cct
