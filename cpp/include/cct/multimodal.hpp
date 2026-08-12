#pragma once

#include "cct/deliberation.hpp"
#include "cct/memory.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cct {

enum class Modality : std::uint8_t { Text = 0, Code = 1, Audio = 2, Vision = 3, Sensor = 4, Action = 5, Tool = 6 };
enum class TransferMode : std::uint8_t { Frozen = 0, Partial = 1, Full = 2 };
enum class ActionKind : std::uint8_t { NoOp = 0, Up = 1, Down = 2, Left = 3, Right = 4, Collect = 5 };
enum class PolicyDecision : std::uint8_t { Allow = 0, Deny = 1 };

struct TimeInterval {
    std::int64_t start_tick = 0;
    std::int64_t end_tick = 0;
};

struct SpatialFrame {
    std::string name;
    std::array<double, 9> transform{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
};

struct ProvenanceRecord {
    std::string source_id;
    std::string license;
    std::string transformation_version;
    std::string content_hash;
};

struct MultimodalUncertainty {
    double confidence = 1.0;
    double timestamp_uncertainty = 0.0;
    std::string reason;
};

struct AvailabilityMask {
    std::array<bool, 7> available{false, false, false, false, false, false, false};
    bool is_available(Modality modality) const noexcept;
};

struct MultimodalEvent {
    static constexpr std::uint32_t kSchemaVersion = 1;
    std::uint64_t event_id = 0;
    Modality modality = Modality::Sensor;
    std::string payload_ref;
    std::vector<double> embedding;
    std::int64_t timestamp = 0;
    bool has_interval = false;
    TimeInterval interval;
    bool has_spatial_frame = false;
    SpatialFrame spatial_frame;
    std::vector<std::uint64_t> causal_parents;
    ProvenanceRecord provenance;
    MultimodalUncertainty uncertainty;
    AvailabilityMask mask;
    std::uint32_t schema_version = kSchemaVersion;

    std::string serialize() const;
    static MultimodalEvent deserialize(const std::string& text);
};

struct AlignmentResult {
    bool aligned = false;
    std::int64_t estimated_offset = 0;
    double error = 0.0;
    bool missing_explicit = false;
    std::string reason;
};

struct FusionResult {
    std::vector<double> fused_embedding;
    double uncertainty = 1.0;
    std::array<bool, 7> used_modalities{false, false, false, false, false, false, false};
    bool silent_substitution = false;
};

struct MemoryEvidence {
    std::uint64_t event_id = 0;
    Modality modality = Modality::Sensor;
    std::string payload_ref;
    ProvenanceRecord provenance;
    std::int64_t timestamp = 0;
    double score = 0.0;
};

struct Action {
    ActionKind kind = ActionKind::NoOp;
    int argument = 0;
};

struct ActionResult {
    bool accepted = false;
    bool terminated = false;
    double reward = 0.0;
    std::string observation;
    std::string error;
};

struct EpisodeState {
    int x = 0;
    int y = 0;
    int target_x = 2;
    int target_y = 2;
    int steps = 0;
    bool terminated = false;
};

struct EnvironmentConfig {
    int width = 3;
    int height = 3;
    int max_steps = 16;
};

class MultimodalEventStore {
public:
    void write(const MultimodalEvent& event);
    std::vector<MemoryEvidence> query(Modality modality, const std::string& payload_query, std::size_t limit) const;
    const MultimodalEvent& get(std::uint64_t event_id) const;
    std::size_t size() const noexcept;
    std::string serialize() const;
    static MultimodalEventStore deserialize(const std::string& text);

private:
    std::vector<MultimodalEvent> events_;
};

class ModalityAdapter {
public:
    static MultimodalEvent text(std::uint64_t event_id, const std::string& payload, std::int64_t timestamp,
                                const ProvenanceRecord& provenance);
    static MultimodalEvent code(std::uint64_t event_id, const std::string& payload, std::int64_t timestamp,
                                const ProvenanceRecord& provenance);
    static MultimodalEvent audio(std::uint64_t event_id, const std::vector<double>& window, std::int64_t timestamp,
                                 const ProvenanceRecord& provenance);
    static MultimodalEvent vision(std::uint64_t event_id, const std::vector<double>& patch, std::int64_t timestamp,
                                  const SpatialFrame& frame, const ProvenanceRecord& provenance);
    static MultimodalEvent sensor(std::uint64_t event_id, const std::vector<double>& values, std::int64_t timestamp,
                                  const ProvenanceRecord& provenance);
    static MultimodalEvent action(std::uint64_t event_id, ActionKind action_kind, std::int64_t timestamp,
                                  const ProvenanceRecord& provenance);
    static MultimodalEvent tool(std::uint64_t event_id, const std::string& payload, std::int64_t timestamp,
                                const ProvenanceRecord& provenance);
};

class TemporalAligner {
public:
    static AlignmentResult align(const std::vector<MultimodalEvent>& events, std::int64_t expected_offset,
                                  std::int64_t tolerance);
};

class SpatialAligner {
public:
    static double round_trip_error(const SpatialFrame& frame);
    static bool invertible(const SpatialFrame& frame);
};

class MaskAwareFusion {
public:
    static FusionResult fuse(const std::vector<MultimodalEvent>& events);
};

class DeterministicGridEnvironment {
public:
    explicit DeterministicGridEnvironment(EnvironmentConfig config = {});
    std::string reset(std::uint64_t seed);
    ActionResult step(const Action& action);
    std::string replay(const std::vector<Action>& actions, std::uint64_t seed);
    const EpisodeState& state() const noexcept;

private:
    EnvironmentConfig config_;
    EpisodeState state_;
    std::uint64_t seed_ = 0;
};

class ActionPolicy {
public:
    static PolicyDecision validate(const Action& action, const EnvironmentConfig& config);
    static Action safe_noop() noexcept;
};

struct TransferReport {
    TransferMode mode = TransferMode::Frozen;
    std::size_t parameter_updates = 0;
    double heldout_score = 0.0;
    double baseline_score = 0.0;
};

struct MultimodalTraceRecord {
    std::string kind;
    std::uint64_t event_id = 0;
    Modality modality = Modality::Sensor;
    std::string detail;
    bool policy_blocked = false;
};

class MultimodalAuditLog {
public:
    void append(const MultimodalTraceRecord& record);
    const std::vector<MultimodalTraceRecord>& records() const noexcept;
    std::string serialize() const;
    static MultimodalAuditLog deserialize(const std::string& text);

private:
    std::vector<MultimodalTraceRecord> records_;
};

}  // namespace cct
