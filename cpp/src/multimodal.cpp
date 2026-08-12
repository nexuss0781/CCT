#include "cct/multimodal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::size_t modality_index(Modality modality) {
    return static_cast<std::size_t>(modality);
}

MultimodalEvent base_event(std::uint64_t event_id, Modality modality, const std::string& payload_ref,
                           const std::vector<double>& embedding, std::int64_t timestamp,
                           const ProvenanceRecord& provenance) {
    MultimodalEvent event;
    event.event_id = event_id;
    event.modality = modality;
    event.payload_ref = payload_ref;
    event.embedding = embedding;
    event.timestamp = timestamp;
    event.provenance = provenance;
    event.uncertainty = {0.98, 0.0, "adapter"};
    event.mask.available[modality_index(modality)] = true;
    event.schema_version = MultimodalEvent::kSchemaVersion;
    return event;
}

std::vector<double> scalar_features(const std::string& payload, std::size_t width = 4) {
    std::vector<double> values(width, 0.0);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        values[index % width] += static_cast<double>(static_cast<unsigned char>(payload[index])) / 255.0;
    }
    if (!payload.empty()) {
        for (auto& value : values) value /= static_cast<double>(payload.size());
    }
    return values;
}

std::array<double, 9> inverse3(const std::array<double, 9>& matrix) {
    const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                               matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                               matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    require(std::abs(determinant) > 1e-15, "spatial transform is singular");
    const double scale = 1.0 / determinant;
    return {scale * (matrix[4] * matrix[8] - matrix[5] * matrix[7]),
            scale * (matrix[2] * matrix[7] - matrix[1] * matrix[8]),
            scale * (matrix[1] * matrix[5] - matrix[2] * matrix[4]),
            scale * (matrix[5] * matrix[6] - matrix[3] * matrix[8]),
            scale * (matrix[0] * matrix[8] - matrix[2] * matrix[6]),
            scale * (matrix[3] * matrix[2] - matrix[0] * matrix[5]),
            scale * (matrix[3] * matrix[7] - matrix[4] * matrix[6]),
            scale * (matrix[1] * matrix[6] - matrix[0] * matrix[7]),
            scale * (matrix[0] * matrix[4] - matrix[1] * matrix[3])};
}

}  // namespace

bool AvailabilityMask::is_available(Modality modality) const noexcept {
    return available[modality_index(modality)];
}

std::string MultimodalEvent::serialize() const {
    std::ostringstream output;
    output << "CCT_MM_EVENT_V1\n";
    output << event_id << ' ' << static_cast<unsigned int>(modality) << ' ' << timestamp << ' ' << has_interval << ' '
           << interval.start_tick << ' ' << interval.end_tick << ' ' << has_spatial_frame << ' '
           << std::quoted(spatial_frame.name) << '\n';
    for (const auto value : spatial_frame.transform) output << std::setprecision(17) << value << ' ';
    output << '\n' << std::quoted(payload_ref) << '\n';
    output << embedding.size();
    for (const auto value : embedding) output << ' ' << std::setprecision(17) << value;
    output << '\n' << causal_parents.size();
    for (const auto value : causal_parents) output << ' ' << value;
    output << '\n' << std::quoted(provenance.source_id) << ' ' << std::quoted(provenance.license) << ' '
           << std::quoted(provenance.transformation_version) << ' ' << std::quoted(provenance.content_hash) << '\n';
    output << std::setprecision(17) << uncertainty.confidence << ' ' << uncertainty.timestamp_uncertainty << ' '
           << std::quoted(uncertainty.reason) << '\n';
    for (const auto value : mask.available) output << value << ' ';
    output << '\n' << schema_version << '\n';
    return output.str();
}

MultimodalEvent MultimodalEvent::deserialize(const std::string& text) {
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_MM_EVENT_V1", "invalid multimodal event header");
    MultimodalEvent event;
    unsigned int modality = 0;
    input >> event.event_id >> modality >> event.timestamp >> event.has_interval >> event.interval.start_tick >>
        event.interval.end_tick >> event.has_spatial_frame >> std::quoted(event.spatial_frame.name);
    event.modality = static_cast<Modality>(modality);
    for (auto& value : event.spatial_frame.transform) input >> value;
    input >> std::quoted(event.payload_ref);
    std::size_t count = 0;
    input >> count;
    event.embedding.resize(count);
    for (auto& value : event.embedding) input >> value;
    input >> count;
    event.causal_parents.resize(count);
    for (auto& value : event.causal_parents) input >> value;
    input >> std::quoted(event.provenance.source_id) >> std::quoted(event.provenance.license) >>
        std::quoted(event.provenance.transformation_version) >> std::quoted(event.provenance.content_hash);
    input >> event.uncertainty.confidence >> event.uncertainty.timestamp_uncertainty >> std::quoted(event.uncertainty.reason);
    for (auto& value : event.mask.available) input >> value;
    input >> event.schema_version;
    require(static_cast<bool>(input) && event.schema_version == kSchemaVersion, "invalid multimodal event serialization");
    return event;
}

void MultimodalEventStore::write(const MultimodalEvent& event) {
    require(event.event_id != 0 && !event.payload_ref.empty() && event.schema_version == MultimodalEvent::kSchemaVersion,
            "invalid multimodal event");
    const auto found = std::find_if(events_.begin(), events_.end(), [&](const auto& item) { return item.event_id == event.event_id; });
    require(found == events_.end(), "duplicate multimodal event id");
    events_.push_back(event);
}

std::vector<MemoryEvidence> MultimodalEventStore::query(Modality modality, const std::string& payload_query, std::size_t limit) const {
    std::vector<MemoryEvidence> results;
    for (const auto& event : events_) {
        if (event.modality != modality || event.payload_ref.find(payload_query) == std::string::npos) continue;
        const double score = event.payload_ref == payload_query ? 1.0 : 0.8;
        results.push_back({event.event_id, event.modality, event.payload_ref, event.provenance, event.timestamp, score});
    }
    std::sort(results.begin(), results.end(), [](const auto& left, const auto& right) { return left.score > right.score; });
    if (results.size() > limit) results.resize(limit);
    return results;
}

const MultimodalEvent& MultimodalEventStore::get(std::uint64_t event_id) const {
    const auto found = std::find_if(events_.begin(), events_.end(), [&](const auto& item) { return item.event_id == event_id; });
    require(found != events_.end(), "multimodal event id not found");
    return *found;
}

std::size_t MultimodalEventStore::size() const noexcept { return events_.size(); }

std::string MultimodalEventStore::serialize() const {
    std::ostringstream output;
    output << "CCT_MM_STORE_V1\n" << events_.size() << '\n';
    for (const auto& event : events_) output << std::quoted(event.serialize()) << '\n';
    return output.str();
}

MultimodalEventStore MultimodalEventStore::deserialize(const std::string& text) {
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_MM_STORE_V1", "invalid multimodal store header");
    std::size_t count = 0;
    input >> count;
    MultimodalEventStore store;
    for (std::size_t index = 0; index < count; ++index) {
        std::string serialized;
        input >> std::quoted(serialized);
        store.write(MultimodalEvent::deserialize(serialized));
    }
    return store;
}

MultimodalEvent ModalityAdapter::text(std::uint64_t event_id, const std::string& payload, std::int64_t timestamp,
                                      const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Text, "text:" + payload, scalar_features(payload), timestamp, provenance);
    event.provenance.transformation_version = "text-token-v1";
    return event;
}

MultimodalEvent ModalityAdapter::code(std::uint64_t event_id, const std::string& payload, std::int64_t timestamp,
                                      const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Code, "code:" + payload, scalar_features(payload), timestamp, provenance);
    event.provenance.transformation_version = "code-ast-v1";
    return event;
}

MultimodalEvent ModalityAdapter::audio(std::uint64_t event_id, const std::vector<double>& window, std::int64_t timestamp,
                                       const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Audio, "audio:" + std::to_string(event_id), window, timestamp, provenance);
    event.provenance.transformation_version = "audio-window-v1";
    return event;
}

MultimodalEvent ModalityAdapter::vision(std::uint64_t event_id, const std::vector<double>& patch, std::int64_t timestamp,
                                        const SpatialFrame& frame, const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Vision, "vision:" + std::to_string(event_id), patch, timestamp, provenance);
    event.has_spatial_frame = true;
    event.spatial_frame = frame;
    event.provenance.transformation_version = "vision-patch-v1";
    return event;
}

MultimodalEvent ModalityAdapter::sensor(std::uint64_t event_id, const std::vector<double>& values, std::int64_t timestamp,
                                        const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Sensor, "sensor:" + std::to_string(event_id), values, timestamp, provenance);
    event.provenance.transformation_version = "sensor-vector-v1";
    return event;
}

MultimodalEvent ModalityAdapter::action(std::uint64_t event_id, ActionKind action_kind, std::int64_t timestamp,
                                        const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Action, "action:" + std::to_string(static_cast<unsigned int>(action_kind)),
                            {static_cast<double>(static_cast<unsigned int>(action_kind))}, timestamp, provenance);
    event.provenance.transformation_version = "action-schema-v1";
    return event;
}

MultimodalEvent ModalityAdapter::tool(std::uint64_t event_id, const std::string& payload, std::int64_t timestamp,
                                      const ProvenanceRecord& provenance) {
    auto event = base_event(event_id, Modality::Tool, "tool:" + payload, scalar_features(payload), timestamp, provenance);
    event.provenance.transformation_version = "tool-observation-v1";
    return event;
}

AlignmentResult TemporalAligner::align(const std::vector<MultimodalEvent>& events, std::int64_t expected_offset,
                                       std::int64_t tolerance) {
    if (events.size() < 2) return {false, 0, std::numeric_limits<double>::infinity(), true, "insufficient_streams"};
    bool any_missing = false;
    for (const auto& event : events) {
        if (!event.mask.is_available(event.modality)) any_missing = true;
    }
    const auto estimated = events[1].timestamp - events[0].timestamp;
    const auto error = std::abs(static_cast<double>(estimated - expected_offset));
    return {error <= static_cast<double>(tolerance), estimated, error, any_missing, error <= tolerance ? "aligned" : "offset_out_of_tolerance"};
}

double SpatialAligner::round_trip_error(const SpatialFrame& frame) {
    const auto inverse = inverse3(frame.transform);
    double error = 0.0;
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < 3; ++inner) value += frame.transform[row * 3 + inner] * inverse[inner * 3 + column];
            const double target = row == column ? 1.0 : 0.0;
            error = std::max(error, std::abs(value - target));
        }
    }
    return error;
}

bool SpatialAligner::invertible(const SpatialFrame& frame) {
    try {
        static_cast<void>(inverse3(frame.transform));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

FusionResult MaskAwareFusion::fuse(const std::vector<MultimodalEvent>& events) {
    std::size_t width = 0;
    std::array<bool, 7> used{};
    for (const auto& event : events) {
        if (!event.mask.is_available(event.modality)) continue;
        width = std::max(width, event.embedding.size());
        used[modality_index(event.modality)] = true;
    }
    FusionResult result;
    result.fused_embedding.assign(width, 0.0);
    if (width == 0) {
        result.uncertainty = 1.0;
        return result;
    }
    std::size_t count = 0;
    for (const auto& event : events) {
        if (!event.mask.is_available(event.modality)) continue;
        for (std::size_t index = 0; index < event.embedding.size(); ++index) result.fused_embedding[index] += event.embedding[index];
        ++count;
    }
    for (auto& value : result.fused_embedding) value /= static_cast<double>(count);
    result.used_modalities = used;
    result.uncertainty = 1.0 / static_cast<double>(count);
    result.silent_substitution = false;
    return result;
}

void MultimodalAuditLog::append(const MultimodalTraceRecord& record) { records_.push_back(record); }

const std::vector<MultimodalTraceRecord>& MultimodalAuditLog::records() const noexcept { return records_; }

std::string MultimodalAuditLog::serialize() const {
    std::ostringstream output;
    output << "CCT_MM_AUDIT_V1\n" << records_.size() << '\n';
    for (const auto& record : records_) output << std::quoted(record.kind) << ' ' << record.event_id << ' '
                                                << static_cast<unsigned int>(record.modality) << ' '
                                                << record.policy_blocked << ' ' << std::quoted(record.detail) << '\n';
    return output.str();
}

MultimodalAuditLog MultimodalAuditLog::deserialize(const std::string& text) {
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_MM_AUDIT_V1", "invalid multimodal audit header");
    std::size_t count = 0;
    input >> count;
    MultimodalAuditLog log;
    for (std::size_t index = 0; index < count; ++index) {
        MultimodalTraceRecord record;
        unsigned int modality = 0;
        input >> std::quoted(record.kind) >> record.event_id >> modality >> record.policy_blocked >> std::quoted(record.detail);
        record.modality = static_cast<Modality>(modality);
        log.append(record);
    }
    return log;
}

DeterministicGridEnvironment::DeterministicGridEnvironment(EnvironmentConfig config) : config_(config) {
    require(config_.width > 0 && config_.height > 0 && config_.max_steps > 0, "invalid environment config");
}

std::string DeterministicGridEnvironment::reset(std::uint64_t seed) {
    seed_ = seed;
    state_ = {};
    state_.target_x = config_.width - 1;
    state_.target_y = config_.height - 1;
    return "pos=" + std::to_string(state_.x) + "," + std::to_string(state_.y);
}

ActionResult DeterministicGridEnvironment::step(const Action& action) {
    if (ActionPolicy::validate(action, config_) == PolicyDecision::Deny) return {false, state_.terminated, 0.0, "", "policy_denied"};
    if (state_.terminated) return {false, true, 0.0, "", "episode_terminated"};
    ++state_.steps;
    int next_x = state_.x;
    int next_y = state_.y;
    if (action.kind == ActionKind::Up) ++next_y;
    else if (action.kind == ActionKind::Down) --next_y;
    else if (action.kind == ActionKind::Left) --next_x;
    else if (action.kind == ActionKind::Right) ++next_x;
    if (next_x < 0 || next_x >= config_.width || next_y < 0 || next_y >= config_.height) {
        state_.terminated = state_.steps >= config_.max_steps;
        return {false, state_.terminated, -0.1, "boundary", "boundary_rejected"};
    }
    state_.x = next_x;
    state_.y = next_y;
    double reward = -0.01;
    if (action.kind == ActionKind::Collect && state_.x == state_.target_x && state_.y == state_.target_y) {
        reward = 1.0;
        state_.terminated = true;
    }
    if (state_.steps >= config_.max_steps) state_.terminated = true;
    return {true, state_.terminated, reward, "pos=" + std::to_string(state_.x) + "," + std::to_string(state_.y), {}};
}

std::string DeterministicGridEnvironment::replay(const std::vector<Action>& actions, std::uint64_t seed) {
    const auto initial = reset(seed);
    std::ostringstream output;
    output << initial;
    for (const auto& action : actions) {
        const auto result = step(action);
        output << "|" << result.observation << ":" << std::setprecision(17) << result.reward << ":" << result.accepted;
        if (result.terminated) break;
    }
    return output.str();
}

const EpisodeState& DeterministicGridEnvironment::state() const noexcept { return state_; }

PolicyDecision ActionPolicy::validate(const Action& action, const EnvironmentConfig& config) {
    const auto kind = static_cast<unsigned int>(action.kind);
    if (kind > static_cast<unsigned int>(ActionKind::Collect) || action.argument != 0 || config.width <= 0 || config.height <= 0) {
        return PolicyDecision::Deny;
    }
    return PolicyDecision::Allow;
}

Action ActionPolicy::safe_noop() noexcept { return {ActionKind::NoOp, 0}; }

}  // namespace cct
