#include "cct/multimodal.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ProvenanceRecord provenance(const std::string& id) {
    return {id, "MIT", "fixture-v1", "hash-" + id};
}

void test_event_schema_and_adapters() {
    const auto p = provenance("fixture");
    const SpatialFrame frame{"camera", {2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0}};
    const std::vector<MultimodalEvent> events{
        ModalityAdapter::text(1, "red square", 10, p),
        ModalityAdapter::code(2, "return 1", 11, p),
        ModalityAdapter::audio(3, {0.1, 0.2}, 12, p),
        ModalityAdapter::vision(4, {1.0, 0.0}, 13, frame, p),
        ModalityAdapter::sensor(5, {0.3, 0.4}, 14, p),
        ModalityAdapter::action(6, ActionKind::Right, 15, p),
        ModalityAdapter::tool(7, "observation", 16, p),
    };
    require(events.size() == 7, "not all modality adapters emitted events");
    for (const auto& event : events) {
        require(event.schema_version == MultimodalEvent::kSchemaVersion && !event.payload_ref.empty() &&
                    !event.provenance.source_id.empty() && !event.provenance.transformation_version.empty() &&
                    event.mask.is_available(event.modality),
                "adapter dropped event identity, provenance, or mask");
    }
    const auto restored = MultimodalEvent::deserialize(events[3].serialize());
    require(restored.event_id == events[3].event_id && restored.spatial_frame.name == "camera" &&
                restored.provenance.content_hash == events[3].provenance.content_hash,
            "multimodal event schema round-trip failed");
}

void test_store_and_cross_modal_retrieval() {
    const auto p = provenance("store");
    MultimodalEventStore store;
    store.write(ModalityAdapter::text(10, "red square", 1, p));
    store.write(ModalityAdapter::vision(11, {1.0, 0.0}, 2, SpatialFrame{"camera", {}}, p));
    store.write(ModalityAdapter::sensor(12, {1.0, 2.0}, 3, p));
    const auto restored = MultimodalEventStore::deserialize(store.serialize());
    require(restored.size() == 3, "typed event store replay changed size");
    const auto hits = restored.query(Modality::Text, "red square", 1);
    require(hits.size() == 1 && hits.front().event_id == 10 && hits.front().modality == Modality::Text &&
                hits.front().provenance.source_id == "store",
            "cross-modal memory omitted modality or provenance");
}

void test_temporal_and_spatial_alignment() {
    const auto p = provenance("alignment");
    auto first = ModalityAdapter::text(20, "a", 100, p);
    auto second = ModalityAdapter::audio(21, {0.2}, 104, p);
    const auto aligned = TemporalAligner::align({first, second}, 4, 1);
    require(aligned.aligned && aligned.estimated_offset == 4 && aligned.error == 0.0,
            "known temporal offset was not recovered");
    second.mask.available[static_cast<std::size_t>(second.modality)] = false;
    const auto missing = TemporalAligner::align({first, second}, 4, 1);
    require(missing.missing_explicit, "missing timestamp modality was not explicit");

    const SpatialFrame invertible{"world", {2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0}};
    require(SpatialAligner::invertible(invertible) && SpatialAligner::round_trip_error(invertible) <= 1e-12,
            "invertible spatial transform failed round-trip");
    const SpatialFrame singular{"bad", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    require(!SpatialAligner::invertible(singular), "singular spatial transform was accepted");
}

void test_masked_fusion_and_uncertainty() {
    const auto p = provenance("fusion");
    auto text = ModalityAdapter::text(30, "red", 1, p);
    auto sensor = ModalityAdapter::sensor(31, {100.0, 100.0, 100.0, 100.0}, 1, p);
    sensor.mask.available[static_cast<std::size_t>(Modality::Sensor)] = false;
    const auto fusion = MaskAwareFusion::fuse({text, sensor});
    require(fusion.used_modalities[static_cast<std::size_t>(Modality::Text)] &&
                !fusion.used_modalities[static_cast<std::size_t>(Modality::Sensor)] && !fusion.silent_substitution &&
                fusion.uncertainty > 0.0,
            "masked fusion silently substituted unavailable modality");
}

void test_environment_policy_and_replay() {
    DeterministicGridEnvironment environment({3, 3, 16});
    const std::vector<Action> path{{ActionKind::Right, 0}, {ActionKind::Right, 0}, {ActionKind::Up, 0},
                                   {ActionKind::Up, 0}, {ActionKind::Collect, 0}};
    const auto first_replay = environment.replay(path, 99);
    const auto second_replay = environment.replay(path, 99);
    require(first_replay == second_replay, "deterministic environment replay diverged");
    environment.reset(99);
    for (const auto& action : path) static_cast<void>(environment.step(action));
    require(environment.state().terminated, "valid deterministic episode did not terminate");
    const auto invalid = environment.step({static_cast<ActionKind>(99), 0});
    require(!invalid.accepted && invalid.error == "policy_denied", "invalid action was not rejected");
    require(ActionPolicy::validate({ActionKind::NoOp, 0}, {3, 3, 16}) == PolicyDecision::Allow,
            "safe no-op was not allowed");
}

void test_bounded_deserialization() {
    bool rejected = false;
    try {
        static_cast<void>(MultimodalEventStore::deserialize("CCT_MM_STORE_V1\n18446744073709551615\n"));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "multimodal store accepted an unbounded event count");
    const std::string event_prefix =
        "CCT_MM_EVENT_V1\n1 0 0 0 0 0 0 \"frame\"\n1 0 0 0 1 0 0 0 1\n\"payload\"\n";
    rejected = false;
    try {
        static_cast<void>(MultimodalEvent::deserialize(event_prefix + "18446744073709551615\n"));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "multimodal event accepted an unbounded embedding count");
    rejected = false;
    try {
        static_cast<void>(MultimodalAuditLog::deserialize("CCT_MM_AUDIT_V1\n18446744073709551615\n"));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "multimodal audit accepted an unbounded record count");
}

void test_audit_and_transfer_metadata() {
    MultimodalAuditLog log;
    log.append({"input", 40, Modality::Vision, "vision-patch-v1", false});
    log.append({"memory_read", 40, Modality::Vision, "typed-evidence", false});
    log.append({"action", 0, Modality::Action, "policy-denied", true});
    const auto restored = MultimodalAuditLog::deserialize(log.serialize());
    require(restored.records().size() == 3 && restored.records().back().policy_blocked,
            "multimodal audit log lost policy incident");
    const TransferReport frozen{TransferMode::Frozen, 0, 0.9, 0.7};
    const TransferReport partial{TransferMode::Partial, 3, 0.95, 0.7};
    require(frozen.parameter_updates == 0 && partial.parameter_updates == 3 && partial.heldout_score > partial.baseline_score,
            "transfer mode metadata was not explicit");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"event_schema_and_adapters", test_event_schema_and_adapters},
        {"store_and_cross_modal_retrieval", test_store_and_cross_modal_retrieval},
        {"temporal_and_spatial_alignment", test_temporal_and_spatial_alignment},
        {"masked_fusion_and_uncertainty", test_masked_fusion_and_uncertainty},
        {"environment_policy_and_replay", test_environment_policy_and_replay},
        {"bounded_deserialization", test_bounded_deserialization},
        {"audit_and_transfer_metadata", test_audit_and_transfer_metadata},
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
