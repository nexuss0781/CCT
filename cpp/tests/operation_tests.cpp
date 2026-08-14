#include "cct/operation.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

OperationFieldSchema string_field(const std::string& name, const std::string& description, const bool required, const std::size_t maximum_bytes,
                                  std::vector<std::string> enum_values = {}) {
    return {name, description, OperationValueKind::String, required, maximum_bytes, false, {}, 0, 0, 0.0, 0.0, std::move(enum_values)};
}

OperationRegistry make_registry() {
    OperationRegistry registry;
    OperationSchema summarize;
    summarize.operation_id = "document.summarize";
    summarize.description = "Create a bounded summary without external effects.";
    summarize.authorization = OperationAuthorizationClass::TenantMember;
    summarize.fields = {
        string_field("document_id", "Stable document identifier.", true, 64U),
        {"max_sentences", "Maximum summary sentence count.", OperationValueKind::Integer, true, 32U, false, {}, 1, 8, 0.0, 0.0, {}},
        {"style", "Summary style.", OperationValueKind::String, false, 32U, true, OperationValue(std::string("concise")), 0, 0, 0.0, 0.0, {"concise", "detailed"}}};
    registry.register_schema(summarize);

    OperationSchema lookup;
    lookup.operation_id = "knowledge.lookup";
    lookup.description = "Read governed evidence for a declared query.";
    lookup.authorization = OperationAuthorizationClass::Reviewer;
    lookup.requires_evidence = true;
    lookup.fields = {string_field("query", "Bounded lookup query.", true, 256U),
                     {"top_k", "Maximum evidence records.", OperationValueKind::Integer, true, 32U, false, {}, 1, 5, 0.0, 0.0, {}}};
    registry.register_schema(lookup);

    OperationSchema draft;
    draft.operation_id = "workflow.draft";
    draft.description = "Draft a reviewable workflow without executing it.";
    draft.authorization = OperationAuthorizationClass::Admin;
    draft.fields = {string_field("workflow_name", "Human-readable workflow name.", true, 128U),
                    {"approval_required", "Explicit human approval marker.", OperationValueKind::Boolean, true, 8U, false, {}, 0, 0, 0.0, 0.0, {}}};
    registry.register_schema(draft);
    return registry;
}

OperationCall make_call(const OperationRegistry& registry, const std::string& operation_id, std::vector<OperationArgument> arguments) {
    OperationCall call;
    call.request_id = "request-" + operation_id;
    call.tenant_id = "tenant-a";
    call.user_id = "user-a";
    call.role = "member";
    call.operation_id = operation_id;
    call.operation_schema_hash = registry.schema(operation_id).identity_hash();
    call.arguments = std::move(arguments);
    return call;
}

OperationTeacher make_teacher() {
    const auto registry = make_registry();
    OperationManifest manifest;
    auto summarize = make_call(registry, "document.summarize", {{"document_id", OperationValue("doc-42")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    OperationDemonstration first{"demo-summarize-train", "document.summarize", "governed-documents", "doc-42:0-128", "train", false, summarize,
                                OperationDecision::Accepted, OperationErrorClass::None, "validated operation document.summarize",
                                "document.summarize (cct-operation-v1)", "", "source-hash-42", ""};
    manifest.demonstrations.push_back(first);
    auto invalid = make_call(registry, "document.summarize", {{"document_id", OperationValue("doc-43")}, {"max_sentences", OperationValue(std::int64_t(0))}});
    OperationDemonstration second{"demo-summarize-eval", "document.summarize", "governed-documents", "doc-43:0-128", "evaluation", true, invalid,
                                 OperationDecision::Rejected, OperationErrorClass::BoundsViolation, "", "", "Use max_sentences in the inclusive range 1..8.", "source-hash-43", ""};
    manifest.demonstrations.push_back(second);
    manifest.finalize();
    OperationCheckpointIdentity identity{"base-model-hash", "tokenizer-hash", registry.identity_hash(), manifest.manifest_hash, "operation-training-config-v1", ""};
    identity.finalize();
    for (auto& demonstration : manifest.demonstrations) {
        demonstration.call.operation_manifest_hash = manifest.manifest_hash;
        demonstration.call.checkpoint_identity_hash = identity.identity_hash;
    }
    return OperationTeacher(registry, manifest, identity);
}

OperationAuthContext member_auth() {
    return {true, "tenant-a", "user-a", {"member"}, {"document.summarize", "knowledge.lookup"}};
}

void test_registry_identity_and_explanation() {
    const auto registry = make_registry();
    require(registry.schemas().size() == 3U && !registry.identity_hash().empty(), "operation registry did not register three schemas");
    require(registry.explain("document.summarize").find("max_sentences required") != std::string::npos, "operation explanation omitted required field");
    const auto restored = OperationRegistry::deserialize(registry.serialize());
    require(restored.identity_hash() == registry.identity_hash(), "operation registry round trip changed identity");
    bool duplicate_rejected = false;
    try { auto copy = registry.schema("document.summarize"); static_cast<void>(copy); auto mutable_registry = OperationRegistry::deserialize(registry.serialize()); mutable_registry.register_schema(copy); }
    catch (const std::exception&) { duplicate_rejected = true; }
    require(duplicate_rejected, "duplicate operation schema was accepted");
}

void test_valid_normalization_and_round_trip() {
    auto teacher = make_teacher();
    auto call = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-42")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    call.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    call.checkpoint_identity_hash = teacher.identity().identity_hash;
    const auto response = teacher.respond(call, member_auth());
    require(response.decision == OperationDecision::Accepted && response.error_class == OperationErrorClass::None &&
                response.normalized_arguments.size() == 3U && response.normalized_arguments.back().name == "style" &&
                response.normalized_arguments.back().value.canonical() == "concise" && !response.side_effect_performed,
            "valid operation call was not accepted with deterministic default normalization");
    const auto restored_call = OperationCall::deserialize(response.serialized_call);
    require(restored_call.arguments.size() == 3U && restored_call.serialize() == response.serialized_call, "normalized operation call did not round trip");
    const auto restored_teacher = OperationTeacher::deserialize(teacher.serialize());
    require(restored_teacher.explain("document.summarize") == teacher.explain("document.summarize"), "operation teacher serialization changed explanation");
}

void test_error_classes_and_authorization() {
    const auto teacher = make_teacher();
    const auto auth = member_auth();
    auto missing = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-42")}});
    missing.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    missing.checkpoint_identity_hash = teacher.identity().identity_hash;
    require(teacher.respond(missing, auth).error_class == OperationErrorClass::RequiredFieldMissing, "missing required field was not rejected");

    auto wrong_type = missing;
    wrong_type.arguments.push_back({"max_sentences", OperationValue("three")});
    require(teacher.respond(wrong_type, auth).error_class == OperationErrorClass::TypeMismatch, "wrong operation field type was not rejected");

    auto unknown_field = missing;
    unknown_field.arguments.push_back({"unexpected", OperationValue("value")});
    require(teacher.respond(unknown_field, auth).error_class == OperationErrorClass::UnknownField, "unknown operation field was not rejected");

    auto out_of_bounds = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-42")}, {"max_sentences", OperationValue(std::int64_t(99))}});
    out_of_bounds.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    out_of_bounds.checkpoint_identity_hash = teacher.identity().identity_hash;
    require(teacher.respond(out_of_bounds, auth).error_class == OperationErrorClass::BoundsViolation, "out-of-bounds operation value was not rejected");

    auto unauthorized = make_call(teacher.registry(), "workflow.draft", {{"workflow_name", OperationValue("release")}, {"approval_required", OperationValue(true)}});
    unauthorized.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    unauthorized.checkpoint_identity_hash = teacher.identity().identity_hash;
    require(teacher.respond(unauthorized, auth).error_class == OperationErrorClass::AuthorizationDenied, "unauthorized operation was accepted");

    auto ambiguous = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-42")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    ambiguous.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    ambiguous.checkpoint_identity_hash = teacher.identity().identity_hash;
    ambiguous.ambiguous = true;
    require(teacher.respond(ambiguous, auth).decision == OperationDecision::Abstained, "ambiguous operation did not abstain");

    auto unknown = ambiguous;
    unknown.operation_id = "unknown.operation";
    require(teacher.respond(unknown, auth).error_class == OperationErrorClass::UnknownOperation, "unknown operation was accepted");
}

void test_evidence_identity_and_side_effect_boundaries() {
    const auto teacher = make_teacher();
    auto lookup = make_call(teacher.registry(), "knowledge.lookup", {{"query", OperationValue("contract retention")}, {"top_k", OperationValue(std::int64_t(2))}});
    lookup.role = "reviewer";
    lookup.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    lookup.checkpoint_identity_hash = teacher.identity().identity_hash;
    auto auth = member_auth();
    auth.roles = {"reviewer"};
    auth.allowed_operations = {"knowledge.lookup"};
    require(teacher.respond(lookup, auth).error_class == OperationErrorClass::EvidenceMissing, "required operation evidence was not enforced");
    lookup.evidence = {{"source-1", "span-4", 0.98}};
    require(teacher.respond(lookup, auth).decision == OperationDecision::Accepted, "supported evidence operation was not accepted");

    auto side_effect = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-42")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    side_effect.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    side_effect.checkpoint_identity_hash = teacher.identity().identity_hash;
    side_effect.requests_external_action = true;
    const auto blocked = teacher.respond(side_effect, member_auth());
    require(blocked.error_class == OperationErrorClass::SideEffectDenied && !blocked.side_effect_performed, "external side effect request was not blocked");
}

void test_lineage_and_corruption_fail_closed() {
    const auto teacher = make_teacher();
    auto call = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-42")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    call.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    call.checkpoint_identity_hash = teacher.identity().identity_hash;
    auto changed = call;
    changed.operation_schema_hash = "foreign-schema";
    require(teacher.respond(changed, member_auth()).error_class == OperationErrorClass::IdentityMismatch, "schema identity mismatch was accepted");
    auto malformed = call.serialize() + "trailing";
    bool trailing_rejected = false;
    try { static_cast<void>(OperationCall::deserialize(malformed)); } catch (const std::exception&) { trailing_rejected = true; }
    require(trailing_rejected, "operation call trailing bytes were accepted");
    auto old_version = call;
    old_version.schema_version = "cct-operation-call-v0";
    require(teacher.respond(old_version, member_auth()).error_class == OperationErrorClass::SchemaVersionMismatch, "operation schema version mismatch was not classified");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"registry_identity_and_explanation", test_registry_identity_and_explanation},
        {"valid_normalization_and_round_trip", test_valid_normalization_and_round_trip},
        {"error_classes_and_authorization", test_error_classes_and_authorization},
        {"evidence_identity_and_side_effect_boundaries", test_evidence_identity_and_side_effect_boundaries},
        {"lineage_and_corruption_fail_closed", test_lineage_and_corruption_fail_closed}};
    std::size_t passed = 0U;
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
