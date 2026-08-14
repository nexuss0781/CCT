#include "cct/operation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

struct Check {
    std::string name;
    std::string status;
    std::string details;
    double duration_seconds = 0.0;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    require(static_cast<bool>(output), "cannot write L1-8 artifact: " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish L1-8 artifact: " + path.string());
}

std::string escape_json(const std::string& value) {
    std::string output;
    for (const char character : value) {
        if (character == '\\') output += "\\\\";
        else if (character == '"') output += "\\\"";
        else if (character == '\n') output += "\\n";
        else if (character == '\r') output += "\\r";
        else if (character == '\t') output += "\\t";
        else output.push_back(character);
    }
    return output;
}

std::string bool_json(const bool value) { return value ? "true" : "false"; }

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
    summarize.fields = {string_field("document_id", "Stable document identifier.", true, 64U),
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
    call.request_id = "gate-" + operation_id;
    call.tenant_id = "tenant-a";
    call.user_id = "user-a";
    call.role = "member";
    call.operation_id = operation_id;
    call.operation_schema_hash = registry.schema(operation_id).identity_hash();
    call.arguments = std::move(arguments);
    return call;
}

OperationAuthContext member_auth() {
    return {true, "tenant-a", "user-a", {"member"}, {"document.summarize", "knowledge.lookup"}};
}

OperationTeacher make_teacher() {
    const auto registry = make_registry();
    OperationManifest manifest;
    auto call = make_call(registry, "document.summarize", {{"document_id", OperationValue("doc-gate")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    OperationDemonstration training{"operation-demo-train", "document.summarize", "governed-operation-corpus", "doc-gate:0-256", "train", false, call,
                                  OperationDecision::Accepted, OperationErrorClass::None, "validated operation document.summarize",
                                  "document.summarize (cct-operation-v1)", "", "source-hash-gate", ""};
    manifest.demonstrations.push_back(training);
    auto evaluation_call = make_call(registry, "document.summarize", {{"document_id", OperationValue("doc-eval")}, {"max_sentences", OperationValue(std::int64_t(4))}});
    OperationDemonstration evaluation{"operation-demo-eval", "document.summarize", "governed-operation-corpus", "doc-eval:0-256", "evaluation", true, evaluation_call,
                                     OperationDecision::Accepted, OperationErrorClass::None, "validated operation document.summarize",
                                     "document.summarize (cct-operation-v1)", "", "source-hash-eval", ""};
    manifest.demonstrations.push_back(evaluation);
    manifest.finalize();
    OperationCheckpointIdentity identity{"base-checkpoint-l1-7", "tokenizer-l1-4", registry.identity_hash(), manifest.manifest_hash,
                                        "operation-training-config-v1", ""};
    identity.finalize();
    for (auto& demonstration : manifest.demonstrations) {
        demonstration.call.operation_manifest_hash = manifest.manifest_hash;
        demonstration.call.checkpoint_identity_hash = identity.identity_hash;
    }
    return OperationTeacher(registry, manifest, identity);
}

OperationCall bound_call(const OperationTeacher& teacher) {
    auto call = make_call(teacher.registry(), "document.summarize", {{"document_id", OperationValue("doc-gate")}, {"max_sentences", OperationValue(std::int64_t(3))}});
    call.operation_manifest_hash = teacher.identity().operation_manifest_hash;
    call.checkpoint_identity_hash = teacher.identity().identity_hash;
    return call;
}

std::string response_json(const OperationResponse& response) {
    return "{\"decision\":\"" + operation_decision_name(response.decision) + "\",\"error\":\"" + operation_error_name(response.error_class) +
           "\",\"error_code\":\"" + escape_json(response.error_code) + "\",\"side_effect_performed\":" + bool_json(response.side_effect_performed) +
           ",\"audit_digest\":\"" + response.audit_digest + "\"}";
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return {name, "PASS", details, elapsed};
    } catch (const std::exception& error) {
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return {name, "FAIL", "{\"error\":\"" + escape_json(error.what()) + "\"}", elapsed};
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/l1-8/cpp-gate";
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--output") output = argv[++index];
    }

    const auto teacher = make_teacher();
    const auto registry_hash = teacher.registry().identity_hash();
    std::vector<Check> checks;
    checks.push_back(run_check("operation_schema_registry", [&]() {
        require(teacher.registry().schemas().size() == 3U && !registry_hash.empty(), "operation schema registry is incomplete");
        return "{\"schema_count\":3,\"registry_hash\":\"" + registry_hash + "\",\"versions\":\"cct-operation-v1\"}";
    }));
    checks.push_back(run_check("required_optional_types_bounds_defaults", [&]() {
        const auto& schema = teacher.registry().schema("document.summarize");
        require(schema.fields.size() == 3U && schema.fields[0].required && schema.fields[1].kind == OperationValueKind::Integer &&
                    schema.fields[1].minimum_integer == 1 && schema.fields[1].maximum_integer == 8 && schema.fields[2].has_default,
                "operation field contract is incomplete");
        return "{\"required_fields\":2,\"optional_fields\":1,\"typed_bounds\":true,\"default_fields\":1}";
    }));
    checks.push_back(run_check("valid_call_serialization_and_validation", [&]() {
        const auto response = teacher.respond(bound_call(teacher), member_auth());
        require(response.decision == OperationDecision::Accepted && response.error_class == OperationErrorClass::None &&
                    response.normalized_arguments.size() == 3U && !response.serialized_call.empty(), "valid operation call failed validation");
        return response_json(response);
    }));
    checks.push_back(run_check("invalid_argument_error_classes", [&]() {
        auto missing = bound_call(teacher);
        missing.arguments.pop_back();
        const auto required = teacher.respond(missing, member_auth());
        auto type = bound_call(teacher);
        type.arguments[1].value = OperationValue("three");
        const auto type_result = teacher.respond(type, member_auth());
        auto bounds = bound_call(teacher);
        bounds.arguments[1].value = OperationValue(std::int64_t(99));
        const auto bounds_result = teacher.respond(bounds, member_auth());
        auto unknown = bound_call(teacher);
        unknown.arguments.push_back({"unexpected", OperationValue("x")});
        const auto unknown_result = teacher.respond(unknown, member_auth());
        require(required.error_class == OperationErrorClass::RequiredFieldMissing && type_result.error_class == OperationErrorClass::TypeMismatch &&
                    bounds_result.error_class == OperationErrorClass::BoundsViolation && unknown_result.error_class == OperationErrorClass::UnknownField,
                "invalid operation calls did not map to the declared error classes");
        return "{\"required_field\":\"required_field_missing\",\"type\":\"type_mismatch\",\"bounds\":\"bounds_violation\",\"unknown_field\":\"unknown_field\"}";
    }));
    checks.push_back(run_check("unknown_operation_rejection", [&]() {
        auto call = bound_call(teacher);
        call.operation_id = "unknown.operation";
        const auto response = teacher.respond(call, member_auth());
        require(response.error_class == OperationErrorClass::UnknownOperation && response.decision == OperationDecision::Rejected, "unknown operation was accepted");
        return response_json(response);
    }));
    checks.push_back(run_check("authorization_class_enforcement", [&]() {
        auto call = bound_call(teacher);
        const auto denied = teacher.respond(call, {true, "tenant-a", "user-a", {"reviewer"}, {"knowledge.lookup"}});
        auto admin = make_call(teacher.registry(), "workflow.draft", {{"workflow_name", OperationValue("release")}, {"approval_required", OperationValue(true)}});
        admin.operation_manifest_hash = teacher.identity().operation_manifest_hash;
        admin.checkpoint_identity_hash = teacher.identity().identity_hash;
        const auto admin_denied = teacher.respond(admin, member_auth());
        require(denied.error_class == OperationErrorClass::AuthorizationDenied && admin_denied.error_class == OperationErrorClass::AuthorizationDenied,
                "operation authorization classes were not enforced");
        return "{\"tenant_member_denied\":true,\"admin_denied\":true,\"default\":\"deny\"}";
    }));
    checks.push_back(run_check("ambiguous_and_evidence_boundaries", [&]() {
        auto ambiguous = bound_call(teacher);
        ambiguous.ambiguous = true;
        const auto ambiguous_response = teacher.respond(ambiguous, member_auth());
        auto lookup = make_call(teacher.registry(), "knowledge.lookup", {{"query", OperationValue("retention")}, {"top_k", OperationValue(std::int64_t(2))}});
        lookup.role = "reviewer";
        lookup.operation_manifest_hash = teacher.identity().operation_manifest_hash;
        lookup.checkpoint_identity_hash = teacher.identity().identity_hash;
        const auto missing = teacher.respond(lookup, {true, "tenant-a", "user-a", {"reviewer"}, {"knowledge.lookup"}});
        lookup.evidence = {{"source-1", "span-2", 0.97}};
        const auto supported = teacher.respond(lookup, {true, "tenant-a", "user-a", {"reviewer"}, {"knowledge.lookup"}});
        require(ambiguous_response.decision == OperationDecision::Abstained && missing.error_class == OperationErrorClass::EvidenceMissing &&
                    supported.decision == OperationDecision::Accepted, "ambiguity or evidence boundary failed");
        return "{\"ambiguous\":\"abstained\",\"missing_evidence\":\"evidence_missing\",\"supported\":\"accepted\"}";
    }));
    checks.push_back(run_check("explanation_and_correction", [&]() {
        const auto explanation = teacher.explain("document.summarize");
        auto invalid = bound_call(teacher);
        invalid.arguments[1].value = OperationValue(std::int64_t(0));
        const auto correction = teacher.correct(invalid, member_auth());
        require(explanation.find("max_sentences") != std::string::npos && correction.correction.find("document.summarize") != std::string::npos,
                "operation explanation or correction omitted schema guidance");
        return "{\"explanation_fields\":true,\"correction_schema_linked\":true}";
    }));
    checks.push_back(run_check("operation_manifest_provenance_and_split_isolation", [&]() {
        require(teacher.manifest().demonstrations.size() == 2U && !teacher.manifest().contains_evaluator_training(), "operation manifest split barrier failed");
        for (const auto& demonstration : teacher.manifest().demonstrations) {
            require(!demonstration.demonstration_hash.empty() && !demonstration.source_hash.empty() && !demonstration.source_id.empty(),
                    "operation demonstration provenance is incomplete");
        }
        return "{\"demonstrations\":2,\"training_evaluator_leakage\":0,\"source_lineage\":true}";
    }));
    checks.push_back(run_check("schema_manifest_checkpoint_identity", [&]() {
        require(teacher.identity().operation_schema_registry_hash == registry_hash && teacher.identity().operation_manifest_hash == teacher.manifest().manifest_hash &&
                    !teacher.identity().identity_hash.empty(), "operation checkpoint identity is incomplete");
        return "{\"registry_hash\":\"" + registry_hash + "\",\"manifest_hash\":\"" + teacher.manifest().manifest_hash +
               "\",\"checkpoint_identity\":\"" + teacher.identity().identity_hash + "\"}";
    }));
    checks.push_back(run_check("serialization_round_trip", [&]() {
        const auto restored = OperationTeacher::deserialize(teacher.serialize());
        const auto original = teacher.respond(bound_call(teacher), member_auth());
        const auto replayed = restored.respond(bound_call(restored), member_auth());
        require(restored.registry().identity_hash() == registry_hash && restored.manifest().manifest_hash == teacher.manifest().manifest_hash &&
                    original.serialized_call == replayed.serialized_call && original.audit_digest == replayed.audit_digest,
                "operation teacher serialization changed replay behavior");
        return "{\"registry_round_trip\":true,\"manifest_round_trip\":true,\"response_replay\":true}";
    }));
    checks.push_back(run_check("identity_mismatch_rejection", [&]() {
        auto call = bound_call(teacher);
        call.operation_schema_hash = "foreign-schema";
        const auto schema = teacher.respond(call, member_auth());
        call = bound_call(teacher);
        call.operation_manifest_hash = "foreign-manifest";
        const auto manifest = teacher.respond(call, member_auth());
        call = bound_call(teacher);
        call.checkpoint_identity_hash = "foreign-checkpoint";
        const auto checkpoint = teacher.respond(call, member_auth());
        require(schema.error_class == OperationErrorClass::IdentityMismatch && manifest.error_class == OperationErrorClass::IdentityMismatch &&
                    checkpoint.error_class == OperationErrorClass::IdentityMismatch, "operation lineage mismatch was accepted");
        return "{\"schema\":\"identity_mismatch\",\"manifest\":\"identity_mismatch\",\"checkpoint\":\"identity_mismatch\"}";
    }));
    checks.push_back(run_check("side_effect_isolation", [&]() {
        auto call = bound_call(teacher);
        call.requests_external_action = true;
        const auto response = teacher.respond(call, member_auth());
        require(response.error_class == OperationErrorClass::SideEffectDenied && !response.side_effect_performed,
                "external side-effect request bypassed Level 1 isolation");
        return "{\"external_actions_allowed\":false,\"host_execution_allowed\":false,\"secret_access_allowed\":false,\"side_effect_performed\":false}";
    }));
    checks.push_back(run_check("strict_corruption_rejection", [&]() {
        auto call = bound_call(teacher);
        bool trailing_rejected = false;
        try { static_cast<void>(OperationCall::deserialize(call.serialize() + " trailing")); } catch (const std::exception&) { trailing_rejected = true; }
        bool registry_rejected = false;
        try { static_cast<void>(OperationRegistry::deserialize(teacher.registry().serialize() + " trailing")); } catch (const std::exception&) { registry_rejected = true; }
        bool identity_rejected = false;
        try { static_cast<void>(OperationCheckpointIdentity::deserialize(teacher.identity().serialize() + " trailing")); } catch (const std::exception&) { identity_rejected = true; }
        require(trailing_rejected && registry_rejected && identity_rejected, "operation serializer accepted trailing corruption");
        return "{\"call_trailing_rejected\":true,\"registry_trailing_rejected\":true,\"identity_trailing_rejected\":true}";
    }));
    checks.push_back(run_check("schema_evolution_invalidates_identity", [&]() {
        auto evolved = make_registry();
        auto changed = evolved.schema("document.summarize");
        changed.description += " evolved";
        OperationRegistry replacement;
        replacement.register_schema(changed);
        replacement.register_schema(evolved.schema("knowledge.lookup"));
        replacement.register_schema(evolved.schema("workflow.draft"));
        require(replacement.identity_hash() != registry_hash, "operation schema evolution did not change registry identity");
        auto call = bound_call(teacher);
        call.operation_schema_hash = replacement.schema("document.summarize").identity_hash();
        require(teacher.respond(call, member_auth()).error_class == OperationErrorClass::IdentityMismatch, "evolved schema call was accepted by old teacher");
        return "{\"old_registry\":\"" + registry_hash + "\",\"new_registry\":\"" + replacement.identity_hash() + "\",\"old_checkpoint_rejected\":true}";
    }));
    checks.push_back(run_check("deterministic_replay", [&]() {
        const auto first = teacher.respond(bound_call(teacher), member_auth());
        const auto second = teacher.respond(bound_call(teacher), member_auth());
        require(first.decision == second.decision && first.serialized_call == second.serialized_call && first.audit_digest == second.audit_digest,
                "operation replay was not deterministic");
        return "{\"same_seed_replay\":true,\"response_identity_equal\":true}";
    }));
    checks.push_back(run_check("negative_control_coverage", [&]() {
        const auto unknown = teacher.respond([] {
            OperationCall call;
            call.request_id = "negative-unknown";
            call.tenant_id = "tenant-a";
            call.user_id = "user-a";
            call.role = "member";
            call.operation_id = "not-declared";
            return call;
        }(), member_auth());
        auto unauth = bound_call(teacher);
        const auto unauthorized = teacher.respond(unauth, {false, "tenant-a", "user-a", {}, {}});
        require(unknown.error_class == OperationErrorClass::UnknownOperation && unauthorized.error_class == OperationErrorClass::IdentityMissing,
                "negative controls did not remain fail-closed");
        return "{\"unknown_operation_rejected\":true,\"unauthenticated_rejected\":true}";
    }));

    const bool passed = checks.size() == 17U && std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0U; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status <<
                       "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    write_file(output / "operation_schema_registry.json", "{\"schema_count\":3,\"registry_hash\":\"" + registry_hash + "\",\"schemas\":[\"document.summarize\",\"knowledge.lookup\",\"workflow.draft\"]}\n");
    write_file(output / "operation_manifest.json", "{\"manifest_hash\":\"" + teacher.manifest().manifest_hash + "\",\"demonstrations\":2,\"evaluator_training\":0}\n");
    write_file(output / "formatter_validator_report.json", "{\"valid_call\":" + checks[2].details + ",\"invalid_error_classes\":" + checks[3].details + ",\"explanation\":" + checks[7].details + "}\n");
    write_file(output / "error_class_report.json", checks[3].details + "\n");
    write_file(output / "authorization_report.json", checks[5].details + "\n");
    write_file(output / "checkpoint_identity_report.json", checks[9].details + "\n");
    write_file(output / "side_effect_isolation_report.json", checks[12].details + "\n");
    write_file(output / "release_record.json", "{\"stage\":\"L1-8\",\"status\":\"" + std::string(passed ? "PASS" : "FAIL") +
               "\",\"mandatory_check_count\":" + std::to_string(checks.size()) + ",\"registry_hash\":\"" + registry_hash +
               "\",\"manifest_hash\":\"" + teacher.manifest().manifest_hash + "\",\"checkpoint_identity\":\"" + teacher.identity().identity_hash +
               "\",\"external_side_effects\":false,\"training_authorized\":false,\"next_stage\":\"L1-9\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# L1-8 Operation and API Teacher Adaptation Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n**Mandatory checks:** " << checks.size() <<
             "  \n**Operation schemas:** 3  \n**Governed demonstrations:** 2  \n\nThe native C++20 gate validates versioned operation schemas, typed required and optional fields, defaults and bounds, deterministic call serialization, structured error classes, unknown-operation rejection, authorization classes, ambiguity and evidence boundaries, explanations and corrections, demonstration provenance, schema/manifest/checkpoint lineage, strict corruption rejection, schema-evolution invalidation, deterministic replay, and Level 1 side-effect isolation.\n\nThis is a bounded operation-contract and teacher-interface result. It does not establish broad API competence, production deployment, unrestricted tool use, autonomous action, or general intelligence. External side effects remain disabled and the L1-9 transition requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\",\"checks\":" << checks.size() << "}\n";
    return passed ? 0 : 1;
}
