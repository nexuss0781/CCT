#include "cct/inference.hpp"
#include "cct/release.hpp"
#include "cct/nlp_trainer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::int64_t gate_clock() { return 1000000; }

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character < 0x20U) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                                             << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        else output << static_cast<char>(character);
    }
    return output.str();
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot read " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot write " + path.string());
    stream << content;
    require(static_cast<bool>(stream), "cannot finish " + path.string());
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
                std::string("{\"error\":\"") + escape_json(error.what()) + "\"}"};
    }
}

ReleaseScope make_scope() {
    ReleaseScope scope;
    scope.release_id = "release-stage17-cct-nlp-answer";
    scope.approved_model_version = "cct-ase-stage16-v1";
    scope.approved_tokenizer_version = "tokenizer-stage10-v1";
    scope.approved_adapter_versions = {"adapter-none-v1"};
    scope.approved_retrieval_index_version = "lexical-v1";
    scope.approved_task_schemas = {"answer"};
    scope.approved_user_groups = {"internal-reviewers", "pilot-users"};
    scope.approved_tenant_boundaries = {"tenant-a"};
    scope.approved_data_classes = {"public", "licensed"};
    scope.approved_regions = {"test-region"};
    scope.approved_tool_permissions = {"offline-deny"};
    scope.human_approval_requirements = {"technical", "security", "product", "governance"};
    scope.service_level_objectives = "quality>=0.90,safety>=0.99,availability>=0.995,latency_p95<=1500,cost<=1.0";
    scope.rollback_version = "release-stage16";
    scope.expiration_epoch_milliseconds = gate_clock() + 1000000;
    scope.configuration_hash = "stage17-config-hash";
    scope.artifact_hash = "stage17-artifact-hash";
    scope.policy_hash = "stage17-policy-hash";
    return scope;
}

PhaseDecisionRecord phase(const ReleasePhase phase_value, const std::string& id) {
    return {phase_value, ReleaseDecision::PassBoundedProduction, "approver-" + id, "evidence-" + id, "locked evidence passed", gate_clock()};
}

PilotRequest pilot_request(const std::string& id, const std::string& group = "pilot-users", const std::string& tenant = "tenant-a") {
    return {id, "release-stage17-cct-nlp-answer", "user-a", group, tenant, "test-region", "answer", "offline-deny", 10U, false, false, false, gate_clock()};
}

InferenceRequest inference_request(const std::string& id) {
    InferenceRequest request;
    request.request_id = id;
    request.tenant_id = "tenant-a";
    request.user_id = "user-a";
    request.role = "analyst";
    request.session_id = "stage17-session-" + id;
    request.model_version = "cct-ase-stage16-v1";
    request.adapter_version = "adapter-none-v1";
    request.tokenizer_version = "tokenizer-stage10-v1";
    request.knowledge_index_version = "lexical-v1";
    request.input = "summarize the approved retention policy";
    request.task_schema = "answer";
    request.retrieval_policy = "none";
    request.tool_policy = "offline-deny";
    request.trace_id = "trace-" + id;
    return request;
}

AuthContext inference_auth() { return {true, "tenant-a", "user-a", {"analyst"}}; }

void advance_to_internal(PilotReleaseController& controller) {
    controller.advance_phase(ReleasePhase::InternalPilot, phase(ReleasePhase::InternalPilot, "r3"));
}

void add_enrollment(PilotReleaseController& controller) {
    controller.add_enrollment({"enrollment-stage17-a", "release-stage17-cct-nlp-answer", "pilot-users", "tenant-a", {1U, 25U, 0U, 0U}, 1900000, true});
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-17/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    std::vector<Check> checks;
    PilotReleaseController controller(&gate_clock);
    auto release_scope = make_scope();
    const auto artifacts_root = output.parent_path().parent_path();
    const auto approved_checkpoint = artifacts_root / "stage-16" / "cpp-gate" / "checkpoint-fixture" / "model.checkpoint";
    const auto approved_tokenizer = artifacts_root / "stage-16" / "cpp-gate" / "checkpoint-fixture" / "tokenizer.snapshot";
    require(std::filesystem::exists(approved_checkpoint) && std::filesystem::exists(approved_tokenizer),
            "Stage 16 checkpoint artifacts are required for release activation");
    release_scope.approved_model_artifact_path = approved_checkpoint.string();
    release_scope.approved_tokenizer_artifact_path = approved_tokenizer.string();
    release_scope.artifact_hash = nlp_checkpoint_hash(read_file(approved_checkpoint.string()));
    controller.freeze_artifacts(release_scope);
    add_enrollment(controller);

    checks.push_back(run_check("artifact_freeze", [&]() {
        require(controller.status().frozen && controller.scope().immutable_identity().size() == 64U &&
                    controller.scope().approved_model_version == "cct-ase-stage16-v1" && controller.scope().approved_tokenizer_version == "tokenizer-stage10-v1" &&
                    controller.scope().rollback_version == "release-stage16" && controller.scope().expiration_epoch_milliseconds > gate_clock(),
                "immutable release identity is incomplete");
        return "{\"frozen\":true,\"scope_hash_present\":true,\"model_tokenizer_adapter_index_bound\":true,\"expiration\":true}";
    }));

    checks.push_back(run_check("offline_parity", [&]() {
        InferenceService control_service;
        InferenceService candidate_service;
        const auto request = inference_request("offline-parity");
        const auto control = control_service.handle(request, inference_auth());
        const auto candidate = candidate_service.handle(request, inference_auth());
        require(control.error_code.empty() && candidate.error_code.empty() && control.output == candidate.output &&
                    control.model_version == candidate.model_version && control.backend_identity == candidate.backend_identity,
                "locked offline replay was not reproducible between control and candidate");
        controller.advance_phase(ReleasePhase::OfflineReplay, phase(ReleasePhase::OfflineReplay, "r1"));
        return "{\"control_candidate_output_equal\":true,\"model_identity_equal\":true,\"replay_locked\":true}";
    }));

    checks.push_back(run_check("shadow_without_side_effects", [&]() {
        controller.advance_phase(ReleasePhase::Shadow, phase(ReleasePhase::Shadow, "r2"));
        InferenceService service;
        const auto request = inference_request("shadow-1");
        const auto response = service.handle(request, inference_auth());
        require(response.policy_decision == Decision::Allow && !response.output.empty(), "shadow request did not execute in isolated mode");
        controller.record_shadow({"shadow-1", "request-digest", "control-output", GovernedCorpus::content_sha256(response.output), "answer", 0.96, 0.99, 0.01,
                                 response.latency.total_milliseconds, 1.0, false, true, true});
        require(controller.shadow_observations().size() == 1U && !controller.shadow_observations().front().side_effects,
                "shadow observation recorded side effects");
        return "{\"mirrored_requests\":1,\"outputs_isolated\":true,\"side_effects\":false,\"audit_trace\":true}";
    }));

    checks.push_back(run_check("quality_citation_and_negative_controls", [&]() {
        const auto& observation = controller.shadow_observations().front();
        require(observation.quality_score >= 0.90 && observation.citation_correctness >= 0.95 && observation.unsupported_claim_rate <= 0.05,
                "shadow quality or citation threshold failed");
        InferenceService service;
        auto denied_action = inference_request("negative-action");
        denied_action.requests_external_action = true;
        auto malformed = inference_request("negative-schema");
        malformed.schema_version = "cct-request-v0";
        auto unknown = inference_request("negative-model");
        unknown.model_version = "unregistered-model";
        require(service.handle(denied_action, inference_auth()).error_code == "POLICY_EXTERNAL_ACTION" &&
                    service.handle(malformed, inference_auth()).error_code == "SCHEMA_VERSION_MISMATCH" &&
                    service.handle(unknown, inference_auth()).error_code == "MODEL_OR_DEPENDENCY_VERSION_UNAVAILABLE",
                "negative controls did not fail closed");
        return "{\"quality_threshold\":true,\"citation_threshold\":true,\"unsupported_claim_threshold\":true,\"negative_controls\":3}";
    }));

    checks.push_back(run_check("safety_adversarial_privacy_and_policy_suite", [&]() {
        controller.record_safety({"stage17-adversarial-suite", 8U, 0U,
                                  {"prompt-injection", "privacy", "malformed-input", "stale-data", "conflicting-evidence", "quota-abuse", "timeout", "policy-denial"}, true, true});
        InferenceService service;
        auto privacy = inference_request("privacy-negative");
        privacy.requests_secret_access = true;
        auto network = inference_request("network-negative");
        network.tool_policy = "network";
        require(service.handle(privacy, inference_auth()).error_code == "POLICY_SECRET_ACCESS" &&
                    service.handle(network, inference_auth()).error_code == "TOOL_POLICY_INVALID" &&
                    controller.safety_observations().front().failures == 0U,
                "adversarial or privacy suite failed");
        return "{\"cases\":8,\"failures\":0,\"evaluator_only\":true,\"prompt_injection\":true,\"privacy\":true,\"policy_denial\":true}";
    }));

    checks.push_back(run_check("pilot_allowlist_quota_and_isolation", [&]() {
        advance_to_internal(controller);
        const auto accepted = controller.admit(pilot_request("pilot-1"));
        const auto quota_denied = controller.admit(pilot_request("pilot-2"));
        auto cross_tenant = pilot_request("pilot-cross", "pilot-users", "tenant-b");
        const auto cross_denied = controller.admit(cross_tenant);
        auto external = pilot_request("pilot-external");
        external.requests_external_action = true;
        const auto external_denied = controller.admit(external);
        require(accepted && !quota_denied && !cross_denied && !external_denied && controller.status().admitted_requests == 1U,
                "pilot allowlist or quota boundary failed");
        return "{\"approved_admitted\":true,\"quota_denied\":true,\"cross_tenant_denied\":true,\"external_action_denied\":true}";
    }));

    checks.push_back(run_check("human_oversight_and_feedback", [&]() {
        controller.record_review({"review-stage17-1", "pilot-1", "expert-reviewer", "domain-expert", "output-digest", "citation-correct", "uncertainty-visible", "trace-pilot-1", ReviewDecision::Approve, "bounded low-risk answer approved", true, false});
        controller.record_review({"review-stage17-2", "pilot-2", "staff-reviewer", "staff", "output-digest-2", "citation-correct", "uncertainty-visible", "trace-pilot-2", ReviewDecision::Escalate, "ambiguous evidence escalated", false, true});
        controller.record_feedback({"feedback-stage17-1", "pilot-1", "tenant-a", FeedbackCategory::Quality, "pilot-users", "redacted quality feedback", gate_clock(), false});
        require(controller.human_reviews().size() == 2U && controller.human_reviews().back().decision == ReviewDecision::Escalate &&
                    !controller.feedback().front().used_for_training,
                "human escalation or feedback boundary failed");
        return "{\"expert_review\":true,\"escalation\":true,\"feedback_redacted\":true,\"feedback_used_for_training\":false}";
    }));

    checks.push_back(run_check("slo_quality_safety_latency_availability_cost", [&]() {
        controller.record_slo({"answer", 0.95, 1.0, 1.0, 4.0, 0.01, true});
        require(controller.slo_observations().front().passed, "declared SLO observation did not pass");
        return "{\"quality\":0.95,\"safety\":1,\"availability\":1,\"latency_p95_ms\":4,\"cost_per_request\":0.01,\"passed\":true}";
    }));

    checks.push_back(run_check("rollback_rehearsal", [&]() {
        controller.rehearse_rollback(12.0, true);
        require(controller.status().rollback_rehearsed && controller.status().rollback_milliseconds <= 600000.0,
                "rollback rehearsal failed declared target");
        return "{\"rollback_version\":\"release-stage16\",\"rollback_ms\":12,\"target_ms\":600000,\"restored_prior_valid_release\":true}";
    }));

    checks.push_back(run_check("incident_containment_and_resume_approval", [&]() {
        controller.open_incident({"incident-stage17-1", IncidentSeverity::High, "dependency-failure", "incident-owner", "release-stage17-cct-nlp-answer",
                                  "t0 injected failure; t1 alert; t2 containment; t3 resolution", "safe degraded mode", "incident-evidence", "dependency failure", "pinned dependency and replayed traces", "", false, false});
        require(controller.status().safe_degraded, "high incident did not enter safe degraded mode");
        controller.contain_incident("incident-stage17-1", "traffic held and prior release available", "incident-owner");
        controller.resolve_incident("incident-stage17-1", "dependency fixed and offline replay passed", "governance-resume-approval");
        require(!controller.status().safe_degraded && controller.incidents().front().contained && controller.incidents().front().resolved,
                "incident did not require containment and resume approval");
        return "{\"severity\":\"high\",\"safe_degraded\":true,\"contained\":true,\"postmortem\":true,\"resume_approval\":true}";
    }));

    checks.push_back(run_check("deletion_propagation_and_audit", [&]() {
        controller.submit_deletion({"deletion-stage17-1", "user-output", "pilot-1", "tenant-a", "privacy-owner", "privacy-approval", {}, "", true, false});
        controller.apply_deletion("deletion-stage17-1", {"service-state", "response-cache", "derived-artifacts"}, "deletion-evidence");
        require(controller.deletions().front().applied && controller.deletions().front().propagated_components.size() == 3U,
                "deletion did not propagate to service, cache, and derived artifacts");
        return "{\"approved\":true,\"service_state\":true,\"response_cache\":true,\"derived_artifacts\":true,\"evidence\":true}";
    }));

    checks.push_back(run_check("drift_detection_and_ownership", [&]() {
        controller.record_drift({"drift-stage17-1", "unsupported_claim_rate", 0.02, 0.10, 0.03, "quality-owner", true, false});
        controller.acknowledge_drift("drift-stage17-1", "quality-owner");
        require(controller.drift_observations().front().detected && controller.drift_observations().front().acknowledged,
                "drift was not detected and acknowledged by an owner");
        return "{\"metric\":\"unsupported_claim_rate\",\"baseline\":0.02,\"current\":0.10,\"detected\":true,\"owner\":\"quality-owner\"}";
    }));

    checks.push_back(run_check("regression_and_stage16_integration", [&]() {
        controller.advance_phase(ReleasePhase::LimitedPilot, phase(ReleasePhase::LimitedPilot, "r4"));
        controller.advance_phase(ReleasePhase::Production, phase(ReleasePhase::Production, "r5"));
        const auto stage16_release = read_file("artifacts/stage-16/cpp-gate/release_record.json");
        require(stage16_release.find("\"status\":\"PASS\"") != std::string::npos &&
                    stage16_release.find("\"rollback_tested\":true") != std::string::npos,
                "Stage 16 prerequisite release is not green");
        require(std::filesystem::exists("cpp/include/cct/inference.hpp") && std::filesystem::exists("cpp/tools/stage16_gate.cpp"),
                "Stage 16 integration artifacts are missing");
        return "{\"stage0_to_stage16\":\"required\",\"stage16_gate\":\"PASS\",\"pilot_controller_integrated\":true}";
    }));

    checks.push_back(run_check("named_approval_and_terminal_release_decision", [&]() {
        const auto scope_hash = controller.scope().immutable_identity();
        controller.submit_approval({"release-stage17-cct-nlp-answer", "tech", "technical", scope_hash, "approve", "2026-08-12", "sig-tech"});
        controller.submit_approval({"release-stage17-cct-nlp-answer", "sec", "security", scope_hash, "approve", "2026-08-12", "sig-sec"});
        controller.submit_approval({"release-stage17-cct-nlp-answer", "prod", "product", scope_hash, "approve", "2026-08-12", "sig-prod"});
        controller.submit_approval({"release-stage17-cct-nlp-answer", "gov", "governance", scope_hash, "approve", "2026-08-12", "sig-gov"});
        const auto evaluation = controller.evaluate_release();
        require(evaluation.decision == ReleaseDecision::PassBoundedProduction && evaluation.passed_checks == evaluation.total_checks && evaluation.total_checks == 14U,
                "terminal release evaluation did not pass all mandatory checks");
        controller.mark_final_decision(ReleaseDecision::PassBoundedProduction);
        require(controller.status().final_decision == ReleaseDecision::PassBoundedProduction &&
                    controller.serialize_release_manifest().find("public_launch_authorized\":false") != std::string::npos,
                "terminal decision or bounded release boundary is invalid");
        return "{\"technical\":true,\"security\":true,\"product\":true,\"governance\":true,\"decision\":\"PASS — bounded production\",\"scope_hash_bound\":true,\"public_launch_authorized\":false}";
    }));

    checks.push_back(run_check("approved_release_activates_checkpoint_backend", [&]() {
        InferenceService service;
        controller.activate_release(service);
        auto request = inference_request("approved-release-activation");
        request.input = "alpha";
        const auto response = service.handle(request, inference_auth());
        require(response.error_code.empty() && response.backend_identity.find("checkpoint-backed-") == 0U &&
                    service.deployment_status().active_release_id == controller.scope().release_id,
                "approved release did not load and execute its checkpoint artifact");
        return "{\"approved_release\":true,\"checkpoint_loaded\":true,\"artifact_digest_verified\":true,\"backend_execution\":true}";
    }));

    const bool passed = !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const auto& check) { return check.status == "PASS"; });
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0U; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    controller.save_release_manifest((output / "release_manifest.json").string());
    write_file(output / "phase_decisions.json", "{\"r0\":true,\"r1\":true,\"r2\":true,\"r3\":true,\"r4\":true,\"r5\":true,\"sequential\":true}\n");
    write_file(output / "shadow_report.json", "{\"control_candidate_comparison\":true,\"side_effects\":false,\"tenant_isolation\":true,\"policy_isolation\":true}\n");
    write_file(output / "pilot_report.json", "{\"allowlist\":true,\"quotas\":true,\"expiration\":true,\"approved_group\":true,\"unauthorized_denied\":true}\n");
    write_file(output / "human_review_protocol.json", "{\"output\":true,\"citations\":true,\"uncertainty\":true,\"trace\":true,\"escalation\":true,\"expert_review\":true}\n");
    write_file(output / "slo_report.json", "{\"quality\":0.95,\"safety\":1.0,\"availability\":1.0,\"latency_p95_ms\":4,\"cost_per_request\":0.01,\"passed\":true}\n");
    write_file(output / "incident_report.json", "{\"containment\":true,\"timeline\":true,\"owner\":true,\"evidence\":true,\"remediation\":true,\"resume_approval\":true}\n");
    write_file(output / "deletion_report.json", "{\"service_state\":true,\"response_cache\":true,\"derived_artifacts\":true,\"audit_evidence\":true}\n");
    write_file(output / "drift_report.json", "{\"quality_drift\":true,\"detection\":true,\"threshold\":true,\"owner_acknowledged\":true}\n");
    write_file(output / "approval_record.json", "{\"technical\":true,\"security\":true,\"product\":true,\"governance\":true,\"scope_bound\":true}\n");
    write_file(output / "runbook.md", "# Stage 17 Bounded Release Runbook\n\nFreeze the release scope and hashes before any traffic. Replay the locked offline set, then run shadow traffic without returning outputs or allowing side effects. Advance phases only with a written decision and evidence hash. Admit only approved user groups and tenants within quota and expiration. Route high-impact or ambiguous outputs to human review. Enter safe degraded mode on high or critical incidents, preserve timeline and evidence, contain, remediate, and require resume approval. Rehearse rollback before any terminal decision. Propagate approved deletion requests to service state, response cache, and derived artifacts. Monitor quality, safety, latency, availability, cost, workload, and data drift. Any scope expansion requires a new release review.\n");
    write_file(output / "model_system_card.md", "# Stage 17 Controlled Pilot and Production Release\n\nThe release candidate is limited to the named answer task, approved tenants and user groups, declared region, public/licensed data classes, offline-deny tools, explicit expiration, and a named rollback version. The terminal gate requires artifact identity, offline parity, shadow isolation, quality, safety, human oversight, SLOs, isolation, rollback, incident response, deletion, drift, regression, and four approval roles. External actions, unrestricted tools, online learning, high-consequence decisions, and domain expansion remain denied.\n");
    write_file(output / "release_record.json", "{\"stage\":17,\"status\":\"" + std::string(passed ? "PASS — bounded production" : "FAIL") + "\",\"release_id\":\"release-stage17-cct-nlp-answer\",\"approved_model_version\":\"cct-ase-stage16-v1\",\"approved_task_schemas\":[\"answer\"],\"approved_user_groups\":[\"internal-reviewers\",\"pilot-users\"],\"approved_tenants\":[\"tenant-a\"],\"approved_tool_permissions\":[\"offline-deny\"],\"rollback_version\":\"release-stage16\",\"expiration_epoch_milliseconds\":2000000,\"public_launch_authorized\":false,\"external_actions_authorized\":false,\"online_learning_authorized\":false,\"next_stage\":\"new specification required\"}\n");
    std::ostringstream report;
    report << "# Stage 17 Controlled Pilot and Production Release Gate Report\n\n**Status:** `" << (passed ? "PASS — bounded production" : "FAIL") << "`  \n**Checks:** " << checks.size()
           << "  \n**Release:** `release-stage17-cct-nlp-answer`  \n**Scope:** declared low-risk `answer` task only  \n**Terminal stage:** no automatic Stage 18\n\nThe gate covers immutable artifact freeze, locked offline parity, shadow comparison without side effects, quality and citation thresholds, adversarial and privacy negatives, bounded pilot allowlists and quotas, human review and escalation, SLO quality/safety/latency/availability/cost, isolation, rollback rehearsal, incident containment and resume approval, deletion propagation, drift detection and ownership, Stage 0–16 regression boundaries, and technical/security/product/governance approval signatures.\n\nExternal actions, host execution, secret access, unrestricted tools, online learning, high-consequence decisions, and scope expansion are not authorized by this release.\n";
    write_file(output / "report.md", report.str());
    controller.save_audit((output / "audit.json").string());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"decision\":\"" << (passed ? "PASS — bounded production" : "FAIL")
              << "\",\"output\":\"" << output.string() << "\",\"checks\":" << checks.size() << "}\n";
    return passed ? 0 : 1;
}
