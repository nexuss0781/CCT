#include "cct/release.hpp"
#include "cct/nlp_trainer.hpp"
#include "cct/tokenizer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::int64_t test_clock() { return 1000000; }

struct CheckpointFixture {
    std::filesystem::path model_path;
    std::filesystem::path tokenizer_path;
    std::string digest;
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "release test artifact could not be read");
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

CheckpointFixture checkpoint_fixture() {
    const auto root = std::filesystem::temp_directory_path() / "cct-release-activation-fixture";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    TokenizerConfig tokenizer_config;
    tokenizer_config.tokenizer_version = "tokenizer-stage10-v1";
    tokenizer_config.candidate = TokenizerCandidate::Byte;
    tokenizer_config.include_bos_eos = false;
    const auto tokenizer = Tokenizer::build(tokenizer_config, {TokenizerTrainingRecord{"release", "alpha beta", true, false}});
    const auto tokenizer_path = root / "tokenizer.snapshot";
    { std::ofstream output(tokenizer_path, std::ios::binary); require(static_cast<bool>(output), "release tokenizer fixture could not be created"); output << tokenizer.serialize_snapshot(); }
    const auto vocabulary_size = static_cast<std::size_t>(tokenizer.vocabulary().back().id) + 1U;
    const NlpModelConfig model_config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 4U, 4U, 16U, 16U};
    NlpOptimizerConfig optimizer;
    optimizer.total_steps = 1U;
    NlpTrainer trainer(model_config, optimizer, tokenizer.snapshot_hash(), "release-activation-fixture");
    auto parameters = trainer.model().parameter_vector();
    parameters.assign(parameters.size(), 0.0);
    const auto recurrent_offset = vocabulary_size * model_config.embedding_dim;
    const auto head_offset = recurrent_offset + 4U * model_config.hidden_dim * model_config.embedding_dim + 3U * model_config.hidden_dim;
    const auto bias_offset = head_offset + vocabulary_size * model_config.hidden_dim;
    parameters[bias_offset + static_cast<TokenId>(Tokenizer::kByteFirstId + static_cast<unsigned int>('a'))] = 10.0;
    parameters[bias_offset + Tokenizer::kEosId] = -10.0;
    trainer.model().set_parameter_vector(parameters);
    const auto model_path = root / "model.checkpoint";
    trainer.save_checkpoint(model_path.string());
    return {model_path, tokenizer_path, nlp_checkpoint_hash(read_file(model_path))};
}

ReleaseScope scope() {
    ReleaseScope value;
    value.release_id = "release-stage17";
    value.approved_model_version = "cct-ase-stage16-v1";
    value.approved_tokenizer_version = "tokenizer-stage10-v1";
    value.approved_adapter_versions = {"adapter-none-v1"};
    value.approved_retrieval_index_version = "lexical-v1";
    value.approved_task_schemas = {"answer"};
    value.approved_user_groups = {"internal-reviewers", "pilot-users"};
    value.approved_tenant_boundaries = {"tenant-a"};
    value.approved_data_classes = {"public", "licensed"};
    value.approved_regions = {"test-region"};
    value.approved_tool_permissions = {"offline-deny"};
    value.human_approval_requirements = {"technical", "security", "product", "governance"};
    value.service_level_objectives = "quality>=0.90,safety>=0.99,availability>=0.995,latency_p95<=1500";
    value.rollback_version = "release-stage16";
    value.expiration_epoch_milliseconds = 2000000;
    value.configuration_hash = "config-hash";
    value.artifact_hash = "artifact-hash";
    value.policy_hash = "policy-hash";
    return value;
}

PhaseDecisionRecord phase(const ReleasePhase value, const std::string& name) {
    return {value, ReleaseDecision::PassBoundedProduction, "approver-" + name, "evidence-" + name, "phase passed", 1000000};
}

void advance_to_limited(PilotReleaseController& controller) {
    controller.advance_phase(ReleasePhase::OfflineReplay, phase(ReleasePhase::OfflineReplay, "r1"));
    controller.advance_phase(ReleasePhase::Shadow, phase(ReleasePhase::Shadow, "r2"));
    controller.advance_phase(ReleasePhase::InternalPilot, phase(ReleasePhase::InternalPilot, "r3"));
    controller.advance_phase(ReleasePhase::LimitedPilot, phase(ReleasePhase::LimitedPilot, "r4"));
}

void add_scope_enrollment(PilotReleaseController& controller) {
    controller.add_enrollment({"enrollment-a", "release-stage17", "pilot-users", "tenant-a", {2U, 100U, 0U, 0U}, 1900000, true});
}

void add_complete_evidence(PilotReleaseController& controller) {
    controller.advance_phase(ReleasePhase::OfflineReplay, phase(ReleasePhase::OfflineReplay, "r1"));
    controller.advance_phase(ReleasePhase::Shadow, phase(ReleasePhase::Shadow, "r2"));
    controller.record_shadow({"shadow-1", "request-digest", "control-digest", "candidate-digest", "answer", 0.96, 0.99, 0.01, 3.0, 1.0, false, true, true});
    controller.record_safety({"safety-suite", 10U, 0U, {"prompt-injection", "privacy", "policy"}, true, true});
    controller.advance_phase(ReleasePhase::InternalPilot, phase(ReleasePhase::InternalPilot, "r3"));
    controller.record_review({"review-1", "request-1", "expert-1", "domain-expert", "output-digest", "citation-correct", "uncertainty-shown", "trace-1", ReviewDecision::Approve, "reviewed bounded answer", true, false});
    controller.record_review({"review-2", "request-2", "staff-1", "staff-reviewer", "output-digest-2", "citation-correct", "uncertainty-shown", "trace-2", ReviewDecision::Escalate, "escalated ambiguity", false, true});
    controller.advance_phase(ReleasePhase::LimitedPilot, phase(ReleasePhase::LimitedPilot, "r4"));
    controller.advance_phase(ReleasePhase::Production, phase(ReleasePhase::Production, "r5"));
    controller.record_feedback({"feedback-1", "request-1", "tenant-a", FeedbackCategory::Quality, "pilot-users", "redacted quality feedback", 1000000, false});
    controller.record_slo({"answer", 0.95, 1.0, 1.0, 4.0, 0.01, true});
    controller.record_drift({"drift-1", "unsupported_claim_rate", 0.02, 0.10, 0.03, "quality-owner", true, false});
    controller.acknowledge_drift("drift-1", "quality-owner");
    controller.submit_deletion({"delete-1", "user", "user-a", "tenant-a", "privacy-owner", "privacy-approval", {}, "", true, false});
    controller.apply_deletion("delete-1", {"service-state", "response-cache", "derived-artifacts"}, "deletion-evidence");
    controller.rehearse_rollback(12.0, true);
    controller.open_incident({"incident-1", IncidentSeverity::High, "dependency-failure", "incident-owner", "release-stage17", "t0 fault; t1 contain; t2 resolve", "safe degraded", "incident-evidence", "dependency failure", "restart and replay", "", false, false});
    controller.contain_incident("incident-1", "traffic stopped and release held", "incident-owner");
    controller.resolve_incident("incident-1", "dependency pinned and replayed", "governance-resume-approval");
}

void test_scope_and_phase_barriers() {
    PilotReleaseController controller(&test_clock);
    bool scope_rejected = false;
    try { controller.freeze_artifacts({}); } catch (const ReleaseError&) { scope_rejected = true; }
    require(scope_rejected, "incomplete release scope was accepted");
    controller.freeze_artifacts(scope());
    require(controller.status().frozen && controller.scope().immutable_identity().size() == 64U, "scope was not frozen with immutable identity");
    bool skipped = false;
    try { controller.advance_phase(ReleasePhase::Shadow, phase(ReleasePhase::Shadow, "skip")); } catch (const ReleaseError&) { skipped = true; }
    require(skipped && controller.status().phase == ReleasePhase::ArtifactFreeze, "phase transition skipped R1");
    controller.advance_phase(ReleasePhase::OfflineReplay, phase(ReleasePhase::OfflineReplay, "r1"));
    require(controller.status().phase == ReleasePhase::OfflineReplay, "R1 did not advance");
}

void test_pilot_allowlist_quota_and_safe_mode() {
    PilotReleaseController controller(&test_clock);
    controller.freeze_artifacts(scope());
    add_scope_enrollment(controller);
    auto request = PilotRequest{"pilot-1", "release-stage17", "user-a", "pilot-users", "tenant-a", "test-region", "answer", "offline-deny", 20U, false, false, false, 1000000};
    require(!controller.admit(request), "pilot admitted before internal-pilot phase");
    advance_to_limited(controller);
    require(controller.admit(request), "approved pilot request was denied");
    request.request_id = "pilot-2";
    require(controller.admit(request), "second request inside quota was denied");
    request.request_id = "pilot-quota";
    require(!controller.admit(request), "request quota was not enforced");
    request.request_id = "pilot-cross-tenant";
    request.tenant_id = "tenant-b";
    require(!controller.admit(request), "cross-tenant pilot request was admitted");
    request.tenant_id = "tenant-a";
    request.request_id = "pilot-action";
    request.requests_external_action = true;
    require(!controller.admit(request), "external action was admitted in pilot");
    controller.enter_safe_degraded_mode("critical safety incident");
    request.requests_external_action = false;
    request.request_id = "pilot-degraded";
    require(!controller.admit(request), "safe-degraded mode did not stop admission");
}

void test_shadow_review_feedback_and_deletion() {
    PilotReleaseController controller(&test_clock);
    controller.freeze_artifacts(scope());
    controller.advance_phase(ReleasePhase::OfflineReplay, phase(ReleasePhase::OfflineReplay, "r1"));
    controller.advance_phase(ReleasePhase::Shadow, phase(ReleasePhase::Shadow, "r2"));
    controller.record_shadow({"shadow-1", "request-digest", "control-digest", "candidate-digest", "answer", 0.96, 0.99, 0.01, 3.0, 1.0, false, true, true});
    controller.record_safety({"adversarial", 12U, 0U, {"prompt-injection", "malformed", "privacy"}, true, true});
    controller.advance_phase(ReleasePhase::InternalPilot, phase(ReleasePhase::InternalPilot, "r3"));
    controller.record_review({"review-1", "request-1", "expert-1", "domain-expert", "output", "citations", "uncertainty", "trace", ReviewDecision::Approve, "approved", true, false});
    controller.record_feedback({"feedback-1", "request-1", "tenant-a", FeedbackCategory::Factuality, "pilot-users", "redacted factuality complaint", 1000000, false});
    controller.submit_deletion({"delete-1", "model-output", "output-1", "tenant-a", "privacy-owner", "approved", {}, "", true, false});
    bool incomplete_deletion = false;
    try { controller.apply_deletion("delete-1", {"service-state"}, "evidence"); } catch (const ReleaseError&) { incomplete_deletion = true; }
    require(incomplete_deletion, "incomplete deletion propagation was accepted");
    controller.apply_deletion("delete-1", {"service-state", "response-cache", "derived-artifacts"}, "evidence");
    require(controller.deletions().front().applied && !controller.feedback().front().used_for_training, "deletion or feedback controls failed");
}

void test_incident_and_rollback_controls() {
    PilotReleaseController controller(&test_clock);
    controller.freeze_artifacts(scope());
    controller.rehearse_rollback(700000.0, true);
    require(!controller.status().rollback_rehearsed, "slow rollback passed the declared target");
    controller.rehearse_rollback(15.0, true);
    controller.open_incident({"incident-1", IncidentSeverity::Critical, "tenant-crossover", "security-owner", "release-stage17", "t0 detected", "safe degraded", "evidence", "isolation defect", "revoke release", "", false, false});
    require(controller.status().safe_degraded, "critical incident did not enter safe degraded mode");
    bool premature = false;
    try { controller.resolve_incident("incident-1", "not contained", "approval"); } catch (const ReleaseError&) { premature = true; }
    require(premature, "uncontained incident was resolved");
    controller.contain_incident("incident-1", "release stopped", "security-owner");
    controller.resolve_incident("incident-1", "isolation fixed", "security-resume");
    require(!controller.status().safe_degraded && controller.incidents().front().resolved, "resolved incident did not restore controlled mode");
}

void test_release_activation_and_atomic_publication() {
    const auto fixture = checkpoint_fixture();
    auto release_scope = scope();
    release_scope.approved_model_artifact_path = fixture.model_path.string();
    release_scope.approved_tokenizer_artifact_path = fixture.tokenizer_path.string();
    release_scope.artifact_hash = fixture.digest;
    PilotReleaseController controller(&test_clock);
    controller.freeze_artifacts(release_scope);
    add_scope_enrollment(controller);
    add_complete_evidence(controller);
    const auto scope_hash = controller.scope().immutable_identity();
    controller.submit_approval({"release-stage17", "tech", "technical", scope_hash, "approve", "2026-08-12", "sig-tech"});
    controller.submit_approval({"release-stage17", "sec", "security", scope_hash, "approve", "2026-08-12", "sig-sec"});
    controller.submit_approval({"release-stage17", "prod", "product", scope_hash, "approve", "2026-08-12", "sig-prod"});
    controller.submit_approval({"release-stage17", "gov", "governance", scope_hash, "approve", "2026-08-12", "sig-gov"});
    controller.mark_final_decision(ReleaseDecision::PassBoundedProduction);
    InferenceService service;
    controller.activate_release(service);
    require(service.deployment_status().active_release_id == "release-stage17" && service.config().backend_mode == InferenceBackendMode::Checkpoint &&
                service.config().model_checkpoint_path == fixture.model_path.string() && service.config().model_version == "cct-ase-stage16-v1",
            "approved release did not activate its checkpoint backend and dependency identity");
    InferenceRequest request;
    request.request_id = "release-activation-request";
    request.tenant_id = "tenant-a";
    request.user_id = "user-a";
    request.role = "analyst";
    request.session_id = "release-session";
    request.input = "alpha";
    request.task_schema = "answer";
    request.retrieval_policy = "none";
    request.tool_policy = "offline-deny";
    request.trace_id = "release-trace";
    const auto response = service.handle(request, {true, "tenant-a", "user-a", {"analyst"}});
    require(response.error_code.empty() && response.backend_identity.find("checkpoint-backed-") == 0U, "activated release did not execute through the checkpoint backend");
    const auto root = std::filesystem::temp_directory_path() / "cct-release-atomic-publication";
    std::filesystem::remove_all(root);
    controller.save_release_manifest((root / "release_manifest.json").string());
    controller.save_audit((root / "audit.json").string());
    require(std::filesystem::exists(root / "release_manifest.json") && std::filesystem::exists(root / "audit.json"),
            "release manifest or audit was not atomically published");
    require(read_file(root / "release_manifest.json").find(fixture.model_path.string()) != std::string::npos,
            "release manifest omitted the approved artifact path");
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        require(entry.path().filename().string().find(".tmp.") == std::string::npos, "release temporary file was left after publication");
    }
    std::filesystem::remove_all(root);
    std::filesystem::remove_all(fixture.model_path.parent_path());
}

void test_durable_release_bundle_black_box() {
    const auto model_path = std::filesystem::path("artifacts/stage-16/cpp-gate/checkpoint-fixture/model.checkpoint");
    const auto tokenizer_path = std::filesystem::path("artifacts/stage-16/cpp-gate/checkpoint-fixture/tokenizer.snapshot");
    require(std::filesystem::exists(model_path) && std::filesystem::exists(tokenizer_path), "durable release-validation artifacts are missing");
    auto release_scope = scope();
    release_scope.approved_model_artifact_path = model_path.string();
    release_scope.approved_tokenizer_artifact_path = tokenizer_path.string();
    release_scope.artifact_hash = nlp_checkpoint_hash(read_file(model_path));
    PilotReleaseController controller(&test_clock);
    controller.freeze_artifacts(release_scope);
    add_scope_enrollment(controller);
    add_complete_evidence(controller);
    const auto scope_hash = controller.scope().immutable_identity();
    controller.submit_approval({"release-stage17", "tech", "technical", scope_hash, "approve", "2026-08-12", "sig-tech"});
    controller.submit_approval({"release-stage17", "sec", "security", scope_hash, "approve", "2026-08-12", "sig-sec"});
    controller.submit_approval({"release-stage17", "prod", "product", scope_hash, "approve", "2026-08-12", "sig-prod"});
    controller.submit_approval({"release-stage17", "gov", "governance", scope_hash, "approve", "2026-08-12", "sig-gov"});
    controller.mark_final_decision(ReleaseDecision::PassBoundedProduction);
    InferenceService service;
    controller.activate_release(service);
    InferenceRequest request;
    request.request_id = "durable-release-black-box";
    request.tenant_id = "tenant-a";
    request.user_id = "user-a";
    request.role = "analyst";
    request.session_id = "durable-release-session";
    request.input = "alpha";
    request.task_schema = "answer";
    request.retrieval_policy = "none";
    request.tool_policy = "offline-deny";
    request.trace_id = "durable-release-trace";
    const auto response = service.handle(request, {true, "tenant-a", "user-a", {"analyst"}});
    require(response.error_code.empty() && response.backend_identity.find("checkpoint-backed-") == 0U && !response.output.empty(),
            "durable release-validation bundle did not execute a black-box inference request");
}

void test_terminal_approval_and_decision() {
    PilotReleaseController controller(&test_clock);
    controller.freeze_artifacts(scope());
    add_scope_enrollment(controller);
    add_complete_evidence(controller);
    const auto scope_hash = controller.scope().immutable_identity();
    controller.submit_approval({"release-stage17", "tech", "technical", scope_hash, "approve", "2026-08-12", "sig-tech"});
    controller.submit_approval({"release-stage17", "sec", "security", scope_hash, "approve", "2026-08-12", "sig-sec"});
    controller.submit_approval({"release-stage17", "prod", "product", scope_hash, "approve", "2026-08-12", "sig-prod"});
    controller.submit_approval({"release-stage17", "gov", "governance", scope_hash, "approve", "2026-08-12", "sig-gov"});
    const auto evaluation = controller.evaluate_release();
    require(evaluation.decision == ReleaseDecision::PassBoundedProduction && evaluation.passed_checks == evaluation.total_checks && evaluation.total_checks == 14U,
            "complete terminal evidence did not pass bounded production evaluation");
    controller.mark_final_decision(ReleaseDecision::PassBoundedProduction);
    require(controller.status().final_decision == ReleaseDecision::PassBoundedProduction && controller.serialize_release_manifest().find("public_launch_authorized\":false") != std::string::npos,
            "terminal release decision or boundary was not serialized");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"scope_and_phase_barriers", test_scope_and_phase_barriers},
        {"pilot_allowlist_quota_and_safe_mode", test_pilot_allowlist_quota_and_safe_mode},
        {"shadow_review_feedback_and_deletion", test_shadow_review_feedback_and_deletion},
        {"incident_and_rollback_controls", test_incident_and_rollback_controls},
        {"release_activation_and_atomic_publication", test_release_activation_and_atomic_publication},
        {"durable_release_bundle_black_box", test_durable_release_bundle_black_box},
        {"terminal_approval_and_decision", test_terminal_approval_and_decision}};
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
