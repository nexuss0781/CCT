#include "cct/release.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw ReleaseError(message);
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
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

std::string json_array(const std::vector<std::string>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        output << '"' << json_escape(values[index]) << '"';
    }
    output << ']';
    return output.str();
}

std::int64_t system_clock_milliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

bool passed_phase(const std::vector<PhaseDecisionRecord>& decisions, const ReleasePhase phase) {
    return std::find_if(decisions.begin(), decisions.end(), [&](const auto& decision) {
               return decision.phase == phase && decision.decision == ReleaseDecision::PassBoundedProduction;
           }) != decisions.end();
}

}  // namespace

std::string release_phase_name(const ReleasePhase phase) {
    if (phase == ReleasePhase::ArtifactFreeze) return "R0_artifact_freeze";
    if (phase == ReleasePhase::OfflineReplay) return "R1_offline_replay";
    if (phase == ReleasePhase::Shadow) return "R2_shadow";
    if (phase == ReleasePhase::InternalPilot) return "R3_internal_pilot";
    if (phase == ReleasePhase::LimitedPilot) return "R4_limited_pilot";
    return "R5_production";
}

std::string release_decision_name(const ReleaseDecision decision) {
    if (decision == ReleaseDecision::PassBoundedProduction) return "PASS — bounded production";
    if (decision == ReleaseDecision::PassLimitedPilot) return "PASS — limited pilot";
    if (decision == ReleaseDecision::Hold) return "HOLD";
    if (decision == ReleaseDecision::Fail) return "FAIL";
    return "BLOCKED";
}

std::string incident_severity_name(const IncidentSeverity severity) {
    if (severity == IncidentSeverity::Low) return "low";
    if (severity == IncidentSeverity::Medium) return "medium";
    if (severity == IncidentSeverity::High) return "high";
    return "critical";
}

std::string feedback_category_name(const FeedbackCategory category) {
    if (category == FeedbackCategory::Quality) return "quality";
    if (category == FeedbackCategory::Factuality) return "factuality";
    if (category == FeedbackCategory::Privacy) return "privacy";
    if (category == FeedbackCategory::UnsafeContent) return "unsafe_content";
    if (category == FeedbackCategory::PolicyFailure) return "policy_failure";
    return "infrastructure";
}

std::string review_decision_name(const ReviewDecision decision) {
    if (decision == ReviewDecision::Approve) return "approve";
    if (decision == ReviewDecision::Reject) return "reject";
    return "escalate";
}

void ReleaseScope::validate() const {
    require(!release_id.empty() && !approved_model_version.empty() && !approved_tokenizer_version.empty() &&
                !approved_retrieval_index_version.empty() && !service_level_objectives.empty() && !rollback_version.empty() &&
                !configuration_hash.empty() && !artifact_hash.empty() && !policy_hash.empty(),
            "release scope identity is incomplete");
    require(!approved_adapter_versions.empty() && !approved_task_schemas.empty() && !approved_user_groups.empty() &&
                !approved_tenant_boundaries.empty() && !approved_data_classes.empty() && !approved_regions.empty() &&
                !approved_tool_permissions.empty() && !human_approval_requirements.empty(),
            "release scope allowlists and approval requirements are incomplete");
    require(expiration_epoch_milliseconds > 0, "release expiration is required");
    require(contains(approved_tool_permissions, "offline-deny"), "offline-deny must be explicitly approved");
    require(!contains(approved_tool_permissions, "external-action") && !contains(approved_tool_permissions, "host-execution") &&
                !contains(approved_tool_permissions, "secret-access"),
            "release scope cannot authorize external, host, or secret actions");
}

std::string ReleaseScope::immutable_identity() const {
    const auto material = release_id + "|" + approved_model_version + "|" + approved_tokenizer_version + "|" +
                          json_array(approved_adapter_versions) + "|" + approved_retrieval_index_version + "|" +
                          json_array(approved_task_schemas) + "|" + json_array(approved_user_groups) + "|" +
                          json_array(approved_tenant_boundaries) + "|" + json_array(approved_data_classes) + "|" +
                          json_array(approved_regions) + "|" + json_array(approved_tool_permissions) + "|" +
                          json_array(human_approval_requirements) + "|" + service_level_objectives + "|" + rollback_version + "|" +
                          std::to_string(expiration_epoch_milliseconds) + "|" + configuration_hash + "|" + artifact_hash + "|" + policy_hash;
    return GovernedCorpus::content_sha256(material);
}

bool ReleaseScope::permits(const std::string& user_group, const std::string& tenant_id, const std::string& region,
                           const std::string& task_schema, const std::string& tool_permission) const {
    return contains(approved_user_groups, user_group) && contains(approved_tenant_boundaries, tenant_id) &&
           contains(approved_regions, region) && contains(approved_task_schemas, task_schema) &&
           contains(approved_tool_permissions, tool_permission);
}

PilotReleaseController::PilotReleaseController(std::int64_t (*clock)()) : clock_(clock) {}

const ReleaseScope& PilotReleaseController::scope() const { require(status_.frozen, "release scope has not been frozen"); return scope_; }
const ReleaseStatus& PilotReleaseController::status() const noexcept { return status_; }
const std::vector<PilotEnrollment>& PilotReleaseController::enrollments() const noexcept { return enrollments_; }
const std::vector<PhaseDecisionRecord>& PilotReleaseController::phase_decisions() const noexcept { return phase_decisions_; }
const std::vector<ShadowObservation>& PilotReleaseController::shadow_observations() const noexcept { return shadow_observations_; }
const std::vector<SafetyObservation>& PilotReleaseController::safety_observations() const noexcept { return safety_observations_; }
const std::vector<HumanReviewRecord>& PilotReleaseController::human_reviews() const noexcept { return human_reviews_; }
const std::vector<FeedbackRecord>& PilotReleaseController::feedback() const noexcept { return feedback_; }
const std::vector<SloObservation>& PilotReleaseController::slo_observations() const noexcept { return slo_observations_; }
const std::vector<IncidentRecord>& PilotReleaseController::incidents() const noexcept { return incidents_; }
const std::vector<DeletionRequest>& PilotReleaseController::deletions() const noexcept { return deletions_; }
const std::vector<DriftObservation>& PilotReleaseController::drift_observations() const noexcept { return drift_observations_; }
const std::vector<ApprovalSignature>& PilotReleaseController::approvals() const noexcept { return approvals_; }

std::int64_t PilotReleaseController::now() const { return clock_ == nullptr ? system_clock_milliseconds() : clock_(); }

void PilotReleaseController::require_unique(const std::string& id, const std::string& kind) const {
    require(!id.empty(), kind + " identifier is required");
    if (kind == "enrollment") require(std::find_if(enrollments_.begin(), enrollments_.end(), [&](const auto& item) { return item.enrollment_id == id; }) == enrollments_.end(), "duplicate enrollment");
    else if (kind == "shadow") require(std::find_if(shadow_observations_.begin(), shadow_observations_.end(), [&](const auto& item) { return item.observation_id == id; }) == shadow_observations_.end(), "duplicate shadow observation");
    else if (kind == "review") require(std::find_if(human_reviews_.begin(), human_reviews_.end(), [&](const auto& item) { return item.review_id == id; }) == human_reviews_.end(), "duplicate human review");
    else if (kind == "feedback") require(std::find_if(feedback_.begin(), feedback_.end(), [&](const auto& item) { return item.feedback_id == id; }) == feedback_.end(), "duplicate feedback");
    else if (kind == "incident") require(std::find_if(incidents_.begin(), incidents_.end(), [&](const auto& item) { return item.incident_id == id; }) == incidents_.end(), "duplicate incident");
    else if (kind == "deletion") require(std::find_if(deletions_.begin(), deletions_.end(), [&](const auto& item) { return item.deletion_id == id; }) == deletions_.end(), "duplicate deletion");
    else if (kind == "drift") require(std::find_if(drift_observations_.begin(), drift_observations_.end(), [&](const auto& item) { return item.drift_id == id; }) == drift_observations_.end(), "duplicate drift observation");
}

void PilotReleaseController::require_frozen() const { require(status_.frozen, "release artifacts are not frozen"); }

void PilotReleaseController::require_phase(const ReleasePhase phase) const {
    require(status_.phase >= phase, "release phase has not been reached");
}

void PilotReleaseController::freeze_artifacts(const ReleaseScope& scope) {
    require(!status_.frozen, "release artifacts are already frozen");
    scope.validate();
    require(scope.expiration_epoch_milliseconds > now(), "release scope is already expired");
    scope_ = scope;
    status_.release_id = scope.release_id;
    status_.phase = ReleasePhase::ArtifactFreeze;
    status_.frozen = true;
}

void PilotReleaseController::add_enrollment(PilotEnrollment enrollment) {
    require_frozen();
    require_unique(enrollment.enrollment_id, "enrollment");
    require(enrollment.release_id == scope_.release_id && enrollment.active, "enrollment release identity or active flag is invalid");
    require(contains(scope_.approved_user_groups, enrollment.user_group) && contains(scope_.approved_tenant_boundaries, enrollment.tenant_id),
            "enrollment is outside approved user or tenant scope");
    require(enrollment.quota.maximum_requests > 0U && enrollment.quota.maximum_input_tokens > 0U &&
                enrollment.expires_epoch_milliseconds > now() && enrollment.expires_epoch_milliseconds <= scope_.expiration_epoch_milliseconds,
            "enrollment quota or expiration is invalid");
    enrollments_.push_back(std::move(enrollment));
}

bool PilotReleaseController::admit(const PilotRequest& request) {
    require_frozen();
    if (status_.safe_degraded || status_.phase < ReleasePhase::InternalPilot || request.release_id != scope_.release_id ||
        request.now_epoch_milliseconds > scope_.expiration_epoch_milliseconds || request.requests_external_action ||
        request.requests_host_execution || request.requests_secret_access || !scope_.permits(request.user_group, request.tenant_id, request.region,
                                                                                             request.task_schema, request.tool_permission)) {
        ++status_.denied_requests;
        return false;
    }
    const auto enrollment = std::find_if(enrollments_.begin(), enrollments_.end(), [&](auto& item) {
        return item.active && item.release_id == request.release_id && item.user_group == request.user_group && item.tenant_id == request.tenant_id &&
               request.now_epoch_milliseconds <= item.expires_epoch_milliseconds;
    });
    if (enrollment == enrollments_.end() || enrollment->quota.used_requests >= enrollment->quota.maximum_requests ||
        enrollment->quota.used_input_tokens + request.input_tokens > enrollment->quota.maximum_input_tokens) {
        ++status_.denied_requests;
        return false;
    }
    auto& mutable_enrollment = const_cast<PilotEnrollment&>(*enrollment);
    ++mutable_enrollment.quota.used_requests;
    mutable_enrollment.quota.used_input_tokens += request.input_tokens;
    ++status_.admitted_requests;
    return true;
}

void PilotReleaseController::advance_phase(const ReleasePhase phase, const PhaseDecisionRecord& decision) {
    require_frozen();
    require(decision.phase == phase && decision.decision == ReleaseDecision::PassBoundedProduction && !decision.evidence_hash.empty() &&
                !decision.rationale.empty() && !decision.approver.empty(),
            "phase decision is incomplete or not a passing decision");
    require(static_cast<unsigned int>(phase) == static_cast<unsigned int>(status_.phase) + 1U, "release phase transition is not sequential");
    phase_decisions_.push_back(decision);
    status_.phase = phase;
}

void PilotReleaseController::record_shadow(const ShadowObservation& observation) {
    require_frozen();
    require_phase(ReleasePhase::Shadow);
    require_unique(observation.observation_id, "shadow");
    require(!observation.request_digest.empty() && !observation.control_output_digest.empty() && !observation.candidate_output_digest.empty() &&
                !observation.task_schema.empty() && observation.quality_score >= 0.0 && observation.quality_score <= 1.0 &&
                observation.citation_correctness >= 0.0 && observation.citation_correctness <= 1.0 && observation.unsupported_claim_rate >= 0.0 &&
                observation.unsupported_claim_rate <= 1.0 && observation.latency_milliseconds >= 0.0 && observation.cost_units >= 0.0 &&
                !observation.side_effects && observation.policy_isolated && observation.tenant_isolated,
            "shadow observation is invalid or side effects were detected");
    shadow_observations_.push_back(observation);
}

void PilotReleaseController::record_safety(const SafetyObservation& observation) {
    require_frozen();
    require_phase(ReleasePhase::Shadow);
    require(!observation.suite_id.empty() && observation.cases > 0U && observation.failures <= observation.cases &&
                !observation.categories.empty() && observation.evaluator_only && observation.passed == (observation.failures == 0U),
            "safety observation is incomplete or inconsistent");
    safety_observations_.push_back(observation);
}

void PilotReleaseController::record_review(const HumanReviewRecord& review) {
    require_frozen();
    require_phase(ReleasePhase::InternalPilot);
    require_unique(review.review_id, "review");
    require(!review.request_id.empty() && !review.reviewer_id.empty() && !review.reviewer_role.empty() && !review.output_digest.empty() &&
                !review.citation_summary.empty() && !review.uncertainty_summary.empty() && !review.trace_id.empty() && !review.rationale.empty(),
            "human review is incomplete");
    if (review.high_impact && review.decision == ReviewDecision::Approve) require(review.expert, "high-impact approval requires an expert reviewer");
    human_reviews_.push_back(review);
}

void PilotReleaseController::record_feedback(const FeedbackRecord& feedback) {
    require_frozen();
    require_unique(feedback.feedback_id, "feedback");
    require(!feedback.request_id.empty() && !feedback.tenant_id.empty() && !feedback.reporter_group.empty() &&
                !feedback.redacted_summary.empty() && !feedback.used_for_training,
            "feedback is incomplete or was incorrectly used for training");
    feedback_.push_back(feedback);
}

void PilotReleaseController::record_slo(const SloObservation& observation) {
    require_frozen();
    require(!observation.use_case.empty() && observation.quality_score >= 0.0 && observation.quality_score <= 1.0 &&
                observation.safety_score >= 0.0 && observation.safety_score <= 1.0 && observation.availability >= 0.0 &&
                observation.availability <= 1.0 && observation.latency_p95_milliseconds >= 0.0 && observation.cost_per_request >= 0.0,
            "SLO observation is invalid");
    require(observation.passed == (observation.quality_score >= 0.90 && observation.safety_score >= 0.99 &&
                                   observation.availability >= 0.995 && observation.latency_p95_milliseconds <= 1500.0),
            "SLO pass flag is inconsistent with declared thresholds");
    slo_observations_.push_back(observation);
}

void PilotReleaseController::open_incident(const IncidentRecord& incident) {
    require_frozen();
    require_unique(incident.incident_id, "incident");
    require(!incident.category.empty() && !incident.owner.empty() && !incident.scope.empty() && !incident.timeline.empty() &&
                !incident.evidence_hash.empty() && !incident.root_cause_hypothesis.empty() && !incident.containment.empty() && !incident.remediation.empty(),
            "incident record is incomplete");
    IncidentRecord copy = incident;
    copy.contained = false;
    copy.resolved = false;
    incidents_.push_back(std::move(copy));
    if (incident.severity >= IncidentSeverity::High) status_.safe_degraded = true;
}

void PilotReleaseController::contain_incident(const std::string& incident_id, const std::string& containment, const std::string& owner) {
    const auto found = std::find_if(incidents_.begin(), incidents_.end(), [&](const auto& item) { return item.incident_id == incident_id; });
    require(found != incidents_.end() && !found->resolved && !containment.empty() && !owner.empty(), "incident cannot be contained");
    found->containment = containment;
    found->owner = owner;
    found->contained = true;
}

void PilotReleaseController::resolve_incident(const std::string& incident_id, const std::string& remediation, const std::string& resume_approval) {
    const auto found = std::find_if(incidents_.begin(), incidents_.end(), [&](const auto& item) { return item.incident_id == incident_id; });
    require(found != incidents_.end() && found->contained && !found->resolved && !remediation.empty() && !resume_approval.empty(),
            "incident cannot be resolved without containment, remediation, and resume approval");
    found->remediation = remediation;
    found->resume_approval = resume_approval;
    found->resolved = true;
    if (!has_open_high_incident()) status_.safe_degraded = false;
}

void PilotReleaseController::submit_deletion(DeletionRequest request) {
    require_frozen();
    require_unique(request.deletion_id, "deletion");
    require(!request.target_type.empty() && !request.target_id.empty() && !request.tenant_id.empty() && !request.requester.empty() &&
                !request.approval.empty() && request.approved && !request.applied,
            "deletion request is not approved or incomplete");
    deletions_.push_back(std::move(request));
}

void PilotReleaseController::apply_deletion(const std::string& deletion_id, const std::vector<std::string>& components,
                                            const std::string& evidence_hash) {
    const auto found = std::find_if(deletions_.begin(), deletions_.end(), [&](const auto& item) { return item.deletion_id == deletion_id; });
    require(found != deletions_.end() && found->approved && !found->applied && components.size() >= 3U && !evidence_hash.empty(),
            "deletion cannot be applied without approval, propagation, and evidence");
    require(contains(components, "service-state") && contains(components, "response-cache") && contains(components, "derived-artifacts"),
            "deletion propagation omitted service state, cache, or derived artifacts");
    found->propagated_components = components;
    found->evidence_hash = evidence_hash;
    found->applied = true;
}

void PilotReleaseController::record_drift(const DriftObservation& observation) {
    require_frozen();
    require_unique(observation.drift_id, "drift");
    require(!observation.metric.empty() && observation.baseline >= 0.0 && observation.current >= 0.0 && observation.threshold > 0.0 &&
                !observation.owner.empty() && observation.detected == (std::abs(observation.current - observation.baseline) > observation.threshold),
            "drift observation is incomplete or detection flag is inconsistent");
    drift_observations_.push_back(observation);
}

void PilotReleaseController::acknowledge_drift(const std::string& drift_id, const std::string& owner) {
    const auto found = std::find_if(drift_observations_.begin(), drift_observations_.end(), [&](const auto& item) { return item.drift_id == drift_id; });
    require(found != drift_observations_.end() && found->detected && !owner.empty(), "drift cannot be acknowledged");
    found->owner = owner;
    found->acknowledged = true;
}

void PilotReleaseController::rehearse_rollback(const double milliseconds, const bool succeeded) {
    require_frozen();
    require(!scope_.rollback_version.empty() && milliseconds >= 0.0, "rollback rehearsal identity is incomplete");
    status_.rollback_rehearsed = succeeded && milliseconds <= 600000.0;
    status_.rollback_milliseconds = milliseconds;
    if (!succeeded) status_.safe_degraded = true;
}

void PilotReleaseController::submit_approval(const ApprovalSignature& approval) {
    require_frozen();
    require(approval.release_id == scope_.release_id && !approval.approver_id.empty() && !approval.role.empty() &&
                !approval.scope_hash.empty() && approval.scope_hash == scope_.immutable_identity() && approval.decision == "approve" &&
                !approval.timestamp.empty() && !approval.signature_reference.empty(),
            "approval signature is incomplete or scope does not match");
    require(std::find_if(approvals_.begin(), approvals_.end(), [&](const auto& item) { return item.role == approval.role; }) == approvals_.end(),
            "duplicate approval role");
    approvals_.push_back(approval);
}

bool PilotReleaseController::has_open_high_incident() const {
    return std::find_if(incidents_.begin(), incidents_.end(), [](const auto& incident) {
               return incident.severity >= IncidentSeverity::High && !incident.resolved;
           }) != incidents_.end();
}

bool PilotReleaseController::has_role_approval(const std::string& role) const {
    return std::find_if(approvals_.begin(), approvals_.end(), [&](const auto& approval) { return approval.role == role && approval.decision == "approve"; }) != approvals_.end();
}

ReleaseEvaluation PilotReleaseController::evaluate_release() const {
    require_frozen();
    ReleaseEvaluation evaluation;
    const auto check = [&](const bool passed, const std::string& name) {
        ++evaluation.total_checks;
        if (passed) ++evaluation.passed_checks;
        else evaluation.failed_checks.push_back(name);
    };
    const bool phase_r1 = passed_phase(phase_decisions_, ReleasePhase::OfflineReplay);
    const bool phase_r2 = passed_phase(phase_decisions_, ReleasePhase::Shadow);
    const bool phase_r3 = passed_phase(phase_decisions_, ReleasePhase::InternalPilot);
    const bool phase_r4 = passed_phase(phase_decisions_, ReleasePhase::LimitedPilot);
    const bool phase_r5 = passed_phase(phase_decisions_, ReleasePhase::Production);
    check(status_.frozen && !scope_.immutable_identity().empty(), "artifact_freeze");
    check(phase_r1, "offline_parity");
    check(phase_r2 && !shadow_observations_.empty() && std::all_of(shadow_observations_.begin(), shadow_observations_.end(), [](const auto& item) {
              return item.quality_score >= 0.90 && item.citation_correctness >= 0.95 && item.unsupported_claim_rate <= 0.05 && !item.side_effects && item.policy_isolated && item.tenant_isolated;
          }), "shadow");
    check(!shadow_observations_.empty() && std::all_of(shadow_observations_.begin(), shadow_observations_.end(), [](const auto& item) { return item.quality_score >= 0.90 && item.citation_correctness >= 0.95; }), "quality");
    check(!safety_observations_.empty() && std::all_of(safety_observations_.begin(), safety_observations_.end(), [](const auto& item) { return item.passed && item.failures == 0U && item.evaluator_only; }), "safety");
    check(phase_r3 && phase_r4 && !human_reviews_.empty() && std::any_of(human_reviews_.begin(), human_reviews_.end(), [](const auto& item) { return item.expert; }), "human_oversight");
    check(!slo_observations_.empty() && std::all_of(slo_observations_.begin(), slo_observations_.end(), [](const auto& item) { return item.passed; }), "slo");
    check(!enrollments_.empty() && std::all_of(enrollments_.begin(), enrollments_.end(), [&](const auto& item) { return scope_.permits(item.user_group, item.tenant_id, scope_.approved_regions.front(), scope_.approved_task_schemas.front(), "offline-deny"); }), "isolation");
    check(status_.rollback_rehearsed && status_.rollback_milliseconds <= 600000.0, "rollback");
    check(!incidents_.empty() && !has_open_high_incident() && std::all_of(incidents_.begin(), incidents_.end(), [](const auto& item) { return item.contained && item.resolved && !item.resume_approval.empty(); }), "incident_response");
    check(!deletions_.empty() && std::all_of(deletions_.begin(), deletions_.end(), [](const auto& item) { return item.approved && item.applied && item.propagated_components.size() >= 3U; }), "deletion");
    check(!drift_observations_.empty() && std::any_of(drift_observations_.begin(), drift_observations_.end(), [](const auto& item) { return item.detected && item.acknowledged; }), "drift");
    check(phase_r5, "regression");
    check(has_role_approval("technical") && has_role_approval("security") && has_role_approval("product") && has_role_approval("governance"), "approval");
    if (evaluation.passed_checks == evaluation.total_checks) {
        evaluation.decision = ReleaseDecision::PassBoundedProduction;
        evaluation.reason = "all declared bounded-release checks passed";
    } else if (phase_r4 && evaluation.passed_checks >= evaluation.total_checks - 3U) {
        evaluation.decision = ReleaseDecision::PassLimitedPilot;
        evaluation.reason = "limited-pilot evidence is incomplete for bounded production";
    } else if (has_open_high_incident() || !status_.rollback_rehearsed) {
        evaluation.decision = ReleaseDecision::Blocked;
        evaluation.reason = "critical incident or rollback readiness is unresolved";
    } else {
        evaluation.decision = ReleaseDecision::Hold;
        evaluation.reason = "release evidence requires remediation";
    }
    return evaluation;
}

void PilotReleaseController::mark_final_decision(const ReleaseDecision decision) {
    const auto evaluation = evaluate_release();
    require(decision == evaluation.decision, "final release decision does not match gate evaluation");
    require(decision == ReleaseDecision::PassBoundedProduction || decision == ReleaseDecision::PassLimitedPilot,
            "release cannot be finalized with a non-passing decision");
    if (decision == ReleaseDecision::PassBoundedProduction) {
        require(status_.phase == ReleasePhase::Production && status_.rollback_rehearsed && !has_open_high_incident(),
                "bounded production requires production phase, rollback, and no critical incident");
    }
    status_.final_decision = decision;
}

void PilotReleaseController::enter_safe_degraded_mode(const std::string& reason) {
    require(!reason.empty(), "safe degraded reason is required");
    status_.safe_degraded = true;
}

std::string PilotReleaseController::serialize_release_manifest() const {
    require_frozen();
    std::ostringstream output;
    output << "{\"release_id\":\"" << json_escape(scope_.release_id) << "\",\"approved_model_version\":\"" << json_escape(scope_.approved_model_version)
           << "\",\"approved_tokenizer_version\":\"" << json_escape(scope_.approved_tokenizer_version) << "\",\"approved_adapter_versions\":" << json_array(scope_.approved_adapter_versions)
           << ",\"approved_retrieval_index_version\":\"" << json_escape(scope_.approved_retrieval_index_version) << "\",\"approved_task_schemas\":" << json_array(scope_.approved_task_schemas)
           << ",\"approved_user_groups\":" << json_array(scope_.approved_user_groups) << ",\"approved_tenant_boundaries\":" << json_array(scope_.approved_tenant_boundaries)
           << ",\"approved_data_classes\":" << json_array(scope_.approved_data_classes) << ",\"approved_regions\":" << json_array(scope_.approved_regions)
           << ",\"approved_tool_permissions\":" << json_array(scope_.approved_tool_permissions) << ",\"human_approval_requirements\":" << json_array(scope_.human_approval_requirements)
           << ",\"service_level_objectives\":\"" << json_escape(scope_.service_level_objectives) << "\",\"rollback_version\":\"" << json_escape(scope_.rollback_version)
           << "\",\"expiration_epoch_milliseconds\":" << scope_.expiration_epoch_milliseconds << ",\"configuration_hash\":\"" << json_escape(scope_.configuration_hash)
           << "\",\"artifact_hash\":\"" << json_escape(scope_.artifact_hash) << "\",\"policy_hash\":\"" << json_escape(scope_.policy_hash)
           << "\",\"scope_hash\":\"" << scope_.immutable_identity() << "\",\"phase\":\"" << release_phase_name(status_.phase)
           << "\",\"final_decision\":\"" << release_decision_name(status_.final_decision) << "\",\"public_launch_authorized\":false,\"external_actions_authorized\":false}";
    return output.str();
}

std::string PilotReleaseController::serialize_audit() const {
    std::ostringstream output;
    output << "{\"phase_decisions\":" << phase_decisions_.size() << ",\"shadow_observations\":" << shadow_observations_.size()
           << ",\"safety_observations\":" << safety_observations_.size() << ",\"human_reviews\":" << human_reviews_.size()
           << ",\"feedback_records\":" << feedback_.size() << ",\"slo_observations\":" << slo_observations_.size()
           << ",\"incidents\":" << incidents_.size() << ",\"deletions\":" << deletions_.size() << ",\"drift_observations\":" << drift_observations_.size()
           << ",\"approval_signatures\":" << approvals_.size() << ",\"safe_degraded\":" << (status_.safe_degraded ? "true" : "false") << "}";
    return output.str();
}

}  // namespace cct
