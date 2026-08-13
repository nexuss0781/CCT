#pragma once

#include "cct/inference.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class ReleasePhase : std::uint8_t {
    ArtifactFreeze = 0,
    OfflineReplay = 1,
    Shadow = 2,
    InternalPilot = 3,
    LimitedPilot = 4,
    Production = 5
};

enum class ReleaseDecision : std::uint8_t {
    PassBoundedProduction = 0,
    PassLimitedPilot = 1,
    Hold = 2,
    Fail = 3,
    Blocked = 4
};

enum class IncidentSeverity : std::uint8_t {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3
};

enum class FeedbackCategory : std::uint8_t {
    Quality = 0,
    Factuality = 1,
    Privacy = 2,
    UnsafeContent = 3,
    PolicyFailure = 4,
    Infrastructure = 5
};

enum class ReviewDecision : std::uint8_t {
    Approve = 0,
    Reject = 1,
    Escalate = 2
};

std::string release_phase_name(ReleasePhase phase);
std::string release_decision_name(ReleaseDecision decision);
std::string incident_severity_name(IncidentSeverity severity);
std::string feedback_category_name(FeedbackCategory category);
std::string review_decision_name(ReviewDecision decision);

class ReleaseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ReleaseScope {
    std::string release_id;
    std::string approved_model_version;
    std::string approved_model_artifact_path;
    std::string approved_tokenizer_version;
    std::string approved_tokenizer_artifact_path;
    std::vector<std::string> approved_adapter_versions;
    std::string approved_retrieval_index_version;
    std::vector<std::string> approved_task_schemas;
    std::vector<std::string> approved_user_groups;
    std::vector<std::string> approved_tenant_boundaries;
    std::vector<std::string> approved_data_classes;
    std::vector<std::string> approved_regions;
    std::vector<std::string> approved_tool_permissions;
    std::vector<std::string> human_approval_requirements;
    std::string service_level_objectives;
    std::string rollback_version;
    std::int64_t expiration_epoch_milliseconds = 0;
    std::string configuration_hash;
    std::string artifact_hash;
    std::string policy_hash;

    void validate() const;
    std::string immutable_identity() const;
    bool permits(const std::string& user_group, const std::string& tenant_id, const std::string& region,
                 const std::string& task_schema, const std::string& tool_permission) const;
};

struct PilotRequest {
    std::string request_id;
    std::string release_id;
    std::string user_id;
    std::string user_group;
    std::string tenant_id;
    std::string region;
    std::string task_schema;
    std::string tool_permission = "offline-deny";
    std::size_t input_tokens = 0;
    bool requests_external_action = false;
    bool requests_host_execution = false;
    bool requests_secret_access = false;
    std::int64_t now_epoch_milliseconds = 0;
};

struct PilotQuota {
    std::size_t maximum_requests = 0;
    std::size_t maximum_input_tokens = 0;
    std::size_t used_requests = 0;
    std::size_t used_input_tokens = 0;
};

struct PilotEnrollment {
    std::string enrollment_id;
    std::string release_id;
    std::string user_group;
    std::string tenant_id;
    PilotQuota quota;
    std::int64_t expires_epoch_milliseconds = 0;
    bool active = false;
};

struct PhaseDecisionRecord {
    ReleasePhase phase = ReleasePhase::ArtifactFreeze;
    ReleaseDecision decision = ReleaseDecision::Hold;
    std::string approver;
    std::string evidence_hash;
    std::string rationale;
    std::int64_t recorded_epoch_milliseconds = 0;
};

struct ShadowObservation {
    std::string observation_id;
    std::string request_digest;
    std::string control_output_digest;
    std::string candidate_output_digest;
    std::string task_schema;
    double quality_score = 0.0;
    double citation_correctness = 0.0;
    double unsupported_claim_rate = 1.0;
    double latency_milliseconds = 0.0;
    double cost_units = 0.0;
    bool side_effects = false;
    bool policy_isolated = false;
    bool tenant_isolated = false;
};

struct SafetyObservation {
    std::string suite_id;
    std::size_t cases = 0;
    std::size_t failures = 0;
    std::vector<std::string> categories;
    bool evaluator_only = true;
    bool passed = false;
};

struct HumanReviewRecord {
    std::string review_id;
    std::string request_id;
    std::string reviewer_id;
    std::string reviewer_role;
    std::string output_digest;
    std::string citation_summary;
    std::string uncertainty_summary;
    std::string trace_id;
    ReviewDecision decision = ReviewDecision::Escalate;
    std::string rationale;
    bool expert = false;
    bool high_impact = false;
};

struct FeedbackRecord {
    std::string feedback_id;
    std::string request_id;
    std::string tenant_id;
    FeedbackCategory category = FeedbackCategory::Quality;
    std::string reporter_group;
    std::string redacted_summary;
    std::int64_t recorded_epoch_milliseconds = 0;
    bool used_for_training = false;
};

struct SloObservation {
    std::string use_case;
    double quality_score = 0.0;
    double safety_score = 0.0;
    double availability = 0.0;
    double latency_p95_milliseconds = 0.0;
    double cost_per_request = 0.0;
    bool passed = false;
};

struct IncidentRecord {
    std::string incident_id;
    IncidentSeverity severity = IncidentSeverity::Low;
    std::string category;
    std::string owner;
    std::string scope;
    std::string timeline;
    std::string containment;
    std::string evidence_hash;
    std::string root_cause_hypothesis;
    std::string remediation;
    std::string resume_approval;
    bool contained = false;
    bool resolved = false;
};

struct DeletionRequest {
    std::string deletion_id;
    std::string target_type;
    std::string target_id;
    std::string tenant_id;
    std::string requester;
    std::string approval;
    std::vector<std::string> propagated_components;
    std::string evidence_hash;
    bool approved = false;
    bool applied = false;
};

struct DriftObservation {
    std::string drift_id;
    std::string metric;
    double baseline = 0.0;
    double current = 0.0;
    double threshold = 0.0;
    std::string owner;
    bool detected = false;
    bool acknowledged = false;
};

struct ApprovalSignature {
    std::string release_id;
    std::string approver_id;
    std::string role;
    std::string scope_hash;
    std::string decision;
    std::string timestamp;
    std::string signature_reference;
};

struct ReleaseEvaluation {
    ReleaseDecision decision = ReleaseDecision::Blocked;
    std::size_t passed_checks = 0;
    std::size_t total_checks = 0;
    std::vector<std::string> failed_checks;
    std::string reason;
};

struct ReleaseStatus {
    std::string release_id;
    ReleasePhase phase = ReleasePhase::ArtifactFreeze;
    bool frozen = false;
    bool safe_degraded = false;
    bool rollback_rehearsed = false;
    double rollback_milliseconds = 0.0;
    ReleaseDecision final_decision = ReleaseDecision::Blocked;
    std::size_t admitted_requests = 0;
    std::size_t denied_requests = 0;
};

class PilotReleaseController {
public:
    explicit PilotReleaseController(std::int64_t (*clock)() = nullptr);

    const ReleaseScope& scope() const;
    const ReleaseStatus& status() const noexcept;
    const std::vector<PilotEnrollment>& enrollments() const noexcept;
    const std::vector<PhaseDecisionRecord>& phase_decisions() const noexcept;
    const std::vector<ShadowObservation>& shadow_observations() const noexcept;
    const std::vector<SafetyObservation>& safety_observations() const noexcept;
    const std::vector<HumanReviewRecord>& human_reviews() const noexcept;
    const std::vector<FeedbackRecord>& feedback() const noexcept;
    const std::vector<SloObservation>& slo_observations() const noexcept;
    const std::vector<IncidentRecord>& incidents() const noexcept;
    const std::vector<DeletionRequest>& deletions() const noexcept;
    const std::vector<DriftObservation>& drift_observations() const noexcept;
    const std::vector<ApprovalSignature>& approvals() const noexcept;

    void freeze_artifacts(const ReleaseScope& scope);
    void add_enrollment(PilotEnrollment enrollment);
    bool admit(const PilotRequest& request);
    void advance_phase(ReleasePhase phase, const PhaseDecisionRecord& decision);
    void record_shadow(const ShadowObservation& observation);
    void record_safety(const SafetyObservation& observation);
    void record_review(const HumanReviewRecord& review);
    void record_feedback(const FeedbackRecord& feedback);
    void record_slo(const SloObservation& observation);
    void open_incident(const IncidentRecord& incident);
    void contain_incident(const std::string& incident_id, const std::string& containment, const std::string& owner);
    void resolve_incident(const std::string& incident_id, const std::string& remediation, const std::string& resume_approval);
    void submit_deletion(DeletionRequest request);
    void apply_deletion(const std::string& deletion_id, const std::vector<std::string>& components,
                        const std::string& evidence_hash);
    void record_drift(const DriftObservation& observation);
    void acknowledge_drift(const std::string& drift_id, const std::string& owner);
    void rehearse_rollback(double milliseconds, bool succeeded);
    void submit_approval(const ApprovalSignature& approval);
    ReleaseEvaluation evaluate_release() const;
    void mark_final_decision(ReleaseDecision decision);
    void enter_safe_degraded_mode(const std::string& reason);
    void activate_release(InferenceService& service) const;

    std::string serialize_release_manifest() const;
    std::string serialize_audit() const;
    void save_release_manifest(const std::string& path) const;
    void save_audit(const std::string& path) const;

private:
    std::int64_t (*clock_)() = nullptr;
    ReleaseScope scope_;
    ReleaseStatus status_;
    std::vector<PilotEnrollment> enrollments_;
    std::vector<PhaseDecisionRecord> phase_decisions_;
    std::vector<ShadowObservation> shadow_observations_;
    std::vector<SafetyObservation> safety_observations_;
    std::vector<HumanReviewRecord> human_reviews_;
    std::vector<FeedbackRecord> feedback_;
    std::vector<SloObservation> slo_observations_;
    std::vector<IncidentRecord> incidents_;
    std::vector<DeletionRequest> deletions_;
    std::vector<DriftObservation> drift_observations_;
    std::vector<ApprovalSignature> approvals_;

    std::int64_t now() const;
    void require_frozen() const;
    void require_phase(ReleasePhase phase) const;
    void require_unique(const std::string& id, const std::string& kind) const;
    bool has_open_high_incident() const;
    bool has_role_approval(const std::string& role) const;
};

}  // namespace cct
