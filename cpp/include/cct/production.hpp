#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cct {

enum class Decision : std::uint8_t { Allow = 0, Deny = 1, Quarantine = 2, Abstain = 3 };
enum class RiskSeverity : std::uint8_t { Low = 0, Medium = 1, High = 2, Critical = 3 };
enum class DataClass : std::uint8_t { Public = 0, Licensed = 1, Private = 2, EvaluatorOnly = 3, Restricted = 4 };
enum class ApplicationKind : std::uint8_t { Extraction = 0, Classification = 1, GroundedAnswer = 2, CodeUnderstanding = 3 };

struct ProductUseCase {
    std::string id;
    std::string name;
    ApplicationKind kind = ApplicationKind::Extraction;
    std::vector<std::string> allowed_outputs;
    std::vector<std::string> denied_actions;
    bool human_review_required = true;
    std::string owner;
    std::string expiration;
};

struct ThreatControl {
    std::string threat_id;
    std::string description;
    RiskSeverity severity = RiskSeverity::Medium;
    std::string control_id;
    std::string test_id;
    std::string owner;
    std::string residual_risk;
    bool accepted = false;
};

struct DataPolicy {
    DataClass data_class = DataClass::Public;
    std::string license_or_consent;
    std::string jurisdiction;
    std::string privacy_class;
    std::string retention;
    bool training_allowed = false;
    bool evaluation_allowed = false;
    bool unresolved = false;
};

struct ExperimentIdentity {
    std::string experiment_id;
    std::string config_hash;
    std::string data_manifest_hash;
    std::string code_commit;
    std::string hardware;
    std::string software;
    std::uint64_t seed = 0;
    std::string status;
};

struct EvaluationSpec {
    std::string evaluation_id;
    std::string task_id;
    std::string split_id;
    std::vector<std::string> baselines;
    std::vector<std::string> metrics;
    std::vector<std::string> negative_controls;
    std::string evaluator_owner;
    bool evaluator_only = true;
};

struct ArtifactManifest {
    std::string stage_id;
    std::string commit;
    std::string configuration_hash;
    std::string data_manifest_hash;
    std::string environment_hash;
    std::string hardware;
    std::vector<std::string> files;
    std::vector<std::string> restricted_files;
    std::string status;
};

struct PolicyRequest {
    std::string tenant_id;
    std::string use_case_id;
    std::string operation;
    std::string resource;
    bool requests_external_action = false;
    bool requests_host_execution = false;
    bool requests_secret_access = false;
    bool requests_online_learning = false;
    bool requests_autonomous_self_modification = false;
};

struct PolicyResult {
    Decision decision = Decision::Deny;
    std::string rule_id;
    std::string reason;
    bool audited = false;
};

struct ApplicationFixture {
    std::string fixture_id;
    std::string use_case_id;
    ApplicationKind kind = ApplicationKind::Extraction;
    std::string tenant_id;
    std::string user_request;
    std::string source_text;
    std::string task_schema;
    std::string expected_behavior;
    std::vector<std::string> required_evidence;
    PolicyRequest policy_request;
    bool contains_prompt_injection = false;
    bool contains_conflict = false;
    bool contains_missing_evidence = false;
    bool contains_sensitive_data = false;
};

struct ApplicationResult {
    std::string fixture_id;
    Decision decision = Decision::Abstain;
    std::string output_contract;
    std::vector<std::string> citations;
    std::string uncertainty;
    bool policy_bypassed = false;
    bool evidence_bypassed = false;
    bool audited = false;
};

struct ReleaseRecord {
    std::string release_id;
    std::string stage_id;
    std::string commit;
    std::string configuration_hash;
    std::string model_scope;
    std::string data_scope;
    std::string user_scope;
    std::string allowed_actions;
    std::string denied_actions;
    std::string rollback_reference;
    std::string approver;
    std::string expiration;
    Decision decision = Decision::Deny;
};

struct AuditRecord {
    std::string event_type;
    std::string subject_id;
    std::string decision;
    std::string detail;
    bool restricted = false;
};

class ProductionRegistry {
public:
    void add_use_case(const ProductUseCase& use_case);
    void add_threat(const ThreatControl& threat);
    void add_data_policy(const DataPolicy& policy);
    void add_experiment(const ExperimentIdentity& experiment);
    void add_evaluation(const EvaluationSpec& evaluation);
    void add_artifact(const ArtifactManifest& artifact);
    void add_release(const ReleaseRecord& release);

    const ProductUseCase& use_case(const std::string& id) const;
    const std::vector<ProductUseCase>& use_cases() const noexcept;
    const std::vector<ThreatControl>& threats() const noexcept;
    const std::vector<DataPolicy>& data_policies() const noexcept;
    const std::vector<ExperimentIdentity>& experiments() const noexcept;
    const std::vector<EvaluationSpec>& evaluations() const noexcept;
    const std::vector<ArtifactManifest>& artifacts() const noexcept;
    const std::vector<ReleaseRecord>& releases() const noexcept;

    std::string serialize() const;
    static ProductionRegistry deserialize(const std::string& text);

private:
    std::vector<ProductUseCase> use_cases_;
    std::vector<ThreatControl> threats_;
    std::vector<DataPolicy> data_policies_;
    std::vector<ExperimentIdentity> experiments_;
    std::vector<EvaluationSpec> evaluations_;
    std::vector<ArtifactManifest> artifacts_;
    std::vector<ReleaseRecord> releases_;
};

class ProductionPolicy {
public:
    static PolicyResult evaluate(const PolicyRequest& request, const ProductUseCase& use_case);
};

class ApplicationReadiness {
public:
    static ApplicationResult evaluate(const ApplicationFixture& fixture, const ProductUseCase& use_case);
};

class ProductionAudit {
public:
    void append(const AuditRecord& record);
    const std::vector<AuditRecord>& records() const noexcept;
    std::string serialize() const;
    static ProductionAudit deserialize(const std::string& text);

private:
    std::vector<AuditRecord> records_;
};

}  // namespace cct
