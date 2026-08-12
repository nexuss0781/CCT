#include "cct/production.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

template <typename T, typename Predicate>
bool has_item(const std::vector<T>& values, Predicate predicate) {
    return std::find_if(values.begin(), values.end(), predicate) != values.end();
}

void require_new_id(const std::vector<std::string>& ids, const std::string& id) {
    require(!id.empty() && !contains(ids, id), "empty or duplicate registry id");
}

std::string quote(const std::string& value) {
    std::ostringstream output;
    output << std::quoted(value);
    return output.str();
}

std::string serialize_strings(const std::vector<std::string>& values) {
    std::ostringstream output;
    output << values.size();
    for (const auto& value : values) output << ' ' << quote(value);
    return output.str();
}

std::vector<std::string> deserialize_strings(std::istringstream& input) {
    std::size_t count = 0;
    input >> count;
    std::vector<std::string> values(count);
    for (auto& value : values) input >> std::quoted(value);
    require(static_cast<bool>(input), "invalid string vector serialization");
    return values;
}

}  // namespace

void ProductionRegistry::add_use_case(const ProductUseCase& use_case) {
    std::vector<std::string> ids;
    for (const auto& item : use_cases_) ids.push_back(item.id);
    require_new_id(ids, use_case.id);
    require(!use_case.owner.empty() && !use_case.expiration.empty(), "use case owner and expiration are required");
    require(!use_case.allowed_outputs.empty(), "use case must declare allowed outputs");
    use_cases_.push_back(use_case);
}

void ProductionRegistry::add_threat(const ThreatControl& threat) {
    std::vector<std::string> ids;
    for (const auto& item : threats_) ids.push_back(item.threat_id);
    require_new_id(ids, threat.threat_id);
    require(!threat.control_id.empty() && !threat.test_id.empty() && !threat.owner.empty(), "threat control, test, and owner are required");
    require(threat.severity < RiskSeverity::Critical || threat.accepted, "critical threat cannot be unaccepted");
    threats_.push_back(threat);
}

void ProductionRegistry::add_data_policy(const DataPolicy& policy) {
    require(!policy.license_or_consent.empty() && !policy.jurisdiction.empty() && !policy.privacy_class.empty() &&
                !policy.retention.empty(),
            "data policy metadata is incomplete");
    require(!policy.unresolved || (!policy.training_allowed && !policy.evaluation_allowed),
            "unresolved data cannot be allowed by policy");
    data_policies_.push_back(policy);
}

void ProductionRegistry::add_experiment(const ExperimentIdentity& experiment) {
    std::vector<std::string> ids;
    for (const auto& item : experiments_) ids.push_back(item.experiment_id);
    require_new_id(ids, experiment.experiment_id);
    require(!experiment.config_hash.empty() && !experiment.data_manifest_hash.empty() && !experiment.code_commit.empty() &&
                !experiment.hardware.empty() && !experiment.software.empty() && !experiment.status.empty(),
            "experiment identity is incomplete");
    experiments_.push_back(experiment);
}

void ProductionRegistry::add_evaluation(const EvaluationSpec& evaluation) {
    std::vector<std::string> ids;
    for (const auto& item : evaluations_) ids.push_back(item.evaluation_id);
    require_new_id(ids, evaluation.evaluation_id);
    require(!evaluation.task_id.empty() && !evaluation.split_id.empty() && !evaluation.evaluator_owner.empty() &&
                !evaluation.baselines.empty() && !evaluation.metrics.empty() && !evaluation.negative_controls.empty() &&
                evaluation.evaluator_only,
            "evaluation specification is incomplete or not evaluator-only");
    evaluations_.push_back(evaluation);
}

void ProductionRegistry::add_artifact(const ArtifactManifest& artifact) {
    require(!artifact.stage_id.empty() && !artifact.commit.empty() && !artifact.configuration_hash.empty() &&
                !artifact.data_manifest_hash.empty() && !artifact.environment_hash.empty() && !artifact.hardware.empty() &&
                !artifact.files.empty() && !artifact.status.empty(),
            "artifact manifest is incomplete");
    artifacts_.push_back(artifact);
}

void ProductionRegistry::add_release(const ReleaseRecord& release) {
    std::vector<std::string> ids;
    for (const auto& item : releases_) ids.push_back(item.release_id);
    require_new_id(ids, release.release_id);
    require(!release.stage_id.empty() && !release.commit.empty() && !release.configuration_hash.empty() &&
                !release.model_scope.empty() && !release.data_scope.empty() && !release.user_scope.empty() &&
                !release.allowed_actions.empty() && !release.denied_actions.empty() && !release.rollback_reference.empty() &&
                !release.approver.empty() && !release.expiration.empty(),
            "release record is incomplete");
    require(release.decision != Decision::Allow || !release.approver.empty(), "allowed release requires approver");
    releases_.push_back(release);
}

const ProductUseCase& ProductionRegistry::use_case(const std::string& id) const {
    const auto found = std::find_if(use_cases_.begin(), use_cases_.end(), [&](const auto& item) { return item.id == id; });
    require(found != use_cases_.end(), "use case not found");
    return *found;
}

const std::vector<ProductUseCase>& ProductionRegistry::use_cases() const noexcept { return use_cases_; }
const std::vector<ThreatControl>& ProductionRegistry::threats() const noexcept { return threats_; }
const std::vector<DataPolicy>& ProductionRegistry::data_policies() const noexcept { return data_policies_; }
const std::vector<ExperimentIdentity>& ProductionRegistry::experiments() const noexcept { return experiments_; }
const std::vector<EvaluationSpec>& ProductionRegistry::evaluations() const noexcept { return evaluations_; }
const std::vector<ArtifactManifest>& ProductionRegistry::artifacts() const noexcept { return artifacts_; }
const std::vector<ReleaseRecord>& ProductionRegistry::releases() const noexcept { return releases_; }

std::string ProductionRegistry::serialize() const {
    std::ostringstream output;
    output << "CCT_PRODUCTION_REGISTRY_V1\n";
    output << use_cases_.size() << '\n';
    for (const auto& item : use_cases_) {
        output << quote(item.id) << ' ' << quote(item.name) << ' ' << static_cast<unsigned int>(item.kind) << ' '
               << serialize_strings(item.allowed_outputs) << ' ' << serialize_strings(item.denied_actions) << ' '
               << item.human_review_required << ' ' << quote(item.owner) << ' ' << quote(item.expiration) << '\n';
    }
    output << threats_.size() << '\n';
    for (const auto& item : threats_) {
        output << quote(item.threat_id) << ' ' << quote(item.description) << ' ' << static_cast<unsigned int>(item.severity) << ' '
               << quote(item.control_id) << ' ' << quote(item.test_id) << ' ' << quote(item.owner) << ' '
               << quote(item.residual_risk) << ' ' << item.accepted << '\n';
    }
    output << data_policies_.size() << '\n';
    for (const auto& item : data_policies_) {
        output << static_cast<unsigned int>(item.data_class) << ' ' << quote(item.license_or_consent) << ' '
               << quote(item.jurisdiction) << ' ' << quote(item.privacy_class) << ' ' << quote(item.retention) << ' '
               << item.training_allowed << ' ' << item.evaluation_allowed << ' ' << item.unresolved << '\n';
    }
    output << experiments_.size() << '\n';
    for (const auto& item : experiments_) {
        output << quote(item.experiment_id) << ' ' << quote(item.config_hash) << ' ' << quote(item.data_manifest_hash) << ' '
               << quote(item.code_commit) << ' ' << quote(item.hardware) << ' ' << quote(item.software) << ' '
               << item.seed << ' ' << quote(item.status) << '\n';
    }
    output << evaluations_.size() << '\n';
    for (const auto& item : evaluations_) {
        output << quote(item.evaluation_id) << ' ' << quote(item.task_id) << ' ' << quote(item.split_id) << ' '
               << serialize_strings(item.baselines) << ' ' << serialize_strings(item.metrics) << ' '
               << serialize_strings(item.negative_controls) << ' ' << quote(item.evaluator_owner) << ' '
               << item.evaluator_only << '\n';
    }
    output << artifacts_.size() << '\n';
    for (const auto& item : artifacts_) {
        output << quote(item.stage_id) << ' ' << quote(item.commit) << ' ' << quote(item.configuration_hash) << ' '
               << quote(item.data_manifest_hash) << ' ' << quote(item.environment_hash) << ' ' << quote(item.hardware) << ' '
               << serialize_strings(item.files) << ' ' << serialize_strings(item.restricted_files) << ' ' << quote(item.status) << '\n';
    }
    output << releases_.size() << '\n';
    for (const auto& item : releases_) {
        output << quote(item.release_id) << ' ' << quote(item.stage_id) << ' ' << quote(item.commit) << ' '
               << quote(item.configuration_hash) << ' ' << quote(item.model_scope) << ' ' << quote(item.data_scope) << ' '
               << quote(item.user_scope) << ' ' << quote(item.allowed_actions) << ' ' << quote(item.denied_actions) << ' '
               << quote(item.rollback_reference) << ' ' << quote(item.approver) << ' ' << quote(item.expiration) << ' '
               << static_cast<unsigned int>(item.decision) << '\n';
    }
    return output.str();
}

ProductionRegistry ProductionRegistry::deserialize(const std::string& text) {
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_PRODUCTION_REGISTRY_V1", "invalid production registry header");
    ProductionRegistry registry;
    std::size_t count = 0;
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        ProductUseCase item;
        unsigned int kind = 0;
        input >> std::quoted(item.id) >> std::quoted(item.name) >> kind;
        item.kind = static_cast<ApplicationKind>(kind);
        item.allowed_outputs = deserialize_strings(input);
        item.denied_actions = deserialize_strings(input);
        input >> item.human_review_required >> std::quoted(item.owner) >> std::quoted(item.expiration);
        registry.add_use_case(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        ThreatControl item;
        unsigned int severity = 0;
        input >> std::quoted(item.threat_id) >> std::quoted(item.description) >> severity >> std::quoted(item.control_id) >>
            std::quoted(item.test_id) >> std::quoted(item.owner) >> std::quoted(item.residual_risk) >> item.accepted;
        item.severity = static_cast<RiskSeverity>(severity);
        registry.add_threat(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        DataPolicy item;
        unsigned int data_class = 0;
        input >> data_class >> std::quoted(item.license_or_consent) >> std::quoted(item.jurisdiction) >>
            std::quoted(item.privacy_class) >> std::quoted(item.retention) >> item.training_allowed >> item.evaluation_allowed >>
            item.unresolved;
        item.data_class = static_cast<DataClass>(data_class);
        registry.add_data_policy(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        ExperimentIdentity item;
        input >> std::quoted(item.experiment_id) >> std::quoted(item.config_hash) >> std::quoted(item.data_manifest_hash) >>
            std::quoted(item.code_commit) >> std::quoted(item.hardware) >> std::quoted(item.software) >> item.seed >>
            std::quoted(item.status);
        registry.add_experiment(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        EvaluationSpec item;
        input >> std::quoted(item.evaluation_id) >> std::quoted(item.task_id) >> std::quoted(item.split_id);
        item.baselines = deserialize_strings(input);
        item.metrics = deserialize_strings(input);
        item.negative_controls = deserialize_strings(input);
        input >> std::quoted(item.evaluator_owner) >> item.evaluator_only;
        registry.add_evaluation(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        ArtifactManifest item;
        input >> std::quoted(item.stage_id) >> std::quoted(item.commit) >> std::quoted(item.configuration_hash) >>
            std::quoted(item.data_manifest_hash) >> std::quoted(item.environment_hash) >> std::quoted(item.hardware);
        item.files = deserialize_strings(input);
        item.restricted_files = deserialize_strings(input);
        input >> std::quoted(item.status);
        registry.add_artifact(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        ReleaseRecord item;
        unsigned int decision = 0;
        input >> std::quoted(item.release_id) >> std::quoted(item.stage_id) >> std::quoted(item.commit) >>
            std::quoted(item.configuration_hash) >> std::quoted(item.model_scope) >> std::quoted(item.data_scope) >>
            std::quoted(item.user_scope) >> std::quoted(item.allowed_actions) >> std::quoted(item.denied_actions) >>
            std::quoted(item.rollback_reference) >> std::quoted(item.approver) >> std::quoted(item.expiration) >> decision;
        item.decision = static_cast<Decision>(decision);
        registry.add_release(item);
    }
    require(static_cast<bool>(input), "invalid production registry serialization");
    return registry;
}

PolicyResult ProductionPolicy::evaluate(const PolicyRequest& request, const ProductUseCase& use_case) {
    if (request.tenant_id.empty() || request.use_case_id != use_case.id) return {Decision::Deny, "POLICY_IDENTITY", "tenant or use-case identity is invalid", true};
    if (request.requests_external_action) return {Decision::Deny, "POLICY_EXTERNAL_ACTION", "external actions are not authorized in Stage 8", true};
    if (request.requests_host_execution) return {Decision::Deny, "POLICY_HOST_EXECUTION", "host execution is denied by default", true};
    if (request.requests_secret_access) return {Decision::Deny, "POLICY_SECRET_ACCESS", "secret access is denied by default", true};
    if (request.requests_online_learning) return {Decision::Deny, "POLICY_ONLINE_LEARNING", "online learning is not authorized", true};
    if (request.requests_autonomous_self_modification) return {Decision::Deny, "POLICY_SELF_MODIFICATION", "autonomous self-modification is not authorized", true};
    if (!contains(use_case.allowed_outputs, request.operation)) return {Decision::Deny, "POLICY_OUTPUT_SCOPE", "operation is outside declared use-case scope", true};
    if (contains(use_case.denied_actions, request.resource)) return {Decision::Deny, "POLICY_RESOURCE", "requested resource is denied", true};
    return {Decision::Allow, "POLICY_DECLARED_SCOPE", "operation is within declared bounded scope", true};
}

ApplicationResult ApplicationReadiness::evaluate(const ApplicationFixture& fixture, const ProductUseCase& use_case) {
    const auto policy = ProductionPolicy::evaluate(fixture.policy_request, use_case);
    ApplicationResult result{fixture.fixture_id, policy.decision, fixture.expected_behavior, {}, "", false, false, policy.audited};
    if (policy.decision == Decision::Deny) {
        result.uncertainty = policy.reason;
        result.policy_bypassed = false;
        return result;
    }
    if (fixture.contains_sensitive_data) {
        result.decision = Decision::Quarantine;
        result.uncertainty = "sensitive input requires governed review";
        return result;
    }
    if (fixture.contains_prompt_injection || fixture.contains_conflict || fixture.contains_missing_evidence) {
        result.decision = Decision::Abstain;
        result.uncertainty = fixture.contains_prompt_injection ? "source prompt injection isolated" :
                             (fixture.contains_conflict ? "conflicting evidence requires review" : "required evidence is missing");
        result.evidence_bypassed = false;
        return result;
    }
    result.decision = Decision::Allow;
    result.citations = fixture.required_evidence;
    result.uncertainty = "bounded application fixture; human review policy applies";
    return result;
}

void ProductionAudit::append(const AuditRecord& record) {
    require(!record.event_type.empty() && !record.subject_id.empty() && !record.decision.empty(), "audit record is incomplete");
    records_.push_back(record);
}

const std::vector<AuditRecord>& ProductionAudit::records() const noexcept { return records_; }

std::string ProductionAudit::serialize() const {
    std::ostringstream output;
    output << "CCT_PRODUCTION_AUDIT_V1\n" << records_.size() << '\n';
    for (const auto& record : records_) {
        output << std::quoted(record.event_type) << ' ' << std::quoted(record.subject_id) << ' '
               << std::quoted(record.decision) << ' ' << std::quoted(record.detail) << ' ' << record.restricted << '\n';
    }
    return output.str();
}

ProductionAudit ProductionAudit::deserialize(const std::string& text) {
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_PRODUCTION_AUDIT_V1", "invalid production audit header");
    std::size_t count = 0;
    input >> count;
    ProductionAudit audit;
    for (std::size_t index = 0; index < count; ++index) {
        AuditRecord record;
        input >> std::quoted(record.event_type) >> std::quoted(record.subject_id) >> std::quoted(record.decision) >>
            std::quoted(record.detail) >> record.restricted;
        audit.append(record);
    }
    require(static_cast<bool>(input), "invalid production audit serialization");
    return audit;
}

}  // namespace cct
