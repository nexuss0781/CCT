#include "cct/production.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ProductUseCase extraction_case() {
    return {"invoice_extraction", "Invoice field extraction", ApplicationKind::Extraction,
            {"extract"}, {"submit_payment", "send_email"}, true, "finance-owner", "2027-12-31"};
}

ProductUseCase answer_case() {
    return {"policy_answer", "Grounded internal policy answer", ApplicationKind::GroundedAnswer,
            {"answer"}, {"approve_request"}, true, "policy-owner", "2027-12-31"};
}

ProductionRegistry populated_registry() {
    ProductionRegistry registry;
    registry.add_use_case(extraction_case());
    registry.add_use_case(answer_case());
    registry.add_threat({"prompt_injection", "Instruction-like source text", RiskSeverity::High, "isolate_source", "app_injection", "safety-owner", "bounded abstention", false});
    registry.add_threat({"secret_exposure", "Request attempts secret access", RiskSeverity::Critical, "deny_secret", "policy_secret", "security-owner", "none", true});
    registry.add_data_policy({DataClass::Public, "public-domain", "global", "none", "retain-release", true, true, false});
    registry.add_data_policy({DataClass::Restricted, "unresolved", "unknown", "high-risk", "quarantine", false, false, true});
    registry.add_experiment({"stage8-foundation", "cfg-hash", "data-hash", "commit-sha", "native-cpu", "gcc-cpp20", 808, "planned"});
    registry.add_evaluation({"eval-app-fixtures", "stage8_application", "evaluator-only-v1", {"dense-transformer", "gru", "cct"}, {"policy_accuracy", "audit_completeness", "abstention_quality"}, {"prompt_injection", "missing_evidence", "conflicting_evidence"}, "eval-owner", true});
    registry.add_artifact({"8", "commit-sha", "cfg-hash", "data-hash", "env-hash", "native-cpu", {"gate.json", "checks.json", "report.md"}, {"evaluator_truth.json"}, "planned"});
    registry.add_release({"stage8-plan", "8", "commit-sha", "cfg-hash", "governance-only", "declared-fixtures", "engineering-team", "none", "external-action,host-execution,online-learning", "not-applicable", "roadmap-owner", "2027-12-31", Decision::Deny});
    return registry;
}

void test_registry_roundtrip_and_contracts() {
    const auto registry = populated_registry();
    const auto restored = ProductionRegistry::deserialize(registry.serialize());
    require(restored.use_cases().size() == 2 && restored.threats().size() == 2 && restored.data_policies().size() == 2 &&
                restored.experiments().size() == 1 && restored.evaluations().size() == 1 && restored.artifacts().size() == 1 &&
                restored.releases().size() == 1,
            "production registry round-trip changed collection sizes");
    require(restored.use_case("invoice_extraction").human_review_required &&
                restored.evaluations().front().evaluator_only && restored.data_policies().back().unresolved,
            "registry round-trip lost governance fields");
}

void test_policy_matrix() {
    const auto use_case = extraction_case();
    const PolicyRequest safe{"tenant-a", use_case.id, "extract", "document", false, false, false, false, false};
    require(ProductionPolicy::evaluate(safe, use_case).decision == Decision::Allow, "declared extraction was denied");
    const std::vector<PolicyRequest> denied{
        {"tenant-a", use_case.id, "extract", "document", true, false, false, false, false},
        {"tenant-a", use_case.id, "extract", "document", false, true, false, false, false},
        {"tenant-a", use_case.id, "extract", "document", false, false, true, false, false},
        {"tenant-a", use_case.id, "extract", "document", false, false, false, true, false},
        {"tenant-a", use_case.id, "extract", "document", false, false, false, false, true},
        {"tenant-a", use_case.id, "classify", "document", false, false, false, false, false},
        {"tenant-a", use_case.id, "extract", "send_email", false, false, false, false, false},
    };
    for (const auto& request : denied) {
        const auto result = ProductionPolicy::evaluate(request, use_case);
        require(result.decision == Decision::Deny && result.audited && !result.rule_id.empty(), "unsafe policy request was not denied and audited");
    }
}

ApplicationFixture fixture(const std::string& id, const ProductUseCase& use_case) {
    ApplicationFixture result;
    result.fixture_id = id;
    result.use_case_id = use_case.id;
    result.kind = use_case.kind;
    result.tenant_id = "tenant-a";
    result.user_request = "extract fields";
    result.source_text = "invoice total is 12.50";
    result.task_schema = "{total:number}";
    result.expected_behavior = "bounded-extract";
    result.required_evidence = {"doc-1#total"};
    result.policy_request = {"tenant-a", use_case.id, "extract", "document", false, false, false, false, false};
    return result;
}

void test_realistic_application_fixtures() {
    const auto use_case = extraction_case();
    auto normal = fixture("normal", use_case);
    const auto normal_result = ApplicationReadiness::evaluate(normal, use_case);
    require(normal_result.decision == Decision::Allow && normal_result.citations.size() == 1 && normal_result.audited &&
                !normal_result.policy_bypassed && !normal_result.evidence_bypassed,
            "normal application fixture did not produce grounded bounded output");
    auto injection = fixture("injection", use_case);
    injection.contains_prompt_injection = true;
    const auto injection_result = ApplicationReadiness::evaluate(injection, use_case);
    require(injection_result.decision == Decision::Abstain && injection_result.uncertainty.find("injection") != std::string::npos,
            "prompt-injection fixture did not abstain");
    auto conflict = fixture("conflict", use_case);
    conflict.contains_conflict = true;
    require(ApplicationReadiness::evaluate(conflict, use_case).decision == Decision::Abstain, "conflict fixture did not abstain");
    auto missing = fixture("missing", use_case);
    missing.contains_missing_evidence = true;
    require(ApplicationReadiness::evaluate(missing, use_case).decision == Decision::Abstain, "missing-evidence fixture did not abstain");
    auto sensitive = fixture("sensitive", use_case);
    sensitive.contains_sensitive_data = true;
    require(ApplicationReadiness::evaluate(sensitive, use_case).decision == Decision::Quarantine, "sensitive fixture did not quarantine");
    auto external = fixture("external", use_case);
    external.policy_request.requests_external_action = true;
    require(ApplicationReadiness::evaluate(external, use_case).decision == Decision::Deny, "external action fixture did not deny");
}

void test_audit_and_release_boundaries() {
    ProductionAudit audit;
    audit.append({"policy_decision", "external", "Deny", "external action denied", false});
    audit.append({"application_fixture", "injection", "Abstain", "source instruction isolated", false});
    audit.append({"evaluator_truth", "stage8", "Restricted", "not publishable", true});
    const auto restored = ProductionAudit::deserialize(audit.serialize());
    require(restored.records().size() == 3 && restored.records().back().restricted && restored.records().front().decision == "Deny",
            "audit serialization lost release-boundary records");
}

void test_fail_closed_validation() {
    ProductionRegistry registry;
    bool rejected = false;
    try {
        registry.add_data_policy({DataClass::Restricted, "unresolved", "unknown", "high", "none", true, false, true});
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "unresolved training data was accepted");
    rejected = false;
    try {
        registry.add_evaluation({"bad", "task", "split", {}, {"metric"}, {"negative"}, "owner", false});
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "non-evaluator-only evaluation was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"registry_roundtrip_and_contracts", test_registry_roundtrip_and_contracts},
        {"policy_matrix", test_policy_matrix},
        {"realistic_application_fixtures", test_realistic_application_fixtures},
        {"audit_and_release_boundaries", test_audit_and_release_boundaries},
        {"fail_closed_validation", test_fail_closed_validation},
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
