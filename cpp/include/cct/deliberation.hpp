#pragma once

#include "cct/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cct {

enum class DeliberationTaskKind : std::uint8_t { Arithmetic = 0, Graph = 1, Planning = 2, Code = 3, Evidence = 4 };
enum class VerificationStatus : std::uint8_t { Verified = 0, PartiallyVerified = 1, Unverified = 2 };
enum class TerminationReason : std::uint8_t { Budget = 0, Success = 1, Abstain = 2, PolicyBlock = 3, Error = 4 };
enum class ToolPolicyDecision : std::uint8_t { Allowed = 0, Denied = 1 };

enum class ToolKind : std::uint8_t { StaticCodeCheck = 0, ExactArithmetic = 1, EvidenceLookup = 2 };

struct EvidenceRef {
    std::string source_id;
    MemoryId memory_id = 0;
    std::uint64_t version = 0;
    std::string span;
    double confidence = 0.0;
};

struct DeliberationUncertainty {
    double confidence = 0.0;
    bool abstained = false;
    std::string reason;
};

struct PlanSubgoal {
    std::size_t id = 0;
    std::size_t parent_id = 0;
    std::string description;
    bool complete = false;
    bool repaired = false;
};

struct PlanSummary {
    std::string goal;
    std::vector<PlanSubgoal> subgoals;
    std::size_t invalid_actions = 0;
};

struct Hypothesis {
    std::string name;
    std::string value;
    double confidence = 0.0;
    bool rejected = false;
};

struct WorkspaceState {
    static constexpr std::uint32_t kSchemaVersion = 1;
    std::uint32_t schema_version = kSchemaVersion;
    std::size_t capacity = 64;
    std::vector<std::string> goals;
    std::vector<PlanSubgoal> subgoals;
    std::vector<Hypothesis> hypotheses;
    std::vector<std::string> observations;
    std::vector<std::string> conflicts;
    std::vector<EvidenceRef> evidence_refs;

    std::string serialize() const;
    static WorkspaceState deserialize(const std::string& text);
    std::size_t slot_count() const noexcept;
    void enforce_capacity();
};

struct ToolRequest {
    ToolKind kind = ToolKind::StaticCodeCheck;
    std::string name;
    std::string arguments;
    std::size_t step = 0;
};

struct ToolCallRecord {
    ToolRequest request;
    ToolPolicyDecision policy = ToolPolicyDecision::Denied;
    bool started = false;
    bool completed = false;
    bool success = false;
    std::string output;
    std::string error;
};

struct VerifierResult {
    bool passed = false;
    double score = 0.0;
    std::vector<std::string> failed_checks;
    std::vector<EvidenceRef> evidence_refs;
    std::string execution_metadata;
};

struct DeliberationRequest {
    DeliberationTaskKind task_kind = DeliberationTaskKind::Arithmetic;
    std::string goal;
    std::string payload;
    std::vector<EvidenceRef> evidence;
    std::size_t step_budget = 8;
    std::size_t tool_budget = 4;
    std::size_t workspace_capacity = 64;
    std::uint64_t seed = 0;
    bool induce_first_answer_error = false;
    bool allow_tools = true;
    bool interrupt_after_step = false;
};

struct DeliberationResult {
    DeliberationRequest request;
    std::string answer;
    std::vector<EvidenceRef> evidence_refs;
    PlanSummary plan_summary;
    VerificationStatus verification_status = VerificationStatus::Unverified;
    DeliberationUncertainty uncertainty;
    std::size_t steps_used = 0;
    std::vector<ToolCallRecord> tool_calls;
    TerminationReason termination_reason = TerminationReason::Error;
    WorkspaceState workspace;
    std::vector<std::string> trace;

    std::string serialize() const;
    static DeliberationResult deserialize(const std::string& text);
};

class ToolRegistry {
public:
    ToolRegistry();

    ToolPolicyDecision policy_for(const ToolRequest& request) const;
    ToolCallRecord execute(const ToolRequest& request) const;

private:
    bool validate_static_code_arguments(const std::string& arguments) const;
    bool validate_arithmetic_arguments(const std::string& arguments) const;
    bool validate_evidence_arguments(const std::string& arguments) const;
};

class DeliberationEngine {
public:
    explicit DeliberationEngine(std::uint64_t seed = 0);

    DeliberationResult run(const DeliberationRequest& request) const;
    DeliberationResult resume(const DeliberationResult& interrupted) const;
    static std::string replay_trace(const DeliberationRequest& request, const DeliberationResult& result);

private:
    std::uint64_t seed_ = 0;

    PlanSummary plan(const DeliberationRequest& request, WorkspaceState& workspace) const;
    VerifierResult verify_arithmetic(const DeliberationRequest& request, const std::string& answer) const;
    VerifierResult verify_graph(const DeliberationRequest& request, const std::string& answer) const;
    VerifierResult verify_planning(const DeliberationRequest& request, const PlanSummary& summary) const;
    VerifierResult verify_code(const DeliberationRequest& request, const std::string& answer) const;
    VerifierResult verify_evidence(const DeliberationRequest& request, const std::string& answer) const;
    std::string first_answer(const DeliberationRequest& request) const;
    std::string verified_answer(const DeliberationRequest& request) const;
};

}  // namespace cct
