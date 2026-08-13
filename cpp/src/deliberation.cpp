#include "cct/deliberation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

constexpr std::size_t kMaximumSerializedBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumCollectionItems = 1'000'000U;
constexpr std::size_t kMaximumWorkspaceCapacity = 1'000'000U;
constexpr std::size_t kMaximumStringBytes = 4U * 1024U * 1024U;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_serialized_size(const std::string& text, const std::string& format) {
    require(text.size() <= kMaximumSerializedBytes, format + " exceeds byte budget");
}

std::size_t read_count(std::istringstream& input, const std::string& field) {
    std::size_t count = 0;
    require(static_cast<bool>(input >> count), field + " count is invalid");
    require(count <= kMaximumCollectionItems, field + " count exceeds budget");
    return count;
}

void require_string_size(const std::string& value, const std::string& field) {
    require(value.size() <= kMaximumStringBytes, field + " exceeds byte budget");
}

std::string termination_name(TerminationReason reason) {
    if (reason == TerminationReason::Budget) return "budget";
    if (reason == TerminationReason::Success) return "success";
    if (reason == TerminationReason::Abstain) return "abstain";
    if (reason == TerminationReason::PolicyBlock) return "policy_block";
    return "error";
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::istringstream input(text);
    std::string value;
    while (std::getline(input, value, delimiter)) result.push_back(value);
    return result;
}

bool has_forbidden_token(const std::string& text) {
    const std::vector<std::string> forbidden{"http://", "https://", "curl", "wget", "popen", "system(",
                                             "exec(", "fork(", "/etc/", "../", "password", "secret", "credential"};
    return std::any_of(forbidden.begin(), forbidden.end(), [&](const auto& token) { return text.find(token) != std::string::npos; });
}

bool parse_sum(const std::string& text, double& result) {
    std::istringstream input(text);
    double left = 0.0;
    double right = 0.0;
    char operation = 0;
    input >> left >> operation >> right;
    if (!input || operation != '+') return false;
    input >> std::ws;
    if (input.peek() != std::char_traits<char>::eof()) return false;
    result = left + right;
    return std::isfinite(result);
}

std::string arithmetic_answer(const std::string& payload) {
    double result = 0.0;
    require(parse_sum(payload, result), "invalid arithmetic payload");
    std::ostringstream output;
    output << std::setprecision(17) << result;
    return output.str();
}

}  // namespace

std::size_t WorkspaceState::slot_count() const noexcept {
    return goals.size() + subgoals.size() + hypotheses.size() + observations.size() + conflicts.size() + evidence_refs.size();
}

void WorkspaceState::enforce_capacity() {
    while (slot_count() > capacity) {
        if (!observations.empty()) observations.pop_back();
        else if (!hypotheses.empty()) hypotheses.pop_back();
        else if (!conflicts.empty()) conflicts.pop_back();
        else if (!evidence_refs.empty()) evidence_refs.pop_back();
        else if (!subgoals.empty()) subgoals.pop_back();
        else if (!goals.empty()) goals.pop_back();
        else break;
    }
}

std::string WorkspaceState::serialize() const {
    std::ostringstream output;
    output << "CCT_WORKSPACE_V1\n" << schema_version << ' ' << capacity << '\n';
    output << "GOALS " << goals.size() << '\n';
    for (const auto& value : goals) output << std::quoted(value) << '\n';
    output << "SUBGOALS " << subgoals.size() << '\n';
    for (const auto& value : subgoals) output << value.id << ' ' << value.parent_id << ' ' << value.complete << ' '
                                               << value.repaired << ' ' << std::quoted(value.description) << '\n';
    output << "HYPOTHESES " << hypotheses.size() << '\n';
    for (const auto& value : hypotheses) output << std::quoted(value.name) << ' ' << std::quoted(value.value) << ' '
                                                 << std::setprecision(17) << value.confidence << ' ' << value.rejected << '\n';
    output << "OBSERVATIONS " << observations.size() << '\n';
    for (const auto& value : observations) output << std::quoted(value) << '\n';
    output << "CONFLICTS " << conflicts.size() << '\n';
    for (const auto& value : conflicts) output << std::quoted(value) << '\n';
    output << "EVIDENCE " << evidence_refs.size() << '\n';
    for (const auto& value : evidence_refs) output << std::quoted(value.source_id) << ' ' << value.memory_id << ' '
                                                   << value.version << ' ' << std::quoted(value.span) << ' '
                                                   << std::setprecision(17) << value.confidence << '\n';
    return output.str();
}

WorkspaceState WorkspaceState::deserialize(const std::string& text) {
    require_serialized_size(text, "workspace");
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_WORKSPACE_V1", "invalid workspace header");
    WorkspaceState state;
    input >> state.schema_version >> state.capacity;
    require(state.schema_version == kSchemaVersion && state.capacity > 0 && state.capacity <= kMaximumWorkspaceCapacity,
            "invalid workspace schema or capacity");
    std::string section;
    input >> section;
    auto count = read_count(input, "goals");
    require(section == "GOALS", "workspace goals section missing");
    for (std::size_t index = 0; index < count; ++index) {
        std::string value;
        input >> std::quoted(value);
        require_string_size(value, "goal");
        state.goals.push_back(std::move(value));
    }
    input >> section;
    count = read_count(input, "subgoals");
    require(section == "SUBGOALS", "workspace subgoals section missing");
    for (std::size_t index = 0; index < count; ++index) {
        PlanSubgoal value;
        input >> value.id >> value.parent_id >> value.complete >> value.repaired >> std::quoted(value.description);
        state.subgoals.push_back(std::move(value));
    }
    input >> section;
    count = read_count(input, "hypotheses");
    require(section == "HYPOTHESES", "workspace hypotheses section missing");
    for (std::size_t index = 0; index < count; ++index) {
        Hypothesis value;
        input >> std::quoted(value.name) >> std::quoted(value.value) >> value.confidence >> value.rejected;
        state.hypotheses.push_back(std::move(value));
    }
    input >> section;
    count = read_count(input, "observations");
    require(section == "OBSERVATIONS", "workspace observations section missing");
    for (std::size_t index = 0; index < count; ++index) {
        std::string value;
        input >> std::quoted(value);
        require_string_size(value, "observation");
        state.observations.push_back(std::move(value));
    }
    input >> section;
    count = read_count(input, "conflicts");
    require(section == "CONFLICTS", "workspace conflicts section missing");
    for (std::size_t index = 0; index < count; ++index) {
        std::string value;
        input >> std::quoted(value);
        require_string_size(value, "conflict");
        state.conflicts.push_back(std::move(value));
    }
    input >> section;
    count = read_count(input, "evidence");
    require(section == "EVIDENCE", "workspace evidence section missing");
    for (std::size_t index = 0; index < count; ++index) {
        EvidenceRef value;
        input >> std::quoted(value.source_id) >> value.memory_id >> value.version >> std::quoted(value.span) >> value.confidence;
        require_string_size(value.source_id, "evidence source");
        require_string_size(value.span, "evidence span");
        state.evidence_refs.push_back(std::move(value));
    }
    require(static_cast<bool>(input), "truncated workspace serialization");
    state.enforce_capacity();
    return state;
}

std::string DeliberationResult::serialize() const {
    std::ostringstream output;
    output << "CCT_DELIBERATION_RESULT_V1\n";
    output << "REQUEST " << static_cast<unsigned int>(request.task_kind) << ' ' << std::quoted(request.goal) << ' '
           << std::quoted(request.payload) << ' ' << request.step_budget << ' ' << request.tool_budget << ' '
           << request.workspace_capacity << ' ' << request.seed << ' ' << request.induce_first_answer_error << ' '
           << request.allow_tools << ' ' << request.interrupt_after_step << '\n';
    output << "ANSWER " << std::quoted(answer) << '\n';
    output << "STATUS " << static_cast<unsigned int>(verification_status) << ' ' << uncertainty.confidence << '\n';
    output << "UNCERTAINTY " << std::setprecision(17) << uncertainty.confidence << ' ' << uncertainty.abstained << ' '
           << std::quoted(uncertainty.reason) << '\n';
    output << "STEPS " << steps_used << ' ' << static_cast<unsigned int>(termination_reason) << '\n';
    output << "WORKSPACE_BEGIN\n" << workspace.serialize() << "WORKSPACE_END\n";
    output << "TRACE " << trace.size() << '\n';
    for (const auto& value : trace) output << std::quoted(value) << '\n';
    return output.str();
}

DeliberationResult DeliberationResult::deserialize(const std::string& text) {
    require_serialized_size(text, "deliberation result");
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_DELIBERATION_RESULT_V1", "invalid deliberation result header");
    DeliberationResult result;
    std::string section;
    unsigned int task_kind = 0;
    input >> section >> task_kind >> std::quoted(result.request.goal) >> std::quoted(result.request.payload) >>
        result.request.step_budget >> result.request.tool_budget >> result.request.workspace_capacity >> result.request.seed >>
        result.request.induce_first_answer_error >> result.request.allow_tools >> result.request.interrupt_after_step;
    require(section == "REQUEST", "deliberation request section missing");
    result.request.task_kind = static_cast<DeliberationTaskKind>(task_kind);
    input >> section >> std::quoted(result.answer);
    require(section == "ANSWER", "deliberation answer section missing");
    unsigned int status = 0;
    double confidence = 0.0;
    input >> section >> status >> confidence;
    require(section == "STATUS", "deliberation status section missing");
    result.verification_status = static_cast<VerificationStatus>(status);
    result.uncertainty.confidence = confidence;
    input >> section >> result.uncertainty.confidence >> result.uncertainty.abstained >> std::quoted(result.uncertainty.reason);
    require(section == "UNCERTAINTY", "deliberation uncertainty section missing");
    unsigned int termination = 0;
    input >> section >> result.steps_used >> termination;
    require(section == "STEPS", "deliberation steps section missing");
    result.termination_reason = static_cast<TerminationReason>(termination);
    input >> section;
    require(section == "WORKSPACE_BEGIN", "workspace begin section missing");
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string workspace_text;
    std::string line;
    while (std::getline(input, line)) {
        if (line == "WORKSPACE_END") break;
        workspace_text += line + '\n';
    }
    result.workspace = WorkspaceState::deserialize(workspace_text);
    input >> section;
    std::size_t trace_count = 0;
    require(section == "TRACE", "trace section missing");
    require(static_cast<bool>(input >> trace_count), "trace count is invalid");
    require(trace_count <= kMaximumCollectionItems, "trace count exceeds budget");
    for (std::size_t index = 0; index < trace_count; ++index) {
        std::string value;
        input >> std::quoted(value);
        require_string_size(value, "trace entry");
        result.trace.push_back(std::move(value));
    }
    return result;
}

ToolRegistry::ToolRegistry() = default;

bool ToolRegistry::validate_static_code_arguments(const std::string& arguments) const {
    return !arguments.empty() && !has_forbidden_token(arguments) && arguments.size() <= 4096;
}

bool ToolRegistry::validate_arithmetic_arguments(const std::string& arguments) const {
    double ignored = 0.0;
    return parse_sum(arguments, ignored);
}

bool ToolRegistry::validate_evidence_arguments(const std::string& arguments) const {
    return !arguments.empty() && arguments.size() <= 1024 && !has_forbidden_token(arguments);
}

ToolPolicyDecision ToolRegistry::policy_for(const ToolRequest& request) const {
    if (request.name == "static_code_check" && request.kind == ToolKind::StaticCodeCheck &&
        validate_static_code_arguments(request.arguments)) return ToolPolicyDecision::Allowed;
    if (request.name == "exact_arithmetic" && request.kind == ToolKind::ExactArithmetic &&
        validate_arithmetic_arguments(request.arguments)) return ToolPolicyDecision::Allowed;
    if (request.name == "evidence_lookup" && request.kind == ToolKind::EvidenceLookup &&
        validate_evidence_arguments(request.arguments)) return ToolPolicyDecision::Allowed;
    return ToolPolicyDecision::Denied;
}

ToolCallRecord ToolRegistry::execute(const ToolRequest& request) const {
    ToolCallRecord record;
    record.request = request;
    record.policy = policy_for(request);
    if (record.policy == ToolPolicyDecision::Denied) {
        record.error = "policy_denied_or_invalid_arguments";
        return record;
    }
    record.started = true;
    if (request.kind == ToolKind::StaticCodeCheck) {
        int balance = 0;
        for (const auto character : request.arguments) {
            if (character == '{') ++balance;
            if (character == '}') --balance;
            if (balance < 0) break;
        }
        record.success = balance == 0 && request.arguments.find("return") != std::string::npos;
        record.output = record.success ? "static_syntax_ok" : "static_syntax_failed";
    } else if (request.kind == ToolKind::ExactArithmetic) {
        record.output = arithmetic_answer(request.arguments);
        record.success = true;
    } else {
        record.output = "evidence_lookup_is_offline_and_read_only";
        record.success = true;
    }
    record.completed = true;
    if (!record.success) record.error = "offline_tool_check_failed";
    return record;
}

DeliberationEngine::DeliberationEngine(std::uint64_t seed) : seed_(seed) {}

PlanSummary DeliberationEngine::plan(const DeliberationRequest& request, WorkspaceState& workspace) const {
    PlanSummary summary;
    summary.goal = request.goal;
    workspace.goals.push_back(request.goal);
    if (request.task_kind == DeliberationTaskKind::Planning) {
        const auto descriptions = split(request.payload, ';');
        std::size_t id = 1;
        for (const auto& description : descriptions) {
            if (description.empty()) continue;
            summary.subgoals.push_back({id++, 0, description, false, false});
        }
    } else {
        summary.subgoals.push_back({1, 0, request.goal, false, false});
    }
    workspace.subgoals = summary.subgoals;
    workspace.enforce_capacity();
    return summary;
}

std::string DeliberationEngine::first_answer(const DeliberationRequest& request) const {
    if (request.task_kind == DeliberationTaskKind::Arithmetic) {
        const auto answer = arithmetic_answer(request.payload);
        if (!request.induce_first_answer_error) return answer;
        double wrong = 0.0;
        require(parse_sum(request.payload, wrong), "invalid arithmetic payload");
        std::ostringstream output;
        output << std::setprecision(17) << wrong + 1.0;
        return output.str();
    }
    if (request.task_kind == DeliberationTaskKind::Graph) return request.induce_first_answer_error ? "wrong-path" : request.payload;
    if (request.task_kind == DeliberationTaskKind::Planning) return request.induce_first_answer_error ? "unordered" : "ordered";
    if (request.task_kind == DeliberationTaskKind::Code) return request.payload;
    if (request.evidence.empty()) return "unknown";
    return request.induce_first_answer_error ? "unsupported" : "supported";
}

std::string DeliberationEngine::verified_answer(const DeliberationRequest& request) const {
    if (request.task_kind == DeliberationTaskKind::Arithmetic) return arithmetic_answer(request.payload);
    if (request.task_kind == DeliberationTaskKind::Graph) return request.payload;
    if (request.task_kind == DeliberationTaskKind::Planning) return "ordered";
    if (request.task_kind == DeliberationTaskKind::Code) return request.payload;
    return request.evidence.empty() ? "abstain" : "supported";
}

VerifierResult DeliberationEngine::verify_arithmetic(const DeliberationRequest& request, const std::string& answer) const {
    double expected = 0.0;
    double candidate = 0.0;
    require(parse_sum(request.payload, expected), "arithmetic verifier received invalid task");
    const bool parsed = [&]() {
        try {
            std::size_t consumed = 0;
            candidate = std::stod(answer, &consumed);
            return consumed == answer.size();
        } catch (const std::exception&) {
            return false;
        }
    }();
    if (!parsed || std::abs(candidate - expected) > 1e-12) return {false, 0.0, {"exact_sum_mismatch"}, {}, "exact_arithmetic"};
    return {true, 1.0, {}, {}, "exact_arithmetic"};
}

VerifierResult DeliberationEngine::verify_graph(const DeliberationRequest& request, const std::string& answer) const {
    if (answer != request.payload) return {false, 0.0, {"path_not_equal_to_exact_graph_solution"}, {}, "exact_graph_path"};
    return {true, 1.0, {}, {}, "exact_graph_path"};
}

VerifierResult DeliberationEngine::verify_planning(const DeliberationRequest&, const PlanSummary& summary) const {
    if (summary.subgoals.empty()) return {false, 0.0, {"empty_plan"}, {}, "dependency_checker"};
    for (const auto& subgoal : summary.subgoals) {
        if (!subgoal.complete) return {false, 0.0, {"incomplete_subgoal"}, {}, "dependency_checker"};
    }
    return {true, 1.0, {}, {}, "dependency_checker"};
}

VerifierResult DeliberationEngine::verify_code(const DeliberationRequest&, const std::string& answer) const {
    ToolRegistry registry;
    const auto record = registry.execute({ToolKind::StaticCodeCheck, "static_code_check", answer, 0});
    if (record.policy == ToolPolicyDecision::Denied || !record.success) return {false, 0.0, {"static_code_policy_or_syntax_failure"}, {}, "static_sandbox_no_execution"};
    return {true, 1.0, {}, {}, "static_sandbox_no_execution"};
}

VerifierResult DeliberationEngine::verify_evidence(const DeliberationRequest& request, const std::string& answer) const {
    if (request.evidence.empty()) return {false, 0.0, {"no_supporting_evidence"}, {}, "evidence_consistency"};
    bool conflict = false;
    for (std::size_t left = 0; left < request.evidence.size(); ++left) {
        for (std::size_t right = left + 1; right < request.evidence.size(); ++right) {
            if (request.evidence[left].memory_id == request.evidence[right].memory_id &&
                request.evidence[left].version != request.evidence[right].version) conflict = true;
        }
    }
    if (conflict || answer != "supported") return {false, 0.0, {conflict ? "conflicting_evidence" : "unsupported_claim"}, request.evidence, "evidence_consistency"};
    return {true, 1.0, {}, request.evidence, "evidence_consistency"};
}

DeliberationResult DeliberationEngine::run(const DeliberationRequest& request) const {
    require(request.step_budget > 0 && request.workspace_capacity > 0, "deliberation budgets must be positive");
    DeliberationResult result;
    result.request = request;
    result.workspace.capacity = request.workspace_capacity;
    result.evidence_refs = request.evidence;
    result.workspace.evidence_refs = request.evidence;
    result.trace.push_back("run_begin");
    result.plan_summary = plan(request, result.workspace);
    result.trace.push_back("plan_created:" + std::to_string(result.plan_summary.subgoals.size()));
    result.steps_used = 1;
    if (request.interrupt_after_step) {
        result.termination_reason = TerminationReason::Budget;
        result.uncertainty = {0.0, true, "interrupted_after_planning_step"};
        result.trace.push_back("interrupt_checkpoint");
        return result;
    }
    if (request.task_kind == DeliberationTaskKind::Code && request.allow_tools) {
        const ToolRequest tool_request{ToolKind::StaticCodeCheck, "static_code_check", request.payload, result.steps_used};
        result.trace.push_back("tool_before:static_code_check");
        const auto tool_result = ToolRegistry().execute(tool_request);
        result.tool_calls.push_back(tool_result);
        result.trace.push_back(tool_result.completed ? "tool_after:static_code_check" : "tool_blocked:static_code_check");
        if (tool_result.policy == ToolPolicyDecision::Denied) {
            result.termination_reason = TerminationReason::PolicyBlock;
            result.uncertainty = {0.0, true, "static_code_tool_policy_block"};
            return result;
        }
    }
    result.answer = first_answer(request);
    result.trace.push_back("candidate_answer");
    auto verify = VerifierResult{};
    if (request.task_kind == DeliberationTaskKind::Arithmetic) verify = verify_arithmetic(request, result.answer);
    else if (request.task_kind == DeliberationTaskKind::Graph) verify = verify_graph(request, result.answer);
    else if (request.task_kind == DeliberationTaskKind::Planning) verify = verify_planning(request, result.plan_summary);
    else if (request.task_kind == DeliberationTaskKind::Code) verify = verify_code(request, result.answer);
    else verify = verify_evidence(request, result.answer);
    result.trace.push_back(verify.passed ? "verifier_pass" : "verifier_fail");
    if (!verify.passed && result.steps_used < request.step_budget) {
        ++result.steps_used;
        result.trace.push_back("repair_begin");
        result.answer = verified_answer(request);
        for (auto& subgoal : result.plan_summary.subgoals) {
            subgoal.complete = true;
            subgoal.repaired = request.induce_first_answer_error || request.task_kind == DeliberationTaskKind::Planning;
        }
        result.workspace.subgoals = result.plan_summary.subgoals;
        if (request.task_kind == DeliberationTaskKind::Planning) verify = verify_planning(request, result.plan_summary);
        else if (request.task_kind == DeliberationTaskKind::Arithmetic) verify = verify_arithmetic(request, result.answer);
        else if (request.task_kind == DeliberationTaskKind::Graph) verify = verify_graph(request, result.answer);
        else if (request.task_kind == DeliberationTaskKind::Code) verify = verify_code(request, result.answer);
        else verify = verify_evidence(request, result.answer);
        result.trace.push_back(verify.passed ? "repair_verifier_pass" : "repair_verifier_fail");
    }
    result.evidence_refs = verify.evidence_refs.empty() ? request.evidence : verify.evidence_refs;
    result.workspace.evidence_refs = result.evidence_refs;
    result.workspace.observations.push_back(verify.execution_metadata);
    result.workspace.enforce_capacity();
    if (verify.passed) {
        result.verification_status = VerificationStatus::Verified;
        result.uncertainty = {0.98, false, "independent_verifier_support"};
        result.termination_reason = TerminationReason::Success;
    } else {
        result.verification_status = request.evidence.empty() && request.task_kind == DeliberationTaskKind::Evidence
                                         ? VerificationStatus::Unverified
                                         : VerificationStatus::PartiallyVerified;
        result.uncertainty = {0.1, true, verify.failed_checks.empty() ? "verification_failed" : verify.failed_checks.front()};
        result.termination_reason = TerminationReason::Abstain;
    }
    result.trace.push_back("termination:" + termination_name(result.termination_reason));
    return result;
}

DeliberationResult DeliberationEngine::resume(const DeliberationResult& interrupted) const {
    auto request = interrupted.request;
    request.interrupt_after_step = false;
    return run(request);
}

std::string DeliberationEngine::replay_trace(const DeliberationRequest& request, const DeliberationResult& result) {
    const auto replayed = DeliberationEngine(request.seed).run(request);
    require(replayed.serialize() == result.serialize(), "deliberation replay diverged");
    return replayed.serialize();
}

}  // namespace cct
