#include "cct/deliberation.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cct::DeliberationEngine;
using cct::DeliberationRequest;
using cct::DeliberationTaskKind;
using cct::EvidenceRef;
using cct::TerminationReason;
using cct::ToolKind;
using cct::ToolPolicyDecision;
using cct::ToolRegistry;
using cct::ToolRequest;
using cct::VerificationStatus;
using cct::WorkspaceState;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_workspace_schema_and_capacity() {
    WorkspaceState workspace;
    workspace.capacity = 4;
    workspace.goals = {"goal", "extra"};
    workspace.observations = {"o1", "o2", "o3"};
    workspace.conflicts = {"conflict"};
    workspace.enforce_capacity();
    require(workspace.slot_count() <= workspace.capacity, "workspace exceeded declared capacity");
    const auto restored = WorkspaceState::deserialize(workspace.serialize());
    require(restored.slot_count() == workspace.slot_count() && restored.capacity == workspace.capacity,
            "workspace serialization changed bounded state");
}

void test_arithmetic_verification_and_repair() {
    DeliberationRequest request;
    request.task_kind = DeliberationTaskKind::Arithmetic;
    request.goal = "compute exact sum";
    request.payload = "2 + 3";
    request.step_budget = 4;
    request.seed = 11;
    request.induce_first_answer_error = true;
    const auto result = DeliberationEngine(11).run(request);
    require(result.answer == "5" && result.verification_status == VerificationStatus::Verified &&
                result.termination_reason == TerminationReason::Success && result.steps_used == 2,
            "independent arithmetic verifier did not repair wrong first answer: answer=" + result.answer + " status=" + std::to_string(static_cast<unsigned int>(result.verification_status)) + " termination=" + std::to_string(static_cast<unsigned int>(result.termination_reason)) + " steps=" + std::to_string(result.steps_used));
    require(result.trace.size() >= 5, "arithmetic trace omitted deliberation events");
    require(DeliberationEngine::replay_trace(DeliberationRequest{request.task_kind, request.goal, request.payload, {}, 4, 4, 64, 11, true, true, false}, result).size() > 0,
            "arithmetic replay did not produce a trace");
}

void test_graph_and_planning_verification() {
    DeliberationRequest graph;
    graph.task_kind = DeliberationTaskKind::Graph;
    graph.goal = "find exact path";
    graph.payload = "1->2->3";
    graph.induce_first_answer_error = true;
    const auto graph_result = DeliberationEngine(12).run(graph);
    require(graph_result.verification_status == VerificationStatus::Verified && graph_result.answer == graph.payload,
            "exact graph verifier failed to repair path");

    DeliberationRequest planning;
    planning.task_kind = DeliberationTaskKind::Planning;
    planning.goal = "complete dependency plan";
    planning.payload = "collect;transform;verify";
    planning.step_budget = 4;
    const auto planning_result = DeliberationEngine(13).run(planning);
    require(planning_result.verification_status == VerificationStatus::Verified && planning_result.plan_summary.subgoals.size() == 3,
            "planning verifier did not complete typed subgoals");
    for (const auto& subgoal : planning_result.plan_summary.subgoals) require(subgoal.complete, "planning subgoal remained incomplete");
}

void test_code_policy_and_tool_containment() {
    DeliberationRequest safe;
    safe.task_kind = DeliberationTaskKind::Code;
    safe.goal = "static-check code";
    safe.payload = "int main() { return 0; }";
    safe.allow_tools = true;
    const auto safe_result = DeliberationEngine(14).run(safe);
    require(safe_result.verification_status == VerificationStatus::Verified && safe_result.tool_calls.size() == 1 &&
                safe_result.tool_calls.front().started && safe_result.tool_calls.front().completed,
            "safe static code tool did not complete under policy");

    DeliberationRequest unsafe = safe;
    unsafe.payload = "http://evil; system(\"secret\");";
    const auto unsafe_result = DeliberationEngine(14).run(unsafe);
    require(unsafe_result.termination_reason == TerminationReason::PolicyBlock && unsafe_result.tool_calls.size() == 1 &&
                unsafe_result.tool_calls.front().policy == ToolPolicyDecision::Denied && !unsafe_result.tool_calls.front().started,
            "unsafe tool request was not deny-by-default");
    const auto unknown = ToolRegistry().execute({ToolKind::StaticCodeCheck, "unknown_tool", "int main() { return 0; }", 0});
    require(unknown.policy == ToolPolicyDecision::Denied && !unknown.started, "unknown tool was not refused");
}

void test_evidence_integrity_and_abstention() {
    DeliberationRequest supported;
    supported.task_kind = DeliberationTaskKind::Evidence;
    supported.goal = "answer from cited evidence";
    supported.evidence = {{"trusted-source", 77, 1, "span-1", 0.98}};
    const auto supported_result = DeliberationEngine(15).run(supported);
    require(supported_result.verification_status == VerificationStatus::Verified && supported_result.evidence_refs.size() == 1,
            "supported evidence was not verified");

    DeliberationRequest conflict = supported;
    conflict.evidence.push_back({"trusted-source", 77, 2, "span-2", 0.91});
    const auto conflict_result = DeliberationEngine(15).run(conflict);
    require(conflict_result.termination_reason == TerminationReason::Abstain &&
                conflict_result.verification_status != VerificationStatus::Verified && conflict_result.uncertainty.abstained,
            "conflicting evidence did not trigger abstention");

    DeliberationRequest missing = supported;
    missing.evidence.clear();
    const auto missing_result = DeliberationEngine(15).run(missing);
    require(missing_result.termination_reason == TerminationReason::Abstain && missing_result.uncertainty.abstained,
            "unsupported evidence claim did not abstain");
}

void test_serialization_replay_and_interrupt_resume() {
    DeliberationRequest uninterrupted;
    uninterrupted.task_kind = DeliberationTaskKind::Arithmetic;
    uninterrupted.goal = "resume exact computation";
    uninterrupted.payload = "8 + 13";
    uninterrupted.step_budget = 4;
    uninterrupted.seed = 16;
    const auto expected = DeliberationEngine(16).run(uninterrupted);
    const auto interrupted_request = [&]() {
        auto value = uninterrupted;
        value.interrupt_after_step = true;
        return value;
    }();
    const auto checkpoint = DeliberationEngine(16).run(interrupted_request);
    require(checkpoint.termination_reason == TerminationReason::Budget, "interruption did not stop at checkpoint");
    const auto resumed = DeliberationEngine(16).resume(checkpoint);
    require(resumed.serialize() == expected.serialize(), "interrupted/resumed result diverged from uninterrupted result");
    const auto restored = cct::DeliberationResult::deserialize(expected.serialize());
    require(restored.serialize() == expected.serialize(), "deliberation result serialization is not round-trip stable");
    auto oversized_trace = expected.serialize();
    const auto trace_marker = oversized_trace.find("TRACE 5");
    require(trace_marker != std::string::npos, "deliberation trace marker changed unexpectedly");
    oversized_trace.replace(trace_marker, std::string("TRACE 5").size(), "TRACE 1000001");
    bool trace_rejected = false;
    try { static_cast<void>(cct::DeliberationResult::deserialize(oversized_trace)); } catch (const std::exception&) { trace_rejected = true; }
    require(trace_rejected, "oversized deliberation trace count was accepted");
    auto oversized_workspace = expected.serialize();
    const auto goals_marker = oversized_workspace.find("GOALS 1");
    require(goals_marker != std::string::npos, "workspace goals marker changed unexpectedly");
    oversized_workspace.replace(goals_marker, std::string("GOALS 1").size(), "GOALS 1000001");
    bool workspace_rejected = false;
    try { static_cast<void>(cct::DeliberationResult::deserialize(oversized_workspace)); } catch (const std::exception&) { workspace_rejected = true; }
    require(workspace_rejected, "oversized deliberation workspace count was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"workspace_schema_and_capacity", test_workspace_schema_and_capacity},
        {"arithmetic_verification_and_repair", test_arithmetic_verification_and_repair},
        {"graph_and_planning_verification", test_graph_and_planning_verification},
        {"code_policy_and_tool_containment", test_code_policy_and_tool_containment},
        {"evidence_integrity_and_abstention", test_evidence_integrity_and_abstention},
        {"serialization_replay_and_interrupt_resume", test_serialization_replay_and_interrupt_resume},
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
