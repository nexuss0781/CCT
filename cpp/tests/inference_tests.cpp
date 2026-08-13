#include "cct/inference.hpp"
#include "cct/nlp_trainer.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

KnowledgeRecord knowledge_record(const std::string& id, const std::string& tenant, const std::string& content) {
    KnowledgeRecord record;
    record.knowledge_id = id;
    record.tenant_id = tenant;
    record.document_id = id + "-document";
    record.document_version = 1U;
    record.source_uri_or_reference = "fixture://stage16/" + id;
    record.content = content;
    record.content_hash = GovernedCorpus::content_sha256(content);
    record.embedding_version = "embedding-v1";
    record.lexical_index_version = "lexical-v1";
    record.created_at = 1;
    record.valid_from = 1;
    record.access_policy = {tenant, {"analyst"}, false};
    record.provenance = "stage16-unit-fixture";
    record.citation_spans.push_back({id + "#span-0", 0U, content.size(), record.content_hash});
    record.quality = {0.99, 0.99, "normal"};
    return record;
}

InferenceRequest request(const std::string& id, const std::string& tenant = "tenant-a", const std::string& session = "session-a") {
    InferenceRequest item;
    item.request_id = id;
    item.tenant_id = tenant;
    item.user_id = tenant + "-user";
    item.role = "analyst";
    item.session_id = session;
    item.input = "retention policy";
    item.task_schema = "answer";
    item.retrieval_policy = "none";
    item.tool_policy = "offline-deny";
    item.stream = false;
    item.trace_id = "trace-" + id;
    return item;
}

AuthContext auth(const std::string& tenant = "tenant-a") {
    return {true, tenant, tenant + "-user", {"analyst"}};
}

InferenceService service_with_knowledge(KnowledgePlane& plane) {
    plane.ingest(knowledge_record("policy", "tenant-a", "The retention policy is seven days."));
    plane.ingest(knowledge_record("beta-policy", "tenant-b", "Tenant beta retention policy is thirty days."));
    InferenceService service({}, &plane);
    service.set_slo_thresholds({1500.0, 150.0, 0.95, 0.2, 600000.0});
    return service;
}

void test_checkpoint_backend_generation_and_streaming() {
    const auto root = std::filesystem::temp_directory_path() / "cct-inference-checkpoint-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    TokenizerConfig tokenizer_config;
    tokenizer_config.tokenizer_version = "tokenizer-stage10-v1";
    tokenizer_config.candidate = TokenizerCandidate::Byte;
    tokenizer_config.include_bos_eos = false;
    const auto tokenizer = Tokenizer::build(tokenizer_config, {TokenizerTrainingRecord{"fixture", "alpha beta", true, false}});
    const auto tokenizer_path = root / "tokenizer.snapshot";
    {
        std::ofstream output(tokenizer_path, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "could not create checkpoint tokenizer snapshot");
        output << tokenizer.serialize_snapshot();
    }
    const auto vocabulary_size = static_cast<std::size_t>(tokenizer.vocabulary().back().id) + 1U;
    const NlpModelConfig model_config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 4U, 4U, 16U, 7U};
    NlpOptimizerConfig optimizer;
    optimizer.total_steps = 1U;
    NlpTrainer trainer(model_config, optimizer, tokenizer.snapshot_hash(), "inference-fixture-dataset");
    auto parameters = trainer.model().parameter_vector();
    const auto recurrent_offset = vocabulary_size * model_config.embedding_dim;
    const auto head_offset = recurrent_offset + 4U * model_config.hidden_dim * model_config.embedding_dim + 3U * model_config.hidden_dim;
    const auto bias_offset = head_offset + vocabulary_size * model_config.hidden_dim;
    const auto token_a = static_cast<TokenId>(Tokenizer::kByteFirstId + static_cast<unsigned int>('a'));
    const auto token_b = static_cast<TokenId>(Tokenizer::kByteFirstId + static_cast<unsigned int>('b'));
    parameters.assign(parameters.size(), 0.0);
    parameters[bias_offset + token_a] = 10.0;
    parameters[bias_offset + Tokenizer::kEosId] = -10.0;
    trainer.model().set_parameter_vector(parameters);
    const auto checkpoint_a = root / "model-a.checkpoint";
    trainer.save_checkpoint(checkpoint_a.string());

    InferenceConfig config;
    config.backend_mode = InferenceBackendMode::Checkpoint;
    config.model_checkpoint_path = checkpoint_a.string();
    config.tokenizer_snapshot_path = tokenizer_path.string();
    config.tokenizer_version = tokenizer.version();
    config.model_version = "checkpoint-model-a";
    config.maximum_input_tokens = 16U;
    config.maximum_output_tokens = 4U;
    InferenceService service(config);
    auto first_request = request("checkpoint-generation");
    first_request.input = "alpha";
    const auto first = service.handle(first_request, auth());
    require(first.error_code.empty() && first.backend_identity.find("checkpoint-backed-") == 0U && !first.output.empty() &&
                first.output.find('a') != std::string::npos && first.latency.first_token_milliseconds >= 0.0,
            "checkpoint backend did not generate a decoded model response");
    auto stream_request = first_request;
    stream_request.request_id = "checkpoint-stream";
    stream_request.trace_id = "trace-checkpoint-stream";
    const auto streamed = service.execute_stream(stream_request, auth(), 4U, false);
    require(!streamed.cancelled && streamed.resources_released && !streamed.events.empty() &&
                streamed.events.back().type == StreamEventType::Completed &&
                std::count_if(streamed.events.begin(), streamed.events.end(), [](const auto& event) { return event.type == StreamEventType::Token; }) > 1U,
            "checkpoint stream did not emit incremental token events");
    auto cancelled_request = first_request;
    cancelled_request.request_id = "checkpoint-cancel";
    cancelled_request.trace_id = "trace-checkpoint-cancel";
    const auto cancelled = service.execute_stream(cancelled_request, auth(), 4U, true);
    std::string cancellation_events;
    for (const auto& event : cancelled.events) cancellation_events += stream_event_type_name(event.type) + ":" + event.payload + ";";
    require(cancelled.cancelled && cancelled.resources_released && !cancelled.events.empty() && cancelled.events.front().type == StreamEventType::Token &&
                cancelled.events.back().type == StreamEventType::Cancelled,
            "checkpoint cancellation did not stop after the first emitted token: " + cancellation_events);

    parameters.assign(parameters.size(), 0.0);
    parameters[bias_offset + token_b] = 10.0;
    parameters[bias_offset + Tokenizer::kEosId] = -10.0;
    trainer.model().set_parameter_vector(parameters);
    const auto checkpoint_b = root / "model-b.checkpoint";
    trainer.save_checkpoint(checkpoint_b.string());
    config.model_checkpoint_path = checkpoint_b.string();
    config.model_version = "checkpoint-model-b";
    InferenceService changed_service(config);
    auto changed_request = first_request;
    changed_request.request_id = "checkpoint-parameter-change";
    changed_request.trace_id = "trace-checkpoint-parameter-change";
    const auto changed = changed_service.handle(changed_request, auth());
    require(changed.error_code.empty() && changed.output != first.output && changed.backend_identity.find("checkpoint-backed-") == 0U,
            "changing checkpoint parameters did not change model output");
}

void test_versioned_api_auth_and_policy() {
    KnowledgePlane plane;
    auto service = service_with_knowledge(plane);
    auto valid = request("api-valid");
    const auto response = service.handle(valid, auth());
    require(response.schema_version == "cct-response-v1" && response.request_id == valid.request_id && !response.output.empty() &&
                response.policy_decision == Decision::Allow && response.backend_identity == "fixture-template-cct" && response.trace_id == valid.trace_id,
            "valid versioned API request did not produce canonical response");
    auto unauthenticated = valid;
    const auto denied = service.handle(unauthenticated, AuthContext{});
    require(denied.error_code == "AUTHENTICATION_REQUIRED" && denied.policy_decision == Decision::Deny && denied.output.empty(),
            "unauthenticated request was not denied with a stable error");
    auto wrong_tenant = valid;
    wrong_tenant.request_id = "api-cross-tenant";
    wrong_tenant.tenant_id = "tenant-b";
    const auto cross_denied = service.handle(wrong_tenant, auth("tenant-a"));
    require(cross_denied.error_code == "TENANT_OR_USER_MISMATCH", "cross-tenant request was not denied");
    auto action = valid;
    action.request_id = "api-action";
    action.requests_external_action = true;
    const auto action_denied = service.handle(action, auth());
    require(action_denied.error_code == "POLICY_EXTERNAL_ACTION" && action_denied.policy_decision == Decision::Deny,
            "external action bypassed Stage 16 policy");
}

void test_dynamic_batch_and_deadlines() {
    KnowledgePlane plane;
    auto service = service_with_knowledge(plane);
    std::vector<InferenceRequest> requests;
    for (std::size_t index = 0U; index < 3U; ++index) {
        auto item = request("batch-" + std::to_string(index), "tenant-a", "batch-session-" + std::to_string(index));
        item.input = "batch input " + std::to_string(index);
        item.retrieval_policy = "none";
        item.deadline_epoch_milliseconds = service.now_epoch_milliseconds() + 10000;
        requests.push_back(item);
        require(service.enqueue(item, auth()).accepted, "valid request was not admitted");
    }
    const auto responses = service.process_pending(3U);
    require(responses.size() == 3U && service.pending_count() == 0U && service.metrics().batches_processed == 1U &&
                service.metrics().maximum_observed_batch_size == 3U,
            "dynamic batch did not drain exactly one bounded batch");
    for (std::size_t index = 0U; index < responses.size(); ++index) {
        require(responses[index].request_id == requests[index].request_id && responses[index].usage.batch_size == 3U &&
                    responses[index].output.find(requests[index].input) != std::string::npos,
                "batch output identity or ordering changed");
    }
    auto expired = request("batch-expired");
    expired.deadline_epoch_milliseconds = service.now_epoch_milliseconds() - 1;
    const auto rejected = service.enqueue(expired, auth());
    require(!rejected.accepted && rejected.rejection.error_code == "DEADLINE_EXPIRED", "expired deadline was admitted");
}

void test_streaming_cancellation_backpressure_and_pending_cancel() {
    KnowledgePlane plane;
    auto service = service_with_knowledge(plane);
    auto stream = request("stream-cancel");
    stream.retrieval_policy = "none";
    stream.stream = true;
    const auto cancelled = service.execute_stream(stream, auth(), 4U, true);
    require(cancelled.cancelled && cancelled.resources_released && cancelled.events.size() >= 2U &&
                cancelled.events.front().type == StreamEventType::Token && cancelled.events.back().type == StreamEventType::Cancelled,
            "client stream cancellation did not release resources");
    auto backpressure = request("stream-backpressure");
    backpressure.retrieval_policy = "none";
    const auto bounded = service.execute_stream(backpressure, auth(), 0U, false);
    require(bounded.backpressure_applied && bounded.cancelled && bounded.resources_released,
            "backpressure budget did not cancel and release the stream");
    auto pending = request("pending-cancel");
    pending.retrieval_policy = "none";
    require(service.enqueue(pending, auth()).accepted && service.cancel(pending.request_id) && service.pending_count() == 0U,
            "queued cancellation did not remove the request");
}

void test_retrieval_verification_and_abstention() {
    KnowledgePlane plane;
    auto service = service_with_knowledge(plane);
    auto grounded = request("retrieval-grounded");
    grounded.retrieval_policy = "required";
    const auto response = service.handle(grounded, auth());
    require(response.policy_decision == Decision::Allow && !response.abstention && !response.citations.empty() &&
                response.output == "The retention policy is seven days.",
            "Stage 15 retrieval and citation verification did not integrate with serving");
    auto missing = request("retrieval-missing");
    missing.input = "a document topic absent from the knowledge plane";
    missing.retrieval_policy = "required";
    const auto abstained = service.handle(missing, auth());
    require(abstained.abstention && abstained.policy_decision == Decision::Abstain && abstained.error_code == "EVIDENCE_MISSING",
            "required missing evidence did not produce a safe abstention");
    service.set_fault(ServiceFault::Verifier);
    auto verifier_fault = request("retrieval-verifier-fault");
    verifier_fault.retrieval_policy = "required";
    const auto fault_response = service.handle(verifier_fault, auth());
    require(fault_response.abstention && fault_response.error_code == "VERIFIER_UNAVAILABLE" && fault_response.output.empty(),
            "verifier dependency fault did not fail closed");
}

void test_state_cache_isolation_and_version_fail_closed() {
    KnowledgePlane plane;
    auto service = service_with_knowledge(plane);
    auto first = request("state-first", "tenant-a", "state-a");
    first.retrieval_policy = "none";
    const auto first_response = service.handle(first, auth());
    auto repeated = first;
    repeated.request_id = "state-cache-hit";
    repeated.trace_id = "trace-state-cache-hit";
    const auto cached = service.handle(repeated, auth());
    require(!first_response.usage.cache_hit && cached.usage.cache_hit && service.metrics().cache_hits == 1U,
            "versioned response cache did not hit on an exact safe key");
    const auto before = service.state_snapshots();
    require(before.size() == 1U && before.front().tenant_id == "tenant-a" && before.front().state_bytes > 0U,
            "recurrent state was not created with tenant identity and bounded bytes");
    auto other_tenant = request("state-tenant-b", "tenant-b", "state-a");
    other_tenant.retrieval_policy = "none";
    const auto other_response = service.handle(other_tenant, auth("tenant-b"));
    require(!other_response.output.empty() && service.state_snapshots().size() == 2U,
            "state or cache crossed tenant boundaries");
    auto wrong_tokenizer = first;
    wrong_tokenizer.request_id = "state-wrong-tokenizer";
    wrong_tokenizer.tokenizer_version = "tokenizer-v2";
    const auto mismatch = service.handle(wrong_tokenizer, auth());
    require(mismatch.error_code == "TOKENIZER_VERSION_MISMATCH", "tokenizer mismatch was accepted");
    service.reset_state("tenant-a", "tenant-a-user", "state-a");
    require(service.state_metrics().resets == 1U && service.state_snapshots().size() == 1U,
            "session reset did not remove only the requested tenant state");
}

void test_model_routing_and_dependency_mismatch() {
    KnowledgePlane plane;
    auto service = service_with_knowledge(plane);
    service.register_release({"hybrid-release", "model-hybrid-v1", "adapter-hybrid-v1", "tokenizer-stage10-v1", "lexical-v1", "digest-hybrid", ModelRoute::Hybrid});
    service.register_release({"transformer-release", "model-transformer-v1", "adapter-transformer-v1", "tokenizer-stage10-v1", "lexical-v1", "digest-transformer", ModelRoute::Transformer});
    auto hybrid = request("route-hybrid");
    hybrid.model_version = "model-hybrid-v1";
    hybrid.adapter_version = "adapter-hybrid-v1";
    hybrid.retrieval_policy = "none";
    const auto hybrid_response = service.handle(hybrid, auth());
    require(hybrid_response.backend_identity == "fixture-template-hybrid" && hybrid_response.model_version == "model-hybrid-v1",
            "hybrid model route did not preserve release identity");
    auto unknown = request("route-unknown");
    unknown.model_version = "model-unknown";
    unknown.retrieval_policy = "none";
    const auto unknown_response = service.handle(unknown, auth());
    require(unknown_response.error_code == "MODEL_OR_DEPENDENCY_VERSION_UNAVAILABLE", "unknown model route was accepted");
    auto transformer = request("route-transformer", "tenant-a", "transformer-session");
    transformer.model_version = "model-transformer-v1";
    transformer.adapter_version = "adapter-transformer-v1";
    transformer.retrieval_policy = "none";
    const auto transformer_response = service.handle(transformer, auth());
    require(transformer_response.backend_identity == "fixture-template-transformer-control", "transformer control route was not observable");
}

void test_concurrent_callers_cache_bound_and_state_accounting() {
    InferenceConfig config;
    config.maximum_cache_entries = 2U;
    KnowledgePlane plane;
    plane.ingest(knowledge_record("policy", "tenant-a", "The retention policy is seven days."));
    plane.ingest(knowledge_record("beta-policy", "tenant-b", "Tenant beta retention policy is thirty days."));
    InferenceService service(config, &plane);
    service.set_slo_thresholds({1500.0, 150.0, 0.95, 0.2, 600000.0});
    std::vector<std::thread> workers;
    for (std::size_t worker = 0U; worker < 4U; ++worker) {
        workers.emplace_back([&service, worker]() {
            for (std::size_t index = 0U; index < 10U; ++index) {
                auto item = request("concurrent-" + std::to_string(worker) + "-" + std::to_string(index), "tenant-a",
                                     "concurrent-session-" + std::to_string(worker) + "-" + std::to_string(index));
                item.input = "concurrent input " + std::to_string(worker) + " " + std::to_string(index);
                static_cast<void>(service.handle(item, auth()));
            }
        });
    }
    for (auto& worker : workers) worker.join();
    const auto metrics = service.metrics();
    require(metrics.submitted_requests == 40U && metrics.successful_requests == 40U && metrics.rejected_requests == 0U,
            "concurrent callers lost or duplicated accepted requests");
    require(metrics.cache_evictions > 0U, "cache entry bound did not evict under concurrent workload");
    require(metrics.total_state_bytes == service.state_metrics().bytes_in_use, "active state byte accounting drifted");
    const auto before_reset = metrics.total_state_bytes;
    service.reset_state("tenant-a", "tenant-a-user", "concurrent-session-0-0");
    const auto after_reset = service.metrics().total_state_bytes;
    require(after_reset < before_reset && after_reset == service.state_metrics().bytes_in_use, "reset did not recompute active state bytes");
}

void test_faults_circuit_and_slo_observability() {
    InferenceConfig config;
    config.circuit_failure_threshold = 2U;
    config.circuit_reset_milliseconds = 1000;
    KnowledgePlane plane;
    plane.ingest(knowledge_record("policy", "tenant-a", "The retention policy is seven days."));
    InferenceService service(config, &plane);
    auto input = request("fault-1");
    input.retrieval_policy = "none";
    service.set_fault(ServiceFault::Worker);
    const auto first = service.handle(input, auth());
    input.request_id = "fault-2";
    input.trace_id = "trace-fault-2";
    const auto second = service.handle(input, auth());
    require(first.abstention && second.abstention && !service.healthy(), "worker fault did not produce bounded failures");
    input.request_id = "fault-circuit";
    const auto circuit = service.handle(input, auth());
    require(circuit.error_code == "CIRCUIT_OPEN" && service.circuit_open(), "circuit breaker did not reject after threshold");
    service.clear_fault();
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    auto recovered = request("fault-recovered");
    recovered.retrieval_policy = "none";
    const auto response = service.handle(recovered, auth());
    require(response.error_code.empty() && service.healthy(), "circuit did not recover after reset interval");
    const auto slo = service.evaluate_slo();
    require(slo.considered_requests > 0U && slo.successful_requests > 0U && slo.first_token_p95_milliseconds >= 0.0 &&
                slo.first_token_p95_milliseconds <= 1500.0 && slo.total_p95_milliseconds >= 0.0 && slo.total_p99_milliseconds >= slo.total_p95_milliseconds &&
                slo.throughput_requests_per_second >= 0.0 && slo.throughput_tokens_per_second >= 0.0 && !service.audit().empty(),
            "SLO percentile, throughput, or audit metrics were incomplete");
    for (const auto& record : service.audit()) require(record.sensitive_data_redacted && record.input_digest.find("retention policy") == std::string::npos,
                                                         "sensitive input was written to service audit");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"checkpoint_backend_generation_and_streaming", test_checkpoint_backend_generation_and_streaming},
        {"versioned_api_auth_and_policy", test_versioned_api_auth_and_policy},
        {"dynamic_batch_and_deadlines", test_dynamic_batch_and_deadlines},
        {"streaming_cancellation_backpressure_and_pending_cancel", test_streaming_cancellation_backpressure_and_pending_cancel},
        {"retrieval_verification_and_abstention", test_retrieval_verification_and_abstention},
        {"state_cache_isolation_and_version_fail_closed", test_state_cache_isolation_and_version_fail_closed},
        {"concurrent_callers_cache_bound_and_state_accounting", test_concurrent_callers_cache_bound_and_state_accounting},
        {"model_routing_and_dependency_mismatch", test_model_routing_and_dependency_mismatch},
        {"faults_circuit_and_slo_observability", test_faults_circuit_and_slo_observability}};
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
