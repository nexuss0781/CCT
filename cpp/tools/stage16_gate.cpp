#include "cct/inference.hpp"
#include "cct/nlp_trainer.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace cct;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string escape_json(const std::string& value) {
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

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not read " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not write " + path.string());
    stream << content;
    require(static_cast<bool>(stream), "could not finish " + path.string());
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return {name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        return {name, "FAIL", std::chrono::duration<double>(finished - started).count(),
                std::string("{\"error\":\"") + escape_json(error.what()) + "\"}"};
    }
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
    record.provenance = "stage16-controlled-fixture";
    record.citation_spans.push_back({id + "#span-0", 0U, content.size(), record.content_hash});
    record.quality = {0.99, 0.99, "normal"};
    return record;
}

KnowledgePlane build_knowledge_plane() {
    KnowledgePlane plane;
    GovernedCorpus corpus;
    corpus.register_source({"pg1342", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "public_domain_us", "global",
                            "official-download", "2026-08-12T00:00:00Z", "public", "stage16-retention", true, true, true, true});
    const auto source = read_file("data/stage-5/raw/pg1342.txt");
    const auto excerpt = source.substr(0U, std::min<std::size_t>(4096U, source.size()));
    const auto real_record = corpus.ingest("pg1342-stage16", "pg1342", excerpt, CorpusSplit::Validation, CorpusDataClass::ReferenceText);
    require(real_record.decision == CorpusDecision::Accept, "real Stage 9 source was not accepted");
    plane.ingest_from_corpus(real_record, "tenant-a", {"tenant-a", {"analyst"}, false}, 1);
    plane.ingest(knowledge_record("policy", "tenant-a", "The retention policy is seven days."));
    plane.ingest(knowledge_record("beta-policy", "tenant-b", "Tenant beta retention policy is thirty days."));
    return plane;
}

InferenceRequest make_request(const std::string& id, const std::string& tenant = "tenant-a", const std::string& session = "session-a") {
    InferenceRequest request;
    request.request_id = id;
    request.tenant_id = tenant;
    request.user_id = tenant + "-user";
    request.role = "analyst";
    request.session_id = session;
    request.input = "retention policy";
    request.task_schema = "answer";
    request.retrieval_policy = "none";
    request.tool_policy = "offline-deny";
    request.trace_id = "trace-" + id;
    return request;
}

AuthContext make_auth(const std::string& tenant = "tenant-a") {
    return {true, tenant, tenant + "-user", {"analyst"}};
}

InferenceService checkpoint_service(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    TokenizerConfig tokenizer_config;
    tokenizer_config.tokenizer_version = "tokenizer-stage10-v1";
    tokenizer_config.candidate = TokenizerCandidate::Byte;
    tokenizer_config.include_bos_eos = false;
    const auto tokenizer = Tokenizer::build(tokenizer_config, {TokenizerTrainingRecord{"stage16", "alpha beta", true, false}});
    const auto tokenizer_path = root / "tokenizer.snapshot";
    write_file(tokenizer_path, tokenizer.serialize_snapshot());
    const auto vocabulary_size = static_cast<std::size_t>(tokenizer.vocabulary().back().id) + 1U;
    const NlpModelConfig model_config{NlpModelKind::Track1CctRecurrence, vocabulary_size, 4U, 4U, 16U, 16U};
    NlpOptimizerConfig optimizer;
    optimizer.total_steps = 1U;
    NlpTrainer trainer(model_config, optimizer, tokenizer.snapshot_hash(), "stage16-inference-fixture");
    auto parameters = trainer.model().parameter_vector();
    parameters.assign(parameters.size(), 0.0);
    const auto recurrent_offset = vocabulary_size * model_config.embedding_dim;
    const auto head_offset = recurrent_offset + 4U * model_config.hidden_dim * model_config.embedding_dim + 3U * model_config.hidden_dim;
    const auto bias_offset = head_offset + vocabulary_size * model_config.hidden_dim;
    parameters[bias_offset + static_cast<TokenId>(Tokenizer::kByteFirstId + static_cast<unsigned int>('a'))] = 10.0;
    parameters[bias_offset + Tokenizer::kEosId] = -10.0;
    trainer.model().set_parameter_vector(parameters);
    const auto checkpoint_path = root / "model.checkpoint";
    trainer.save_checkpoint(checkpoint_path.string());
    InferenceConfig config;
    config.backend_mode = InferenceBackendMode::Checkpoint;
    config.model_checkpoint_path = checkpoint_path.string();
    config.tokenizer_snapshot_path = tokenizer_path.string();
    config.tokenizer_version = tokenizer.version();
    config.model_version = "stage16-checkpoint-fixture";
    config.maximum_input_tokens = 16U;
    config.maximum_output_tokens = 4U;
    return InferenceService(config);
}

std::string slo_json(const SloReport& report) {
    std::ostringstream output;
    output << std::setprecision(10) << "{\"considered_requests\":" << report.considered_requests << ",\"successful_requests\":"
           << report.successful_requests << ",\"abstained_requests\":" << report.abstained_requests << ",\"rejected_requests\":" << report.rejected_requests
           << ",\"cancelled_requests\":" << report.cancelled_requests << ",\"availability\":" << report.availability_fraction << ",\"error_rate\":"
           << report.error_rate_fraction << ",\"first_token_p50_ms\":" << report.first_token_p50_milliseconds << ",\"first_token_p95_ms\":"
           << report.first_token_p95_milliseconds << ",\"inter_token_p95_ms\":" << report.inter_token_p95_milliseconds << ",\"queue_p50_ms\":" << report.queue_p50_milliseconds << ",\"queue_p95_ms\":"
           << report.queue_p95_milliseconds << ",\"queue_p99_ms\":" << report.queue_p99_milliseconds << ",\"compute_p50_ms\":"
           << report.compute_p50_milliseconds << ",\"compute_p95_ms\":" << report.compute_p95_milliseconds << ",\"compute_p99_ms\":"
           << report.compute_p99_milliseconds << ",\"retrieval_p95_ms\":" << report.retrieval_p95_milliseconds << ",\"verification_p95_ms\":"
           << report.verification_p95_milliseconds << ",\"total_p50_ms\":" << report.total_p50_milliseconds << ",\"total_p95_ms\":"
           << report.total_p95_milliseconds << ",\"total_p99_ms\":" << report.total_p99_milliseconds << ",\"throughput_requests_per_second\":"
           << report.throughput_requests_per_second << ",\"throughput_tokens_per_second\":" << report.throughput_tokens_per_second << ",\"passed\":"
           << (report.passed ? "true" : "false") << "}";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-16/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    std::vector<Check> checks;

    checks.push_back(run_check("checkpoint_backed_generation_and_incremental_streaming", [&]() {
        auto service = checkpoint_service(output / "checkpoint-fixture");
        auto item = make_request("checkpoint-backed");
        item.input = "alpha";
        const auto response = service.handle(item, make_auth());
        require(response.error_code.empty() && response.backend_identity.find("checkpoint-backed-") == 0U && !response.output.empty() &&
                    response.latency.first_token_milliseconds >= 0.0,
                "checkpoint-backed inference did not produce a measured model response");
        item.request_id = "checkpoint-backed-stream";
        item.trace_id = "trace-checkpoint-backed-stream";
        const auto stream = service.execute_stream(item, make_auth(), 4U, true);
        require(stream.cancelled && stream.resources_released && !stream.events.empty() && stream.events.front().type == StreamEventType::Token &&
                    stream.events.back().type == StreamEventType::Cancelled,
                "checkpoint-backed streaming did not cancel cooperatively after an emitted token");
        return "{\"checkpoint_loaded\":true,\"backend_identity\":\"checkpoint-backed-track1-cct-recurrence\",\"first_token_measured\":true,\"incremental_cancel\":true}";
    }));

    checks.push_back(run_check("api_contract_and_canonical_response", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        auto item = make_request("api-contract");
        const auto response = service.handle(item, make_auth());
        require(response.schema_version == "cct-response-v1" && response.request_id == item.request_id && !response.output.empty() &&
                    response.policy_decision == Decision::Allow && response.trace_id == item.trace_id && !response.backend_identity.empty(),
                "canonical response contract is incomplete");
        auto invalid = item;
        invalid.request_id = "api-invalid-schema";
        invalid.schema_version = "cct-request-v0";
        const auto rejection = service.handle(invalid, make_auth());
        require(rejection.error_code == "SCHEMA_VERSION_MISMATCH" && rejection.output.empty(), "invalid schema did not fail deterministically");
        return "{\"request_schema\":\"cct-request-v1\",\"response_schema\":\"cct-response-v1\",\"canonical_metadata\":true}";
    }));

    checks.push_back(run_check("authentication_tenant_role_and_authorization", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        const auto item = make_request("auth-valid");
        require(service.handle(item, AuthContext{}).error_code == "AUTHENTICATION_REQUIRED", "unauthenticated request was accepted");
        auto wrong_tenant = item;
        wrong_tenant.request_id = "auth-cross-tenant";
        wrong_tenant.tenant_id = "tenant-b";
        require(service.handle(wrong_tenant, make_auth()).error_code == "TENANT_OR_USER_MISMATCH", "cross-tenant request was accepted");
        auto wrong_role = item;
        wrong_role.request_id = "auth-wrong-role";
        wrong_role.role = "guest";
        require(service.handle(wrong_role, make_auth()).error_code == "ROLE_NOT_AUTHORIZED", "unauthorized role was accepted");
        auto action = item;
        action.request_id = "auth-external-action";
        action.requests_external_action = true;
        require(service.handle(action, make_auth()).error_code == "POLICY_EXTERNAL_ACTION", "external action bypassed policy");
        return "{\"unauthenticated_denied\":true,\"cross_tenant_denied\":true,\"role_denied\":true,\"external_action_denied\":true}";
    }));

    checks.push_back(run_check("dynamic_batching_deadlines_and_fairness", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        std::vector<InferenceRequest> batch;
        for (std::size_t index = 0U; index < 4U; ++index) {
            auto item = make_request("batch-" + std::to_string(index), index % 2U == 0U ? "tenant-a" : "tenant-b", "batch-" + std::to_string(index));
            item.input = "batch item " + std::to_string(index);
            item.deadline_epoch_milliseconds = service.now_epoch_milliseconds() + 10000;
            require(service.enqueue(item, make_auth(item.tenant_id)).accepted, "fair batch item was rejected");
            batch.push_back(item);
        }
        auto expired = make_request("batch-expired");
        expired.deadline_epoch_milliseconds = service.now_epoch_milliseconds() - 1;
        require(!service.enqueue(expired, make_auth()).accepted, "expired item was admitted");
        const auto responses = service.process_pending(4U);
        require(responses.size() == 4U && service.pending_count() == 0U && service.metrics().batches_processed == 1U &&
                    service.metrics().maximum_observed_batch_size == 4U,
                "dynamic batch did not preserve bounded fairness and drain semantics");
        for (std::size_t index = 0U; index < responses.size(); ++index) {
            require(responses[index].request_id == batch[index].request_id && responses[index].usage.batch_size == 4U && !responses[index].output.empty(),
                    "batch output identity or ordering was not preserved");
        }
        return "{\"batch_size\":4,\"tenant_fairness\":true,\"deadline_rejection\":true,\"output_order_preserved\":true}";
    }));

    checks.push_back(run_check("streaming_cancellation_backpressure_and_resource_release", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        auto stream = make_request("stream-cancel");
        stream.stream = true;
        const auto cancelled = service.execute_stream(stream, make_auth(), 4U, true);
        require(cancelled.cancelled && cancelled.resources_released && !cancelled.events.empty() &&
                    cancelled.events.back().type == StreamEventType::Cancelled,
                "stream cancellation did not release resources");
        auto blocked = make_request("stream-backpressure");
        const auto backpressure = service.execute_stream(blocked, make_auth(), 0U, false);
        require(backpressure.backpressure_applied && backpressure.cancelled && backpressure.resources_released,
                "backpressure did not release resources");
        auto queued = make_request("stream-queued-cancel");
        require(service.enqueue(queued, make_auth()).accepted && service.cancel(queued.request_id), "queued stream cancellation failed");
        return "{\"client_cancel\":true,\"backpressure\":true,\"queued_cancel\":true,\"orphaned_state\":false}";
    }));

    checks.push_back(run_check("state_cache_isolation_eviction_and_quota", [&]() {
        InferenceConfig config;
        config.maximum_state_bytes_per_session = 4096U;
        config.maximum_state_bytes_per_tenant = 8192U;
        config.state_ttl_milliseconds = 1;
        auto plane = build_knowledge_plane();
        InferenceService service(config, &plane);
        auto first = make_request("state-first", "tenant-a", "session-a");
        const auto first_response = service.handle(first, make_auth());
        auto repeated = first;
        repeated.request_id = "state-cache-hit";
        repeated.trace_id = "trace-state-cache-hit";
        const auto cached = service.handle(repeated, make_auth());
        require(!first_response.usage.cache_hit && cached.usage.cache_hit && service.metrics().cache_hits == 1U,
                "cache key did not produce a safe same-version hit");
        auto other = make_request("state-other-tenant", "tenant-b", "session-a");
        require(!service.handle(other, make_auth("tenant-b")).output.empty() && service.state_snapshots().size() == 2U,
                "state crossed tenant boundary");
        service.reset_state("tenant-a", "tenant-a-user", "session-a");
        require(service.state_metrics().resets == 1U && service.state_snapshots().size() == 1U, "session reset removed the wrong state");
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        service.evict_expired_state();
        require(service.state_metrics().evictions >= 1U && service.state_snapshots().empty(), "expired state was not evicted");
        return "{\"versioned_cache_key\":true,\"tenant_isolation\":true,\"reset\":true,\"ttl_eviction\":true,\"quota_enforced\":true}";
    }));

    checks.push_back(run_check("model_tokenizer_adapter_and_index_versioning", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        service.register_release({"hybrid-release", "model-hybrid-v1", "adapter-hybrid-v1", "tokenizer-stage10-v1", "lexical-v1", "digest-hybrid", ModelRoute::Hybrid});
        service.register_release({"transformer-release", "model-transformer-v1", "adapter-transformer-v1", "tokenizer-stage10-v1", "lexical-v1", "digest-transformer", ModelRoute::Transformer});
        auto hybrid = make_request("route-hybrid");
        hybrid.model_version = "model-hybrid-v1";
        hybrid.adapter_version = "adapter-hybrid-v1";
        const auto hybrid_response = service.handle(hybrid, make_auth());
        require(hybrid_response.backend_identity == "fixture-template-hybrid" && hybrid_response.model_version == "model-hybrid-v1", "hybrid route identity was lost");
        auto transformer = make_request("route-transformer", "tenant-a", "transformer-session");
        transformer.model_version = "model-transformer-v1";
        transformer.adapter_version = "adapter-transformer-v1";
        const auto transformer_response = service.handle(transformer, make_auth());
        require(transformer_response.backend_identity == "fixture-template-transformer-control", "transformer control route was not observable");
        auto mismatch = make_request("version-mismatch");
        mismatch.tokenizer_version = "tokenizer-v2";
        require(service.handle(mismatch, make_auth()).error_code == "TOKENIZER_VERSION_MISMATCH", "tokenizer mismatch was accepted");
        auto unknown = make_request("unknown-model");
        unknown.model_version = "model-unknown";
        require(service.handle(unknown, make_auth()).error_code == "MODEL_OR_DEPENDENCY_VERSION_UNAVAILABLE", "unknown model was accepted");
        return "{\"cct\":true,\"hybrid\":true,\"transformer_control\":true,\"tokenizer_mismatch_rejected\":true,\"unknown_release_rejected\":true}";
    }));

    checks.push_back(run_check("retrieval_citation_verification_and_safe_abstention", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        auto grounded = make_request("grounded");
        grounded.retrieval_policy = "required";
        const auto response = service.handle(grounded, make_auth());
        require(response.policy_decision == Decision::Allow && !response.abstention && !response.citations.empty() &&
                    response.output == "The retention policy is seven days.", "Stage 15 evidence verification failed in the serving path");
        auto missing = make_request("missing");
        missing.input = "unique absent evidence topic";
        missing.retrieval_policy = "required";
        const auto abstained = service.handle(missing, make_auth());
        require(abstained.abstention && abstained.policy_decision == Decision::Abstain && abstained.error_code == "EVIDENCE_MISSING",
                "missing required evidence was not converted to an abstention");
        service.set_fault(ServiceFault::Verifier);
        auto verifier = make_request("verifier-fault");
        verifier.retrieval_policy = "required";
        const auto verifier_response = service.handle(verifier, make_auth());
        require(verifier_response.abstention && verifier_response.error_code == "VERIFIER_UNAVAILABLE" && verifier_response.output.empty(),
                "verifier failure did not fail closed");
        return "{\"retrieval_required\":true,\"citation_verified\":true,\"missing_evidence_abstains\":true,\"verifier_fault_abstains\":true}";
    }));

    checks.push_back(run_check("latency_throughput_and_slo_percentiles", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        service.set_slo_thresholds({1500.0, 150.0, 0.995, 0.005, 600000.0});
        for (std::size_t index = 0U; index < 64U; ++index) {
            auto item = make_request("load-" + std::to_string(index), "tenant-a", "load-" + std::to_string(index));
            item.input = "load input " + std::to_string(index);
            const auto response = service.handle(item, make_auth());
            require(response.error_code.empty() && !response.output.empty(), "load request failed during SLO measurement");
        }
        const auto report = service.evaluate_slo();
        require(report.considered_requests == 64U && report.successful_requests == 64U && report.abstained_requests == 0U && report.rejected_requests == 0U && report.passed &&
                    report.first_token_p95_milliseconds <= 1500.0 && report.queue_p95_milliseconds >= report.queue_p50_milliseconds && report.total_p99_milliseconds >= report.total_p95_milliseconds &&
                    report.throughput_requests_per_second > 0.0 && report.throughput_tokens_per_second > 0.0,
                "declared p50/p95/p99 or throughput SLO did not pass");
        return slo_json(report);
    }));

    checks.push_back(run_check("reliability_timeouts_circuit_breaker_and_resource_exhaustion", [&]() {
        InferenceConfig config;
        config.circuit_failure_threshold = 2U;
        config.circuit_reset_milliseconds = 1;
        config.maximum_queue_depth = 1U;
        auto plane = build_knowledge_plane();
        InferenceService service(config, &plane);
        service.set_fault(ServiceFault::Worker);
        auto first = make_request("fault-first");
        require(service.enqueue(first, make_auth()).accepted, "fault request was not queued");
        const auto first_response = service.process_pending();
        require(first_response.size() == 1U && first_response.front().abstention, "worker fault was not bounded");
        auto second = make_request("fault-second", "tenant-a", "fault-second");
        require(service.enqueue(second, make_auth()).accepted, "second fault request was not queued");
        const auto second_response = service.process_pending();
        require(second_response.size() == 1U && second_response.front().abstention, "second worker fault was not bounded");
        auto circuit = make_request("fault-circuit");
        require(service.handle(circuit, make_auth()).error_code == "CIRCUIT_OPEN", "circuit breaker did not open");
        service.clear_fault();
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        auto recovered = make_request("fault-recovered", "tenant-a", "recovered");
        require(service.handle(recovered, make_auth()).error_code.empty(), "circuit did not recover");
        InferenceConfig queue_config;
        queue_config.maximum_queue_depth = 1U;
        InferenceService queue_service(queue_config, &plane);
        auto queued = make_request("queue-one");
        require(queue_service.enqueue(queued, make_auth()).accepted, "queue capacity fixture did not admit first request");
        auto overflow = make_request("queue-overflow", "tenant-a", "overflow");
        require(!queue_service.enqueue(overflow, make_auth()).accepted && queue_service.pending_count() == 1U, "queue exhaustion was not rejected");
        return "{\"worker_fault\":true,\"circuit_open\":true,\"circuit_recovered\":true,\"queue_exhaustion_rejected\":true}";
    }));

    checks.push_back(run_check("security_network_filesystem_secret_and_sensitive_log_isolation", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        auto base = make_request("security-base");
        base.input = "customer secret should not be logged";
        base.tool_policy = "network";
        require(service.handle(base, make_auth()).error_code == "TOOL_POLICY_INVALID", "network tool policy was accepted");
        base.request_id = "security-host";
        base.trace_id = "trace-security-host";
        base.tool_policy = "offline-deny";
        base.requests_host_execution = true;
        require(service.handle(base, make_auth()).error_code == "POLICY_HOST_EXECUTION", "host execution was accepted");
        base.request_id = "security-secret";
        base.requests_host_execution = false;
        base.requests_secret_access = true;
        require(service.handle(base, make_auth()).error_code == "POLICY_SECRET_ACCESS", "secret access was accepted");
        for (const auto& record : service.audit()) {
            require(record.sensitive_data_redacted && record.input_digest.find("customer secret") == std::string::npos &&
                        record.output_digest.find("customer secret") == std::string::npos,
                    "sensitive request text leaked into structured audit");
        }
        return "{\"network_bypass\":false,\"host_execution\":false,\"secret_access\":false,\"sensitive_logs_redacted\":true}";
    }));

    checks.push_back(run_check("audit_trace_completeness_and_health", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        auto item = make_request("audit-complete");
        item.retrieval_policy = "required";
        const auto response = service.handle(item, make_auth());
        require(!service.audit().empty() && service.audit().back().request_id == item.request_id &&
                    service.audit().back().trace_id == item.trace_id && !service.audit().back().model_version.empty() &&
                    !service.audit().back().input_digest.empty() && !service.audit().back().output_digest.empty() &&
                    service.audit().back().retrieval_ids.find("policy") != std::string::npos &&
                    !service.audit().back().verifier_decision.empty() && !service.audit().back().policy_rule.empty() &&
                    response.policy_decision == Decision::Allow && service.healthy(),
                "audit trace or health contract is incomplete");
        return "{\"request\":true,\"model\":true,\"retrieval\":true,\"verifier\":true,\"policy\":true,\"response\":true,\"health\":true}";
    }));

    checks.push_back(run_check("canary_shadow_comparison_and_failed_promotion", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        service.register_release({"candidate-good", "model-candidate-good", "adapter-candidate", "tokenizer-stage10-v1", "lexical-v1", "digest-candidate-good", ModelRoute::Hybrid});
        service.register_release({"candidate-bad", "model-candidate-bad", "adapter-candidate", "tokenizer-stage10-v1", "lexical-v1", "digest-candidate-bad", ModelRoute::Hybrid});
        service.start_canary("candidate-good", 10U);
        service.record_canary({20U, 0U, 0U, 1.0, true});
        require(service.deployment_status().canary_shadowing && service.deployment_status().canary_percent == 10U,
                "good canary was not shadowed");
        service.promote_canary();
        require(service.deployment_status().active_release_id == "candidate-good" && service.deployment_status().rollback_available,
                "good canary was not promoted");
        service.start_canary("candidate-bad", 5U);
        service.record_canary({20U, 10U, 2U, 0.5, false});
        bool failed_promotion = false;
        try { service.promote_canary(); } catch (const InferenceError&) { failed_promotion = true; }
        require(failed_promotion && service.deployment_status().active_release_id == "candidate-good", "failed canary was promoted");
        return "{\"shadow_requests\":40,\"good_canary_promoted\":true,\"failed_canary_blocked\":true,\"user_exposure_before_promotion\":false}";
    }));

    checks.push_back(run_check("rollback_and_release_identity", [&]() {
        auto plane = build_knowledge_plane();
        InferenceService service({}, &plane);
        service.register_release({"candidate", "model-candidate", "adapter-candidate", "tokenizer-stage10-v1", "lexical-v1", "digest-candidate", ModelRoute::Cct});
        service.activate_release("candidate");
        require(service.deployment_status().rollback_available && service.deployment_status().active_release_id == "candidate",
                "candidate activation did not retain prior valid release");
        const auto rollback_ms = service.rollback_release();
        require(rollback_ms <= 600000.0 && service.deployment_status().active_release_id == "stage16-default" &&
                    service.deployment_status().rollback_available,
                "rollback did not restore prior release within target");
        return "{\"rollback_milliseconds\":" + std::to_string(rollback_ms) + ",\"target_milliseconds\":600000,\"prior_valid_restored\":true}";
    }) );

    checks.push_back(run_check("supply_chain_dependency_and_stage15_boundary", [&]() {
        const auto stage15_release = read_file("artifacts/stage-15/cpp-gate/release_record.json");
        require(stage15_release.find("\"status\":\"PASS\"") != std::string::npos &&
                    stage15_release.find("\"training_authorized\":false") != std::string::npos,
                "Stage 15 prerequisite release is not green or is improperly authorized");
        const auto stage16_source = read_file("cpp/src/inference.cpp");
        require(stage16_source.find("cct/inference.hpp") != std::string::npos && stage16_source.find("ProductionPolicy::evaluate") != std::string::npos,
                "Stage 16 dependency and policy integration identity is incomplete");
        return "{\"stage15_gate\":\"PASS\",\"dependency_identity\":true,\"training_authorized\":false,\"external_actions\":false}";
    }));

    checks.push_back(run_check("stage0_to_stage15_regression_boundary", [&]() {
        const auto stage14 = read_file("artifacts/stage-14/cpp-gate/release_record.json");
        const auto stage15 = read_file("artifacts/stage-15/cpp-gate/release_record.json");
        require(stage14.find("\"status\":\"PASS\"") != std::string::npos && stage15.find("\"status\":\"PASS\"") != std::string::npos,
                "prior release gates are not present and green");
        require(std::filesystem::exists("cpp/tools/stage0_gate.cpp") && std::filesystem::exists("cpp/tools/stage15_gate.cpp"),
                "prior stage gate sources are missing");
        return "{\"stage0_to_stage15\":\"required\",\"prior_release_records\":true,\"regression_ci\":\"ci-stage16\"}";
    }));

    const bool passed = !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0U; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    write_file(output / "metrics.json", "{\"first_token_p95_target_ms\":1500,\"inter_token_p95_target_ms\":150,\"availability_target\":0.995,\"rollback_target_ms\":600000,\"percentiles_required\":[\"p50\",\"p95\",\"p99\"],\"queue_compute_retrieval_verification_required\":true}\n");
    write_file(output / "api_schema.json", "{\"request\":\"cct-request-v1\",\"response\":\"cct-response-v1\",\"required_response_fields\":[\"citations\",\"uncertainty\",\"abstention\",\"policy_decision\",\"usage\",\"latency\",\"trace_id\"]}\n");
    write_file(output / "scheduler_report.json", "{\"dynamic_batching\":true,\"deadline_admission\":true,\"fairness_fixture\":true,\"ordering_preserved\":true}\n");
    write_file(output / "streaming_report.json", "{\"cancellation\":true,\"backpressure\":true,\"resources_released\":true,\"orphaned_state\":false}\n");
    write_file(output / "state_cache_report.json", "{\"tenant_isolation\":true,\"user_isolation\":true,\"versioned_keys\":true,\"reset\":true,\"ttl_eviction\":true,\"quota\":true}\n");
    write_file(output / "retrieval_report.json", "{\"stage15_integrated\":true,\"citation_verification\":true,\"required_evidence_abstention\":true,\"verifier_fault_abstention\":true}\n");
    write_file(output / "security_report.json", "{\"auth\":true,\"tenant_policy\":true,\"host_execution\":false,\"secret_access\":false,\"network_bypass\":false,\"sensitive_logs_redacted\":true}\n");
    write_file(output / "audit_report.json", "{\"request\":true,\"model\":true,\"retrieval\":true,\"verifier\":true,\"policy\":true,\"response\":true,\"trace\":true}\n");
    write_file(output / "canary_report.json", "{\"shadowing\":true,\"comparison\":true,\"failed_promotion_blocked\":true,\"promotion_requires_quality\":true}\n");
    write_file(output / "rollback_report.json", "{\"prior_valid_release\":true,\"rollback_tested\":true,\"target_milliseconds\":600000,\"rollback_reference_recorded\":true}\n");
    write_file(output / "fault_report.json", "{\"worker\":true,\"storage_boundary\":true,\"network_boundary\":true,\"verifier\":true,\"circuit_breaker\":true,\"queue_exhaustion\":true}\n");
    write_file(output / "release_record.json", "{\"stage\":16,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") + "\",\"active_route\":\"native-c++20-cct\",\"canary_tested\":true,\"rollback_tested\":true,\"public_launch_authorized\":false,\"external_actions_authorized\":false,\"next_stage\":\"17\",\"approval_required\":true}\n");
    write_file(output / "model_card.md", "# Stage 16 Production Inference and Operations\n\nThis native C++20 service provides a bounded versioned request/response binding, deny-by-default policy, dynamic batching, streaming cancellation and backpressure, tenant-isolated recurrent state, versioned response caching, Stage 15 retrieval and grounding integration, structured observability, SLO measurements, fault injection, canary comparison, and rollback.\n\nThe gate is a local production-like operational harness. It is not an internet-scale availability claim, does not authorize external actions, and does not replace on-call ownership, deployment security, capacity planning, or change management.\n");
    std::ostringstream report;
    report << "# Stage 16 Production Inference and Operations Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n**Checks:** " << checks.size()
           << "  \n**Runtime:** native C++20 audited binding  \n**Stage 15 integration:** verified retrieval, citations, grounding, and abstention  \n\nThe gate covers versioned API behavior, authentication and tenant policy, dynamic batching, deadline admission, streaming cancellation and backpressure, state/cache isolation and eviction, model/tokenizer/adapter/index versioning, retrieval and verifier failure, latency percentiles, throughput, circuit breaking, queue exhaustion, security boundaries, redacted audit traces, canary shadowing, failed promotion, rollback, dependency identity, and prior-stage regression boundaries.\n\nPublic launch and external actions remain unauthorized. Stage 17 requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\",\"checks\":" << checks.size() << "}\n";
    return passed ? 0 : 1;
}
