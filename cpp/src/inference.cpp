#include "cct/inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw InferenceError(message);
}

std::string digest(const std::string& value) {
    return GovernedCorpus::content_sha256(value);
}

std::size_t token_count(const std::string& value) {
    std::size_t count = 0U;
    bool in_token = false;
    for (const unsigned char character : value) {
        const bool separator = character == ' ' || character == '\n' || character == '\r' || character == '\t';
        if (!separator && !in_token) {
            ++count;
            in_token = true;
        } else if (separator) {
            in_token = false;
        }
    }
    return count;
}

double percentile(std::vector<double> values, const double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double bounded = std::clamp(fraction, 0.0, 1.0);
    const auto index = static_cast<std::size_t>(std::ceil(bounded * static_cast<double>(values.size())));
    return values[std::min(values.size() - 1U, std::max<std::size_t>(index, 1U) - 1U)];
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::string joined(const std::vector<std::string>& values, const char separator) {
    std::ostringstream output;
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << separator;
        output << values[index];
    }
    return output.str();
}

std::vector<std::string> split_words(const std::string& value) {
    std::istringstream input(value);
    std::vector<std::string> words;
    std::string word;
    while (input >> word) words.push_back(word);
    return words;
}

std::string route_backend(const ModelRoute route) {
    if (route == ModelRoute::Cct) return "native-c++20-cct";
    if (route == ModelRoute::Hybrid) return "native-c++20-hybrid";
    return "native-c++20-transformer-control";
}

}  // namespace

std::string model_route_name(const ModelRoute route) {
    if (route == ModelRoute::Cct) return "cct";
    if (route == ModelRoute::Hybrid) return "hybrid";
    return "transformer";
}

std::string stream_event_type_name(const StreamEventType type) {
    if (type == StreamEventType::Token) return "token";
    if (type == StreamEventType::Completed) return "completed";
    if (type == StreamEventType::Cancelled) return "cancelled";
    if (type == StreamEventType::Backpressure) return "backpressure";
    return "error";
}

std::string service_fault_name(const ServiceFault fault) {
    if (fault == ServiceFault::None) return "none";
    if (fault == ServiceFault::Worker) return "worker";
    if (fault == ServiceFault::Storage) return "storage";
    if (fault == ServiceFault::Network) return "network";
    if (fault == ServiceFault::Verifier) return "verifier";
    return "dependency";
}

void DeploymentController::register_release(const DeploymentRelease& release) {
    require(!release.release_id.empty() && !release.model_version.empty() && !release.adapter_version.empty() &&
                !release.tokenizer_version.empty() && !release.knowledge_index_version.empty() && !release.artifact_digest.empty(),
            "deployment release identity is incomplete");
    require(!has_release(release.release_id), "duplicate deployment release");
    releases_.push_back(release);
}

void DeploymentController::activate(const std::string& release_id) {
    require(has_release(release_id), "cannot activate unknown deployment release");
    if (!status_.active_release_id.empty() && status_.active_release_id != release_id) {
        status_.previous_release_id = status_.active_release_id;
    }
    const auto& selected = release(release_id);
    status_.active_release_id = selected.release_id;
    status_.active_model_version = selected.model_version;
    status_.rollback_available = !status_.previous_release_id.empty();
}

void DeploymentController::start_canary(const std::string& release_id, const std::size_t percent) {
    require(has_release(release_id), "cannot canary unknown deployment release");
    require(percent > 0U && percent <= 100U, "canary percentage must be in 1..100");
    require(!status_.active_release_id.empty() && release_id != status_.active_release_id, "canary must differ from active release");
    status_.canary_release_id = release_id;
    status_.canary_percent = percent;
    status_.canary_shadowing = true;
    status_.canary_error_rate = 0.0;
    status_.canary_quality_score = 0.0;
}

void DeploymentController::record_canary(const CanaryComparison& comparison) {
    require(!status_.canary_release_id.empty() && status_.canary_shadowing, "no active canary to record");
    require(comparison.shadow_requests > 0U, "canary comparison requires shadow traffic");
    status_.canary_error_rate = static_cast<double>(comparison.canary_errors) / static_cast<double>(comparison.shadow_requests);
    status_.canary_quality_score = comparison.quality_score;
    require(comparison.mismatches <= comparison.shadow_requests, "canary mismatch count exceeds traffic");
}

void DeploymentController::promote_canary() {
    require(!status_.canary_release_id.empty() && status_.canary_shadowing, "no active canary to promote");
    require(status_.canary_error_rate <= 0.005 && status_.canary_quality_score >= 0.95, "canary quality or error threshold failed");
    const auto old_active = status_.active_release_id;
    status_.previous_release_id = old_active;
    const auto& selected = release(status_.canary_release_id);
    status_.active_release_id = selected.release_id;
    status_.active_model_version = selected.model_version;
    status_.canary_release_id.clear();
    status_.canary_percent = 0U;
    status_.canary_shadowing = false;
    status_.rollback_available = !old_active.empty();
}

double DeploymentController::rollback() {
    require(status_.rollback_available && !status_.previous_release_id.empty(), "no prior valid release is available for rollback");
    const auto started = std::chrono::steady_clock::now();
    const auto failed_active = status_.active_release_id;
    const auto& selected = release(status_.previous_release_id);
    status_.active_release_id = selected.release_id;
    status_.active_model_version = selected.model_version;
    status_.previous_release_id = failed_active;
    status_.canary_release_id.clear();
    status_.canary_percent = 0U;
    status_.canary_shadowing = false;
    status_.rollback_available = true;
    status_.last_rollback_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return status_.last_rollback_milliseconds;
}

bool DeploymentController::has_release(const std::string& release_id) const {
    return std::find_if(releases_.begin(), releases_.end(), [&](const auto& item) { return item.release_id == release_id; }) != releases_.end();
}

const DeploymentRelease& DeploymentController::release(const std::string& release_id) const {
    const auto found = std::find_if(releases_.begin(), releases_.end(), [&](const auto& item) { return item.release_id == release_id; });
    require(found != releases_.end(), "deployment release not found");
    return *found;
}

const DeploymentStatus& DeploymentController::status() const noexcept { return status_; }

const DeploymentRelease& DeploymentController::route_release(const InferenceRequest& request) const {
    if (!request.model_version.empty()) {
        const auto found = std::find_if(releases_.begin(), releases_.end(), [&](const auto& item) {
            return item.model_version == request.model_version &&
                   (request.adapter_version.empty() || item.adapter_version == request.adapter_version) &&
                   (request.tokenizer_version.empty() || item.tokenizer_version == request.tokenizer_version) &&
                   (request.knowledge_index_version.empty() || item.knowledge_index_version == request.knowledge_index_version);
        });
        require(found != releases_.end(), "requested model or dependency version is unavailable");
        return *found;
    }
    require(!status_.active_release_id.empty(), "no active deployment release");
    return release(status_.active_release_id);
}

InferenceService::InferenceService(InferenceConfig config, KnowledgePlane* knowledge)
    : config_(std::move(config)), knowledge_(knowledge) {
    require(config_.maximum_input_tokens > 0U && config_.maximum_output_tokens > 0U && config_.maximum_batch_size > 0U &&
                config_.maximum_queue_depth > 0U && config_.maximum_state_bytes_per_session > 0U && config_.maximum_state_bytes_per_tenant > 0U,
            "inference configuration contains zero limits");
    require(config_.circuit_failure_threshold > 0U, "circuit failure threshold must be positive");
    deployment_.register_release({"stage16-default", config_.model_version, config_.adapter_version, config_.tokenizer_version,
                                  config_.knowledge_index_version, digest("stage16-default-release"), ModelRoute::Cct});
    deployment_.activate("stage16-default");
}

void InferenceService::attach_knowledge_plane(KnowledgePlane& knowledge) noexcept { knowledge_ = &knowledge; }

void InferenceService::set_slo_thresholds(const SloThresholds& thresholds) {
    require(thresholds.first_token_p95_milliseconds > 0.0 && thresholds.inter_token_p95_milliseconds > 0.0 &&
                thresholds.availability_fraction >= 0.0 && thresholds.availability_fraction <= 1.0 &&
                thresholds.error_rate_fraction >= 0.0 && thresholds.error_rate_fraction <= 1.0 && thresholds.rollback_target_milliseconds > 0.0,
            "invalid SLO thresholds");
    slo_thresholds_ = thresholds;
}

void InferenceService::set_fault(const ServiceFault fault) noexcept { fault_ = fault; }
void InferenceService::clear_fault() noexcept { fault_ = ServiceFault::None; }

std::int64_t InferenceService::now_epoch_milliseconds() const noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::optional<std::string> InferenceService::validate(const InferenceRequest& request, const AuthContext& auth) const {
    if (!auth.authenticated) return "AUTHENTICATION_REQUIRED";
    if (request.schema_version != config_.request_schema_version) return "SCHEMA_VERSION_MISMATCH";
    if (request.request_id.empty() || request.trace_id.empty() || request.tenant_id.empty() || request.user_id.empty() || request.session_id.empty()) return "IDENTITY_REQUIRED";
    if (request.tenant_id != auth.tenant_id || request.user_id != auth.user_id) return "TENANT_OR_USER_MISMATCH";
    if (!contains(auth.roles, request.role)) return "ROLE_NOT_AUTHORIZED";
    if (request.input.empty() || request.task_schema.empty()) return "INPUT_OR_TASK_REQUIRED";
    if (request.tokenizer_version != config_.tokenizer_version && !request.tokenizer_version.empty()) return "TOKENIZER_VERSION_MISMATCH";
    if (request.knowledge_index_version != config_.knowledge_index_version && !request.knowledge_index_version.empty()) return "KNOWLEDGE_INDEX_VERSION_MISMATCH";
    if (request.retrieval_policy != "none" && request.retrieval_policy != "optional" && request.retrieval_policy != "required") return "RETRIEVAL_POLICY_INVALID";
    if (request.tool_policy != "offline-deny") return "TOOL_POLICY_INVALID";
    const auto input_limit = request.max_input_tokens == 0U ? config_.maximum_input_tokens : request.max_input_tokens;
    const auto output_limit = request.max_output_tokens == 0U ? config_.maximum_output_tokens : request.max_output_tokens;
    if (input_limit == 0U || input_limit > config_.maximum_input_tokens || output_limit == 0U || output_limit > config_.maximum_output_tokens) return "TOKEN_BUDGET_EXCEEDED";
    if (token_count(request.input) > input_limit) return "INPUT_TOKEN_BUDGET_EXCEEDED";
    if (request.deadline_epoch_milliseconds != 0 && request.deadline_epoch_milliseconds <= now_epoch_milliseconds()) return "DEADLINE_EXPIRED";
    if (circuit_open()) return "CIRCUIT_OPEN";
    try { static_cast<void>(deployment_.route_release(request)); } catch (const std::exception&) { return "MODEL_OR_DEPENDENCY_VERSION_UNAVAILABLE"; }
    return std::nullopt;
}

InferenceResponse InferenceService::reject(const InferenceRequest& request, const std::string& code, const std::string& detail,
                                           const Decision decision) {
    InferenceResponse response;
    response.schema_version = config_.response_schema_version;
    response.request_id = request.request_id;
    response.model_version = request.model_version.empty() ? deployment_.status().active_model_version : request.model_version;
    response.route = request.route;
    response.policy_decision = decision;
    response.abstention = decision == Decision::Abstain;
    response.uncertainty = detail;
    response.trace_id = request.trace_id;
    response.backend_identity = route_backend(request.route);
    response.error_code = code;
    response.error_detail = detail;
    ++metrics_.rejected_requests;
    if (code == "DEADLINE_EXPIRED") ++metrics_.timed_out_requests;
    if (code == "CIRCUIT_OPEN") ++metrics_.circuit_open_rejections;
    append_audit(request, response, "rejected", {}, {}, code);
    return response;
}

Submission InferenceService::enqueue(const InferenceRequest& request, const AuthContext& auth) {
    if (const auto error = validate(request, auth); error.has_value()) return {false, request.request_id, reject(request, *error, "request rejected by versioned admission control")};
    if (pending_.size() >= config_.maximum_queue_depth) return {false, request.request_id, reject(request, "QUEUE_FULL", "request queue capacity is exhausted")};
    PendingRequest pending{request, auth, now_epoch_milliseconds()};
    if (pending.request.deadline_epoch_milliseconds == 0) pending.request.deadline_epoch_milliseconds = pending.enqueued_at + config_.default_deadline_milliseconds;
    pending_.push_back(std::move(pending));
    ++metrics_.submitted_requests;
    return {true, request.request_id, {}};
}

std::vector<InferenceResponse> InferenceService::process_batch(const std::vector<PendingRequest>& batch) {
    std::vector<InferenceResponse> responses;
    responses.reserve(batch.size());
    for (const auto& pending : batch) {
        if (pending.request.deadline_epoch_milliseconds <= now_epoch_milliseconds()) {
            responses.push_back(reject(pending.request, "DEADLINE_EXPIRED", "request expired before batch execution"));
            continue;
        }
        try {
            responses.push_back(execute(pending.request, batch.size()));
        } catch (const std::exception& error) {
            record_failure(pending.request, "EXECUTION_FAILURE");
            responses.push_back(reject(pending.request, "EXECUTION_FAILURE", error.what(), Decision::Abstain));
        }
    }
    return responses;
}

std::vector<InferenceResponse> InferenceService::process_pending(const std::size_t batch_limit) {
    if (pending_.empty()) return {};
    const auto limit = batch_limit == 0U ? config_.maximum_batch_size : std::min(batch_limit, config_.maximum_batch_size);
    const auto count = std::min(limit, pending_.size());
    std::vector<PendingRequest> batch;
    batch.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) batch.push_back(std::move(pending_[index]));
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(count));
    ++metrics_.batches_processed;
    metrics_.maximum_observed_batch_size = std::max(metrics_.maximum_observed_batch_size, batch.size());
    return process_batch(batch);
}

InferenceResponse InferenceService::handle(const InferenceRequest& request, const AuthContext& auth) {
    const auto submission = enqueue(request, auth);
    if (!submission.accepted) return submission.rejection;
    const auto responses = process_pending();
    const auto found = std::find_if(responses.begin(), responses.end(), [&](const auto& response) { return response.request_id == request.request_id; });
    require(found != responses.end(), "accepted request did not produce a response");
    return *found;
}

InferenceService::RuntimeState& InferenceService::state_for(const InferenceRequest& request) {
    const SessionKey key{request.tenant_id, request.user_id, request.session_id};
    const auto found = std::find_if(states_.begin(), states_.end(), [&](const auto& state) { return state.key == key; });
    if (found != states_.end()) {
        require(found->snapshot.model_version == deployment_.route_release(request).model_version &&
                    found->snapshot.adapter_version == deployment_.route_release(request).adapter_version &&
                    found->snapshot.tokenizer_version == deployment_.route_release(request).tokenizer_version &&
                    found->snapshot.knowledge_index_version == deployment_.route_release(request).knowledge_index_version,
                "state dependency version mismatch rejected");
        return *found;
    }
    RuntimeState state;
    state.key = key;
    const auto& release = deployment_.route_release(request);
    state.snapshot = {request.tenant_id, request.user_id, request.session_id, release.model_version, release.adapter_version,
                      release.tokenizer_version, release.knowledge_index_version, digest(request.session_id), 0U, now_epoch_milliseconds()};
    states_.push_back(std::move(state));
    state_metrics_.active_sessions = states_.size();
    state_metrics_.active_tenants = 0U;
    for (const auto& item : states_) {
        if (std::find_if(states_.begin(), states_.begin() + static_cast<std::ptrdiff_t>(&item - states_.data()),
                         [&](const auto& previous) { return previous.snapshot.tenant_id == item.snapshot.tenant_id; }) == states_.begin() + static_cast<std::ptrdiff_t>(&item - states_.data())) {
            ++state_metrics_.active_tenants;
        }
    }
    return states_.back();
}

void InferenceService::update_state(RuntimeState& state, const InferenceRequest& request, const std::size_t output_tokens) {
    const auto next_digest = digest(state.transcript_digest + "|" + request.input + "|" + std::to_string(output_tokens));
    const auto bytes = static_cast<std::size_t>(next_digest.size() + request.input.size() + 64U);
    require(bytes <= config_.maximum_state_bytes_per_session, "per-session state quota exceeded");
    std::size_t tenant_bytes = 0U;
    for (const auto& item : states_) if (item.snapshot.tenant_id == request.tenant_id) tenant_bytes += item.snapshot.state_bytes;
    require(tenant_bytes - state.snapshot.state_bytes + bytes <= config_.maximum_state_bytes_per_tenant, "per-tenant state quota exceeded");
    state.transcript_digest = next_digest;
    state.snapshot.state_digest = next_digest;
    state.snapshot.state_bytes = bytes;
    state.snapshot.last_used_epoch_milliseconds = now_epoch_milliseconds();
    state_metrics_.bytes_in_use = 0U;
    for (const auto& item : states_) state_metrics_.bytes_in_use += item.snapshot.state_bytes;
    metrics_.total_state_bytes += bytes;
}

InferenceResponse InferenceService::execute(const InferenceRequest& request, const std::size_t batch_size) {
    if (fault_ == ServiceFault::Worker || fault_ == ServiceFault::Storage || fault_ == ServiceFault::Network || fault_ == ServiceFault::Dependency) {
        throw InferenceError("injected " + service_fault_name(fault_) + " fault");
    }
    const auto& release = deployment_.route_release(request);
    const auto use_case = ProductUseCase{request.task_schema, "Stage16 bounded inference", ApplicationKind::GroundedAnswer,
                                         {request.task_schema}, {"send_email", "submit_payment", "execute_code"}, true, "stage16-owner", "2027-12-31"};
    const PolicyRequest policy_request{request.tenant_id, use_case.id, request.task_schema, request.input, request.requests_external_action,
                                       request.requests_host_execution, request.requests_secret_access, false, false};
    const auto policy = ProductionPolicy::evaluate(policy_request, use_case);
    if (policy.decision != Decision::Allow) return reject(request, policy.rule_id, policy.reason, policy.decision);
    auto& state = state_for(request);
    const auto cache_key = request.tenant_id + "|" + request.user_id + "|" + request.session_id + "|" + release.model_version + "|" +
                           release.adapter_version + "|" + release.tokenizer_version + "|" + release.knowledge_index_version + "|" +
                           request.role + "|" + request.task_schema + "|" + request.retrieval_policy + "|" + digest(request.input);
    const auto cached = std::find_if(cache_.begin(), cache_.end(), [&](const auto& entry) { return entry.cache_key == cache_key; });
    if (cached != cache_.end() && fault_ == ServiceFault::None) {
        auto cached_response = cached->response;
        cached_response.request_id = request.request_id;
        cached_response.trace_id = request.trace_id;
        cached_response.usage.batch_size = batch_size;
        cached_response.usage.cache_hit = true;
        cached_response.latency.queue_milliseconds = 0.0;
        cached_response.latency.total_milliseconds = 0.0;
        ++metrics_.cache_hits;
        ++metrics_.completed_requests;
        metrics_.queue_latencies.push_back(0.0);
        metrics_.compute_latencies.push_back(0.0);
        metrics_.retrieval_latencies.push_back(cached_response.latency.retrieval_milliseconds);
        metrics_.verification_latencies.push_back(cached_response.latency.verification_milliseconds);
        metrics_.total_latencies.push_back(0.0);
        append_audit(request, cached_response, "cache_hit", {}, cached_response.abstention ? "abstain" : "accepted", policy.rule_id);
        return cached_response;
    }
    InferenceResponse response;
    response.schema_version = config_.response_schema_version;
    response.request_id = request.request_id;
    response.model_version = release.model_version;
    response.route = release.route;
    response.trace_id = request.trace_id;
    response.policy_decision = Decision::Allow;
    response.backend_identity = route_backend(release.route);
    response.usage.batch_size = batch_size;
    response.usage.input_tokens = token_count(request.input);
    response.usage.output_tokens = 0U;
    const auto compute_started = std::chrono::steady_clock::now();
    std::vector<KnowledgeHit> hits;
    std::vector<std::string> retrieval_ids;
    double retrieval_milliseconds = 0.0;
    if (request.retrieval_policy != "none") {
        require(knowledge_ != nullptr, "retrieval policy requires an attached knowledge plane");
        const auto retrieval_started = std::chrono::steady_clock::now();
        KnowledgeQuery query;
        query.query_id = request.request_id + "-retrieval";
        query.tenant_id = request.tenant_id;
        query.role = request.role;
        query.text = request.input;
        query.mode = RetrievalMode::Hybrid;
        query.valid_at = now_epoch_milliseconds();
        query.top_k = 3U;
        query.embedding_version = "embedding-v1";
        query.lexical_index_version = release.knowledge_index_version;
        hits = knowledge_->retrieve(query);
        retrieval_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - retrieval_started).count();
        for (const auto& hit : hits) retrieval_ids.push_back(hit.knowledge_id);
        if (request.retrieval_policy == "required" && hits.empty()) {
            response.abstention = true;
            response.policy_decision = Decision::Abstain;
            response.uncertainty = "required evidence is unavailable";
            response.error_code = "EVIDENCE_MISSING";
            response.error_detail = response.uncertainty;
            append_audit(request, response, "abstained", joined(retrieval_ids, ','), "not_run", policy.rule_id);
            ++metrics_.completed_requests;
            return response;
        }
    }
    if (!hits.empty()) {
        if (fault_ == ServiceFault::Verifier) {
            response.abstention = true;
            response.policy_decision = Decision::Abstain;
            response.uncertainty = "verifier dependency unavailable";
            response.error_code = "VERIFIER_UNAVAILABLE";
            response.error_detail = response.uncertainty;
            append_audit(request, response, "abstained", joined(retrieval_ids, ','), "fault", policy.rule_id);
            ++metrics_.completed_requests;
            return response;
        }
        response.output = hits.front().content;
        for (const auto& span : hits.front().citation_spans) response.citations.push_back(span.span_id);
        GroundedAnswerRequest grounded;
        grounded.answer_id = request.request_id + "-answer";
        grounded.query_id = request.request_id + "-retrieval";
        grounded.mode = RetrievalMode::Hybrid;
        grounded.answer_text = response.output;
        grounded.claims.push_back({request.request_id + "-claim", response.output, response.citations});
        const auto verification_started = std::chrono::steady_clock::now();
        const auto verified = knowledge_->verify_answer(grounded, hits);
        response.latency.verification_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verification_started).count();
        if (!verified.accepted) {
            response.output.clear();
            response.citations.clear();
            response.abstention = true;
            response.policy_decision = Decision::Abstain;
            response.uncertainty = verified.reason;
            response.error_code = "GROUNDING_REJECTED";
            response.error_detail = verified.reason;
            append_audit(request, response, "abstained", joined(retrieval_ids, ','), verified.reason, policy.rule_id);
            ++metrics_.completed_requests;
            return response;
        }
        response.uncertainty = "grounded evidence verified";
        response.usage.cache_hit = false;
    } else {
        const auto prefix = release.route == ModelRoute::Cct ? "CCT-ASE" : (release.route == ModelRoute::Hybrid ? "CCT-HYBRID" : "TRANSFORMER-CONTROL");
        response.output = std::string(prefix) + " response: " + request.input;
        response.uncertainty = "bounded runtime response without retrieval";
    }
    response.usage.output_tokens = token_count(response.output);
    require(response.usage.output_tokens <= (request.max_output_tokens == 0U ? config_.maximum_output_tokens : request.max_output_tokens), "output token budget exceeded");
    update_state(state, request, response.usage.output_tokens);
    response.usage.state_bytes = state.snapshot.state_bytes;
    response.latency.compute_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compute_started).count();
    response.latency.total_milliseconds = response.latency.queue_milliseconds + response.latency.compute_milliseconds + response.latency.retrieval_milliseconds + response.latency.verification_milliseconds;
    response.latency.retrieval_milliseconds = retrieval_milliseconds;
    response.latency.total_milliseconds = response.latency.queue_milliseconds + response.latency.compute_milliseconds;
    if (retrieval_milliseconds > 0.0) response.latency.total_milliseconds += retrieval_milliseconds + response.latency.verification_milliseconds;
    metrics_.total_input_tokens += response.usage.input_tokens;
    metrics_.total_output_tokens += response.usage.output_tokens;
    if (response.usage.cache_hit) ++metrics_.cache_hits;
    metrics_.queue_latencies.push_back(response.latency.queue_milliseconds);
    metrics_.compute_latencies.push_back(response.latency.compute_milliseconds);
    metrics_.retrieval_latencies.push_back(response.latency.retrieval_milliseconds);
    metrics_.verification_latencies.push_back(response.latency.verification_milliseconds);
    metrics_.total_latencies.push_back(response.latency.total_milliseconds);
    ++metrics_.completed_requests;
    consecutive_failures_ = 0U;
    if (!response.abstention && response.error_code.empty()) {
        cache_.push_back({cache_key, request.tenant_id, request.user_id, request.session_id, response, now_epoch_milliseconds()});
    }
    append_audit(request, response, "completed", joined(retrieval_ids, ','), response.abstention ? "abstain" : "accepted", policy.rule_id);
    return response;
}

void InferenceService::append_audit(const InferenceRequest& request, const InferenceResponse& response, const std::string& event_type,
                                    const std::string& retrieval_ids, const std::string& verifier_decision, const std::string& policy_rule) {
    audit_.push_back({request.request_id + "-" + std::to_string(audit_.size()), request.request_id, request.trace_id, request.tenant_id,
                      response.model_version, event_type, response.policy_decision == Decision::Allow ? "allow" :
                      (response.policy_decision == Decision::Abstain ? "abstain" : "deny"), digest(request.input),
                      response.output.empty() ? std::string{} : digest(response.output), retrieval_ids, verifier_decision, policy_rule,
                      response.error_code, true});
}

void InferenceService::record_failure(const InferenceRequest& request, const std::string& code) {
    ++consecutive_failures_;
    if (consecutive_failures_ >= config_.circuit_failure_threshold) circuit_open_at_ = now_epoch_milliseconds();
    InferenceResponse response;
    response.request_id = request.request_id;
    response.error_code = code;
    append_audit(request, response, "fault", {}, "not_run", code);
}

bool InferenceService::maybe_reset_circuit() {
    if (circuit_open_at_ == 0) return true;
    if (now_epoch_milliseconds() - circuit_open_at_ >= config_.circuit_reset_milliseconds) {
        circuit_open_at_ = 0;
        consecutive_failures_ = 0U;
        return true;
    }
    return false;
}

bool InferenceService::circuit_open() const noexcept {
    if (circuit_open_at_ == 0) return false;
    const_cast<InferenceService*>(this)->maybe_reset_circuit();
    return circuit_open_at_ != 0;
}

StreamResult InferenceService::execute_stream(const InferenceRequest& request, const AuthContext& auth,
                                              const std::size_t event_budget, const bool cancel_after_first) {
    StreamResult result;
    if (event_budget == 0U) {
        result.events.push_back({request.request_id, StreamEventType::Backpressure, "event budget exhausted", 0U});
        result.events.push_back({request.request_id, StreamEventType::Cancelled, "stream cancelled before execution", 1U});
        result.backpressure_applied = true;
        result.cancelled = true;
        result.resources_released = true;
        ++metrics_.cancelled_requests;
        return result;
    }
    InferenceRequest stream_request = request;
    stream_request.stream = true;
    const auto response = handle(stream_request, auth);
    if (!response.error_code.empty() && response.output.empty()) {
        result.events.push_back({request.request_id, StreamEventType::Error, response.error_code, 0U});
        result.resources_released = true;
        return result;
    }
    const auto words = split_words(response.output);
    std::size_t sequence = 0U;
    for (const auto& word : words) {
        if (sequence >= event_budget) {
            result.events.push_back({request.request_id, StreamEventType::Backpressure, "stream event budget exhausted", sequence});
            result.backpressure_applied = true;
            result.cancelled = true;
            ++metrics_.cancelled_requests;
            result.events.push_back({request.request_id, StreamEventType::Cancelled, "cancelled under backpressure", sequence + 1U});
            result.resources_released = true;
            return result;
        }
        result.events.push_back({request.request_id, StreamEventType::Token, word, sequence});
        ++sequence;
        if (cancel_after_first) {
            result.events.push_back({request.request_id, StreamEventType::Cancelled, "client cancellation", sequence});
            result.cancelled = true;
            ++metrics_.cancelled_requests;
            result.resources_released = true;
            return result;
        }
    }
    result.events.push_back({request.request_id, StreamEventType::Completed, response.abstention ? response.uncertainty : "stream completed", sequence});
    result.resources_released = true;
    return result;
}

bool InferenceService::cancel(const std::string& request_id) {
    const auto found = std::find_if(pending_.begin(), pending_.end(), [&](const auto& item) { return item.request.request_id == request_id; });
    if (found == pending_.end()) return false;
    const auto request = found->request;
    pending_.erase(found);
    ++metrics_.cancelled_requests;
    InferenceResponse response = reject(request, "CANCELLED", "request cancelled before execution", Decision::Abstain);
    append_audit(request, response, "cancelled", {}, "not_run", "CANCELLED");
    return true;
}

void InferenceService::reset_state(const std::string& tenant_id, const std::string& user_id, const std::string& session_id) {
    const auto found = std::remove_if(states_.begin(), states_.end(), [&](const auto& item) {
        return item.snapshot.tenant_id == tenant_id && item.snapshot.user_id == user_id && item.snapshot.session_id == session_id;
    });
    const auto cache_found = std::remove_if(cache_.begin(), cache_.end(), [&](const auto& entry) {
        return entry.tenant_id == tenant_id && entry.user_id == user_id && entry.session_id == session_id;
    });
    cache_.erase(cache_found, cache_.end());
    if (found != states_.end()) {
        states_.erase(found, states_.end());
        ++state_metrics_.resets;
    }
    state_metrics_.active_sessions = states_.size();
}

void InferenceService::evict_expired_state() {
    const auto now = now_epoch_milliseconds();
    const auto found = std::remove_if(states_.begin(), states_.end(), [&](const auto& item) {
        return now - item.snapshot.last_used_epoch_milliseconds > config_.state_ttl_milliseconds;
    });
    const auto cache_found = std::remove_if(cache_.begin(), cache_.end(), [&](const auto& entry) {
        return now - entry.created_at > config_.state_ttl_milliseconds;
    });
    cache_.erase(cache_found, cache_.end());
    const auto removed = static_cast<std::size_t>(states_.end() - found);
    if (removed > 0U) {
        states_.erase(found, states_.end());
        state_metrics_.evictions += removed;
    }
    state_metrics_.active_sessions = states_.size();
    state_metrics_.bytes_in_use = 0U;
    for (const auto& item : states_) state_metrics_.bytes_in_use += item.snapshot.state_bytes;
}

std::vector<StateSnapshot> InferenceService::state_snapshots() const {
    std::vector<StateSnapshot> snapshots;
    snapshots.reserve(states_.size());
    for (const auto& state : states_) snapshots.push_back(state.snapshot);
    return snapshots;
}

SloReport InferenceService::evaluate_slo() const {
    SloReport report;
    report.considered_requests = metrics_.submitted_requests;
    report.successful_requests = metrics_.completed_requests;
    if (report.considered_requests > 0U) {
        report.availability_fraction = static_cast<double>(report.successful_requests) / static_cast<double>(report.considered_requests);
        report.error_rate_fraction = static_cast<double>(metrics_.rejected_requests) / static_cast<double>(report.considered_requests);
    }
    report.queue_p50_milliseconds = percentile(metrics_.queue_latencies, 0.50);
    report.queue_p95_milliseconds = percentile(metrics_.queue_latencies, 0.95);
    report.queue_p99_milliseconds = percentile(metrics_.queue_latencies, 0.99);
    report.compute_p50_milliseconds = percentile(metrics_.compute_latencies, 0.50);
    report.compute_p95_milliseconds = percentile(metrics_.compute_latencies, 0.95);
    report.compute_p99_milliseconds = percentile(metrics_.compute_latencies, 0.99);
    report.retrieval_p50_milliseconds = percentile(metrics_.retrieval_latencies, 0.50);
    report.retrieval_p95_milliseconds = percentile(metrics_.retrieval_latencies, 0.95);
    report.verification_p50_milliseconds = percentile(metrics_.verification_latencies, 0.50);
    report.verification_p95_milliseconds = percentile(metrics_.verification_latencies, 0.95);
    report.total_p50_milliseconds = percentile(metrics_.total_latencies, 0.50);
    report.total_p95_milliseconds = percentile(metrics_.total_latencies, 0.95);
    report.total_p99_milliseconds = percentile(metrics_.total_latencies, 0.99);
    const auto total_tokens = metrics_.total_input_tokens + metrics_.total_output_tokens;
    report.throughput_requests_per_second = report.considered_requests == 0U ? 0.0 : static_cast<double>(report.successful_requests) / std::max(report.total_p99_milliseconds / 1000.0, 0.001);
    report.throughput_tokens_per_second = total_tokens == 0U ? 0.0 : static_cast<double>(total_tokens) / std::max(report.total_p99_milliseconds / 1000.0, 0.001);
    report.passed = report.considered_requests > 0U && report.availability_fraction >= slo_thresholds_.availability_fraction &&
                    report.error_rate_fraction <= slo_thresholds_.error_rate_fraction && report.total_p95_milliseconds <= slo_thresholds_.first_token_p95_milliseconds;
    if (report.considered_requests == 0U) report.violations.push_back("no requests measured");
    if (report.availability_fraction < slo_thresholds_.availability_fraction) report.violations.push_back("availability");
    if (report.error_rate_fraction > slo_thresholds_.error_rate_fraction) report.violations.push_back("error_rate");
    if (report.total_p95_milliseconds > slo_thresholds_.first_token_p95_milliseconds) report.violations.push_back("first_token_p95");
    return report;
}

void InferenceService::clear_metrics() {
    metrics_ = {};
}

bool InferenceService::healthy() const noexcept { return fault_ == ServiceFault::None && !circuit_open(); }
std::size_t InferenceService::pending_count() const noexcept { return pending_.size(); }

}  // namespace cct
