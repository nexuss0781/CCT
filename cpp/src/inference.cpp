#include "cct/inference.hpp"
#include "cct/nlp_trainer.hpp"

#include <algorithm>
#include <fstream>
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

double monotonic_milliseconds() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::size_t token_count(const std::string& value) {
    std::size_t count = 0U;
    bool in_token = false;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
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

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read inference artifact " + path);
    std::ostringstream output;
    output << input.rdbuf();
    require(static_cast<bool>(output), "cannot read inference artifact bytes " + path);
    return output.str();
}

std::string route_backend(const ModelRoute route) {
    if (route == ModelRoute::Cct) return "fixture-template-cct";
    if (route == ModelRoute::Hybrid) return "fixture-template-hybrid";
    return "fixture-template-transformer-control";
}

class CheckpointInferenceBackend final : public InferenceBackend {
public:
    CheckpointInferenceBackend(const std::string& checkpoint_path, const std::string& tokenizer_path,
                               const std::string& expected_tokenizer_version)
        : tokenizer_(Tokenizer::from_snapshot(read_file(tokenizer_path))),
          trainer_(NlpTrainer::load_checkpoint(checkpoint_path, tokenizer_.snapshot_hash())) {
        require(tokenizer_.version() == expected_tokenizer_version, "checkpoint tokenizer version does not match inference configuration");
        require(trainer_.model().kind() == NlpModelKind::Track1CctRecurrence,
                "checkpoint inference currently requires the Track 1 CCT recurrence model kind");
        const auto vocabulary_size = static_cast<std::size_t>(tokenizer_.vocabulary().back().id) + 1U;
        require(trainer_.model().config().vocabulary_size == vocabulary_size, "checkpoint and tokenizer vocabulary sizes differ");
        identity_ = "checkpoint-backed-" + trainer_.model().name() + "-" + nlp_checkpoint_hash(read_file(checkpoint_path));
    }

    std::string identity() const override { return identity_; }

    BackendGenerationResult generate(const std::string& input, const std::size_t maximum_input_tokens,
                                     const std::size_t maximum_output_tokens, const std::vector<std::uint32_t>& prior_context,
                                     const InferenceTokenCallback& callback) const override {
        const auto encoded = tokenizer_.encode(input, "inference", false);
        require(!encoded.tokens.empty() && encoded.tokens.size() <= maximum_input_tokens,
                "checkpoint inference input exceeds the token budget");
        const auto maximum_context = trainer_.model().config().context_length;
        require(encoded.tokens.size() <= maximum_context, "checkpoint inference input exceeds the model context contract");
        std::vector<TokenId> context(prior_context.begin(), prior_context.end());
        for (const auto& token : encoded.tokens) context.push_back(token.id);
        if (context.size() > maximum_context) context.erase(context.begin(), context.end() - static_cast<std::ptrdiff_t>(maximum_context));
        require(!context.empty(), "checkpoint inference context is empty after windowing");
        BackendGenerationResult result;
        result.input_tokens = encoded.tokens.size();
        const auto started = std::chrono::steady_clock::now();
        auto previous_token_time = started;
        double inter_token_sum = 0.0;
        std::size_t inter_token_count = 0U;
        for (std::size_t step = 0U; step < maximum_output_tokens; ++step) {
            const auto logits = trainer_.model().next_logits(context);
            const auto next = static_cast<TokenId>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
            if (next == Tokenizer::kEosId) break;
            const auto token_text = tokenizer_.decode(std::vector<TokenId>{next}, true);
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(now - started).count();
            const auto inter = std::chrono::duration<double, std::milli>(now - previous_token_time).count();
            if (result.output_tokens == 0U) result.first_token_milliseconds = elapsed;
            else {
                inter_token_sum += inter;
                ++inter_token_count;
            }
            result.output += token_text;
            ++result.output_tokens;
            previous_token_time = now;
            if (callback && !callback(token_text, result.output_tokens - 1U, result.output_tokens == 1U ? elapsed : inter)) {
                result.cancelled = true;
                break;
            }
            if (context.size() == trainer_.model().config().context_length) context.erase(context.begin());
            context.push_back(next);
        }
        result.inter_token_milliseconds = inter_token_count == 0U ? 0.0 : inter_token_sum / static_cast<double>(inter_token_count);
        result.state_token_ids.assign(context.begin(), context.end());
        return result;
    }

private:
    Tokenizer tokenizer_;
    NlpTrainer trainer_;
    std::string identity_;
};

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
    require(release.model_artifact_path.empty() == release.tokenizer_artifact_path.empty(),
            "deployment release artifact paths must be supplied together");
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
                config_.maximum_queue_depth > 0U && config_.maximum_state_bytes_per_session > 0U && config_.maximum_state_bytes_per_tenant > 0U &&
                config_.maximum_cache_entries > 0U && config_.maximum_cache_bytes > 0U,
            "inference configuration contains zero limits");
    require(config_.circuit_failure_threshold > 0U && !config_.default_release_id.empty() && !config_.default_release_digest.empty(),
            "inference release configuration is incomplete");
    if (config_.backend_mode == InferenceBackendMode::Checkpoint) {
        require(!config_.model_checkpoint_path.empty() && !config_.tokenizer_snapshot_path.empty(),
                "checkpoint inference requires model and tokenizer artifact paths");
        backend_ = std::make_shared<CheckpointInferenceBackend>(config_.model_checkpoint_path, config_.tokenizer_snapshot_path,
                                                                 config_.tokenizer_version);
    }
    deployment_.register_release({config_.default_release_id, config_.model_version, config_.adapter_version, config_.tokenizer_version,
                                  config_.knowledge_index_version, config_.default_release_digest, ModelRoute::Cct, {}, {}});
    deployment_.activate(config_.default_release_id);
}

InferenceService::InferenceService(InferenceService&& other) noexcept
    : config_(std::move(other.config_)), slo_thresholds_(other.slo_thresholds_), knowledge_(other.knowledge_),
      deployment_(std::move(other.deployment_)), fault_(other.fault_), consecutive_failures_(other.consecutive_failures_),
      circuit_open_at_(other.circuit_open_at_), pending_(std::move(other.pending_)), states_(std::move(other.states_)),
      cache_(std::move(other.cache_)), audit_(std::move(other.audit_)), metrics_(std::move(other.metrics_)),
      state_metrics_(other.state_metrics_), backend_(std::move(other.backend_)) {}

ServiceMetrics InferenceService::metrics() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return metrics_;
}

StateMetrics InferenceService::state_metrics() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return state_metrics_;
}

std::vector<ServiceAuditRecord> InferenceService::audit() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return audit_;
}

DeploymentStatus InferenceService::deployment_status() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return deployment_.status();
}

void InferenceService::attach_knowledge_plane(KnowledgePlane& knowledge) noexcept {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    knowledge_ = &knowledge;
}

void InferenceService::set_slo_thresholds(const SloThresholds& thresholds) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    require(thresholds.first_token_p95_milliseconds > 0.0 && thresholds.inter_token_p95_milliseconds > 0.0 &&
                thresholds.availability_fraction >= 0.0 && thresholds.availability_fraction <= 1.0 &&
                thresholds.error_rate_fraction >= 0.0 && thresholds.error_rate_fraction <= 1.0 && thresholds.rollback_target_milliseconds > 0.0,
            "invalid SLO thresholds");
    slo_thresholds_ = thresholds;
}

void InferenceService::set_fault(const ServiceFault fault) noexcept {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    fault_ = fault;
}
void InferenceService::clear_fault() noexcept {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    fault_ = ServiceFault::None;
}

void InferenceService::register_release(const DeploymentRelease& release) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    deployment_.register_release(release);
}

void InferenceService::activate_release(const std::string& release_id) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    const auto& selected = deployment_.release(release_id);
    std::shared_ptr<InferenceBackend> candidate = backend_;
    if (!selected.model_artifact_path.empty()) {
        const auto serialized_checkpoint = read_file(selected.model_artifact_path);
        require(nlp_checkpoint_hash(serialized_checkpoint) == selected.artifact_digest,
                "deployment release checkpoint digest does not match its approved artifact digest");
        candidate = std::make_shared<CheckpointInferenceBackend>(selected.model_artifact_path, selected.tokenizer_artifact_path,
                                                                 selected.tokenizer_version);
    }
    deployment_.activate(release_id);
    const bool backend_changed = candidate != backend_;
    backend_ = std::move(candidate);
    config_.model_version = selected.model_version;
    config_.adapter_version = selected.adapter_version;
    config_.tokenizer_version = selected.tokenizer_version;
    config_.knowledge_index_version = selected.knowledge_index_version;
    if (!selected.model_artifact_path.empty()) {
        config_.backend_mode = InferenceBackendMode::Checkpoint;
        config_.model_checkpoint_path = selected.model_artifact_path;
        config_.tokenizer_snapshot_path = selected.tokenizer_artifact_path;
    }
    if (backend_changed) {
        states_.clear();
        cache_.clear();
        cache_bytes_ = 0U;
        state_metrics_.active_sessions = 0U;
        state_metrics_.active_tenants = 0U;
        state_metrics_.bytes_in_use = 0U;
    }
}

void InferenceService::start_canary(const std::string& release_id, const std::size_t percent) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    deployment_.start_canary(release_id, percent);
}

void InferenceService::record_canary(const CanaryComparison& comparison) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    deployment_.record_canary(comparison);
}

void InferenceService::promote_canary() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    deployment_.promote_canary();
}

double InferenceService::rollback_release() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return deployment_.rollback();
}

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
    metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
    if (code == "DEADLINE_EXPIRED") ++metrics_.timed_out_requests;
    if (code == "CIRCUIT_OPEN") ++metrics_.circuit_open_rejections;
    append_audit(request, response, "rejected", {}, {}, code);
    return response;
}

Submission InferenceService::enqueue(const InferenceRequest& request, const AuthContext& auth) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (const auto error = validate(request, auth); error.has_value()) return {false, request.request_id, reject(request, *error, "request rejected by versioned admission control")};
    if (pending_.size() >= config_.maximum_queue_depth) return {false, request.request_id, reject(request, "QUEUE_FULL", "request queue capacity is exhausted")};
    PendingRequest pending{request, auth, monotonic_milliseconds()};
    const auto enqueued_at_milliseconds = pending.enqueued_at_milliseconds;
    if (pending.request.deadline_epoch_milliseconds == 0) pending.request.deadline_epoch_milliseconds = now_epoch_milliseconds() + config_.default_deadline_milliseconds;
    pending_.push_back(std::move(pending));
    ++metrics_.submitted_requests;
    if (metrics_.measurement_started_milliseconds == 0.0) metrics_.measurement_started_milliseconds = enqueued_at_milliseconds;
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
            const auto queue_milliseconds = monotonic_milliseconds() - pending.enqueued_at_milliseconds;
            responses.push_back(execute(pending.request, batch.size(), {}, queue_milliseconds));
        } catch (const std::exception& error) {
            record_failure(pending.request, "EXECUTION_FAILURE");
            responses.push_back(reject(pending.request, "EXECUTION_FAILURE", error.what(), Decision::Abstain));
        }
    }
    return responses;
}

std::vector<InferenceResponse> InferenceService::process_pending(const std::size_t batch_limit) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
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
    std::lock_guard<std::recursive_mutex> guard(mutex_);
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

void InferenceService::update_state(RuntimeState& state, const InferenceRequest& request, const std::size_t output_tokens,
                                     const std::vector<std::uint32_t>& model_context) {
    if (!model_context.empty()) state.model_context = model_context;
    const auto next_digest = digest(state.transcript_digest + "|" + request.input + "|" + std::to_string(output_tokens));
    const auto bytes = static_cast<std::size_t>(next_digest.size() + request.input.size() + 64U + state.model_context.size() * sizeof(std::uint32_t));
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
    metrics_.total_state_bytes = state_metrics_.bytes_in_use;
}

InferenceResponse InferenceService::execute(const InferenceRequest& request, const std::size_t batch_size,
                                              const InferenceTokenCallback& callback, const double queue_milliseconds) {
    if (fault_ == ServiceFault::Worker || fault_ == ServiceFault::Storage || fault_ == ServiceFault::Network || fault_ == ServiceFault::Dependency) {
        throw InferenceError("injected " + service_fault_name(fault_) + " fault");
    }
    const auto& release = deployment_.route_release(request);
    require(!config_.policy_use_case_id.empty() && !config_.policy_use_case_name.empty() && !config_.policy_owner.empty() &&
                !config_.policy_expiration.empty() && !config_.policy_allowed_outputs.empty(),
            "inference policy configuration is incomplete");
    const auto use_case = ProductUseCase{config_.policy_use_case_id, config_.policy_use_case_name, config_.policy_application_kind,
                                         config_.policy_allowed_outputs, config_.policy_denied_actions, config_.policy_human_review_required,
                                         config_.policy_owner, config_.policy_expiration};
    const PolicyRequest policy_request{request.tenant_id, use_case.id, request.task_schema, request.input, request.requests_external_action,
                                       request.requests_host_execution, request.requests_secret_access, false, false};
    const auto policy = ProductionPolicy::evaluate(policy_request, use_case);
    if (policy.decision != Decision::Allow) return reject(request, policy.rule_id, policy.reason, policy.decision);
    auto& state = state_for(request);
    const auto cache_key = request.tenant_id + "|" + request.user_id + "|" + request.session_id + "|" + release.model_version + "|" +
                           release.adapter_version + "|" + release.tokenizer_version + "|" + release.knowledge_index_version + "|" +
                           request.role + "|" + request.task_schema + "|" + request.retrieval_policy + "|" + digest(request.input);
    const auto cached = std::find_if(cache_.begin(), cache_.end(), [&](const auto& entry) { return entry.cache_key == cache_key; });
    if (cached != cache_.end() && fault_ == ServiceFault::None && !callback) {
        auto cached_response = cached->response;
        cached_response.request_id = request.request_id;
        cached_response.trace_id = request.trace_id;
        cached_response.usage.batch_size = batch_size;
        cached_response.usage.cache_hit = true;
        update_state(state, request, cached_response.usage.output_tokens);
        cached_response.usage.state_bytes = state.snapshot.state_bytes;
        cached_response.latency.queue_milliseconds = 0.0;
        cached_response.latency.total_milliseconds = 0.0;
        ++metrics_.cache_hits;
        ++metrics_.completed_requests;
        ++metrics_.successful_requests;
        metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
        metrics_.queue_latencies.push_back(0.0);
        metrics_.first_token_latencies.push_back(cached_response.latency.first_token_milliseconds);
        metrics_.inter_token_latencies.push_back(cached_response.latency.inter_token_milliseconds);
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
    response.backend_identity = backend_ != nullptr ? backend_->identity() : route_backend(release.route);
    response.usage.batch_size = batch_size;
    response.usage.input_tokens = token_count(request.input);
    response.latency.queue_milliseconds = queue_milliseconds;
    response.usage.output_tokens = 0U;
    std::vector<std::uint32_t> next_model_context;
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
            ++metrics_.abstained_requests;
            metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
            return response;
        }
    }
    const auto compute_started = std::chrono::steady_clock::now();
    if (!hits.empty()) {
        if (fault_ == ServiceFault::Verifier) {
            response.abstention = true;
            response.policy_decision = Decision::Abstain;
            response.uncertainty = "verifier dependency unavailable";
            response.error_code = "VERIFIER_UNAVAILABLE";
            response.error_detail = response.uncertainty;
            append_audit(request, response, "abstained", joined(retrieval_ids, ','), "fault", policy.rule_id);
            ++metrics_.completed_requests;
            ++metrics_.abstained_requests;
            metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
            return response;
        }
        response.backend_identity = "knowledge-grounded-retrieval";
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
            ++metrics_.abstained_requests;
            metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
            return response;
        }
        response.uncertainty = "grounded evidence verified";
        response.usage.cache_hit = false;
        if (callback) {
            const auto stream_started = std::chrono::steady_clock::now();
            const auto words = split_words(response.output);
            for (std::size_t index = 0U; index < words.size(); ++index) {
                const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stream_started).count();
                if (!callback(words[index], index, elapsed)) {
                    response.abstention = true;
                    response.policy_decision = Decision::Abstain;
                    response.error_code = "CANCELLED";
                    response.error_detail = "grounded response stream cancelled before state commit";
                    response.uncertainty = response.error_detail;
                    ++metrics_.cancelled_requests;
                    append_audit(request, response, "cancelled", joined(retrieval_ids, ','), "not_run", policy.rule_id);
                    return response;
                }
            }
            response.latency.first_token_milliseconds = words.empty() ? 0.0 : 0.0;
        }
    } else if (backend_ != nullptr) {
        const auto output_limit = request.max_output_tokens == 0U ? config_.maximum_output_tokens : request.max_output_tokens;
        const auto generated = backend_->generate(request.input, request.max_input_tokens == 0U ? config_.maximum_input_tokens : request.max_input_tokens,
                                                  output_limit, state.model_context, callback);
        response.output = generated.output;
        response.usage.input_tokens = generated.input_tokens;
        response.usage.output_tokens = generated.output_tokens;
        response.latency.first_token_milliseconds = generated.first_token_milliseconds;
        response.latency.inter_token_milliseconds = generated.inter_token_milliseconds;
        next_model_context = generated.state_token_ids;
        response.uncertainty = "checkpoint-backed greedy generation";
        if (generated.cancelled) {
            response.abstention = true;
            response.policy_decision = Decision::Abstain;
            response.error_code = "CANCELLED";
            response.error_detail = "checkpoint generation cancelled before state commit";
            response.uncertainty = response.error_detail;
            metrics_.total_input_tokens += response.usage.input_tokens;
            metrics_.total_output_tokens += response.usage.output_tokens;
            ++metrics_.cancelled_requests;
            metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
            append_audit(request, response, "cancelled", joined(retrieval_ids, ','), "not_run", policy.rule_id);
            return response;
        }
    } else {
        const auto prefix = release.route == ModelRoute::Cct ? "CCT-ASE" : (release.route == ModelRoute::Hybrid ? "CCT-HYBRID" : "TRANSFORMER-CONTROL");
        response.output = std::string(prefix) + " response: " + request.input;
        response.uncertainty = "explicit fixture-template backend; no model checkpoint loaded";
        response.usage.output_tokens = token_count(response.output);
        if (callback) {
            const auto words = split_words(response.output);
            for (std::size_t index = 0U; index < words.size(); ++index) {
                if (!callback(words[index], index, 0.0)) {
                    response.abstention = true;
                    response.policy_decision = Decision::Abstain;
                    response.error_code = "CANCELLED";
                    response.error_detail = "fixture response stream cancelled before state commit";
                    response.uncertainty = response.error_detail;
                    ++metrics_.cancelled_requests;
                    append_audit(request, response, "cancelled", joined(retrieval_ids, ','), "not_run", policy.rule_id);
                    return response;
                }
            }
        }
    }
    if (response.usage.output_tokens == 0U && !response.output.empty()) response.usage.output_tokens = token_count(response.output);
    require(response.usage.output_tokens <= (request.max_output_tokens == 0U ? config_.maximum_output_tokens : request.max_output_tokens), "output token budget exceeded");
    update_state(state, request, response.usage.output_tokens, next_model_context);
    response.usage.state_bytes = state.snapshot.state_bytes;
    response.latency.compute_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compute_started).count();
    response.latency.retrieval_milliseconds = retrieval_milliseconds;
    if (response.latency.first_token_milliseconds == 0.0 && response.usage.output_tokens > 0U)
        response.latency.first_token_milliseconds = response.latency.compute_milliseconds;
    response.latency.total_milliseconds = response.latency.queue_milliseconds + response.latency.compute_milliseconds +
                                          response.latency.retrieval_milliseconds + response.latency.verification_milliseconds;
    metrics_.total_input_tokens += response.usage.input_tokens;
    metrics_.total_output_tokens += response.usage.output_tokens;
    if (response.usage.cache_hit) ++metrics_.cache_hits;
    metrics_.queue_latencies.push_back(response.latency.queue_milliseconds);
    metrics_.first_token_latencies.push_back(response.latency.first_token_milliseconds);
    metrics_.inter_token_latencies.push_back(response.latency.inter_token_milliseconds);
    metrics_.compute_latencies.push_back(response.latency.compute_milliseconds);
    metrics_.retrieval_latencies.push_back(response.latency.retrieval_milliseconds);
    metrics_.verification_latencies.push_back(response.latency.verification_milliseconds);
    metrics_.total_latencies.push_back(response.latency.total_milliseconds);
    ++metrics_.completed_requests;
    if (response.abstention) ++metrics_.abstained_requests;
    else ++metrics_.successful_requests;
    metrics_.measurement_ended_milliseconds = monotonic_milliseconds();
    consecutive_failures_ = 0U;
    if (!response.abstention && response.error_code.empty() && backend_ == nullptr) {
        cache_.push_back({cache_key, request.tenant_id, request.user_id, request.session_id, response, now_epoch_milliseconds()});
        cache_bytes_ += response.output.size() + response.model_version.size() + 256U;
        while (cache_.size() > config_.maximum_cache_entries || cache_bytes_ > config_.maximum_cache_bytes) {
            require(!cache_.empty(), "inference cache bound enforcement encountered an empty cache");
            cache_.erase(cache_.begin());
            cache_bytes_ = 0U;
            for (const auto& entry : cache_) cache_bytes_ += entry.response.output.size() + entry.response.model_version.size() + 256U;
            ++metrics_.cache_evictions;
        }
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
    std::lock_guard<std::recursive_mutex> guard(mutex_);
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
    if (const auto error = validate(stream_request, auth); error.has_value()) {
        const auto response = reject(stream_request, *error, "request rejected by versioned admission control");
        result.events.push_back({request.request_id, StreamEventType::Error, response.error_code, 0U});
        result.resources_released = true;
        return result;
    }
    ++metrics_.submitted_requests;
    std::size_t sequence = 0U;
    const InferenceTokenCallback callback = [&](const std::string& token, const std::size_t, const double) {
        if (sequence >= event_budget) {
            result.events.push_back({request.request_id, StreamEventType::Backpressure, "stream event budget exhausted", sequence});
            result.backpressure_applied = true;
            result.cancelled = true;
            return false;
        }
        result.events.push_back({request.request_id, StreamEventType::Token, token, sequence});
        ++sequence;
        if (cancel_after_first) {
            result.events.push_back({request.request_id, StreamEventType::Cancelled, "client cancellation", sequence});
            result.cancelled = true;
            return false;
        }
        return true;
    };
    try {
        const auto response = execute(stream_request, 1U, callback, 0.0);
        if (!response.error_code.empty() && !result.cancelled) {
            result.events.push_back({request.request_id, StreamEventType::Error, response.error_code, sequence});
        } else if (!result.cancelled) {
            result.events.push_back({request.request_id, StreamEventType::Completed, "stream completed", sequence});
        }
    } catch (const std::exception& error) {
        record_failure(stream_request, "EXECUTION_FAILURE");
        result.events.push_back({request.request_id, StreamEventType::Error, error.what(), sequence});
    }
    result.resources_released = true;
    return result;
}

bool InferenceService::cancel(const std::string& request_id) {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
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
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    const auto found = std::remove_if(states_.begin(), states_.end(), [&](const auto& item) {
        return item.snapshot.tenant_id == tenant_id && item.snapshot.user_id == user_id && item.snapshot.session_id == session_id;
    });
    const auto cache_found = std::remove_if(cache_.begin(), cache_.end(), [&](const auto& entry) {
        return entry.tenant_id == tenant_id && entry.user_id == user_id && entry.session_id == session_id;
    });
    cache_.erase(cache_found, cache_.end());
    cache_bytes_ = 0U;
    for (const auto& entry : cache_) cache_bytes_ += entry.response.output.size() + entry.response.model_version.size() + 256U;
    if (found != states_.end()) {
        states_.erase(found, states_.end());
        ++state_metrics_.resets;
    }
    state_metrics_.active_sessions = states_.size();
    state_metrics_.bytes_in_use = 0U;
    for (const auto& item : states_) state_metrics_.bytes_in_use += item.snapshot.state_bytes;
    metrics_.total_state_bytes = state_metrics_.bytes_in_use;
}

void InferenceService::evict_expired_state() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    const auto now = now_epoch_milliseconds();
    const auto found = std::remove_if(states_.begin(), states_.end(), [&](const auto& item) {
        return now - item.snapshot.last_used_epoch_milliseconds > config_.state_ttl_milliseconds;
    });
    const auto cache_found = std::remove_if(cache_.begin(), cache_.end(), [&](const auto& entry) {
        return now - entry.created_at > config_.state_ttl_milliseconds;
    });
    cache_.erase(cache_found, cache_.end());
    cache_bytes_ = 0U;
    for (const auto& entry : cache_) cache_bytes_ += entry.response.output.size() + entry.response.model_version.size() + 256U;
    const auto removed = static_cast<std::size_t>(states_.end() - found);
    if (removed > 0U) {
        states_.erase(found, states_.end());
        state_metrics_.evictions += removed;
    }
    state_metrics_.active_sessions = states_.size();
    state_metrics_.bytes_in_use = 0U;
    for (const auto& item : states_) state_metrics_.bytes_in_use += item.snapshot.state_bytes;
    metrics_.total_state_bytes = state_metrics_.bytes_in_use;
}

std::vector<StateSnapshot> InferenceService::state_snapshots() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    std::vector<StateSnapshot> snapshots;
    snapshots.reserve(states_.size());
    for (const auto& state : states_) snapshots.push_back(state.snapshot);
    return snapshots;
}

SloReport InferenceService::evaluate_slo() const {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    SloReport report;
    report.considered_requests = metrics_.submitted_requests;
    report.successful_requests = metrics_.successful_requests;
    report.abstained_requests = metrics_.abstained_requests;
    report.rejected_requests = metrics_.rejected_requests;
    report.cancelled_requests = metrics_.cancelled_requests;
    const auto attempted_requests = metrics_.submitted_requests + metrics_.rejected_requests;
    if (report.considered_requests > 0U) report.availability_fraction = static_cast<double>(report.successful_requests) / static_cast<double>(report.considered_requests);
    if (attempted_requests > 0U) report.error_rate_fraction = static_cast<double>(metrics_.rejected_requests) / static_cast<double>(attempted_requests);
    report.first_token_p50_milliseconds = percentile(metrics_.first_token_latencies, 0.50);
    report.first_token_p95_milliseconds = percentile(metrics_.first_token_latencies, 0.95);
    report.inter_token_p95_milliseconds = percentile(metrics_.inter_token_latencies, 0.95);
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
    const auto wall_clock_seconds = metrics_.measurement_started_milliseconds == 0.0 || metrics_.measurement_ended_milliseconds <= metrics_.measurement_started_milliseconds
                                         ? 0.0
                                         : (metrics_.measurement_ended_milliseconds - metrics_.measurement_started_milliseconds) / 1000.0;
    report.throughput_requests_per_second = wall_clock_seconds <= 0.0 ? 0.0 : static_cast<double>(report.successful_requests) / wall_clock_seconds;
    report.throughput_tokens_per_second = wall_clock_seconds <= 0.0 ? 0.0 : static_cast<double>(total_tokens) / wall_clock_seconds;
    const auto inter_token_measured = !metrics_.inter_token_latencies.empty() &&
                                      std::any_of(metrics_.inter_token_latencies.begin(), metrics_.inter_token_latencies.end(), [](const double value) { return value > 0.0; });
    report.passed = report.considered_requests > 0U && report.availability_fraction >= slo_thresholds_.availability_fraction &&
                    report.error_rate_fraction <= slo_thresholds_.error_rate_fraction && report.first_token_p95_milliseconds <= slo_thresholds_.first_token_p95_milliseconds &&
                    (!inter_token_measured || report.inter_token_p95_milliseconds <= slo_thresholds_.inter_token_p95_milliseconds);
    if (report.considered_requests == 0U) report.violations.push_back("no requests measured");
    if (report.availability_fraction < slo_thresholds_.availability_fraction) report.violations.push_back("availability");
    if (report.error_rate_fraction > slo_thresholds_.error_rate_fraction) report.violations.push_back("error_rate");
    if (report.first_token_p95_milliseconds > slo_thresholds_.first_token_p95_milliseconds) report.violations.push_back("first_token_p95");
    if (inter_token_measured && report.inter_token_p95_milliseconds > slo_thresholds_.inter_token_p95_milliseconds) report.violations.push_back("inter_token_p95");
    return report;
}

void InferenceService::clear_metrics() {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    metrics_ = {};
}

bool InferenceService::healthy() const noexcept {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return fault_ == ServiceFault::None && !circuit_open();
}
std::size_t InferenceService::pending_count() const noexcept {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return pending_.size();
}

}  // namespace cct
