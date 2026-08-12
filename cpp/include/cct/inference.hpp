#pragma once

#include "cct/knowledge.hpp"
#include "cct/production.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class ModelRoute : std::uint8_t {
    Cct = 0,
    Hybrid = 1,
    Transformer = 2
};

enum class StreamEventType : std::uint8_t {
    Token = 0,
    Completed = 1,
    Cancelled = 2,
    Backpressure = 3,
    Error = 4
};

enum class ServiceFault : std::uint8_t {
    None = 0,
    Worker = 1,
    Storage = 2,
    Network = 3,
    Verifier = 4,
    Dependency = 5
};

std::string model_route_name(ModelRoute route);
std::string stream_event_type_name(StreamEventType type);
std::string service_fault_name(ServiceFault fault);

class InferenceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct InferenceConfig {
    std::string request_schema_version = "cct-request-v1";
    std::string response_schema_version = "cct-response-v1";
    std::string model_version = "cct-ase-stage16-v1";
    std::string adapter_version = "adapter-none-v1";
    std::string tokenizer_version = "tokenizer-stage10-v1";
    std::string knowledge_index_version = "lexical-v1";
    std::size_t maximum_input_tokens = 512;
    std::size_t maximum_output_tokens = 128;
    std::size_t maximum_batch_size = 8;
    std::size_t maximum_queue_depth = 64;
    std::size_t maximum_state_bytes_per_session = 16384;
    std::size_t maximum_state_bytes_per_tenant = 131072;
    std::int64_t default_deadline_milliseconds = 1500;
    std::int64_t state_ttl_milliseconds = 300000;
    std::size_t circuit_failure_threshold = 2;
    std::int64_t circuit_reset_milliseconds = 1000;
};

struct AuthContext {
    bool authenticated = false;
    std::string tenant_id;
    std::string user_id;
    std::vector<std::string> roles;
};

struct InferenceRequest {
    std::string schema_version = "cct-request-v1";
    std::string request_id;
    std::string tenant_id;
    std::string user_id;
    std::string role;
    std::string session_id;
    std::string model_version;
    std::string adapter_version;
    std::string tokenizer_version;
    std::string knowledge_index_version;
    std::string input;
    std::string task_schema;
    std::string retrieval_policy = "optional";
    std::string tool_policy = "offline-deny";
    std::size_t max_input_tokens = 0;
    std::size_t max_output_tokens = 0;
    std::int64_t deadline_epoch_milliseconds = 0;
    bool stream = false;
    ModelRoute route = ModelRoute::Cct;
    bool requests_external_action = false;
    bool requests_host_execution = false;
    bool requests_secret_access = false;
    std::string trace_id;
};

struct InferenceUsage {
    std::size_t input_tokens = 0;
    std::size_t output_tokens = 0;
    std::size_t batch_size = 1;
    std::size_t state_bytes = 0;
    bool cache_hit = false;
    bool degraded = false;
};

struct InferenceLatency {
    double queue_milliseconds = 0.0;
    double compute_milliseconds = 0.0;
    double retrieval_milliseconds = 0.0;
    double verification_milliseconds = 0.0;
    double total_milliseconds = 0.0;
};

struct InferenceResponse {
    std::string schema_version;
    std::string request_id;
    std::string model_version;
    ModelRoute route = ModelRoute::Cct;
    std::string output;
    std::vector<std::string> citations;
    std::string uncertainty;
    bool abstention = false;
    Decision policy_decision = Decision::Deny;
    InferenceUsage usage;
    InferenceLatency latency;
    std::string trace_id;
    std::string backend_identity;
    std::string error_code;
    std::string error_detail;
};

struct StreamEvent {
    std::string request_id;
    StreamEventType type = StreamEventType::Error;
    std::string payload;
    std::size_t sequence = 0;
};

struct StreamResult {
    std::vector<StreamEvent> events;
    bool cancelled = false;
    bool backpressure_applied = false;
    bool resources_released = false;
};

struct Submission {
    bool accepted = false;
    std::string request_id;
    InferenceResponse rejection;
};

struct ServiceAuditRecord {
    std::string event_id;
    std::string request_id;
    std::string trace_id;
    std::string tenant_id;
    std::string model_version;
    std::string event_type;
    std::string decision;
    std::string input_digest;
    std::string output_digest;
    std::string retrieval_ids;
    std::string verifier_decision;
    std::string policy_rule;
    std::string error_code;
    bool sensitive_data_redacted = true;
};

struct ServiceMetrics {
    std::size_t submitted_requests = 0;
    std::size_t completed_requests = 0;
    std::size_t rejected_requests = 0;
    std::size_t cancelled_requests = 0;
    std::size_t timed_out_requests = 0;
    std::size_t degraded_requests = 0;
    std::size_t batches_processed = 0;
    std::size_t maximum_observed_batch_size = 0;
    std::size_t circuit_open_rejections = 0;
    std::size_t total_input_tokens = 0;
    std::size_t total_output_tokens = 0;
    std::size_t total_state_bytes = 0;
    std::size_t cache_hits = 0;
    std::vector<double> queue_latencies;
    std::vector<double> compute_latencies;
    std::vector<double> retrieval_latencies;
    std::vector<double> verification_latencies;
    std::vector<double> total_latencies;
};

struct SloThresholds {
    double first_token_p95_milliseconds = 1500.0;
    double inter_token_p95_milliseconds = 150.0;
    double availability_fraction = 0.995;
    double error_rate_fraction = 0.005;
    double rollback_target_milliseconds = 600000.0;
};

struct SloReport {
    std::size_t considered_requests = 0;
    std::size_t successful_requests = 0;
    double availability_fraction = 0.0;
    double error_rate_fraction = 0.0;
    double queue_p50_milliseconds = 0.0;
    double queue_p95_milliseconds = 0.0;
    double queue_p99_milliseconds = 0.0;
    double compute_p50_milliseconds = 0.0;
    double compute_p95_milliseconds = 0.0;
    double compute_p99_milliseconds = 0.0;
    double retrieval_p50_milliseconds = 0.0;
    double retrieval_p95_milliseconds = 0.0;
    double verification_p50_milliseconds = 0.0;
    double verification_p95_milliseconds = 0.0;
    double total_p50_milliseconds = 0.0;
    double total_p95_milliseconds = 0.0;
    double total_p99_milliseconds = 0.0;
    double throughput_requests_per_second = 0.0;
    double throughput_tokens_per_second = 0.0;
    bool passed = false;
    std::vector<std::string> violations;
};

struct StateSnapshot {
    std::string tenant_id;
    std::string user_id;
    std::string session_id;
    std::string model_version;
    std::string adapter_version;
    std::string tokenizer_version;
    std::string knowledge_index_version;
    std::string state_digest;
    std::size_t state_bytes = 0;
    std::int64_t last_used_epoch_milliseconds = 0;
};

struct StateMetrics {
    std::size_t active_sessions = 0;
    std::size_t active_tenants = 0;
    std::size_t bytes_in_use = 0;
    std::size_t evictions = 0;
    std::size_t resets = 0;
    std::size_t mismatches_rejected = 0;
};

struct DeploymentRelease {
    std::string release_id;
    std::string model_version;
    std::string adapter_version;
    std::string tokenizer_version;
    std::string knowledge_index_version;
    std::string artifact_digest;
    ModelRoute route = ModelRoute::Cct;
};

struct DeploymentStatus {
    std::string active_release_id;
    std::string active_model_version;
    std::string previous_release_id;
    std::string canary_release_id;
    std::size_t canary_percent = 0;
    bool canary_shadowing = false;
    bool rollback_available = false;
    double canary_error_rate = 0.0;
    double canary_quality_score = 0.0;
    double last_rollback_milliseconds = 0.0;
};

struct CanaryComparison {
    std::size_t shadow_requests = 0;
    std::size_t mismatches = 0;
    std::size_t canary_errors = 0;
    double quality_score = 0.0;
    bool passed = false;
};

class DeploymentController {
public:
    void register_release(const DeploymentRelease& release);
    void activate(const std::string& release_id);
    void start_canary(const std::string& release_id, std::size_t percent);
    void record_canary(const CanaryComparison& comparison);
    void promote_canary();
    double rollback();
    bool has_release(const std::string& release_id) const;
    const DeploymentRelease& release(const std::string& release_id) const;
    const DeploymentStatus& status() const noexcept;
    const DeploymentRelease& route_release(const InferenceRequest& request) const;

private:
    std::vector<DeploymentRelease> releases_;
    DeploymentStatus status_;
};

class InferenceService {
public:
    explicit InferenceService(InferenceConfig config = {}, KnowledgePlane* knowledge = nullptr);

    const InferenceConfig& config() const noexcept { return config_; }
    const ServiceMetrics& metrics() const noexcept { return metrics_; }
    const StateMetrics& state_metrics() const noexcept { return state_metrics_; }
    const std::vector<ServiceAuditRecord>& audit() const noexcept { return audit_; }
    const DeploymentController& deployment() const noexcept { return deployment_; }
    DeploymentController& deployment() noexcept { return deployment_; }

    void attach_knowledge_plane(KnowledgePlane& knowledge) noexcept;
    void set_slo_thresholds(const SloThresholds& thresholds);
    void set_fault(ServiceFault fault) noexcept;
    void clear_fault() noexcept;
    std::int64_t now_epoch_milliseconds() const noexcept;

    Submission enqueue(const InferenceRequest& request, const AuthContext& auth);
    std::vector<InferenceResponse> process_pending(std::size_t batch_limit = 0);
    InferenceResponse handle(const InferenceRequest& request, const AuthContext& auth);
    StreamResult execute_stream(const InferenceRequest& request, const AuthContext& auth,
                                std::size_t event_budget, bool cancel_after_first = false);
    bool cancel(const std::string& request_id);
    std::size_t pending_count() const noexcept;

    void reset_state(const std::string& tenant_id, const std::string& user_id, const std::string& session_id);
    void evict_expired_state();
    std::vector<StateSnapshot> state_snapshots() const;

    SloReport evaluate_slo() const;
    void clear_metrics();
    bool healthy() const noexcept;
    bool circuit_open() const noexcept;

private:
    struct PendingRequest {
        InferenceRequest request;
        AuthContext auth;
        std::int64_t enqueued_at = 0;
    };
    struct SessionKey {
        std::string tenant_id;
        std::string user_id;
        std::string session_id;
        bool operator==(const SessionKey& other) const {
            return tenant_id == other.tenant_id && user_id == other.user_id && session_id == other.session_id;
        }
    };
    struct RuntimeState {
        StateSnapshot snapshot;
        SessionKey key;
        std::string transcript_digest;
    };
    struct CachedResponse {
        std::string cache_key;
        std::string tenant_id;
        std::string user_id;
        std::string session_id;
        InferenceResponse response;
        std::int64_t created_at = 0;
    };

    InferenceConfig config_;
    SloThresholds slo_thresholds_;
    KnowledgePlane* knowledge_ = nullptr;
    DeploymentController deployment_;
    ServiceFault fault_ = ServiceFault::None;
    std::size_t consecutive_failures_ = 0;
    std::int64_t circuit_open_at_ = 0;
    std::vector<PendingRequest> pending_;
    std::vector<RuntimeState> states_;
    std::vector<CachedResponse> cache_;
    std::vector<ServiceAuditRecord> audit_;
    ServiceMetrics metrics_;
    StateMetrics state_metrics_;

    InferenceResponse reject(const InferenceRequest& request, const std::string& code, const std::string& detail,
                             Decision decision = Decision::Deny);
    std::optional<std::string> validate(const InferenceRequest& request, const AuthContext& auth) const;
    std::vector<InferenceResponse> process_batch(const std::vector<PendingRequest>& batch);
    InferenceResponse execute(const InferenceRequest& request, std::size_t batch_size);
    RuntimeState& state_for(const InferenceRequest& request);
    void update_state(RuntimeState& state, const InferenceRequest& request, std::size_t output_tokens);
    void append_audit(const InferenceRequest& request, const InferenceResponse& response,
                      const std::string& event_type, const std::string& retrieval_ids,
                      const std::string& verifier_decision, const std::string& policy_rule);
    void record_failure(const InferenceRequest& request, const std::string& code);
    bool maybe_reset_circuit();
};

class NativeInferenceApi {
public:
    explicit NativeInferenceApi(InferenceService& service) noexcept : service_(service) {}
    InferenceResponse handle_v1(const InferenceRequest& request, const AuthContext& auth) { return service_.handle(request, auth); }

private:
    InferenceService& service_;
};

}  // namespace cct
