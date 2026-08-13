#pragma once

#include "cct/corpus.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class RetrievalMode : std::uint8_t {
    Lexical = 0,
    Vector = 1,
    Hybrid = 2
};

using KnowledgeEmbeddingProvider = std::function<std::vector<double>(const std::string&)>;

enum class KnowledgeState : std::uint8_t {
    Active = 0,
    Superseded = 1,
    Deleted = 2,
    Quarantined = 3
};

std::string retrieval_mode_name(RetrievalMode mode);
std::string knowledge_state_name(KnowledgeState state);

class KnowledgeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct KnowledgeAccessPolicy {
    std::string tenant_id;
    std::vector<std::string> allowed_roles;
    bool public_read = false;

    bool allows(const std::string& query_tenant, const std::string& query_role) const;
};

struct KnowledgeCitationSpan {
    std::string span_id;
    std::size_t start = 0;
    std::size_t end = 0;
    std::string span_hash;
};

struct KnowledgeQuality {
    double quality = 1.0;
    double confidence = 1.0;
    std::string source_risk = "normal";
};

struct KnowledgeRecord {
    std::string knowledge_id;
    std::string tenant_id;
    std::string document_id;
    std::uint64_t document_version = 1;
    std::string source_uri_or_reference;
    std::string content;
    std::string content_hash;
    std::string embedding_version;
    std::string lexical_index_version;
    std::int64_t created_at = 0;
    std::int64_t valid_from = 0;
    std::optional<std::int64_t> valid_until;
    KnowledgeAccessPolicy access_policy;
    std::string provenance;
    std::vector<KnowledgeCitationSpan> citation_spans;
    KnowledgeQuality quality;
    std::vector<std::string> supersedes_or_conflicts;
    KnowledgeState retention_and_deletion_state = KnowledgeState::Active;
    bool superseded = false;
};

struct KnowledgeIndexConfig {
    std::string embedding_version = "embedding-v1";
    std::string lexical_index_version = "lexical-v1";
    std::string ranking_version = "hybrid-v1";
    std::string transformation_version = "knowledge-transform-v1";
    std::size_t embedding_dimension = 8;
    double lexical_weight = 0.6;
    double vector_weight = 0.4;
    double minimum_quality = 0.0;
    double minimum_confidence = 0.0;
    std::size_t maximum_hits = 8;
    std::size_t maximum_snapshot_bytes = 64U * 1024U * 1024U;
    std::size_t maximum_records = 1'000'000U;
    std::size_t maximum_roles_per_record = 4096U;
    std::size_t maximum_spans_per_record = 8192U;
    std::size_t maximum_relations_per_record = 8192U;
    std::string embedding_backend = "deterministic-hash-baseline-v1";
};

struct KnowledgeQuery {
    std::string query_id;
    std::string tenant_id;
    std::string role;
    std::string text;
    std::vector<double> embedding;
    RetrievalMode mode = RetrievalMode::Hybrid;
    std::int64_t valid_at = 0;
    std::size_t top_k = 5;
    bool include_stale = false;
    std::string embedding_version;
    std::string lexical_index_version;
};

struct KnowledgeHit {
    std::string knowledge_id;
    std::string tenant_id;
    std::string document_id;
    std::uint64_t document_version = 0;
    std::string source_uri_or_reference;
    std::string content;
    std::string content_hash;
    bool access_allowed = false;
    double lexical_score = 0.0;
    double vector_score = 0.0;
    double combined_score = 0.0;
    bool temporally_valid = false;
    bool stale = false;
    bool conflict_visible = false;
    std::string conflict_group;
    std::string source_risk;
    std::string embedding_version;
    std::string lexical_index_version;
    std::string transformation_version;
    std::vector<KnowledgeCitationSpan> citation_spans;
};

struct KnowledgeQueryAudit {
    std::string query_id;
    RetrievalMode mode = RetrievalMode::Hybrid;
    std::string tenant_id;
    std::string role;
    std::size_t scanned_records = 0;
    std::size_t unauthorized_records = 0;
    std::size_t stale_records = 0;
    std::vector<std::string> returned_knowledge_ids;
    std::string decision;
    std::string reason;
};

struct KnowledgeClaim {
    std::string claim_id;
    std::string text;
    std::vector<std::string> citation_span_ids;
};

struct GroundedAnswerRequest {
    std::string answer_id;
    std::string query_id;
    RetrievalMode mode = RetrievalMode::Hybrid;
    std::string answer_text;
    std::vector<KnowledgeClaim> claims;
    bool allow_conflicts = false;
};

struct VerifiedAnswer {
    std::string answer_id;
    std::string query_id;
    RetrievalMode mode = RetrievalMode::Hybrid;
    bool accepted = false;
    bool abstained = false;
    bool conflict_detected = false;
    std::size_t claim_count = 0;
    std::size_t supported_claim_count = 0;
    std::size_t cited_claim_count = 0;
    double citation_precision = 0.0;
    double citation_recall = 0.0;
    std::string reason;
};

struct RetrievalMetrics {
    std::size_t query_count = 0;
    std::size_t total_scanned_records = 0;
    std::size_t total_returned_hits = 0;
    std::size_t unauthorized_hits_returned = 0;
    std::size_t stale_hits_returned = 0;
    double last_latency_milliseconds = 0.0;
    std::size_t estimated_memory_bytes = 0;
};

struct GroundingReview {
    std::string review_id;
    std::string answer_id;
    std::string reviewer_class;
    bool blind = false;
    bool grounded = false;
    bool citation_correct = false;
    bool uncertainty_appropriate = false;
    bool domain_expert = false;
};

struct GroundingReviewSummary {
    std::size_t review_count = 0;
    std::size_t grounded_count = 0;
    std::size_t citation_correct_count = 0;
    std::size_t uncertainty_appropriate_count = 0;
    double grounded_rate = 0.0;
    bool blind_protocol_valid = false;
    bool expert_review_present = false;
};

/**
 * Thread-safety contract: KnowledgePlane has no internal mutex. Concurrent const
 * operations are valid only while no thread mutates the instance; every mutation
 * and any const operation that updates audit or metrics requires caller-provided
 * exclusive synchronization. `records()`, `audit()`, and `metrics()` return
 * borrowed references and must not outlive or overlap a mutation. Serialization
 * and deserialization are deterministic V1 operations and reject malformed input.
 * Failure contract: invalid records, queries, and snapshots throw KnowledgeError.
 */
class KnowledgePlane {
public:
    explicit KnowledgePlane(KnowledgeIndexConfig config = {}, KnowledgeEmbeddingProvider embedding_provider = {});

    const KnowledgeIndexConfig& config() const noexcept { return config_; }
    const std::vector<KnowledgeRecord>& records() const noexcept { return records_; }
    const std::vector<KnowledgeQueryAudit>& audit() const noexcept { return audit_; }
    const RetrievalMetrics& metrics() const noexcept { return metrics_; }

    void ingest(KnowledgeRecord record);
    void ingest_from_corpus(const CorpusRecord& record, const std::string& tenant_id,
                            const KnowledgeAccessPolicy& access_policy, std::int64_t valid_from,
                            std::optional<std::int64_t> valid_until = std::nullopt);
    void supersede(const std::string& knowledge_id, const std::string& reason);
    void tombstone(const std::string& knowledge_id, const std::string& reason);
    void rebuild();

    std::vector<KnowledgeHit> retrieve(const KnowledgeQuery& query) const;
    VerifiedAnswer verify_answer(const GroundedAnswerRequest& request,
                                 const std::vector<KnowledgeHit>& hits) const;
    GroundingReviewSummary review_grounded_answers(const std::vector<GroundingReview>& reviews) const;

    std::string serialize_snapshot() const;
    static KnowledgePlane deserialize_snapshot(const std::string& snapshot);
    void save_snapshot(const std::string& path) const;
    static KnowledgePlane load_snapshot(const std::string& path);
    bool contains_active(const std::string& knowledge_id) const;
    bool can_access(const KnowledgeRecord& record, const KnowledgeQuery& query) const;

private:
    KnowledgeIndexConfig config_;
    std::vector<KnowledgeRecord> records_;
    mutable std::vector<KnowledgeQueryAudit> audit_;
    mutable RetrievalMetrics metrics_;
    bool rebuilt_ = false;

    void validate_record(const KnowledgeRecord& record) const;
    std::vector<double> embed(const std::string& text) const;
    double lexical_score(const std::string& query, const std::string& content) const;
    double vector_score(const std::vector<double>& query, const std::vector<double>& content) const;
    bool temporally_valid(const KnowledgeRecord& record, std::int64_t valid_at) const;
    bool is_stale(const KnowledgeRecord& record, std::int64_t valid_at) const;
    static std::vector<std::string> terms(const std::string& text);
    static std::string join_terms(const std::vector<std::string>& terms);
    static bool claim_supported(const KnowledgeClaim& claim, const std::string& evidence_text);
    KnowledgeEmbeddingProvider embedding_provider_;
};

}  // namespace cct
