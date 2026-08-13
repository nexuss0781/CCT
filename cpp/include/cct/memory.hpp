#pragma once

#include "cct/causal.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class MemoryStatus : std::uint8_t { Active = 0, Superseded = 1, Deleted = 2, Quarantined = 3 };
enum class RetentionClass : std::uint8_t { Ephemeral = 0, Standard = 1, Priority = 2, LegalHold = 3 };
enum class MemoryEventType : std::uint8_t { Append = 0, Update = 1, Tombstone = 2, Quarantine = 3 };
enum class MemoryDecisionKind : std::uint8_t { Write = 0, Update = 1, Ignore = 2, Quarantine = 3 };
enum class CitationSupport : std::uint8_t { Supported = 0, Contradicted = 1, Unsupported = 2, Abstained = 3 };

using MemoryId = std::uint64_t;
using LogicalTime = std::int64_t;

struct SourceRef {
    std::string source_id;
    std::size_t span_start = 0;
    std::size_t span_end = 0;
};

struct MemoryRecord {
    static constexpr std::uint32_t kSchemaVersion = 1;

    std::uint32_t schema_version = kSchemaVersion;
    MemoryId memory_id = 0;
    std::uint64_t version = 1;
    std::string content;
    std::vector<double> embedding;
    std::vector<EventId> event_ids;
    std::vector<EventId> causal_parents;
    LogicalTime created_at = 0;
    LogicalTime valid_from = 0;
    std::optional<LogicalTime> valid_until;
    SourceRef source;
    double confidence = 1.0;
    MemoryStatus status = MemoryStatus::Active;
    RetentionClass retention = RetentionClass::Standard;
    std::string conflict_group;
    std::uint64_t checksum = 0;
    std::string checksum_digest;
};

struct MemoryConfig {
    std::size_t embedding_dim = 4;
    std::size_t max_active_records = 1024;
    double minimum_confidence = 0.0;
    std::uint64_t chain_seed = 1469598103934665603ULL;
    bool immediate_deletion = true;
    double novelty_threshold = 0.0;
    double quarantine_threshold = 0.05;
};

struct MemoryQuery {
    std::vector<double> embedding;
    std::optional<LogicalTime> valid_at;
    std::optional<LogicalTime> created_after;
    std::optional<LogicalTime> created_before;
    std::optional<std::string> source_id;
    std::optional<EventId> event_id;
    std::optional<std::string> conflict_group;
    std::size_t budget = 5;
    double minimum_confidence = 0.0;
    bool include_history = false;
    bool include_expired = false;
};

struct MemoryHit {
    MemoryId memory_id = 0;
    std::uint64_t version = 0;
    double score = 0.0;
    SourceRef source;
    std::size_t evidence_span_start = 0;
    std::size_t evidence_span_end = 0;
    double confidence = 0.0;
    MemoryStatus status = MemoryStatus::Active;
    std::string conflict_group;
    std::uint64_t checksum = 0;
    std::string checksum_digest;
};

struct MemoryEvent {
    std::uint64_t sequence = 0;
    MemoryEventType type = MemoryEventType::Append;
    MemoryRecord record;
    MemoryId target_id = 0;
    std::uint64_t previous_version = 0;
    std::string reason;
    std::uint64_t previous_event_checksum = 0;
    std::uint64_t event_checksum = 0;
    std::string previous_event_digest;
    std::string event_digest;
};

struct MemoryDecision {
    MemoryDecisionKind kind = MemoryDecisionKind::Ignore;
    MemoryId memory_id = 0;
    std::uint64_t version = 0;
    std::string reason;
};

struct CitationBinding {
    std::string claim_id;
    CitationSupport support = CitationSupport::Unsupported;
    std::vector<MemoryId> memory_ids;
    std::vector<SourceRef> evidence;
    std::string reason;
};

struct EvidenceContext {
    std::vector<MemoryHit> hits;
    bool memory_enabled = true;
};

class MemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class MemoryEncoder {
public:
    explicit MemoryEncoder(std::size_t embedding_dim, std::uint32_t schema_version = MemoryRecord::kSchemaVersion);

    std::size_t embedding_dim() const noexcept { return embedding_dim_; }
    std::uint32_t schema_version() const noexcept { return schema_version_; }
    std::vector<double> encode(const MemoryRecord& record) const;
    std::uint64_t content_checksum(const MemoryRecord& record) const;
    std::string content_digest(const MemoryRecord& record) const;

private:
    std::size_t embedding_dim_;
    std::uint32_t schema_version_;
};

class MemoryWriteController {
public:
    explicit MemoryWriteController(double novelty_threshold = 1e-12, double quarantine_threshold = 0.05);

    MemoryDecision decide(const MemoryRecord& candidate,
                          const std::vector<MemoryRecord>& existing_active) const;

private:
    double novelty_threshold_;
    double quarantine_threshold_;
};

/**
 * Thread-safety contract: PersistentMemory has no internal mutex. Callers must
 * externally serialize all mutations, retrievals that overlap mutation, snapshot
 * operations, and log rebuilds. Read-only calls may run concurrently only when
 * the instance is otherwise quiescent. `active_record()` and `event_log()` return
 * borrowed references that are invalidated by mutation. Snapshot loading is
 * fail-closed and does not publish partial state.
 * Failure contract: invalid records, policy violations, and malformed snapshots
 * throw MemoryError; no operation silently repairs an invalid state.
 */
class PersistentMemory {
public:
    explicit PersistentMemory(MemoryConfig config = {});

    const MemoryConfig& config() const noexcept { return config_; }
    const MemoryEncoder& encoder() const noexcept { return encoder_; }

    MemoryDecision write(MemoryRecord record, const std::string& reason = "new_record");
    MemoryDecision update(MemoryRecord record, const std::string& reason = "new_version");
    MemoryDecision delete_memory(MemoryId memory_id, const std::string& reason = "user_delete");
    MemoryDecision quarantine(MemoryId memory_id, const std::string& reason = "policy_quarantine");
    std::size_t expire(LogicalTime now, const std::string& reason = "validity_expired");
    std::size_t process_deferred_deletions(const std::string& reason = "deferred_delete");
    std::size_t enforce_capacity(const std::string& reason = "capacity_policy");
    MemoryDecision write_event(const CausalEvent& event, const std::vector<double>& embedding,
                               MemoryId memory_id, const std::string& reason = "causal_event");

    std::vector<MemoryHit> retrieve(const MemoryQuery& query) const;
    CitationBinding bind_citation(const std::string& claim_id, const std::vector<MemoryHit>& hits,
                                  CitationSupport support = CitationSupport::Supported) const;
    EvidenceContext evidence_context(const MemoryQuery& query) const;
    static EvidenceContext no_memory_context();

    bool contains(MemoryId memory_id) const noexcept;
    const MemoryRecord& active_record(MemoryId memory_id) const;
    std::vector<MemoryRecord> active_records() const;
    std::vector<MemoryRecord> history(MemoryId memory_id) const;
    std::vector<MemoryRecord> conflict_set(const std::string& conflict_group) const;
    const std::vector<MemoryEvent>& event_log() const noexcept { return event_log_; }
    std::size_t active_size() const noexcept { return active_.size(); }

    void verify_log() const;
    void rebuild_from_log();
    std::string canonical_state_export() const;
    std::string log_export() const;
    std::string serialize_snapshot() const;
    static PersistentMemory deserialize_snapshot(const std::string& snapshot);
    void save_snapshot(const std::string& path) const;
    static PersistentMemory load_snapshot(const std::string& path);

private:
    MemoryConfig config_;
    MemoryEncoder encoder_;
    MemoryWriteController write_controller_;
    std::vector<MemoryEvent> event_log_;
    std::map<MemoryId, std::vector<MemoryRecord>> versions_;
    std::map<MemoryId, MemoryRecord> active_;
    std::uint64_t next_sequence_ = 1;

    void validate_record(const MemoryRecord& record) const;
    void append_event(MemoryEvent event);
    void apply_event(const MemoryEvent& event, bool validate_chain);
    MemoryEvent make_event(MemoryEventType type, const MemoryRecord& record, MemoryId target_id,
                           std::uint64_t previous_version, const std::string& reason) const;
    MemoryDecision delete_memory_now(MemoryId memory_id, const std::string& reason);
    std::uint64_t event_checksum(const MemoryEvent& event) const;
    std::string event_digest(const MemoryEvent& event) const;
    bool valid_at(const MemoryRecord& record, const MemoryQuery& query) const;
    double cosine_similarity(const std::vector<double>& left, const std::vector<double>& right) const;
    void reset_state();
    std::vector<MemoryId> deferred_deletions_;
};

}  // namespace cct
