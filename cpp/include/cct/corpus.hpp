#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cct {

enum class CorpusDecision : std::uint8_t { Accept = 0, Quarantine = 1, Reject = 2 };
enum class CorpusSplit : std::uint8_t { Train = 0, Validation = 1, Test = 2, EvaluatorOnly = 3 };
enum class CorpusDataClass : std::uint8_t { GeneralText = 0, ReferenceText = 1, Code = 2, Instruction = 3, Preference = 4, Enterprise = 5, Safety = 6, EvaluatorOnly = 7 };

struct SourcePolicy {
    std::string source_id;
    std::string source_uri;
    std::string license_or_consent;
    std::string jurisdiction;
    std::string collection_method;
    std::string collection_timestamp;
    std::string privacy_classification;
    std::string retention_policy;
    bool license_resolved = false;
    bool training_allowed = false;
    bool evaluation_allowed = false;
    bool human_reviewed = false;
};

struct CorpusRecord {
    std::string record_id;
    std::string source_id;
    std::string source_uri;
    std::string license_or_consent;
    std::string jurisdiction;
    std::string collection_method;
    std::string collection_timestamp;
    std::string privacy_classification;
    std::string content;
    std::string normalized_content;
    std::string content_hash;
    std::string normalized_hash;
    std::vector<std::string> transformation_chain;
    std::vector<std::string> language_and_domain_labels;
    std::vector<std::string> quality_labels;
    std::string split_assignment;
    std::string retention_policy;
    std::string delete_after;
    bool opt_out = false;
    bool evaluator_only = false;
    bool pii_detected = false;
    bool redacted = false;
    bool near_duplicate = false;
    bool quarantined = false;
    bool deleted = false;
    CorpusDecision decision = CorpusDecision::Reject;
    CorpusSplit split = CorpusSplit::Train;
    CorpusDataClass data_class = CorpusDataClass::GeneralText;
    std::vector<std::string> reason_codes;
};

struct CorpusShard {
    std::string shard_id;
    CorpusSplit split = CorpusSplit::Train;
    std::vector<std::string> record_ids;
    std::string content_hash;
    std::size_t byte_count = 0;
};

struct CorpusAuditEvent {
    std::string event_type;
    std::string record_id;
    std::string source_id;
    CorpusDecision decision = CorpusDecision::Reject;
    std::string reason;
    std::string content_hash;
    std::string normalized_hash;
};

class GovernedCorpus {
public:
    void register_source(const SourcePolicy& policy);
    CorpusRecord ingest(const std::string& record_id, const std::string& source_id, const std::string& content,
                        CorpusSplit split, CorpusDataClass data_class, bool evaluator_only = false);
    CorpusRecord ingest_file(const std::string& record_id, const std::string& source_id, const std::string& path,
                             CorpusSplit split, CorpusDataClass data_class, std::size_t max_bytes = 0,
                             bool evaluator_only = false);
    void add_evaluator_canary(const std::string& record_id, const std::string& source_id, const std::string& content);
    bool detect_contamination(const std::string& candidate_content) const;
    bool tombstone(const std::string& record_id, const std::string& reason);
    std::vector<CorpusShard> build_shards(std::size_t max_records_per_shard) const;
    std::vector<CorpusRecord> training_records() const;
    std::vector<CorpusRecord> evaluation_records() const;
    std::vector<CorpusRecord> all_records() const;
    static std::string content_sha256(const std::string& content);
    const std::vector<SourcePolicy>& sources() const noexcept;
    const std::vector<CorpusAuditEvent>& audit() const noexcept;
    std::string serialize() const;
    static GovernedCorpus deserialize(const std::string& text);
    bool equivalent(const GovernedCorpus& other) const;

private:
    const SourcePolicy& source(const std::string& source_id) const;
    CorpusRecord process_record(CorpusRecord record) const;
    static std::string normalize(const std::string& content);
    static std::string sha256(const std::string& content);
    static bool detect_pii(const std::string& content);
    static bool looks_like_code(const std::string& content);
    static std::vector<std::string> labels_for(const std::string& content, CorpusDataClass data_class);
    bool has_exact_hash(const std::string& normalized_hash) const;
    bool has_near_duplicate(const std::string& normalized_content) const;
    void audit(const CorpusRecord& record, const std::string& event_type, const std::string& reason);

    std::vector<SourcePolicy> sources_;
    std::vector<CorpusRecord> records_;
    std::vector<CorpusAuditEvent> audit_;
};

}  // namespace cct
