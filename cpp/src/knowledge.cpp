#include "cct/knowledge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace cct {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw KnowledgeError(message);
}

std::string hex_encode(const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

std::string hex_decode(const std::string& value) {
    require(value.size() % 2U == 0U, "invalid knowledge hex field length");
    const auto nibble = [](const char character) -> unsigned char {
        if (character >= '0' && character <= '9') return static_cast<unsigned char>(character - '0');
        if (character >= 'a' && character <= 'f') return static_cast<unsigned char>(character - 'a' + 10);
        if (character >= 'A' && character <= 'F') return static_cast<unsigned char>(character - 'A' + 10);
        throw KnowledgeError("invalid knowledge hex field character");
    };
    std::string output;
    output.reserve(value.size() / 2U);
    for (std::size_t index = 0U; index < value.size(); index += 2U) {
        output.push_back(static_cast<char>((nibble(value[index]) << 4U) | nibble(value[index + 1U])));
    }
    return output;
}

std::vector<std::string> split(const std::string& value, const char delimiter) {
    std::vector<std::string> fields;
    std::size_t start = 0U;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        fields.push_back(value.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1U;
    }
    return fields;
}

std::string field(const std::vector<std::string>& fields, const std::size_t index) {
    require(index < fields.size(), "serialized knowledge record is truncated");
    return hex_decode(fields[index]);
}

std::string bool_text(const bool value) { return value ? "1" : "0"; }

bool parse_bool(const std::string& value) {
    require(value == "0" || value == "1", "invalid serialized knowledge boolean");
    return value == "1";
}

std::string optional_time(const std::optional<std::int64_t>& value) {
    return value.has_value() ? std::to_string(value.value()) : "none";
}

std::optional<std::int64_t> parse_optional_time(const std::string& value) {
    return value == "none" ? std::nullopt : std::optional<std::int64_t>(std::stoll(value));
}

std::string conflict_key(const KnowledgeRecord& record) {
    for (const auto& value : record.supersedes_or_conflicts) {
        if (value.rfind("conflict:", 0U) == 0U) return value.substr(9U);
    }
    return {};
}

std::string record_line(const KnowledgeRecord& record) {
    std::vector<std::string> fields{
        record.knowledge_id, record.tenant_id, record.document_id, std::to_string(record.document_version), record.source_uri_or_reference,
        record.content, record.content_hash, record.embedding_version, record.lexical_index_version, std::to_string(record.created_at),
        std::to_string(record.valid_from), optional_time(record.valid_until), record.access_policy.tenant_id, bool_text(record.access_policy.public_read),
        std::to_string(record.access_policy.allowed_roles.size())
    };
    fields.insert(fields.end(), record.access_policy.allowed_roles.begin(), record.access_policy.allowed_roles.end());
    fields.push_back(record.provenance);
    fields.push_back(std::to_string(record.citation_spans.size()));
    for (const auto& span : record.citation_spans) {
        fields.push_back(span.span_id);
        fields.push_back(std::to_string(span.start));
        fields.push_back(std::to_string(span.end));
        fields.push_back(span.span_hash);
    }
    fields.push_back(std::to_string(record.quality.quality));
    fields.push_back(std::to_string(record.quality.confidence));
    fields.push_back(record.quality.source_risk);
    fields.push_back(std::to_string(record.supersedes_or_conflicts.size()));
    fields.insert(fields.end(), record.supersedes_or_conflicts.begin(), record.supersedes_or_conflicts.end());
    fields.push_back(knowledge_state_name(record.retention_and_deletion_state));
    fields.push_back(bool_text(record.superseded));
    std::ostringstream output;
    output << "R|";
    for (std::size_t index = 0U; index < fields.size(); ++index) {
        if (index != 0U) output << '|';
        output << hex_encode(fields[index]);
    }
    output << '\n';
    return output.str();
}

std::string config_line(const KnowledgeIndexConfig& config) {
    std::ostringstream output;
    output << "C|" << hex_encode(config.embedding_version) << '|' << hex_encode(config.lexical_index_version) << '|'
           << hex_encode(config.ranking_version) << '|' << hex_encode(config.transformation_version) << '|'
           << config.embedding_dimension << '|' << std::setprecision(17) << config.lexical_weight << '|' << config.vector_weight << '|'
           << config.minimum_quality << '|' << config.minimum_confidence << '|' << config.maximum_hits << '\n';
    return output.str();
}

std::string lower_ascii(const std::string& value) {
    std::string output;
    output.reserve(value.size());
    for (const unsigned char character : value) output.push_back(static_cast<char>(std::tolower(character)));
    return output;
}

}  // namespace

std::string retrieval_mode_name(const RetrievalMode mode) {
    switch (mode) {
        case RetrievalMode::Lexical: return "lexical";
        case RetrievalMode::Vector: return "vector";
        case RetrievalMode::Hybrid: return "hybrid";
    }
    throw KnowledgeError("unknown retrieval mode");
}

std::string knowledge_state_name(const KnowledgeState state) {
    switch (state) {
        case KnowledgeState::Active: return "active";
        case KnowledgeState::Superseded: return "superseded";
        case KnowledgeState::Deleted: return "deleted";
        case KnowledgeState::Quarantined: return "quarantined";
    }
    throw KnowledgeError("unknown knowledge state");
}

bool KnowledgeAccessPolicy::allows(const std::string& query_tenant, const std::string& query_role) const {
    if (public_read) return tenant_id == query_tenant;
    if (tenant_id != query_tenant) return false;
    return std::find(allowed_roles.begin(), allowed_roles.end(), query_role) != allowed_roles.end();
}

KnowledgePlane::KnowledgePlane(KnowledgeIndexConfig config) : config_(std::move(config)) {
    require(!config_.embedding_version.empty() && !config_.lexical_index_version.empty() && !config_.ranking_version.empty() &&
                !config_.transformation_version.empty() && config_.embedding_dimension > 0U && config_.lexical_weight >= 0.0 &&
                config_.vector_weight >= 0.0 && config_.lexical_weight + config_.vector_weight > 0.0 && config_.maximum_hits > 0U,
            "invalid knowledge index configuration");
}

void KnowledgePlane::validate_record(const KnowledgeRecord& record) const {
    require(!record.knowledge_id.empty() && !record.tenant_id.empty() && !record.document_id.empty() && !record.source_uri_or_reference.empty() &&
                !record.content.empty() && !record.content_hash.empty() && !record.embedding_version.empty() && !record.lexical_index_version.empty() &&
                !record.provenance.empty() && record.access_policy.tenant_id == record.tenant_id && record.valid_from <= record.created_at,
            "knowledge record identity or governance fields are incomplete");
    require(record.content_hash == GovernedCorpus::content_sha256(record.content), "knowledge content hash mismatch");
    require(record.embedding_version == config_.embedding_version && record.lexical_index_version == config_.lexical_index_version,
            "knowledge index version mismatch");
    require(record.quality.quality >= 0.0 && record.quality.quality <= 1.0 && record.quality.confidence >= 0.0 && record.quality.confidence <= 1.0,
            "knowledge quality or confidence is outside [0,1]");
    require(!record.valid_until.has_value() || record.valid_until.value() > record.valid_from, "knowledge validity interval is invalid");
    for (const auto& span : record.citation_spans) {
        require(!span.span_id.empty() && span.start < span.end && span.end <= record.content.size() &&
                    span.span_hash == GovernedCorpus::content_sha256(record.content.substr(span.start, span.end - span.start)),
                "knowledge citation span is invalid");
    }
}

void KnowledgePlane::ingest(KnowledgeRecord record) {
    validate_record(record);
    require(!std::any_of(records_.begin(), records_.end(), [&](const KnowledgeRecord& existing) {
        return existing.knowledge_id == record.knowledge_id;
    }), "duplicate knowledge ID");
    for (auto& existing : records_) {
        if (existing.document_id == record.document_id && existing.document_version < record.document_version &&
            existing.retention_and_deletion_state == KnowledgeState::Active) {
            existing.retention_and_deletion_state = KnowledgeState::Superseded;
            existing.superseded = true;
        }
    }
    records_.push_back(std::move(record));
    metrics_.estimated_memory_bytes = serialize_snapshot().size();
}

void KnowledgePlane::ingest_from_corpus(const CorpusRecord& record, const std::string& tenant_id,
                                        const KnowledgeAccessPolicy& access_policy, const std::int64_t valid_from,
                                        const std::optional<std::int64_t> valid_until) {
    require(record.decision == CorpusDecision::Accept && !record.deleted && !record.quarantined, "corpus record is not admissible knowledge");
    require(record.source_uri != "" && !tenant_id.empty(), "corpus-to-knowledge provenance is incomplete");
    KnowledgeRecord knowledge;
    knowledge.knowledge_id = record.record_id;
    knowledge.tenant_id = tenant_id;
    knowledge.document_id = record.source_id + "/" + record.record_id;
    knowledge.document_version = 1U;
    knowledge.source_uri_or_reference = record.source_uri;
    knowledge.content = record.content;
    knowledge.content_hash = record.content_hash;
    knowledge.embedding_version = config_.embedding_version;
    knowledge.lexical_index_version = config_.lexical_index_version;
    knowledge.created_at = valid_from;
    knowledge.valid_from = valid_from;
    knowledge.valid_until = valid_until;
    knowledge.access_policy = access_policy;
    knowledge.provenance = record.source_id + "|" + record.source_uri + "|" + record.content_hash;
    knowledge.citation_spans.push_back({record.record_id + "#span-0", 0U, record.content.size(), record.content_hash});
    knowledge.quality.quality = record.quarantined ? 0.0 : 0.95;
    knowledge.quality.confidence = 0.9;
    knowledge.quality.source_risk = record.pii_detected || record.redacted ? "sensitive" : "normal";
    ingest(std::move(knowledge));
}

void KnowledgePlane::supersede(const std::string& knowledge_id, const std::string&) {
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const KnowledgeRecord& record) { return record.knowledge_id == knowledge_id; });
    require(found != records_.end(), "cannot supersede unknown knowledge record");
    found->retention_and_deletion_state = KnowledgeState::Superseded;
    found->superseded = true;
}

void KnowledgePlane::tombstone(const std::string& knowledge_id, const std::string&) {
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const KnowledgeRecord& record) { return record.knowledge_id == knowledge_id; });
    require(found != records_.end(), "cannot delete unknown knowledge record");
    found->retention_and_deletion_state = KnowledgeState::Deleted;
    found->superseded = false;
}

void KnowledgePlane::rebuild() {
    for (const auto& record : records_) validate_record(record);
    rebuilt_ = true;
    metrics_.estimated_memory_bytes = serialize_snapshot().size();
}

std::vector<std::string> KnowledgePlane::terms(const std::string& text) {
    std::vector<std::string> output;
    std::string current;
    for (const unsigned char character : lower_ascii(text)) {
        if (std::isalnum(character) != 0U || character == '_') current.push_back(static_cast<char>(character));
        else if (!current.empty()) { output.push_back(current); current.clear(); }
    }
    if (!current.empty()) output.push_back(current);
    return output;
}

std::string KnowledgePlane::join_terms(const std::vector<std::string>& values) {
    std::ostringstream output;
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) output << ' ';
        output << values[index];
    }
    return output.str();
}

std::vector<double> KnowledgePlane::embed(const std::string& text) const {
    std::vector<double> output(config_.embedding_dimension, 0.0);
    for (const auto& term : terms(text)) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const unsigned char byte : term) { hash ^= byte; hash *= 1099511628211ULL; }
        const auto index = static_cast<std::size_t>(hash % config_.embedding_dimension);
        output[index] += 1.0;
    }
    double norm = 0.0;
    for (const auto value : output) norm += value * value;
    norm = std::sqrt(norm);
    if (norm > 0.0) for (auto& value : output) value /= norm;
    return output;
}

double KnowledgePlane::lexical_score(const std::string& query, const std::string& content) const {
    const auto query_terms = terms(query);
    const auto content_terms = terms(content);
    if (query_terms.empty() || content_terms.empty()) return 0.0;
    std::set<std::string> content_set(content_terms.begin(), content_terms.end());
    std::size_t matches = 0U;
    for (const auto& term : query_terms) if (content_set.contains(term)) ++matches;
    return static_cast<double>(matches) / static_cast<double>(query_terms.size());
}

double KnowledgePlane::vector_score(const std::vector<double>& query, const std::vector<double>& content) const {
    require(query.size() == content.size(), "knowledge vector dimensions differ");
    double dot = 0.0;
    double query_norm = 0.0;
    double content_norm = 0.0;
    for (std::size_t index = 0U; index < query.size(); ++index) {
        dot += query[index] * content[index];
        query_norm += query[index] * query[index];
        content_norm += content[index] * content[index];
    }
    if (query_norm == 0.0 || content_norm == 0.0) return 0.0;
    return dot / std::sqrt(query_norm * content_norm);
}

bool KnowledgePlane::temporally_valid(const KnowledgeRecord& record, const std::int64_t valid_at) const {
    return record.valid_from <= valid_at && (!record.valid_until.has_value() || valid_at < record.valid_until.value());
}

bool KnowledgePlane::is_stale(const KnowledgeRecord& record, const std::int64_t valid_at) const {
    return record.superseded || record.retention_and_deletion_state != KnowledgeState::Active || !temporally_valid(record, valid_at);
}

bool KnowledgePlane::can_access(const KnowledgeRecord& record, const KnowledgeQuery& query) const {
    return record.access_policy.allows(query.tenant_id, query.role);
}

std::vector<KnowledgeHit> KnowledgePlane::retrieve(const KnowledgeQuery& query) const {
    require(!query.query_id.empty() && !query.tenant_id.empty() && !query.role.empty() && !query.text.empty(), "knowledge query is incomplete");
    if (query.mode != RetrievalMode::Lexical) require(query.embedding_version == config_.embedding_version, "embedding version mismatch");
    if (query.mode != RetrievalMode::Vector) require(query.lexical_index_version == config_.lexical_index_version, "lexical index version mismatch");
    const auto started = std::chrono::steady_clock::now();
    const auto query_embedding = query.embedding.empty() ? embed(query.text) : query.embedding;
    require(query_embedding.size() == config_.embedding_dimension, "knowledge query embedding dimension mismatch");
    std::vector<KnowledgeHit> hits;
    KnowledgeQueryAudit audit;
    audit.query_id = query.query_id;
    audit.mode = query.mode;
    audit.tenant_id = query.tenant_id;
    audit.role = query.role;
    audit.decision = "allow";
    for (const auto& record : records_) {
        ++audit.scanned_records;
        if (!can_access(record, query)) { ++audit.unauthorized_records; continue; }
        const auto stale = is_stale(record, query.valid_at);
        if (stale) {
            ++audit.stale_records;
            if (!query.include_stale || record.retention_and_deletion_state == KnowledgeState::Deleted ||
                record.retention_and_deletion_state == KnowledgeState::Quarantined) continue;
        }
        if (record.quality.quality < config_.minimum_quality || record.quality.confidence < config_.minimum_confidence) continue;
        const auto lexical = lexical_score(query.text, record.content);
        const auto vector = vector_score(query_embedding, embed(record.content));
        double combined = 0.0;
        if (query.mode == RetrievalMode::Lexical) combined = lexical;
        else if (query.mode == RetrievalMode::Vector) combined = vector;
        else combined = config_.lexical_weight * lexical + config_.vector_weight * vector;
        if (query.mode == RetrievalMode::Hybrid && lexical < 0.5 && vector < 0.85) continue;
        if (combined <= 0.0) continue;
        KnowledgeHit hit;
        hit.knowledge_id = record.knowledge_id;
        hit.tenant_id = record.tenant_id;
        hit.document_id = record.document_id;
        hit.document_version = record.document_version;
        hit.source_uri_or_reference = record.source_uri_or_reference;
        hit.content = record.content;
        hit.content_hash = record.content_hash;
        hit.access_allowed = true;
        hit.lexical_score = lexical;
        hit.vector_score = vector;
        hit.combined_score = combined;
        hit.temporally_valid = temporally_valid(record, query.valid_at);
        hit.stale = stale;
        hit.conflict_group = conflict_key(record);
        hit.source_risk = record.quality.source_risk;
        hit.embedding_version = record.embedding_version;
        hit.lexical_index_version = record.lexical_index_version;
        hit.transformation_version = config_.transformation_version;
        hit.citation_spans = record.citation_spans;
        hits.push_back(std::move(hit));
    }
    std::map<std::string, std::size_t> conflict_counts;
    for (const auto& hit : hits) if (!hit.conflict_group.empty()) ++conflict_counts[hit.conflict_group];
    for (auto& hit : hits) hit.conflict_visible = !hit.conflict_group.empty() && conflict_counts[hit.conflict_group] > 1U;
    std::sort(hits.begin(), hits.end(), [](const KnowledgeHit& left, const KnowledgeHit& right) {
        if (left.combined_score != right.combined_score) return left.combined_score > right.combined_score;
        if (left.document_version != right.document_version) return left.document_version > right.document_version;
        return left.knowledge_id < right.knowledge_id;
    });
    const auto limit = std::min({query.top_k, config_.maximum_hits, hits.size()});
    hits.resize(limit);
    for (const auto& hit : hits) audit.returned_knowledge_ids.push_back(hit.knowledge_id);
    audit.reason = hits.empty() ? "no authorized valid evidence" : "typed evidence returned";
    audit_.push_back(audit);
    ++metrics_.query_count;
    metrics_.total_scanned_records += audit.scanned_records;
    metrics_.total_returned_hits += hits.size();
    metrics_.unauthorized_hits_returned += 0U;
    metrics_.stale_hits_returned += static_cast<std::size_t>(std::count_if(hits.begin(), hits.end(), [](const KnowledgeHit& hit) { return hit.stale; }));
    metrics_.last_latency_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return hits;
}

bool KnowledgePlane::claim_supported(const KnowledgeClaim& claim, const KnowledgeHit& hit) {
    const auto stop_word = [](const std::string& term) {
        static const std::set<std::string> stop_words{"a", "an", "and", "are", "by", "do", "does", "for", "from", "has", "i", "in", "is", "it", "of", "on", "that", "the", "this", "to", "what", "with", "will"};
        return stop_words.contains(term);
    };
    const auto all_claim_terms = terms(claim.text);
    const auto all_evidence_terms = terms(hit.content);
    std::vector<std::string> claim_terms;
    std::vector<std::string> evidence_terms;
    for (const auto& term : all_claim_terms) if (!stop_word(term)) claim_terms.push_back(term);
    for (const auto& term : all_evidence_terms) if (!stop_word(term)) evidence_terms.push_back(term);
    if (claim_terms.empty() || evidence_terms.empty()) return false;
    std::set<std::string> evidence(evidence_terms.begin(), evidence_terms.end());
    std::size_t matches = 0U;
    for (const auto& term : claim_terms) if (evidence.contains(term)) ++matches;
    return matches * 3U >= claim_terms.size() * 2U;
}

VerifiedAnswer KnowledgePlane::verify_answer(const GroundedAnswerRequest& request, const std::vector<KnowledgeHit>& hits) const {
    require(!request.answer_id.empty() && !request.query_id.empty() && !request.answer_text.empty(), "grounded answer request is incomplete");
    VerifiedAnswer result;
    result.answer_id = request.answer_id;
    result.query_id = request.query_id;
    result.mode = request.mode;
    result.claim_count = request.claims.size();
    std::set<std::string> span_ids;
    std::map<std::string, const KnowledgeHit*> spans;
    for (const auto& hit : hits) {
        for (const auto& span : hit.citation_spans) {
            span_ids.insert(span.span_id);
            spans[span.span_id] = &hit;
        }
    }
    std::map<std::string, std::size_t> conflicts;
    for (const auto& hit : hits) if (hit.conflict_visible) ++conflicts[hit.conflict_group];
    result.conflict_detected = std::any_of(conflicts.begin(), conflicts.end(), [](const auto& item) { return item.second > 1U; });
    for (const auto& claim : request.claims) {
        if (!claim.citation_span_ids.empty()) ++result.cited_claim_count;
        bool supported = false;
        for (const auto& span_id : claim.citation_span_ids) {
            const auto found = spans.find(span_id);
            if (found != spans.end() && found->second != nullptr && found->second->source_risk != "poisoned" && claim_supported(claim, *found->second)) supported = true;
        }
        if (supported) ++result.supported_claim_count;
    }
    result.citation_precision = result.cited_claim_count == 0U ? 0.0 : static_cast<double>(result.supported_claim_count) / static_cast<double>(result.cited_claim_count);
    result.citation_recall = result.claim_count == 0U ? 0.0 : static_cast<double>(result.supported_claim_count) / static_cast<double>(result.claim_count);
    result.accepted = result.claim_count > 0U && result.supported_claim_count == result.claim_count && result.cited_claim_count == result.claim_count &&
                      (!result.conflict_detected || request.allow_conflicts);
    result.abstained = !result.accepted;
    if (result.accepted) result.reason = "all answer claims bound to verified evidence";
    else if (result.conflict_detected && !request.allow_conflicts) result.reason = "conflicting evidence requires explicit uncertainty";
    else if (result.cited_claim_count != result.claim_count) result.reason = "uncited claim requires abstention";
    else result.reason = "one or more claims are unsupported by cited evidence";
    return result;
}

GroundingReviewSummary KnowledgePlane::review_grounded_answers(const std::vector<GroundingReview>& reviews) const {
    require(!reviews.empty(), "grounding review set cannot be empty");
    std::set<std::string> ids;
    GroundingReviewSummary summary;
    summary.review_count = reviews.size();
    for (const auto& review : reviews) {
        require(!review.review_id.empty() && ids.insert(review.review_id).second && !review.answer_id.empty() && !review.reviewer_class.empty() && review.blind,
                "grounding review protocol is invalid");
        if (review.grounded) ++summary.grounded_count;
        if (review.citation_correct) ++summary.citation_correct_count;
        if (review.uncertainty_appropriate) ++summary.uncertainty_appropriate_count;
        if (review.domain_expert) summary.expert_review_present = true;
    }
    summary.grounded_rate = static_cast<double>(summary.grounded_count) / static_cast<double>(summary.review_count);
    summary.blind_protocol_valid = ids.size() == reviews.size();
    return summary;
}

std::string KnowledgePlane::serialize_snapshot() const {
    std::ostringstream output;
    output << "CCT_KNOWLEDGE_SNAPSHOT_V1\n" << config_line(config_);
    for (const auto& record : records_) output << record_line(record);
    return output.str();
}

KnowledgePlane KnowledgePlane::deserialize_snapshot(const std::string& snapshot) {
    std::istringstream input(snapshot);
    std::string line;
    require(std::getline(input, line) && line == "CCT_KNOWLEDGE_SNAPSHOT_V1", "unsupported knowledge snapshot version");
    require(static_cast<bool>(std::getline(input, line)), "knowledge snapshot has no configuration");
    const auto config_fields = split(line, '|');
    require(config_fields.size() == 11U && config_fields[0] == "C", "malformed knowledge snapshot configuration");
    KnowledgeIndexConfig config;
    config.embedding_version = hex_decode(config_fields[1]); config.lexical_index_version = hex_decode(config_fields[2]);
    config.ranking_version = hex_decode(config_fields[3]); config.transformation_version = hex_decode(config_fields[4]);
    config.embedding_dimension = static_cast<std::size_t>(std::stoull(config_fields[5])); config.lexical_weight = std::stod(config_fields[6]);
    config.vector_weight = std::stod(config_fields[7]); config.minimum_quality = std::stod(config_fields[8]);
    config.minimum_confidence = std::stod(config_fields[9]); config.maximum_hits = static_cast<std::size_t>(std::stoull(config_fields[10]));
    KnowledgePlane plane(config);
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto parts = split(line, '|');
        require(parts.size() >= 2U && parts[0] == "R", "unknown knowledge snapshot record");
        const std::vector<std::string> values(parts.begin() + 1, parts.end());
        KnowledgeRecord record;
        std::size_t index = 0U;
        record.knowledge_id = field(values, index++); record.tenant_id = field(values, index++); record.document_id = field(values, index++);
        record.document_version = static_cast<std::uint64_t>(std::stoull(field(values, index++))); record.source_uri_or_reference = field(values, index++);
        record.content = field(values, index++); record.content_hash = field(values, index++); record.embedding_version = field(values, index++);
        record.lexical_index_version = field(values, index++); record.created_at = std::stoll(field(values, index++)); record.valid_from = std::stoll(field(values, index++));
        record.valid_until = parse_optional_time(field(values, index++)); record.access_policy.tenant_id = field(values, index++);
        record.access_policy.public_read = parse_bool(field(values, index++));
        const auto role_count = static_cast<std::size_t>(std::stoull(field(values, index++)));
        for (std::size_t role = 0U; role < role_count; ++role) record.access_policy.allowed_roles.push_back(field(values, index++));
        record.provenance = field(values, index++);
        const auto span_count = static_cast<std::size_t>(std::stoull(field(values, index++)));
        for (std::size_t span = 0U; span < span_count; ++span) {
            KnowledgeCitationSpan citation;
            citation.span_id = field(values, index++); citation.start = static_cast<std::size_t>(std::stoull(field(values, index++)));
            citation.end = static_cast<std::size_t>(std::stoull(field(values, index++))); citation.span_hash = field(values, index++);
            record.citation_spans.push_back(std::move(citation));
        }
        record.quality.quality = std::stod(field(values, index++)); record.quality.confidence = std::stod(field(values, index++));
        record.quality.source_risk = field(values, index++);
        const auto relation_count = static_cast<std::size_t>(std::stoull(field(values, index++)));
        for (std::size_t relation = 0U; relation < relation_count; ++relation) record.supersedes_or_conflicts.push_back(field(values, index++));
        const auto state = field(values, index++);
        if (state == "active") record.retention_and_deletion_state = KnowledgeState::Active;
        else if (state == "superseded") record.retention_and_deletion_state = KnowledgeState::Superseded;
        else if (state == "deleted") record.retention_and_deletion_state = KnowledgeState::Deleted;
        else if (state == "quarantined") record.retention_and_deletion_state = KnowledgeState::Quarantined;
        else throw KnowledgeError("unknown serialized knowledge state");
        record.superseded = parse_bool(field(values, index++));
        require(index == values.size(), "extra serialized knowledge fields");
        plane.validate_record(record);
        require(!std::any_of(plane.records_.begin(), plane.records_.end(), [&](const KnowledgeRecord& existing) { return existing.knowledge_id == record.knowledge_id; }),
                "duplicate serialized knowledge ID");
        plane.records_.push_back(std::move(record));
    }
    plane.rebuild();
    return plane;
}

bool KnowledgePlane::contains_active(const std::string& knowledge_id) const {
    return std::any_of(records_.begin(), records_.end(), [&](const KnowledgeRecord& record) {
        return record.knowledge_id == knowledge_id && record.retention_and_deletion_state == KnowledgeState::Active && !record.superseded;
    });
}

}  // namespace cct
