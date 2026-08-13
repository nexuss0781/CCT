#include "cct/memory.hpp"
#include "cct/corpus.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fcntl.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>

namespace cct {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw MemoryError(message);
}

bool finite(double value) { return std::isfinite(value); }

std::uint64_t mix_hash(std::uint64_t state, std::uint64_t value) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    return state;
}

std::uint64_t hash_bytes(std::uint64_t state, const std::string& value) {
    for (const auto character : value) state = mix_hash(state, static_cast<unsigned char>(character));
    return state;
}

std::vector<std::string> memory_terms(const std::string& text) {
    std::vector<std::string> result;
    std::string current;
    for (const char raw_character : text) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (std::isalnum(character) != 0U) {
            current.push_back(static_cast<char>(std::tolower(character)));
        } else if (!current.empty()) {
            result.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

std::uint64_t double_bits(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

bool vector_equal(const std::vector<double>& left, const std::vector<double>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

unsigned int status_value(MemoryStatus value) { return static_cast<unsigned int>(value); }
unsigned int retention_value(RetentionClass value) { return static_cast<unsigned int>(value); }
unsigned int event_type_value(MemoryEventType value) { return static_cast<unsigned int>(value); }

std::string record_payload(const MemoryRecord& record) {
    std::ostringstream output;
    output << record.schema_version << '|' << record.memory_id << '|' << record.version << '|' << std::quoted(record.content) << '|';
    output << record.embedding.size();
    for (const auto value : record.embedding) output << '|' << std::setprecision(17) << value;
    output << '|' << record.event_ids.size();
    for (const auto value : record.event_ids) output << '|' << value;
    output << '|' << record.causal_parents.size();
    for (const auto value : record.causal_parents) output << '|' << value;
    output << '|' << record.created_at << '|' << record.valid_from << '|' << (record.valid_until.has_value() ? 1 : 0);
    if (record.valid_until.has_value()) output << '|' << *record.valid_until;
    output << '|' << std::quoted(record.source.source_id) << '|' << record.source.span_start << '|' << record.source.span_end
           << '|' << std::setprecision(17) << record.confidence << '|' << status_value(record.status) << '|' << retention_value(record.retention)
           << '|' << std::quoted(record.conflict_group);
    return output.str();
}

std::string digest_event_payload(const MemoryEvent& event) {
    std::ostringstream output;
    output << event.sequence << '|' << event_type_value(event.type) << '|' << event.target_id << '|' << event.previous_version << '|'
           << std::quoted(event.reason) << '|' << event.previous_event_digest << '|' << event.record.checksum_digest << '|'
           << event.record.memory_id << '|' << event.record.version;
    return output.str();
}

std::uint64_t digest_prefix(const std::string& digest) {
    require(digest.size() >= 16U, "memory SHA-256 digest is truncated");
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index < 16U; index += 2U) {
        const auto hex = [](const char character) -> std::uint64_t {
            if (character >= '0' && character <= '9') return static_cast<std::uint64_t>(character - '0');
            if (character >= 'a' && character <= 'f') return static_cast<std::uint64_t>(character - 'a' + 10);
            if (character >= 'A' && character <= 'F') return static_cast<std::uint64_t>(character - 'A' + 10);
            throw MemoryError("memory SHA-256 digest contains a non-hex character");
        };
        value = (value << 8U) | (hex(digest[index]) << 4U) | hex(digest[index + 1U]);
    }
    return value;
}

void atomic_write_file(const std::string& path, const std::string& content) {
    const std::filesystem::path target(path);
    const auto parent = target.parent_path().empty() ? std::filesystem::path(".") : target.parent_path();
    std::filesystem::create_directories(parent);
    const auto template_path = (parent / (target.filename().string() + ".tmp.XXXXXX")).string();
    std::vector<char> template_bytes(template_path.begin(), template_path.end());
    template_bytes.push_back('\0');
    const auto descriptor = ::mkstemp(template_bytes.data());
    require(descriptor >= 0, "could not create memory snapshot temporary file");
    const auto temporary_path = std::string(template_bytes.data());
    auto cleanup = [&]() {
        ::close(descriptor);
        static_cast<void>(::unlink(temporary_path.c_str()));
    };
    std::size_t written = 0U;
    while (written < content.size()) {
        const auto count = ::write(descriptor, content.data() + written, content.size() - written);
        if (count <= 0) {
            cleanup();
            throw MemoryError("could not write memory snapshot temporary file");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
        static_cast<void>(::unlink(temporary_path.c_str()));
        throw MemoryError("could not durably flush memory snapshot temporary file");
    }
    if (::rename(temporary_path.c_str(), target.c_str()) != 0) {
        static_cast<void>(::unlink(temporary_path.c_str()));
        throw MemoryError("could not atomically publish memory snapshot");
    }
    const auto directory_descriptor = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(directory_descriptor >= 0, "could not open memory snapshot parent directory");
    const auto directory_sync = ::fsync(directory_descriptor);
    const auto directory_close = ::close(directory_descriptor);
    require(directory_sync == 0 && directory_close == 0, "could not durably publish memory snapshot directory entry");
}

void write_record(std::ostream& output, const MemoryRecord& record) {
    output << "RECORD " << record.schema_version << ' ' << record.memory_id << ' ' << record.version << ' '
           << std::quoted(record.content) << ' ' << record.embedding.size();
    for (const auto value : record.embedding) output << ' ' << std::setprecision(17) << value;
    output << ' ' << record.event_ids.size();
    for (const auto value : record.event_ids) output << ' ' << value;
    output << ' ' << record.causal_parents.size();
    for (const auto value : record.causal_parents) output << ' ' << value;
    output << ' ' << record.created_at << ' ' << record.valid_from << ' ' << (record.valid_until.has_value() ? 1 : 0);
    if (record.valid_until.has_value()) output << ' ' << *record.valid_until;
    output << ' ' << std::quoted(record.source.source_id) << ' ' << record.source.span_start << ' ' << record.source.span_end
           << ' ' << std::setprecision(17) << record.confidence << ' ' << status_value(record.status) << ' '
           << retention_value(record.retention) << ' ' << std::quoted(record.conflict_group) << ' ' << record.checksum << ' ' << std::quoted(record.checksum_digest) << '\n';
}

MemoryRecord read_record(std::istream& input, const bool has_digest) {
    std::string token;
    input >> token;
    require(token == "RECORD", "invalid memory record marker");
    MemoryRecord record;
    unsigned int status = 0;
    unsigned int retention = 0;
    int has_valid_until = 0;
    std::size_t count = 0;
    input >> record.schema_version >> record.memory_id >> record.version >> std::quoted(record.content) >> count;
    require(static_cast<bool>(input) && count <= 4096U, "memory embedding count exceeds budget");
    record.embedding.resize(count);
    for (auto& value : record.embedding) input >> value;
    input >> count;
    require(static_cast<bool>(input) && count <= 1'000'000U, "memory event ID count exceeds budget");
    record.event_ids.resize(count);
    for (auto& value : record.event_ids) input >> value;
    input >> count;
    require(static_cast<bool>(input) && count <= 1'000'000U, "memory causal parent count exceeds budget");
    record.causal_parents.resize(count);
    for (auto& value : record.causal_parents) input >> value;
    input >> record.created_at >> record.valid_from >> has_valid_until;
    if (has_valid_until != 0) {
        LogicalTime until = 0;
        input >> until;
        record.valid_until = until;
    }
    input >> std::quoted(record.source.source_id) >> record.source.span_start >> record.source.span_end >> record.confidence >>
        status >> retention >> std::quoted(record.conflict_group) >> record.checksum;
    if (has_digest) input >> std::quoted(record.checksum_digest);
    record.status = static_cast<MemoryStatus>(status);
    record.retention = static_cast<RetentionClass>(retention);
    require(static_cast<bool>(input), "truncated memory record");
    return record;
}

}  // namespace

MemoryEncoder::MemoryEncoder(std::size_t embedding_dim, std::uint32_t schema_version)
    : embedding_dim_(embedding_dim), schema_version_(schema_version) {
    require(embedding_dim_ > 0, "memory embedding dimension must be positive");
    require(schema_version_ == MemoryRecord::kSchemaVersion, "unsupported memory encoder schema");
}

std::string MemoryEncoder::content_digest(const MemoryRecord& record) const {
    return GovernedCorpus::content_sha256(record_payload(record));
}

std::uint64_t MemoryEncoder::content_checksum(const MemoryRecord& record) const {
    const auto digest = content_digest(record);
    std::uint64_t hash = 0U;
    for (std::size_t index = 0U; index < 16U; index += 2U) {
        const auto hex = [](const char value) -> std::uint64_t {
            if (value >= '0' && value <= '9') return static_cast<std::uint64_t>(value - '0');
            if (value >= 'a' && value <= 'f') return static_cast<std::uint64_t>(value - 'a' + 10);
            return static_cast<std::uint64_t>(value - 'A' + 10);
        };
        hash = (hash << 8U) | (hex(digest[index]) << 4U) | hex(digest[index + 1U]);
    }
    return hash;
}

std::uint64_t legacy_content_checksum(const MemoryRecord& record) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = mix_hash(hash, record.schema_version);
    hash = mix_hash(hash, record.memory_id);
    hash = mix_hash(hash, record.version);
    hash = hash_bytes(hash, record.content);
    for (const auto value : record.embedding) hash = mix_hash(hash, double_bits(value));
    for (const auto value : record.event_ids) hash = mix_hash(hash, value);
    for (const auto value : record.causal_parents) hash = mix_hash(hash, value);
    hash = mix_hash(hash, static_cast<std::uint64_t>(record.created_at));
    hash = mix_hash(hash, static_cast<std::uint64_t>(record.valid_from));
    hash = mix_hash(hash, record.valid_until.has_value() ? static_cast<std::uint64_t>(*record.valid_until) : 0ULL);
    hash = hash_bytes(hash, record.source.source_id);
    hash = mix_hash(hash, record.source.span_start);
    hash = mix_hash(hash, record.source.span_end);
    hash = mix_hash(hash, double_bits(record.confidence));
    hash = mix_hash(hash, status_value(record.status));
    hash = mix_hash(hash, retention_value(record.retention));
    hash = hash_bytes(hash, record.conflict_group);
    return hash;
}

std::uint64_t legacy_event_checksum(const MemoryEvent& event) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = mix_hash(hash, event.sequence);
    hash = mix_hash(hash, event_type_value(event.type));
    hash = mix_hash(hash, event.target_id);
    hash = mix_hash(hash, event.previous_version);
    hash = mix_hash(hash, event.previous_event_checksum);
    hash = hash_bytes(hash, event.reason);
    hash = mix_hash(hash, event.record.checksum);
    hash = mix_hash(hash, event.record.memory_id);
    hash = mix_hash(hash, event.record.version);
    return hash;
}

std::vector<double> MemoryEncoder::encode(const MemoryRecord& record) const {
    require(record.embedding.size() == embedding_dim_, "memory embedding dimension mismatch");
    std::vector<double> result(embedding_dim_, 0.0);
    const auto seed = content_checksum(record);
    for (std::size_t index = 0; index < embedding_dim_; ++index) {
        const auto bits = seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(index + 1));
        const auto normalized = static_cast<double>(bits % 1000003ULL) / 1000003.0;
        result[index] = 0.7 * record.embedding[index] + 0.1 * normalized +
                        0.02 * static_cast<double>(record.event_ids.size()) +
                        0.01 * static_cast<double>(record.status == MemoryStatus::Active);
    }
    return result;
}

MemoryWriteController::MemoryWriteController(double novelty_threshold, double quarantine_threshold)
    : novelty_threshold_(novelty_threshold), quarantine_threshold_(quarantine_threshold) {
    require(novelty_threshold_ >= 0.0 && quarantine_threshold_ >= 0.0 && quarantine_threshold_ <= 1.0,
            "invalid memory controller thresholds");
}

MemoryDecision MemoryWriteController::decide(const MemoryRecord& candidate,
                                             const std::vector<MemoryRecord>& existing_active) const {
    if (candidate.confidence < quarantine_threshold_) {
        return {MemoryDecisionKind::Quarantine, candidate.memory_id, candidate.version, "confidence_below_quarantine_threshold"};
    }
    for (const auto& existing : existing_active) {
        if (existing.memory_id == candidate.memory_id) {
            if (existing.content == candidate.content && vector_equal(existing.embedding, candidate.embedding)) {
                return {MemoryDecisionKind::Ignore, candidate.memory_id, existing.version, "duplicate_content_and_embedding"};
            }
            return {MemoryDecisionKind::Update, candidate.memory_id, existing.version + 1, "same_identity_new_version"};
        }
        if (!candidate.conflict_group.empty() && candidate.conflict_group == existing.conflict_group &&
            existing.content == candidate.content && vector_equal(existing.embedding, candidate.embedding)) {
            return {MemoryDecisionKind::Ignore, existing.memory_id, existing.version, "duplicate_conflict_group_content"};
        }
        if (candidate.embedding.size() == existing.embedding.size()) {
            double squared_distance = 0.0;
            for (std::size_t index = 0U; index < candidate.embedding.size(); ++index) {
                const auto delta = candidate.embedding[index] - existing.embedding[index];
                squared_distance += delta * delta;
            }
            if (novelty_threshold_ > 0.0 && squared_distance <= novelty_threshold_)
                return {MemoryDecisionKind::Ignore, existing.memory_id, existing.version, "below_novelty_threshold"};
        }
    }
    return {MemoryDecisionKind::Write, candidate.memory_id, 1, "novel_record"};
}

PersistentMemory::PersistentMemory(MemoryConfig config)
    : config_(std::move(config)), encoder_(config_.embedding_dim), write_controller_(config_.novelty_threshold, config_.quarantine_threshold), next_sequence_(1) {
    require(config_.embedding_dim > 0, "memory embedding dimension must be positive");
    require(!config_.embedding_backend.empty(), "memory embedding backend must be named");
    require(config_.max_active_records > 0, "memory capacity must be positive");
    require(finite(config_.minimum_confidence) && config_.minimum_confidence >= 0.0 && config_.minimum_confidence <= 1.0,
            "invalid memory confidence threshold");
    require(finite(config_.novelty_threshold) && config_.novelty_threshold >= 0.0 && finite(config_.quarantine_threshold) &&
                config_.quarantine_threshold >= 0.0 && config_.quarantine_threshold <= 1.0,
            "invalid memory write thresholds");
}

void PersistentMemory::validate_record(const MemoryRecord& record) const {
    require(record.schema_version == MemoryRecord::kSchemaVersion, "unsupported memory record schema");
    require(record.memory_id != 0, "memory ID must be nonzero");
    require(record.version > 0, "memory version must be positive");
    require(!record.content.empty(), "memory content must not be empty");
    require(record.embedding.size() == config_.embedding_dim, "memory embedding dimension mismatch");
    for (const auto value : record.embedding) require(finite(value), "memory embedding is non-finite");
    require(std::is_sorted(record.event_ids.begin(), record.event_ids.end()), "memory event IDs must be sorted");
    require(std::adjacent_find(record.event_ids.begin(), record.event_ids.end()) == record.event_ids.end(),
            "memory event IDs must be unique");
    require(std::is_sorted(record.causal_parents.begin(), record.causal_parents.end()),
            "memory causal parents must be sorted");
    require(std::adjacent_find(record.causal_parents.begin(), record.causal_parents.end()) == record.causal_parents.end(),
            "memory causal parents must be unique");
    require(record.valid_from <= record.created_at || record.created_at == 0, "memory validity starts after creation");
    if (record.valid_until.has_value()) require(*record.valid_until > record.valid_from, "memory validity interval is empty");
    require(finite(record.confidence) && record.confidence >= 0.0 && record.confidence <= 1.0,
            "memory confidence must be in [0,1]");
    require(record.source.span_start <= record.source.span_end, "memory source span is reversed");
    if (record.status == MemoryStatus::Active) require(record.confidence >= config_.minimum_confidence, "memory confidence below policy");
}

MemoryEvent PersistentMemory::make_event(MemoryEventType type, const MemoryRecord& record, MemoryId target_id,
                                          std::uint64_t previous_version, const std::string& reason) const {
    MemoryEvent event;
    event.sequence = next_sequence_;
    event.type = type;
    event.record = record;
    event.target_id = target_id;
    event.previous_version = previous_version;
    event.reason = reason;
    event.previous_event_checksum = event_log_.empty() ? digest_prefix(GovernedCorpus::content_sha256(std::to_string(config_.chain_seed))) : event_log_.back().event_checksum;
    event.previous_event_digest = event_log_.empty() ? GovernedCorpus::content_sha256(std::to_string(config_.chain_seed)) : event_log_.back().event_digest;
    event.event_digest = event_digest(event);
    event.event_checksum = digest_prefix(event.event_digest);
    return event;
}

std::string PersistentMemory::event_digest(const MemoryEvent& event) const {
    return GovernedCorpus::content_sha256(digest_event_payload(event));
}

std::uint64_t PersistentMemory::event_checksum(const MemoryEvent& event) const {
    return digest_prefix(event_digest(event));
}

void PersistentMemory::apply_event(const MemoryEvent& event, bool validate_chain) {
    require(event.record.checksum_digest == encoder_.content_digest(event.record), "memory record SHA-256 digest mismatch");
    require(event.record.checksum == encoder_.content_checksum(event.record), "memory record compatibility checksum mismatch");
    if (validate_chain) {
        const auto expected_previous_digest = event.sequence == 1 ? GovernedCorpus::content_sha256(std::to_string(config_.chain_seed)) : event_log_.back().event_digest;
        const auto expected_previous = digest_prefix(expected_previous_digest);
        require(event.sequence == next_sequence_, "memory event sequence mismatch");
        require(event.previous_event_digest == expected_previous_digest && event.previous_event_checksum == expected_previous, "memory event chain mismatch");
        require(event.event_digest == event_digest(event) && event.event_checksum == digest_prefix(event.event_digest), "memory event digest mismatch");
    }
    auto& versions = versions_[event.target_id];
    if (event.type == MemoryEventType::Append || (event.type == MemoryEventType::Quarantine && versions.empty())) {
        require(versions.empty(), "append event targets an existing memory");
        versions.push_back(event.record);
        index_record_terms(event.record);
        if (event.record.status == MemoryStatus::Active) active_[event.record.memory_id] = event.record;
        return;
    }
    require(!versions.empty(), "mutation event targets an unknown memory");
    if (event.type == MemoryEventType::Update) {
        require(active_.count(event.target_id) != 0U, "update event targets inactive memory");
        require(event.previous_version == active_.at(event.target_id).version, "update version predecessor mismatch");
        versions.back().status = MemoryStatus::Superseded;
        active_.erase(event.target_id);
        versions.push_back(event.record);
        index_record_terms(event.record);
        if (event.record.status == MemoryStatus::Active) active_[event.target_id] = event.record;
        return;
    }
    require(event.previous_version == versions.back().version, "status mutation version predecessor mismatch");
    if (active_.count(event.target_id) != 0U) {
        versions.back().status = MemoryStatus::Superseded;
        active_.erase(event.target_id);
    }
    versions.push_back(event.record);
    index_record_terms(event.record);
    if (event.record.status == MemoryStatus::Active) active_[event.target_id] = event.record;
}

void PersistentMemory::append_event(MemoryEvent event) {
    require(event.sequence == next_sequence_, "event append sequence mismatch");
    require(event.event_checksum == event_checksum(event), "event checksum not finalized");
    apply_event(event, true);
    event_log_.push_back(std::move(event));
    ++next_sequence_;
}

MemoryDecision PersistentMemory::write(MemoryRecord record, const std::string& reason) {
    validate_record(record);
    const auto decision = write_controller_.decide(record, active_records());
    if (decision.kind == MemoryDecisionKind::Ignore) return decision;
    if (decision.kind == MemoryDecisionKind::Update) return update(std::move(record), reason);
    if (decision.kind == MemoryDecisionKind::Quarantine) {
        record.status = MemoryStatus::Quarantined;
        record.checksum_digest = encoder_.content_digest(record);
        record.checksum = encoder_.content_checksum(record);
        const auto event = make_event(MemoryEventType::Quarantine, record, record.memory_id, 0, reason + ":quarantine");
        append_event(event);
        return {MemoryDecisionKind::Quarantine, record.memory_id, record.version, "quarantined_by_write_controller"};
    }
    record.version = 1;
    record.status = MemoryStatus::Active;
    record.checksum_digest = encoder_.content_digest(record);
    record.checksum = encoder_.content_checksum(record);
    const auto event = make_event(MemoryEventType::Append, record, record.memory_id, 0, reason);
    append_event(event);
    return {MemoryDecisionKind::Write, record.memory_id, record.version, reason};
}

MemoryDecision PersistentMemory::update(MemoryRecord record, const std::string& reason) {
    validate_record(record);
    require(active_.count(record.memory_id) != 0U, "cannot update inactive or unknown memory");
    const auto& previous = active_.at(record.memory_id);
    if (previous.content == record.content && vector_equal(previous.embedding, record.embedding)) {
        return {MemoryDecisionKind::Ignore, record.memory_id, previous.version, "duplicate_update"};
    }
    record.version = previous.version + 1;
    record.status = MemoryStatus::Active;
    record.checksum_digest = encoder_.content_digest(record);
    record.checksum = encoder_.content_checksum(record);
    const auto event = make_event(MemoryEventType::Update, record, record.memory_id, previous.version, reason);
    append_event(event);
    return {MemoryDecisionKind::Update, record.memory_id, record.version, reason};
}

MemoryDecision PersistentMemory::delete_memory_now(const MemoryId memory_id, const std::string& reason) {
    require(memory_id != 0, "memory ID must be nonzero");
    const auto iterator = active_.find(memory_id);
    if (iterator == active_.end()) return {MemoryDecisionKind::Ignore, memory_id, 0, "memory_not_active"};
    MemoryRecord tombstone = iterator->second;
    tombstone.version += 1;
    tombstone.status = MemoryStatus::Deleted;
    tombstone.checksum_digest = encoder_.content_digest(tombstone);
    tombstone.checksum = encoder_.content_checksum(tombstone);
    const auto event = make_event(MemoryEventType::Tombstone, tombstone, memory_id, iterator->second.version, reason);
    append_event(event);
    return {MemoryDecisionKind::Update, memory_id, tombstone.version, "deleted_immediately"};
}

MemoryDecision PersistentMemory::delete_memory(const MemoryId memory_id, const std::string& reason) {
    require(memory_id != 0, "memory ID must be nonzero");
    if (config_.immediate_deletion) return delete_memory_now(memory_id, reason);
    if (active_.count(memory_id) == 0U) return {MemoryDecisionKind::Ignore, memory_id, 0, "memory_not_active"};
    if (std::find(deferred_deletions_.begin(), deferred_deletions_.end(), memory_id) == deferred_deletions_.end()) deferred_deletions_.push_back(memory_id);
    return {MemoryDecisionKind::Update, memory_id, active_.at(memory_id).version + 1U, "deletion_deferred"};
}

std::size_t PersistentMemory::process_deferred_deletions(const std::string& reason) {
    const auto queued = std::move(deferred_deletions_);
    deferred_deletions_.clear();
    std::size_t processed = 0U;
    for (const auto memory_id : queued) {
        if (delete_memory_now(memory_id, reason).kind != MemoryDecisionKind::Ignore) ++processed;
    }
    return processed;
}

MemoryDecision PersistentMemory::quarantine(MemoryId memory_id, const std::string& reason) {
    require(memory_id != 0, "memory ID must be nonzero");
    const auto iterator = active_.find(memory_id);
    if (iterator == active_.end()) return {MemoryDecisionKind::Ignore, memory_id, 0, "memory_not_active"};
    MemoryRecord quarantined = iterator->second;
    quarantined.version += 1;
    quarantined.status = MemoryStatus::Quarantined;
    quarantined.checksum_digest = encoder_.content_digest(quarantined);
    quarantined.checksum = encoder_.content_checksum(quarantined);
    const auto event = make_event(MemoryEventType::Quarantine, quarantined, memory_id, iterator->second.version, reason);
    append_event(event);
    return {MemoryDecisionKind::Quarantine, memory_id, quarantined.version, "quarantined_immediately"};
}

std::size_t PersistentMemory::expire(LogicalTime now, const std::string& reason) {
    std::vector<MemoryId> expired;
    for (const auto& [memory_id, record] : active_) {
        if (record.valid_until.has_value() && *record.valid_until <= now && record.retention != RetentionClass::LegalHold) {
            expired.push_back(memory_id);
        }
    }
    for (const auto memory_id : expired) (void)delete_memory(memory_id, reason);
    return expired.size();
}

std::size_t PersistentMemory::enforce_capacity(const std::string& reason) {
    std::size_t deleted = 0U;
    std::size_t deferred_count = 0U;
    while (active_.size() > config_.max_active_records + deferred_count) {
        auto candidate = active_.end();
        for (auto iterator = active_.begin(); iterator != active_.end(); ++iterator) {
            if (iterator->second.retention == RetentionClass::LegalHold) continue;
            if (candidate == active_.end() ||
                std::tie(iterator->second.retention, iterator->second.created_at, iterator->second.confidence, iterator->first) <
                    std::tie(candidate->second.retention, candidate->second.created_at, candidate->second.confidence, candidate->first)) {
                candidate = iterator;
            }
        }
        if (candidate == active_.end()) break;
        const auto decision = delete_memory(candidate->first, reason);
        if (decision.kind == MemoryDecisionKind::Ignore) break;
        ++deleted;
        if (!config_.immediate_deletion) ++deferred_count;
    }
    return deleted;
}

MemoryDecision PersistentMemory::write_event(const CausalEvent& event, const std::vector<double>& embedding,
                                             MemoryId memory_id, const std::string& reason) {
    require(memory_id != 0, "memory ID must be nonzero");
    require(event.semantic_payload.size() == 1, "causal event memory adapter expects scalar payload");
    MemoryRecord record;
    record.memory_id = memory_id;
    record.content = "causal_event:" + std::to_string(event.id) + ":" + std::to_string(event.semantic_payload.front());
    record.embedding = embedding;
    record.event_ids = {event.id};
    record.causal_parents = event.causal_parents;
    record.created_at = event.timestamp;
    record.valid_from = event.timestamp;
    record.source = {"causal_event", 0, record.content.size()};
    record.confidence = event.uncertainty.confidence;
    record.retention = RetentionClass::Standard;
    return write(std::move(record), reason);
}

bool PersistentMemory::valid_at(const MemoryRecord& record, const MemoryQuery& query) const {
    if (query.valid_at.has_value()) {
        if (record.valid_from > *query.valid_at) return false;
        if (!query.include_expired && record.valid_until.has_value() && *record.valid_until <= *query.valid_at) return false;
    }
    if (query.created_after.has_value() && record.created_at < *query.created_after) return false;
    if (query.created_before.has_value() && record.created_at > *query.created_before) return false;
    if (!query.include_expired && record.valid_until.has_value() && !query.valid_at.has_value() && record.valid_until.value() <= record.created_at) return false;
    return true;
}

double PersistentMemory::cosine_similarity(const std::vector<double>& left, const std::vector<double>& right) const {
    require(left.size() == right.size(), "memory query embedding dimension mismatch");
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
    }
    if (left_norm == 0.0 || right_norm == 0.0) return 0.0;
    return dot / std::sqrt(left_norm * right_norm);
}

double PersistentMemory::lexical_similarity(const std::string& query, const std::string& content) const {
    const auto query_values = memory_terms(query);
    const auto content_values = memory_terms(content);
    if (query_values.empty() || content_values.empty()) return 0.0;
    const std::unordered_set<std::string> content_set(content_values.begin(), content_values.end());
    std::size_t matches = 0U;
    const std::unordered_set<std::string> query_set(query_values.begin(), query_values.end());
    for (const auto& term : query_set) {
        if (content_set.contains(term)) ++matches;
    }
    const auto query_unique = query_set.size();
    return query_unique == 0U ? 0.0 : static_cast<double>(matches) / static_cast<double>(query_unique);
}

void PersistentMemory::index_record_terms(const MemoryRecord& record) {
    if (record.status == MemoryStatus::Deleted || record.status == MemoryStatus::Quarantined) return;
    const auto values = memory_terms(record.content);
    const std::unordered_set<std::string> unique_values(values.begin(), values.end());
    for (const auto& term : unique_values) content_index_[term].insert(record.memory_id);
}

void PersistentMemory::rebuild_retrieval_index() {
    content_index_.clear();
    for (const auto& [memory_id, record] : active_) {
        (void)memory_id;
        index_record_terms(record);
    }
    for (const auto& [memory_id, versions] : versions_) {
        (void)memory_id;
        for (const auto& record : versions) index_record_terms(record);
    }
}

std::vector<MemoryHit> PersistentMemory::retrieve(const MemoryQuery& query) const { return retrieve_internal(query, true); }

std::vector<MemoryHit> PersistentMemory::retrieve_linear_oracle(const MemoryQuery& query) const { return retrieve_internal(query, false); }

std::vector<MemoryHit> PersistentMemory::retrieve_internal(const MemoryQuery& query, const bool use_index) const {
    require(query.embedding.empty() || query.embedding.size() == config_.embedding_dim,
            "memory query embedding dimension mismatch");
    std::unordered_set<MemoryId> indexed_ids;
    if (use_index && query.text.has_value()) {
        for (const auto& term : memory_terms(*query.text)) {
            const auto found = content_index_.find(term);
            if (found != content_index_.end()) indexed_ids.insert(found->second.begin(), found->second.end());
        }
    }
    const auto candidate_allowed = [&](const MemoryId id) {
        return !use_index || !query.text.has_value() || indexed_ids.contains(id);
    };
    struct Candidate { MemoryHit hit; };
    std::vector<Candidate> candidates;
    const auto append_candidate = [&](const MemoryRecord& record) {
        if (record.status == MemoryStatus::Deleted || record.status == MemoryStatus::Quarantined ||
            (record.status == MemoryStatus::Superseded && !query.include_history) ||
            record.confidence < query.minimum_confidence || !valid_at(record, query) ||
            (query.source_id.has_value() && record.source.source_id != *query.source_id) ||
            (query.event_id.has_value() && std::find(record.event_ids.begin(), record.event_ids.end(), *query.event_id) == record.event_ids.end()) ||
            (query.conflict_group.has_value() && record.conflict_group != *query.conflict_group)) return;
        const auto lexical = query.text.has_value() ? lexical_similarity(*query.text, record.content) : 0.0;
        if (query.text.has_value() && lexical <= 0.0) return;
        const auto vector = query.embedding.empty() ? 0.0 : cosine_similarity(query.embedding, record.embedding);
        const auto score = query.embedding.empty() ? lexical : (query.text.has_value() ? 0.5 * lexical + 0.5 * vector : vector);
        candidates.push_back({{record.memory_id, record.version, score, record.source, record.source.span_start,
                               record.source.span_end, record.confidence, record.status, record.conflict_group, record.checksum, record.checksum_digest}});
    };
    if (query.include_history) {
        for (const auto& [memory_id, versions] : versions_) {
            if (!candidate_allowed(memory_id)) continue;
            for (const auto& record : versions) append_candidate(record);
        }
    } else {
        for (const auto& [memory_id, record] : active_) {
            if (!candidate_allowed(memory_id)) continue;
            append_candidate(record);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
        if (left.hit.score != right.hit.score) return left.hit.score > right.hit.score;
        if (left.hit.confidence != right.hit.confidence) return left.hit.confidence > right.hit.confidence;
        if (left.hit.version != right.hit.version) return left.hit.version > right.hit.version;
        return left.hit.memory_id < right.hit.memory_id;
    });
    std::vector<MemoryHit> result;
    const auto limit = std::min(query.budget, candidates.size());
    result.reserve(limit);
    for (std::size_t index = 0; index < limit; ++index) result.push_back(candidates[index].hit);
    return result;
}

CitationBinding PersistentMemory::bind_citation(const std::string& claim_id, const std::vector<MemoryHit>& hits,
                                                CitationSupport support) const {
    CitationBinding binding;
    binding.claim_id = claim_id;
    binding.support = hits.empty() ? CitationSupport::Unsupported : support;
    binding.reason = hits.empty() ? "no_retrieved_evidence" : "evidence_bound_to_memory_records";
    for (const auto& hit : hits) {
        const auto iterator = active_.find(hit.memory_id);
        if (iterator == active_.end() || iterator->second.version != hit.version || iterator->second.checksum != hit.checksum) {
            binding.support = CitationSupport::Abstained;
            binding.reason = "evidence_record_no_longer_active_or_checksum_mismatch";
            binding.memory_ids.clear();
            binding.evidence.clear();
            return binding;
        }
        binding.memory_ids.push_back(hit.memory_id);
        binding.evidence.push_back(hit.source);
    }
    return binding;
}

EvidenceContext PersistentMemory::evidence_context(const MemoryQuery& query) const {
    return {retrieve(query), true};
}

EvidenceContext PersistentMemory::no_memory_context() { return {{}, false}; }

bool PersistentMemory::contains(MemoryId memory_id) const noexcept { return active_.count(memory_id) != 0U; }

const MemoryRecord& PersistentMemory::active_record(MemoryId memory_id) const {
    const auto iterator = active_.find(memory_id);
    if (iterator == active_.end()) throw MemoryError("active memory ID not found");
    return iterator->second;
}

std::vector<MemoryRecord> PersistentMemory::active_records() const {
    std::vector<MemoryRecord> result;
    result.reserve(active_.size());
    for (const auto& [memory_id, record] : active_) {
        (void)memory_id;
        result.push_back(record);
    }
    return result;
}

std::vector<MemoryRecord> PersistentMemory::history(MemoryId memory_id) const {
    const auto iterator = versions_.find(memory_id);
    if (iterator == versions_.end()) return {};
    return iterator->second;
}

std::vector<MemoryRecord> PersistentMemory::conflict_set(const std::string& conflict_group) const {
    std::vector<MemoryRecord> result;
    for (const auto& [memory_id, record] : active_) {
        (void)memory_id;
        if (!conflict_group.empty() && record.conflict_group == conflict_group) result.push_back(record);
    }
    std::sort(result.begin(), result.end(), [](const MemoryRecord& left, const MemoryRecord& right) {
        if (left.confidence != right.confidence) return left.confidence > right.confidence;
        if (left.created_at != right.created_at) return left.created_at > right.created_at;
        return left.memory_id < right.memory_id;
    });
    return result;
}

void PersistentMemory::verify_log() const {
    PersistentMemory replayed(config_);
    for (const auto& event : event_log_) {
        replayed.apply_event(event, true);
        replayed.event_log_.push_back(event);
        replayed.next_sequence_ = event.sequence + 1;
    }
    require(replayed.canonical_state_export() == canonical_state_export(), "memory replay changed canonical state");
    require(replayed.log_export() == log_export(), "memory replay changed event log");
}

void PersistentMemory::reset_state() {
    event_log_.clear();
    versions_.clear();
    active_.clear();
    content_index_.clear();
    next_sequence_ = 1;
    deferred_deletions_.clear();
}

void PersistentMemory::rebuild_from_log() {
    const auto saved_log = event_log_;
    reset_state();
    for (const auto& event : saved_log) {
        apply_event(event, true);
        event_log_.push_back(event);
        next_sequence_ = event.sequence + 1;
    }
    rebuild_retrieval_index();
}

std::string PersistentMemory::canonical_state_export() const {
    std::ostringstream output;
    output << "CCT_MEMORY_CANONICAL_V1\n";
    for (const auto& [memory_id, record] : active_) {
        (void)memory_id;
        write_record(output, record);
    }
    return output.str();
}

std::string PersistentMemory::log_export() const {
    std::ostringstream output;
    output << "CCT_MEMORY_LOG_V2\n";
    for (const auto& event : event_log_) {
        output << "EVENT " << event.sequence << ' ' << event_type_value(event.type) << ' ' << event.target_id << ' '
               << event.previous_version << ' ' << std::quoted(event.reason) << ' ' << event.previous_event_checksum << ' '
               << event.event_checksum << ' ' << std::quoted(event.previous_event_digest) << ' ' << std::quoted(event.event_digest) << '\n';
        write_record(output, event.record);
    }
    return output.str();
}

std::string PersistentMemory::serialize_snapshot() const {
    std::ostringstream output;
    output << "CCT_MEMORY_SNAPSHOT_V3\n" << std::setprecision(17);
    output << "CONFIG " << config_.embedding_dim << ' ' << config_.max_active_records << ' '
           << config_.minimum_confidence << ' ' << config_.chain_seed << ' ' << config_.novelty_threshold << ' '
           << config_.quarantine_threshold << ' ' << (config_.immediate_deletion ? 1 : 0) << ' ' << std::quoted(config_.embedding_backend) << '\n';
    for (const auto& event : event_log_) {
        output << "EVENT " << event.sequence << ' ' << event_type_value(event.type) << ' ' << event.target_id << ' '
               << event.previous_version << ' ' << std::quoted(event.reason) << ' ' << event.previous_event_checksum << ' '
               << event.event_checksum << ' ' << std::quoted(event.previous_event_digest) << ' ' << std::quoted(event.event_digest) << '\n';
        write_record(output, event.record);
    }
    return output.str();
}

PersistentMemory PersistentMemory::deserialize_snapshot(const std::string& snapshot) {
    constexpr std::size_t maximum_snapshot_bytes = 64U * 1024U * 1024U;
    constexpr std::size_t maximum_records = 1'000'000U;
    require(snapshot.size() <= maximum_snapshot_bytes, "memory snapshot exceeds byte budget");
    std::istringstream input(snapshot);
    std::string header;
    std::getline(input, header);
    const bool version_one = header == "CCT_MEMORY_SNAPSHOT_V1";
    const bool version_two = header == "CCT_MEMORY_SNAPSHOT_V2";
    const bool version_three = header == "CCT_MEMORY_SNAPSHOT_V3";
    require(version_one || version_two || version_three, "invalid memory snapshot header");
    std::string token;
    MemoryConfig config;
    int immediate_deletion = 0;
    input >> token >> config.embedding_dim >> config.max_active_records >> config.minimum_confidence >> config.chain_seed;
    require(token == "CONFIG" && config.embedding_dim > 0U && config.embedding_dim <= 4096U && config.max_active_records > 0U &&
                config.max_active_records <= maximum_records && finite(config.minimum_confidence) && config.minimum_confidence >= 0.0 && config.minimum_confidence <= 1.0,
            "invalid memory snapshot configuration");
    if (version_two || version_three) {
        input >> config.novelty_threshold >> config.quarantine_threshold >> immediate_deletion;
        require(static_cast<bool>(input) && finite(config.novelty_threshold) && config.novelty_threshold >= 0.0 && finite(config.quarantine_threshold) &&
                    config.quarantine_threshold >= 0.0 && config.quarantine_threshold <= 1.0,
                "invalid V2/V3 memory snapshot thresholds");
        if (version_three) input >> std::quoted(config.embedding_backend);
    } else {
        input >> immediate_deletion;
    }
    require(static_cast<bool>(input) && !config.embedding_backend.empty(), "truncated memory snapshot configuration");
    config.immediate_deletion = immediate_deletion != 0;
    PersistentMemory memory(config);
    std::uint64_t legacy_previous_checksum = config.chain_seed;
    std::size_t event_count = 0U;
    while (input >> token) {
        require(token == "EVENT" && event_count < maximum_records, "invalid or excessive memory snapshot event count");
        MemoryEvent event;
        unsigned int event_type = 0U;
        input >> event.sequence >> event_type >> event.target_id >> event.previous_version >> std::quoted(event.reason) >>
            event.previous_event_checksum >> event.event_checksum;
        require(static_cast<bool>(input) && event_type <= event_type_value(MemoryEventType::Quarantine), "invalid memory event header");
        event.type = static_cast<MemoryEventType>(event_type);
        if (version_two || version_three) {
            input >> std::quoted(event.previous_event_digest) >> std::quoted(event.event_digest);
        }
        event.record = read_record(input, version_two || version_three);
        require(static_cast<bool>(input), "truncated memory snapshot event");
        require(event.record.embedding.size() <= 4096U && event.record.event_ids.size() <= maximum_records &&
                    event.record.causal_parents.size() <= maximum_records,
                "memory snapshot record exceeds vector budget");
        if (version_one) {
            const auto legacy_event_value = event.event_checksum;
            const auto legacy_previous_value = event.previous_event_checksum;
            require(event.record.checksum == legacy_content_checksum(event.record), "legacy memory record checksum mismatch");
            require(legacy_previous_value == legacy_previous_checksum && legacy_event_value == legacy_event_checksum(event),
                    "legacy memory event chain mismatch");
            event.record.checksum_digest = memory.encoder_.content_digest(event.record);
            event.record.checksum = memory.encoder_.content_checksum(event.record);
            event.previous_event_digest = event.sequence == 1U ? GovernedCorpus::content_sha256(std::to_string(config.chain_seed)) : memory.event_log_.back().event_digest;
            event.previous_event_checksum = digest_prefix(event.previous_event_digest);
            event.event_digest = memory.event_digest(event);
            event.event_checksum = digest_prefix(event.event_digest);
        }
        memory.apply_event(event, true);
        memory.event_log_.push_back(event);
        memory.next_sequence_ = event.sequence + 1U;
        legacy_previous_checksum = event.event_checksum;
        ++event_count;
    }
    require(input.eof(), "trailing invalid memory snapshot data");
    memory.verify_log();
    return memory;
}

void PersistentMemory::save_snapshot(const std::string& path) const {
    atomic_write_file(path, serialize_snapshot());
}

PersistentMemory PersistentMemory::load_snapshot(const std::string& path) {
    constexpr std::uintmax_t maximum_snapshot_bytes = 64U * 1024U * 1024U;
    const auto size = std::filesystem::file_size(path);
    require(size <= maximum_snapshot_bytes, "memory snapshot file exceeds byte budget");
    std::ifstream stream(path);
    require(static_cast<bool>(stream), "could not read memory snapshot");
    std::ostringstream content;
    content << stream.rdbuf();
    return deserialize_snapshot(content.str());
}

}  // namespace cct
