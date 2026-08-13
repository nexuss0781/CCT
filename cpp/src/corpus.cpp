#include "cct/corpus.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cct {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}

std::string quote(const std::string& value) {
    std::ostringstream output;
    output << std::quoted(value);
    return output.str();
}

void write_strings(std::ostringstream& output, const std::vector<std::string>& values) {
    output << values.size();
    for (const auto& value : values) output << ' ' << quote(value);
}

std::vector<std::string> read_strings(std::istringstream& input) {
    std::size_t count = 0;
    input >> count;
    std::vector<std::string> values(count);
    for (auto& value : values) input >> std::quoted(value);
    require(static_cast<bool>(input), "invalid corpus string vector");
    return values;
}

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t amount) {
    return (value >> amount) | (value << (32U - amount));
}

std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (~x & z);
}

std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

std::uint32_t big_sigma_zero(std::uint32_t x) {
    return rotate_right(x, 2U) ^ rotate_right(x, 13U) ^ rotate_right(x, 22U);
}

std::uint32_t big_sigma_one(std::uint32_t x) {
    return rotate_right(x, 6U) ^ rotate_right(x, 11U) ^ rotate_right(x, 25U);
}

std::uint32_t small_sigma_zero(std::uint32_t x) {
    return rotate_right(x, 7U) ^ rotate_right(x, 18U) ^ (x >> 3U);
}

std::uint32_t small_sigma_one(std::uint32_t x) {
    return rotate_right(x, 17U) ^ rotate_right(x, 19U) ^ (x >> 10U);
}

}  // namespace

void GovernedCorpus::register_source(const SourcePolicy& policy) {
    require(!policy.source_id.empty() && !policy.source_uri.empty() && !policy.license_or_consent.empty() &&
                !policy.jurisdiction.empty() && !policy.collection_method.empty() && !policy.collection_timestamp.empty() &&
                !policy.privacy_classification.empty() && !policy.retention_policy.empty(),
            "source policy is incomplete");
    const auto duplicate = std::find_if(sources_.begin(), sources_.end(), [&](const auto& item) { return item.source_id == policy.source_id; });
    require(duplicate == sources_.end(), "duplicate corpus source id");
    require(!policy.training_allowed || policy.license_resolved, "training source requires resolved license or consent");
    require(!policy.evaluation_allowed || policy.license_resolved || policy.human_reviewed,
            "evaluation source requires resolved rights or human review");
    sources_.push_back(policy);
}

CorpusRecord GovernedCorpus::ingest(const std::string& record_id, const std::string& source_id, const std::string& content,
                                    CorpusSplit split, CorpusDataClass data_class, bool evaluator_only) {
    require(!record_id.empty() && !content.empty(), "corpus record id and content are required");
    const auto& policy = source(source_id);
    CorpusRecord record;
    record.record_id = record_id;
    record.source_id = source_id;
    record.source_uri = policy.source_uri;
    record.license_or_consent = policy.license_or_consent;
    record.jurisdiction = policy.jurisdiction;
    record.collection_method = policy.collection_method;
    record.collection_timestamp = policy.collection_timestamp;
    record.privacy_classification = policy.privacy_classification;
    record.retention_policy = policy.retention_policy;
    record.content = content;
    record.split = evaluator_only ? CorpusSplit::EvaluatorOnly : split;
    record.split_assignment = evaluator_only ? "evaluator_only" : std::to_string(static_cast<unsigned int>(split));
    record.data_class = data_class;
    record.evaluator_only = evaluator_only;
    record.delete_after = "review-required";
    record = process_record(std::move(record));
    records_.push_back(record);
    audit(records_.back(), "ingest", records_.back().reason_codes.empty() ? "accepted" : records_.back().reason_codes.front());
    return records_.back();
}

CorpusRecord GovernedCorpus::ingest_file(const std::string& record_id, const std::string& source_id, const std::string& path,
                                         CorpusSplit split, CorpusDataClass data_class, std::size_t max_bytes, bool evaluator_only) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not open corpus source file: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    std::string value = content.str();
    if (max_bytes != 0 && value.size() > max_bytes) value.resize(max_bytes);
    require(!value.empty(), "corpus source file is empty");
    return ingest(record_id, source_id, value, split, data_class, evaluator_only);
}

void GovernedCorpus::add_evaluator_canary(const std::string& record_id, const std::string& source_id, const std::string& content) {
    const auto record = ingest(record_id, source_id, content, CorpusSplit::EvaluatorOnly, CorpusDataClass::EvaluatorOnly, true);
    require(record.decision == CorpusDecision::Accept && record.evaluator_only, "evaluator canary was not accepted as evaluator-only");
}

bool GovernedCorpus::detect_contamination(const std::string& candidate_content) const {
    const auto normalized = normalize(candidate_content);
    const auto candidate_hash = sha256(normalized);
    for (const auto& record : records_) {
        if (!record.evaluator_only || record.deleted) continue;
        if (record.normalized_hash == candidate_hash || contains(normalized, record.normalized_content) ||
            contains(record.normalized_content, normalized)) return true;
    }
    return false;
}

bool GovernedCorpus::tombstone(const std::string& record_id, const std::string& reason) {
    const auto found = std::find_if(records_.begin(), records_.end(), [&](const auto& item) { return item.record_id == record_id && !item.deleted; });
    if (found == records_.end()) return false;
    found->deleted = true;
    found->decision = CorpusDecision::Reject;
    found->reason_codes.push_back("tombstone:" + reason);
    audit(*found, "delete", reason);
    return true;
}

std::vector<CorpusShard> GovernedCorpus::build_shards(std::size_t max_records_per_shard) const {
    require(max_records_per_shard > 0, "shard size must be positive");
    std::vector<CorpusRecord> active;
    for (const auto& record : records_) {
        if (record.decision == CorpusDecision::Accept && !record.deleted && !record.evaluator_only) active.push_back(record);
    }
    std::sort(active.begin(), active.end(), [](const auto& left, const auto& right) { return left.record_id < right.record_id; });
    std::vector<CorpusShard> shards;
    std::array<std::vector<CorpusRecord>, 3> by_split;
    for (const auto& record : active) {
        const auto split = static_cast<unsigned int>(record.split);
        require(split < by_split.size(), "accepted training record has evaluator-only split");
        by_split[split].push_back(record);
    }
    const std::array<CorpusSplit, 3> splits{CorpusSplit::Train, CorpusSplit::Validation, CorpusSplit::Test};
    for (std::size_t split_index = 0; split_index < splits.size(); ++split_index) {
        const auto& records = by_split[split_index];
        for (std::size_t offset = 0; offset < records.size(); offset += max_records_per_shard) {
            CorpusShard shard;
            shard.split = splits[split_index];
            shard.shard_id = "split-" + std::to_string(split_index) + "-" + std::to_string(shards.size());
            const auto end = std::min(offset + max_records_per_shard, records.size());
            std::ostringstream content;
            for (std::size_t index = offset; index < end; ++index) {
                shard.record_ids.push_back(records[index].record_id);
                content << records[index].normalized_content << '\n';
            }
            const auto serialized = content.str();
            shard.byte_count = serialized.size();
            shard.content_hash = sha256(serialized);
            shards.push_back(shard);
        }
    }
    return shards;
}

std::vector<CorpusRecord> GovernedCorpus::training_records() const {
    std::vector<CorpusRecord> result;
    for (const auto& record : records_) {
        if (record.decision == CorpusDecision::Accept && !record.deleted && !record.evaluator_only && record.split == CorpusSplit::Train) result.push_back(record);
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.record_id < right.record_id; });
    return result;
}

std::vector<CorpusRecord> GovernedCorpus::evaluation_records() const {
    std::vector<CorpusRecord> result;
    for (const auto& record : records_) {
        if (record.decision == CorpusDecision::Accept && !record.deleted &&
            (record.evaluator_only || record.split != CorpusSplit::Train)) result.push_back(record);
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.record_id < right.record_id; });
    return result;
}

std::vector<CorpusRecord> GovernedCorpus::all_records() const { return records_; }
std::string GovernedCorpus::content_sha256(const std::string& content) { return sha256(content); }
const std::vector<SourcePolicy>& GovernedCorpus::sources() const noexcept { return sources_; }
const std::vector<CorpusAuditEvent>& GovernedCorpus::audit() const noexcept { return audit_; }

std::string GovernedCorpus::normalize(const std::string& content) {
    std::string result;
    result.reserve(content.size());
    bool whitespace = false;
    for (const unsigned char character : content) {
        if (std::isspace(character) != 0) {
            whitespace = !result.empty();
            continue;
        }
        if (whitespace && !result.empty()) result.push_back(' ');
        whitespace = false;
        result.push_back(static_cast<char>(std::tolower(character)));
    }
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

std::string GovernedCorpus::sha256(const std::string& content) {
    static constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::vector<std::uint8_t> bytes(content.begin(), content.end());
    const auto bit_length = static_cast<std::uint64_t>(bytes.size()) * 8U;
    bytes.push_back(0x80U);
    while ((bytes.size() % 64U) != 56U) bytes.push_back(0U);
    for (int shift = 7; shift >= 0; --shift) bytes.push_back(static_cast<std::uint8_t>((bit_length >> (shift * 8)) & 0xffU));
    std::array<std::uint32_t, 8> hash{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const auto base = offset + index * 4U;
            schedule[index] = (static_cast<std::uint32_t>(bytes[base]) << 24U) |
                              (static_cast<std::uint32_t>(bytes[base + 1U]) << 16U) |
                              (static_cast<std::uint32_t>(bytes[base + 2U]) << 8U) |
                              static_cast<std::uint32_t>(bytes[base + 3U]);
        }
        for (std::size_t index = 16U; index < schedule.size(); ++index) {
            schedule[index] = small_sigma_one(schedule[index - 2U]) + schedule[index - 7U] +
                              small_sigma_zero(schedule[index - 15U]) + schedule[index - 16U];
        }
        auto a = hash[0];
        auto b = hash[1];
        auto c = hash[2];
        auto d = hash[3];
        auto e = hash[4];
        auto f = hash[5];
        auto g = hash[6];
        auto h = hash[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const auto t1 = h + big_sigma_one(e) + choose(e, f, g) + constants[index] + schedule[index];
            const auto t2 = big_sigma_zero(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto value : hash) output << std::setw(8) << value;
    return output.str();
}

bool GovernedCorpus::detect_pii(const std::string& content) {
    const auto lower = normalize(content);
    if (contains(lower, "social security") || contains(lower, "ssn") || contains(lower, "bank account") ||
        contains(lower, "password") || contains(lower, "secret key")) return true;
    const auto at = lower.find('@');
    if (at != std::string::npos && lower.find('.', at) != std::string::npos) return true;
    std::size_t consecutive_digits = 0;
    for (const auto character : lower) {
        if (std::isdigit(static_cast<unsigned char>(character)) != 0) {
            ++consecutive_digits;
            if (consecutive_digits >= 10U) return true;
        } else {
            consecutive_digits = 0;
        }
    }
    return false;
}

bool GovernedCorpus::looks_like_code(const std::string& content) {
    const auto lower = normalize(content);
    return contains(lower, "#include") || contains(lower, "std::") || contains(lower, "def ") ||
           contains(lower, "class ") || (contains(lower, "{") && contains(lower, "}"));
}

std::vector<std::string> GovernedCorpus::labels_for(const std::string& content, CorpusDataClass data_class) {
    std::vector<std::string> labels{"english"};
    if (looks_like_code(content) || data_class == CorpusDataClass::Code) labels.push_back("code");
    switch (data_class) {
        case CorpusDataClass::GeneralText: labels.push_back("general"); break;
        case CorpusDataClass::ReferenceText: labels.push_back("reference"); break;
        case CorpusDataClass::Code: labels.push_back("software"); break;
        case CorpusDataClass::Instruction: labels.push_back("instruction"); break;
        case CorpusDataClass::Preference: labels.push_back("preference"); break;
        case CorpusDataClass::Enterprise: labels.push_back("enterprise"); break;
        case CorpusDataClass::Safety: labels.push_back("safety"); break;
        case CorpusDataClass::EvaluatorOnly: labels.push_back("evaluator-only"); break;
    }
    return labels;
}

bool GovernedCorpus::has_exact_hash(const std::string& normalized_hash) const {
    return std::any_of(records_.begin(), records_.end(), [&](const auto& record) {
        return !record.deleted && record.decision == CorpusDecision::Accept && record.normalized_hash == normalized_hash;
    });
}

bool GovernedCorpus::has_near_duplicate(const std::string& normalized_content) const {
    std::istringstream candidate_stream(normalized_content);
    std::unordered_set<std::string> candidate_words;
    std::string word;
    while (candidate_stream >> word) candidate_words.insert(word);
    if (candidate_words.size() < 4U) return false;
    for (const auto& record : records_) {
        if (record.deleted || record.decision != CorpusDecision::Accept) continue;
        std::istringstream existing_stream(record.normalized_content);
        std::unordered_set<std::string> existing_words;
        while (existing_stream >> word) existing_words.insert(word);
        if (existing_words.size() < 4U) continue;
        std::size_t intersection = 0;
        for (const auto& item : candidate_words) {
            if (existing_words.find(item) != existing_words.end()) ++intersection;
        }
        const auto union_size = candidate_words.size() + existing_words.size() - intersection;
        if (union_size != 0U && static_cast<double>(intersection) / static_cast<double>(union_size) >= 0.8) return true;
    }
    return false;
}

CorpusRecord GovernedCorpus::process_record(CorpusRecord record) const {
    const auto& policy = source(record.source_id);
    record.content_hash = sha256(record.content);
    record.normalized_content = normalize(record.content);
    record.pii_detected = detect_pii(record.content);
    if (record.pii_detected) {
        record.redacted = true;
        record.normalized_content = "[redacted]";
        record.reason_codes.push_back("pii_detected");
        record.reason_codes.push_back("raw_content_purged");
        record.content.clear();
    }
    record.normalized_hash = sha256(record.normalized_content);
    record.transformation_chain = {"utf8-whitespace-v1", "lowercase-v1"};
    record.language_and_domain_labels = labels_for(record.content, record.data_class);
    record.quality_labels.push_back(record.normalized_content.size() >= 8U ? "length_adequate" : "length_short");
    record.quality_labels.push_back(record.normalized_content.find('\0') == std::string::npos ? "encoding_valid" : "encoding_invalid");
    const bool needs_training_rights = !record.evaluator_only && record.split == CorpusSplit::Train;
    const bool needs_evaluation_rights = record.evaluator_only || record.split != CorpusSplit::Train;
    if (!policy.license_resolved || (needs_training_rights && !policy.training_allowed) ||
        (needs_evaluation_rights && !policy.evaluation_allowed)) {
        record.decision = CorpusDecision::Quarantine;
        record.quarantined = true;
        record.reason_codes.push_back("unresolved_or_unapproved_rights");
        return record;
    }
    if (record.evaluator_only || record.split == CorpusSplit::EvaluatorOnly) {
        require(policy.evaluation_allowed || policy.human_reviewed, "evaluator-only source lacks evaluation approval");
        record.evaluator_only = true;
        record.split = CorpusSplit::EvaluatorOnly;
        record.split_assignment = "evaluator_only";
        record.decision = CorpusDecision::Accept;
        return record;
    }
    if (record.pii_detected) {
        record.decision = CorpusDecision::Quarantine;
        record.quarantined = true;
        return record;
    }
    if (record.normalized_content.size() < 8U) {
        record.decision = CorpusDecision::Reject;
        record.reason_codes.push_back("quality_length");
        return record;
    }
    if (has_exact_hash(record.normalized_hash)) {
        record.decision = CorpusDecision::Reject;
        record.reason_codes.push_back("exact_duplicate");
        return record;
    }
    if (has_near_duplicate(record.normalized_content)) {
        record.near_duplicate = true;
        record.decision = CorpusDecision::Reject;
        record.reason_codes.push_back("near_duplicate");
        return record;
    }
    record.decision = CorpusDecision::Accept;
    return record;
}

void GovernedCorpus::audit(const CorpusRecord& record, const std::string& event_type, const std::string& reason) {
    audit_.push_back({event_type, record.record_id, record.source_id, record.decision, reason, record.content_hash, record.normalized_hash});
}

const SourcePolicy& GovernedCorpus::source(const std::string& source_id) const {
    const auto found = std::find_if(sources_.begin(), sources_.end(), [&](const auto& item) { return item.source_id == source_id; });
    require(found != sources_.end(), "corpus source is not registered: " + source_id);
    return *found;
}

std::string GovernedCorpus::serialize() const {
    std::ostringstream output;
    output << "CCT_GOVERNED_CORPUS_V2\n";
    output << sources_.size() << '\n';
    for (const auto& item : sources_) {
        output << quote(item.source_id) << ' ' << quote(item.source_uri) << ' ' << quote(item.license_or_consent) << ' '
               << quote(item.jurisdiction) << ' ' << quote(item.collection_method) << ' ' << quote(item.collection_timestamp) << ' '
               << quote(item.privacy_classification) << ' ' << quote(item.retention_policy) << ' ' << item.license_resolved << ' '
               << item.training_allowed << ' ' << item.evaluation_allowed << ' ' << item.human_reviewed << '\n';
    }
    output << records_.size() << '\n';
    for (const auto& item : records_) {
        const auto persisted_content = item.pii_detected ? std::string{} : item.content;
        output << quote(item.record_id) << ' ' << quote(item.source_id) << ' ' << quote(item.source_uri) << ' '
               << quote(item.license_or_consent) << ' ' << quote(item.jurisdiction) << ' ' << quote(item.collection_method) << ' '
               << quote(item.collection_timestamp) << ' ' << quote(item.privacy_classification) << ' ' << quote(persisted_content) << ' '
               << quote(item.normalized_content) << ' ' << quote(item.content_hash) << ' ' << quote(item.normalized_hash) << ' ';
        write_strings(output, item.transformation_chain);
        output << ' ';
        write_strings(output, item.language_and_domain_labels);
        output << ' ';
        write_strings(output, item.quality_labels);
        output << ' ' << quote(item.split_assignment) << ' ' << quote(item.retention_policy) << ' ' << quote(item.delete_after) << ' '
               << item.opt_out << ' ' << item.evaluator_only << ' ' << item.pii_detected << ' ' << item.redacted << ' '
               << item.near_duplicate << ' ' << item.quarantined << ' ' << item.deleted << ' '
               << static_cast<unsigned int>(item.decision) << ' ' << static_cast<unsigned int>(item.split) << ' '
               << static_cast<unsigned int>(item.data_class) << ' ';
        write_strings(output, item.reason_codes);
        output << '\n';
    }
    output << audit_.size() << '\n';
    for (const auto& item : audit_) {
        output << quote(item.event_type) << ' ' << quote(item.record_id) << ' ' << quote(item.source_id) << ' '
               << static_cast<unsigned int>(item.decision) << ' ' << quote(item.reason) << ' ' << quote(item.content_hash) << ' '
               << quote(item.normalized_hash) << '\n';
    }
    return output.str();
}

GovernedCorpus GovernedCorpus::deserialize(const std::string& text) {
    std::istringstream input(text);
    std::string header;
    std::getline(input, header);
    const bool version_one = header == "CCT_GOVERNED_CORPUS_V1";
    require(version_one || header == "CCT_GOVERNED_CORPUS_V2", "invalid governed corpus header");
    GovernedCorpus corpus;
    std::size_t count = 0;
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        SourcePolicy item;
        input >> std::quoted(item.source_id) >> std::quoted(item.source_uri) >> std::quoted(item.license_or_consent) >>
            std::quoted(item.jurisdiction) >> std::quoted(item.collection_method) >> std::quoted(item.collection_timestamp) >>
            std::quoted(item.privacy_classification) >> std::quoted(item.retention_policy) >> item.license_resolved >>
            item.training_allowed >> item.evaluation_allowed >> item.human_reviewed;
        corpus.register_source(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        CorpusRecord item;
        unsigned int decision = 0;
        unsigned int split = 0;
        unsigned int data_class = 0;
        input >> std::quoted(item.record_id) >> std::quoted(item.source_id) >> std::quoted(item.source_uri) >>
            std::quoted(item.license_or_consent) >> std::quoted(item.jurisdiction) >> std::quoted(item.collection_method) >>
            std::quoted(item.collection_timestamp) >> std::quoted(item.privacy_classification) >> std::quoted(item.content) >>
            std::quoted(item.normalized_content) >> std::quoted(item.content_hash) >> std::quoted(item.normalized_hash);
        item.transformation_chain = read_strings(input);
        item.language_and_domain_labels = read_strings(input);
        item.quality_labels = read_strings(input);
        input >> std::quoted(item.split_assignment) >> std::quoted(item.retention_policy) >> std::quoted(item.delete_after) >>
            item.opt_out >> item.evaluator_only >> item.pii_detected >> item.redacted >> item.near_duplicate >> item.quarantined >>
            item.deleted >> decision >> split >> data_class;
        item.reason_codes = read_strings(input);
        if (item.pii_detected) {
            item.content.clear();
            item.normalized_content = "[redacted]";
            if (std::find(item.reason_codes.begin(), item.reason_codes.end(), "raw_content_purged") == item.reason_codes.end())
                item.reason_codes.push_back("raw_content_purged");
        }
        item.decision = static_cast<CorpusDecision>(decision);
        item.split = static_cast<CorpusSplit>(split);
        item.data_class = static_cast<CorpusDataClass>(data_class);
        corpus.records_.push_back(item);
    }
    input >> count;
    for (std::size_t index = 0; index < count; ++index) {
        CorpusAuditEvent item;
        unsigned int decision = 0;
        input >> std::quoted(item.event_type) >> std::quoted(item.record_id) >> std::quoted(item.source_id) >> decision >>
            std::quoted(item.reason) >> std::quoted(item.content_hash) >> std::quoted(item.normalized_hash);
        item.decision = static_cast<CorpusDecision>(decision);
        corpus.audit_.push_back(item);
    }
    require(static_cast<bool>(input), "invalid governed corpus serialization");
    return corpus;
}

bool GovernedCorpus::equivalent(const GovernedCorpus& other) const {
    return serialize() == other.serialize();
}

}  // namespace cct
