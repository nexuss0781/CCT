#include "cct/tokenizer.hpp"

#include "cct/corpus.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <unistd.h>

namespace cct {
namespace {

using FrequencyMap = std::map<std::string, std::uint64_t>;

std::string hex_encode(const std::string& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(value.size() * 2U);
    for (const char raw_byte : value) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        encoded.push_back(digits[byte >> 4U]);
        encoded.push_back(digits[byte & 0x0fU]);
    }
    return encoded;
}

std::string hex_decode(const std::string& value) {
    if (value.size() % 2U != 0U) throw TokenizerError("snapshot hex field has odd length");
    auto nibble = [](const char character) -> unsigned char {
        if (character >= '0' && character <= '9') return static_cast<unsigned char>(character - '0');
        if (character >= 'a' && character <= 'f') return static_cast<unsigned char>(character - 'a' + 10);
        if (character >= 'A' && character <= 'F') return static_cast<unsigned char>(character - 'A' + 10);
        throw TokenizerError("snapshot contains a non-hex character");
    };
    std::string decoded;
    decoded.reserve(value.size() / 2U);
    for (std::size_t index = 0; index < value.size(); index += 2U) {
        decoded.push_back(static_cast<char>((nibble(value[index]) << 4U) | nibble(value[index + 1U])));
    }
    return decoded;
}

std::uint64_t parse_unsigned(const std::string& value, const std::string& field) {
    if (value.empty()) throw TokenizerError("snapshot has empty numeric field: " + field);
    std::size_t consumed = 0;
    std::uint64_t parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (const std::exception&) {
        throw TokenizerError("snapshot has invalid numeric field: " + field);
    }
    if (consumed != value.size()) throw TokenizerError("snapshot has invalid numeric suffix: " + field);
    return parsed;
}

std::vector<std::string> split(const std::string& value, const char delimiter) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(value);
    while (std::getline(stream, part, delimiter)) parts.push_back(part);
    if (!value.empty() && value.back() == delimiter) parts.emplace_back();
    return parts;
}

bool is_word_byte(const unsigned char byte) {
    return byte >= 0x80U || byte == '_' || (byte >= '0' && byte <= '9') ||
           (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z');
}

bool is_reserved_string(const std::string& value) {
    static const std::unordered_set<std::string> reserved = {
        "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<TASK>", "<SCHEMA>", "<CITATION>",
        "<DOC_BOUNDARY>", "<SEQ_BOUNDARY>"};
    return reserved.find(value) != reserved.end();
}

void add_ngrams(FrequencyMap& frequencies, const std::string& content, const std::size_t maximum_piece_bytes,
                const bool word_runs_only) {
    if (content.size() < 2U) return;
    std::size_t position = 0;
    while (position < content.size()) {
        const std::size_t run_start = position;
        if (word_runs_only && !is_word_byte(static_cast<unsigned char>(content[position]))) {
            ++position;
            continue;
        }
        if (word_runs_only) {
            while (position < content.size() && is_word_byte(static_cast<unsigned char>(content[position]))) ++position;
        } else {
            position = content.size();
        }
        const std::size_t run_end = word_runs_only ? position : content.size();
        for (std::size_t start = run_start; start < run_end; ++start) {
            const std::size_t available = run_end - start;
            const std::size_t largest = std::min(maximum_piece_bytes, available);
            for (std::size_t length = 2U; length <= largest; ++length) {
                frequencies[content.substr(start, length)] += 1U;
            }
        }
        if (!word_runs_only) break;
    }
}

std::size_t current_resident_memory_bytes() {
    std::ifstream stream("/proc/self/statm");
    std::uint64_t total_pages = 0;
    std::uint64_t resident_pages = 0;
    if (!(stream >> total_pages >> resident_pages)) return 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0;
    return static_cast<std::size_t>(resident_pages * static_cast<std::uint64_t>(page_size));
}

std::string candidate_number(const TokenizerCandidate candidate) {
    return std::to_string(static_cast<unsigned int>(candidate));
}

TokenizerCandidate candidate_from_number(const std::string& value) {
    const auto parsed = parse_unsigned(value, "candidate");
    if (parsed > static_cast<std::uint64_t>(TokenizerCandidate::Hybrid)) {
        throw TokenizerError("snapshot contains unsupported tokenizer candidate");
    }
    return static_cast<TokenizerCandidate>(parsed);
}

TokenKind token_kind_from_number(const std::string& value) {
    const auto parsed = parse_unsigned(value, "token kind");
    if (parsed > static_cast<std::uint64_t>(TokenKind::Control)) throw TokenizerError("snapshot token kind is invalid");
    return static_cast<TokenKind>(parsed);
}

ControlKind control_from_number(const std::string& value) {
    const auto parsed = parse_unsigned(value, "control kind");
    if (parsed > static_cast<std::uint64_t>(ControlKind::SequenceBoundary)) {
        throw TokenizerError("snapshot control kind is invalid");
    }
    return static_cast<ControlKind>(parsed);
}

}  // namespace

Tokenizer::Tokenizer(TokenizerConfig config, std::vector<VocabularyEntry> vocabulary,
                     std::vector<std::size_t> piece_order, std::vector<std::string> training_ids)
    : config_(std::move(config)),
      vocabulary_(std::move(vocabulary)),
      piece_order_(std::move(piece_order)),
      normalized_training_ids_(std::move(training_ids)) {
    validate_config(config_);
    validate_vocabulary(config_, vocabulary_);
    std::vector<bool> seen(vocabulary_.size(), false);
    for (const auto index : piece_order_) {
        if (index >= vocabulary_.size() || seen[index] || vocabulary_[index].kind != TokenKind::Content) {
            throw TokenizerError("snapshot piece order is invalid");
        }
        seen[index] = true;
    }
    for (std::size_t index = 0; index < vocabulary_.size(); ++index) {
        if (vocabulary_[index].kind == TokenKind::Content && !seen[index]) {
            throw TokenizerError("snapshot omits a learned vocabulary entry from piece order");
        }
    }
}

void Tokenizer::validate_config(const TokenizerConfig& config) {
    if (config.tokenizer_version.empty()) throw TokenizerError("tokenizer version cannot be empty");
    if (config.snapshot_format_version != kSupportedSnapshotFormat) {
        throw TokenizerError("unsupported tokenizer snapshot format");
    }
    if (config.normalization_version != "preserve-bytes-v1") {
        throw TokenizerError("unsupported normalization version");
    }
    if (config.minimum_piece_frequency == 0U) throw TokenizerError("minimum piece frequency must be positive");
    if (config.maximum_piece_count == 0U) throw TokenizerError("maximum piece count must be positive");
    if (config.maximum_piece_bytes < 2U) throw TokenizerError("maximum piece bytes must be at least two");
}

std::vector<VocabularyEntry> Tokenizer::reserved_vocabulary() {
    return {
        {kPadId, "<PAD>", 0, TokenKind::Control, ControlKind::Pad},
        {kBosId, "<BOS>", 0, TokenKind::Control, ControlKind::Bos},
        {kEosId, "<EOS>", 0, TokenKind::Control, ControlKind::Eos},
        {kUnkId, "<UNK>", 0, TokenKind::Control, ControlKind::Unknown},
        {kTaskId, "<TASK>", 0, TokenKind::Control, ControlKind::Task},
        {kSchemaId, "<SCHEMA>", 0, TokenKind::Control, ControlKind::Schema},
        {kCitationId, "<CITATION>", 0, TokenKind::Control, ControlKind::Citation},
        {kDocumentBoundaryId, "<DOC_BOUNDARY>", 0, TokenKind::Control, ControlKind::DocumentBoundary},
        {kSequenceBoundaryId, "<SEQ_BOUNDARY>", 0, TokenKind::Control, ControlKind::SequenceBoundary}};
}

void Tokenizer::validate_vocabulary(const TokenizerConfig&, const std::vector<VocabularyEntry>& vocabulary) {
    const auto reserved = reserved_vocabulary();
    if (vocabulary.size() < reserved.size() + 256U) throw TokenizerError("vocabulary is missing reserved byte entries");
    TokenId previous = 0;
    for (std::size_t index = 0; index < vocabulary.size(); ++index) {
        const auto& entry = vocabulary[index];
        if (index != 0U && entry.id <= previous) throw TokenizerError("vocabulary IDs are not strictly increasing");
        previous = entry.id;
        if (index < reserved.size()) {
            const auto& expected = reserved[index];
            if (entry.id != expected.id || entry.bytes != expected.bytes || entry.kind != expected.kind ||
                entry.control != expected.control) {
                throw TokenizerError("reserved vocabulary entry changed");
            }
        } else if (entry.id >= kByteFirstId && entry.id < kByteFirstId + 256U) {
            const auto byte_value = entry.id - kByteFirstId;
            if (entry.bytes.size() != 1U || static_cast<unsigned char>(entry.bytes[0]) != byte_value ||
                entry.kind != TokenKind::ByteFallback || entry.control != ControlKind::None) {
                throw TokenizerError("byte fallback vocabulary entry is invalid");
            }
        } else if (entry.id < kLearnedFirstId || entry.kind != TokenKind::Content || entry.bytes.empty() ||
                   entry.bytes.size() > 1024U || entry.control != ControlKind::None || is_reserved_string(entry.bytes)) {
            throw TokenizerError("learned vocabulary entry is invalid");
        }
    }
    for (TokenId byte = kByteFirstId; byte < kByteFirstId + 256U; ++byte) {
        const auto index = static_cast<std::size_t>(byte - kByteFirstId) + reserved.size();
        if (index >= vocabulary.size() || vocabulary[index].id != byte) throw TokenizerError("byte IDs are incomplete");
    }
}

Tokenizer Tokenizer::build(const TokenizerConfig& config,
                           const std::vector<TokenizerTrainingRecord>& training_records) {
    validate_config(config);
    if (training_records.empty()) throw TokenizerError("cannot build a tokenizer from an empty corpus");
    std::vector<TokenizerTrainingRecord> records = training_records;
    std::sort(records.begin(), records.end(), [](const auto& left, const auto& right) {
        return left.record_id < right.record_id;
    });
    std::unordered_set<std::string> ids;
    for (const auto& record : records) {
        if (record.record_id.empty() || !ids.insert(record.record_id).second) {
            throw TokenizerError("training record IDs must be non-empty and unique");
        }
        if (!record.training_allowed || record.evaluator_only) {
            throw TokenizerError("tokenizer construction received a non-training record: " + record.record_id);
        }
    }

    auto vocabulary = reserved_vocabulary();
    for (unsigned int byte = 0; byte < 256U; ++byte) {
        vocabulary.push_back({kByteFirstId + byte, std::string(1, static_cast<char>(byte)), 0,
                              TokenKind::ByteFallback, ControlKind::None});
    }

    FrequencyMap frequencies;
    for (const auto& record : records) {
        const auto normalized = normalize(record.content, config.normalization_version);
        if (config.candidate == TokenizerCandidate::Subword) {
            add_ngrams(frequencies, normalized, config.maximum_piece_bytes, true);
        } else if (config.candidate == TokenizerCandidate::Hybrid) {
            add_ngrams(frequencies, normalized, config.maximum_piece_bytes, false);
        }
    }
    std::vector<std::pair<std::string, std::uint64_t>> candidates;
    for (const auto& [piece, frequency] : frequencies) {
        if (frequency >= config.minimum_piece_frequency && !is_reserved_string(piece)) {
            candidates.emplace_back(piece, frequency);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second > right.second;
        if (left.first.size() != right.first.size()) return left.first.size() > right.first.size();
        return left.first < right.first;
    });
    if (candidates.size() > config.maximum_piece_count) candidates.resize(config.maximum_piece_count);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        vocabulary.push_back({kLearnedFirstId + static_cast<TokenId>(index), candidates[index].first,
                              candidates[index].second, TokenKind::Content, ControlKind::None});
    }
    std::vector<std::size_t> piece_order;
    for (std::size_t index = reserved_vocabulary().size() + 256U; index < vocabulary.size(); ++index) {
        piece_order.push_back(index);
    }
    std::sort(piece_order.begin(), piece_order.end(), [&](const std::size_t left, const std::size_t right) {
        const auto& lhs = vocabulary[left];
        const auto& rhs = vocabulary[right];
        if (lhs.bytes.size() != rhs.bytes.size()) return lhs.bytes.size() > rhs.bytes.size();
        if (lhs.frequency != rhs.frequency) return lhs.frequency > rhs.frequency;
        return lhs.bytes < rhs.bytes;
    });
    std::vector<std::string> training_ids;
    for (const auto& record : records) {
        const auto normalized = normalize(record.content, config.normalization_version);
        training_ids.push_back(record.record_id + "\t" + GovernedCorpus::content_sha256(normalized));
    }
    return Tokenizer(config, std::move(vocabulary), std::move(piece_order), std::move(training_ids));
}

std::string Tokenizer::normalize(const std::string& content, const std::string& normalization_version) {
    if (normalization_version != "preserve-bytes-v1") throw TokenizerError("unsupported normalization version");
    return content;
}

std::string Tokenizer::candidate_name(const TokenizerCandidate candidate) {
    if (candidate == TokenizerCandidate::Byte) return "byte";
    if (candidate == TokenizerCandidate::Subword) return "subword";
    return "hybrid";
}

std::string Tokenizer::token_kind_name(const TokenKind kind) {
    if (kind == TokenKind::Content) return "content";
    if (kind == TokenKind::ByteFallback) return "byte_fallback";
    return "control";
}

std::string Tokenizer::control_name(const ControlKind control) {
    switch (control) {
        case ControlKind::None: return "none";
        case ControlKind::Pad: return "pad";
        case ControlKind::Bos: return "bos";
        case ControlKind::Eos: return "eos";
        case ControlKind::Unknown: return "unknown";
        case ControlKind::Task: return "task";
        case ControlKind::Schema: return "schema";
        case ControlKind::Citation: return "citation";
        case ControlKind::DocumentBoundary: return "document_boundary";
        case ControlKind::SequenceBoundary: return "sequence_boundary";
    }
    throw TokenizerError("unknown control kind");
}

ControlKind Tokenizer::control_for_id(const TokenId id) {
    if (id > kSequenceBoundaryId) return ControlKind::None;
    return static_cast<ControlKind>(id + 1U);
}

bool Tokenizer::is_reserved_id(const TokenId id) {
    return id <= kSequenceBoundaryId;
}

bool Tokenizer::is_control_id(const TokenId id) {
    return is_reserved_id(id);
}

const VocabularyEntry& Tokenizer::entry_for_id(const TokenId id) const {
    const auto iterator = std::lower_bound(vocabulary_.begin(), vocabulary_.end(), id,
                                           [](const VocabularyEntry& entry, const TokenId value) { return entry.id < value; });
    if (iterator == vocabulary_.end() || iterator->id != id) throw TokenizerError("invalid token ID");
    return *iterator;
}

void Tokenizer::validate_token_id(const TokenId id) const {
    static_cast<void>(entry_for_id(id));
}

std::vector<Token> Tokenizer::encode_content(const std::string& normalized, const std::string& record_id) const {
    std::vector<Token> tokens;
    std::size_t position = 0;
    while (position < normalized.size()) {
        const VocabularyEntry* matched = nullptr;
        if (config_.candidate != TokenizerCandidate::Byte) {
            for (const auto index : piece_order_) {
                const auto& entry = vocabulary_[index];
                if (entry.bytes.size() <= normalized.size() - position &&
                    normalized.compare(position, entry.bytes.size(), entry.bytes) == 0) {
                    matched = &entry;
                    break;
                }
            }
        }
        if (matched != nullptr) {
            tokens.push_back({matched->id, TokenKind::Content, ControlKind::None, record_id, position,
                              position + matched->bytes.size()});
            position += matched->bytes.size();
        } else {
            const auto byte = static_cast<unsigned char>(normalized[position]);
            tokens.push_back({kByteFirstId + byte, TokenKind::ByteFallback, ControlKind::None, record_id, position,
                              position + 1U});
            ++position;
        }
    }
    return tokens;
}

EncodedDocument Tokenizer::encode(const std::string& content, const std::string& record_id,
                                   const bool include_controls) const {
    if (record_id.empty()) throw TokenizerError("record ID cannot be empty");
    const auto normalized = normalize(content, config_.normalization_version);
    EncodedDocument document;
    document.record_id = record_id;
    document.source_bytes = normalized;
    document.tokenizer_version = config_.tokenizer_version;
    if (include_controls && config_.include_bos_eos) {
        document.tokens.push_back({kBosId, TokenKind::Control, ControlKind::Bos, record_id, 0, 0});
    }
    const auto content_tokens = encode_content(normalized, record_id);
    document.tokens.insert(document.tokens.end(), content_tokens.begin(), content_tokens.end());
    if (include_controls && config_.include_bos_eos) {
        document.tokens.push_back({kEosId, TokenKind::Control, ControlKind::Eos, record_id, normalized.size(), normalized.size()});
    }
    if (document.tokens.empty()) throw TokenizerError("encoding unexpectedly produced an empty document");
    return document;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids, const bool ignore_controls) const {
    std::string decoded;
    for (const auto id : ids) {
        const auto& entry = entry_for_id(id);
        if (entry.kind == TokenKind::Control) {
            if (!ignore_controls) decoded += entry.bytes;
        } else {
            decoded += entry.bytes;
        }
    }
    return decoded;
}

std::string Tokenizer::decode(const EncodedDocument& document, const bool ignore_controls) const {
    if (document.tokenizer_version != config_.tokenizer_version) {
        throw TokenizerError("encoded document tokenizer version mismatch");
    }
    std::vector<TokenId> ids;
    ids.reserve(document.tokens.size());
    for (const auto& token : document.tokens) {
        validate_token_id(token.id);
        ids.push_back(token.id);
    }
    return decode(ids, ignore_controls);
}

std::size_t Tokenizer::estimated_memory_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + vocabulary_.capacity() * sizeof(VocabularyEntry) +
                        piece_order_.capacity() * sizeof(std::size_t) + normalized_training_ids_.capacity() * sizeof(std::string);
    for (const auto& entry : vocabulary_) bytes += entry.bytes.capacity();
    for (const auto& id : normalized_training_ids_) bytes += id.capacity();
    return bytes;
}

ThroughputMeasurement Tokenizer::measure_throughput(const std::vector<std::string>& documents,
                                                    const std::size_t repetitions) const {
    if (documents.empty() || repetitions == 0U) throw TokenizerError("throughput measurement needs documents and repetitions");
    std::size_t source_bytes = 0;
    for (const auto& document : documents) source_bytes += document.size();
    volatile TokenId sink = 0;
    std::size_t token_count = 0;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (std::size_t index = 0; index < documents.size(); ++index) {
            const auto encoded = encode(documents[index], "throughput-" + std::to_string(index), false);
            token_count += encoded.tokens.size();
            if (!encoded.tokens.empty()) sink ^= encoded.tokens.front().id;
        }
    }
    const auto finished = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(finished - started).count();
    if (sink == std::numeric_limits<TokenId>::max()) throw TokenizerError("throughput sink invariant failed");
    ThroughputMeasurement measurement;
    measurement.source_bytes = source_bytes * repetitions;
    measurement.token_count = token_count;
    measurement.repetitions = repetitions;
    measurement.elapsed_seconds = elapsed;
    measurement.bytes_per_second = elapsed > 0.0 ? static_cast<double>(measurement.source_bytes) / elapsed : 0.0;
    measurement.tokens_per_second = elapsed > 0.0 ? static_cast<double>(measurement.token_count) / elapsed : 0.0;
    measurement.estimated_memory_bytes = estimated_memory_bytes();
    measurement.resident_memory_bytes = current_resident_memory_bytes();
    return measurement;
}

std::string Tokenizer::serialize_snapshot() const {
    validate_vocabulary(config_, vocabulary_);
    std::ostringstream output;
    output << "CCT-ASE-TOKENIZER-SNAPSHOT-V1\n";
    output << "format=" << config_.snapshot_format_version << "\n";
    output << "tokenizer_version=" << hex_encode(config_.tokenizer_version) << "\n";
    output << "normalization_version=" << hex_encode(config_.normalization_version) << "\n";
    output << "candidate=" << candidate_number(config_.candidate) << "\n";
    output << "seed=" << config_.seed << "\n";
    output << "minimum_piece_frequency=" << config_.minimum_piece_frequency << "\n";
    output << "maximum_piece_count=" << config_.maximum_piece_count << "\n";
    output << "maximum_piece_bytes=" << config_.maximum_piece_bytes << "\n";
    output << "include_bos_eos=" << (config_.include_bos_eos ? 1 : 0) << "\n";
    output << "training_count=" << normalized_training_ids_.size() << "\n";
    for (const auto& training_id : normalized_training_ids_) output << "training=" << hex_encode(training_id) << "\n";
    output << "vocabulary_count=" << vocabulary_.size() << "\n";
    for (const auto& entry : vocabulary_) {
        output << "vocab=" << entry.id << ',' << static_cast<unsigned int>(entry.kind) << ','
               << static_cast<unsigned int>(entry.control) << ',' << entry.frequency << ',' << hex_encode(entry.bytes) << "\n";
    }
    output << "piece_order_count=" << piece_order_.size() << "\n";
    output << "piece_order=";
    for (std::size_t index = 0; index < piece_order_.size(); ++index) {
        if (index != 0U) output << ',';
        output << piece_order_[index];
    }
    output << "\nend=1\n";
    return output.str();
}

std::string Tokenizer::snapshot_hash() const {
    return GovernedCorpus::content_sha256(serialize_snapshot());
}

Tokenizer Tokenizer::from_snapshot(const std::string& snapshot, const std::string& expected_hash) {
    constexpr std::size_t maximum_snapshot_bytes = 64U * 1024U * 1024U;
    constexpr std::size_t maximum_line_bytes = 16U * 1024U * 1024U;
    constexpr std::size_t maximum_training_records = 1'000'000U;
    constexpr std::size_t maximum_vocabulary_entries = 1'000'000U;
    constexpr std::size_t maximum_piece_order_entries = 1'000'000U;
    if (snapshot.empty()) throw TokenizerError("cannot load an empty tokenizer snapshot");
    if (snapshot.size() > maximum_snapshot_bytes) throw TokenizerError("tokenizer snapshot exceeds byte budget");
    if (!expected_hash.empty() && GovernedCorpus::content_sha256(snapshot) != expected_hash) {
        throw TokenizerError("tokenizer snapshot hash mismatch");
    }
    std::istringstream input(snapshot);
    std::string line;
    if (!std::getline(input, line) || line != "CCT-ASE-TOKENIZER-SNAPSHOT-V1") {
        throw TokenizerError("unsupported tokenizer snapshot header");
    }
    TokenizerConfig config;
    std::vector<std::string> training_ids;
    std::vector<VocabularyEntry> vocabulary;
    std::vector<std::size_t> piece_order;
    std::size_t expected_training = 0;
    std::size_t expected_vocabulary = 0;
    std::size_t expected_order = 0;
    bool saw_format = false;
    bool saw_end = false;
    while (std::getline(input, line)) {
        if (line.size() > maximum_line_bytes) throw TokenizerError("tokenizer snapshot line exceeds byte budget");
        if (line == "end=1") {
            saw_end = true;
            break;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) throw TokenizerError("snapshot line has no field separator");
        const auto key = line.substr(0, separator);
        const auto value = line.substr(separator + 1U);
        if (key == "format") {
            config.snapshot_format_version = static_cast<std::uint32_t>(parse_unsigned(value, key));
            saw_format = true;
        } else if (key == "tokenizer_version") {
            config.tokenizer_version = hex_decode(value);
        } else if (key == "normalization_version") {
            config.normalization_version = hex_decode(value);
        } else if (key == "candidate") {
            config.candidate = candidate_from_number(value);
        } else if (key == "seed") {
            config.seed = parse_unsigned(value, key);
        } else if (key == "minimum_piece_frequency") {
            config.minimum_piece_frequency = static_cast<std::size_t>(parse_unsigned(value, key));
            if (config.minimum_piece_frequency > maximum_vocabulary_entries) throw TokenizerError("tokenizer minimum frequency exceeds budget");
        } else if (key == "maximum_piece_count") {
            config.maximum_piece_count = static_cast<std::size_t>(parse_unsigned(value, key));
            if (config.maximum_piece_count > maximum_vocabulary_entries) throw TokenizerError("tokenizer piece count exceeds budget");
        } else if (key == "maximum_piece_bytes") {
            config.maximum_piece_bytes = static_cast<std::size_t>(parse_unsigned(value, key));
            if (config.maximum_piece_bytes > maximum_line_bytes) throw TokenizerError("tokenizer piece bytes exceed budget");
        } else if (key == "include_bos_eos") {
            const auto parsed = parse_unsigned(value, key);
            if (parsed > 1U) throw TokenizerError("snapshot include_bos_eos flag is invalid");
            config.include_bos_eos = parsed == 1U;
        } else if (key == "training_count") {
            expected_training = static_cast<std::size_t>(parse_unsigned(value, key));
            if (expected_training > maximum_training_records) throw TokenizerError("tokenizer training count exceeds budget");
        } else if (key == "training") {
            if (training_ids.size() >= maximum_training_records) throw TokenizerError("tokenizer training records exceed budget");
            training_ids.push_back(hex_decode(value));
        } else if (key == "vocabulary_count") {
            expected_vocabulary = static_cast<std::size_t>(parse_unsigned(value, key));
            if (expected_vocabulary > maximum_vocabulary_entries) throw TokenizerError("tokenizer vocabulary count exceeds budget");
        } else if (key == "vocab") {
            const auto fields = split(value, ',');
            if (fields.size() != 5U) throw TokenizerError("snapshot vocabulary row is malformed");
            if (vocabulary.size() >= maximum_vocabulary_entries) throw TokenizerError("tokenizer vocabulary entries exceed budget");
            const auto parsed_id = parse_unsigned(fields[0], "vocabulary ID");
            if (parsed_id > std::numeric_limits<TokenId>::max()) throw TokenizerError("tokenizer vocabulary ID exceeds budget");
            vocabulary.push_back({static_cast<TokenId>(parsed_id), hex_decode(fields[4]), parse_unsigned(fields[3], "frequency"),
                                 token_kind_from_number(fields[1]), control_from_number(fields[2])});
        } else if (key == "piece_order_count") {
            expected_order = static_cast<std::size_t>(parse_unsigned(value, key));
            if (expected_order > maximum_piece_order_entries) throw TokenizerError("tokenizer piece order count exceeds budget");
        } else if (key == "piece_order") {
            if (!value.empty()) {
                for (const auto& field : split(value, ',')) {
                    if (piece_order.size() >= maximum_piece_order_entries) throw TokenizerError("tokenizer piece order exceeds budget");
                    piece_order.push_back(static_cast<std::size_t>(parse_unsigned(field, "piece order")));
                }
            }
        } else {
            throw TokenizerError("snapshot contains an unknown field: " + key);
        }
    }
    if (!saw_format || !saw_end || expected_training != training_ids.size() ||
        expected_vocabulary != vocabulary.size() || expected_order != piece_order.size()) {
        throw TokenizerError("snapshot counts or terminator are inconsistent");
    }
    if (config.snapshot_format_version != kSupportedSnapshotFormat) throw TokenizerError("snapshot format is unsupported");
    return Tokenizer(std::move(config), std::move(vocabulary), std::move(piece_order), std::move(training_ids));
}

PackedBatch CausalBatchPacker::pack(const std::vector<EncodedDocument>& documents, const std::size_t maximum_tokens) {
    if (documents.empty()) throw TokenizerError("cannot pack an empty document list");
    PackedBatch batch;
    batch.tokenizer_version = documents.front().tokenizer_version;
    if (batch.tokenizer_version.empty()) throw TokenizerError("packed batch tokenizer version is empty");
    for (const auto& document : documents) {
        if (document.tokenizer_version != batch.tokenizer_version || document.tokens.empty()) {
            throw TokenizerError("packed documents have incompatible versions or empty tokens");
        }
        if (maximum_tokens != 0U && batch.input_ids.size() + document.tokens.size() > maximum_tokens) {
            throw TokenizerError("packed batch exceeds maximum token capacity");
        }
        const auto start = batch.input_ids.size();
        batch.sequence_starts.push_back(start);
        for (std::size_t index = 0; index < document.tokens.size(); ++index) {
            const auto& token = document.tokens[index];
            batch.input_ids.push_back(token.id);
            batch.target_ids.push_back(index + 1U < document.tokens.size() ? document.tokens[index + 1U].id : Tokenizer::kPadId);
            batch.loss_mask.push_back(index + 1U < document.tokens.size() ? 1U : 0U);
            batch.padding_mask.push_back(1U);
            batch.boundary_mask.push_back(token.kind == TokenKind::Control ? 1U : 0U);
            batch.record_ids.push_back(token.record_id);
            batch.source_starts.push_back(token.source_start);
            batch.source_ends.push_back(token.source_end);
            batch.control_categories.push_back(token.control);
        }
        batch.sequence_ends.push_back(batch.input_ids.size());
    }
    validate(batch);
    return batch;
}

PaddedBatch CausalBatchPacker::pad(const std::vector<EncodedDocument>& documents, const std::size_t maximum_length) {
    if (documents.empty()) throw TokenizerError("cannot pad an empty document list");
    PaddedBatch batch;
    batch.tokenizer_version = documents.front().tokenizer_version;
    if (batch.tokenizer_version.empty()) throw TokenizerError("padded batch tokenizer version is empty");
    std::size_t length = 0;
    for (const auto& document : documents) {
        if (document.tokenizer_version != batch.tokenizer_version || document.tokens.empty()) {
            throw TokenizerError("padded documents have incompatible versions or empty tokens");
        }
        length = std::max(length, document.tokens.size());
    }
    if (maximum_length != 0U) {
        if (length > maximum_length) throw TokenizerError("padded batch exceeds maximum sequence length");
        length = maximum_length;
    }
    for (const auto& document : documents) {
        const auto valid = document.tokens.size();
        batch.sequence_lengths.push_back(valid);
        batch.input_ids.emplace_back(length, Tokenizer::kPadId);
        batch.target_ids.emplace_back(length, Tokenizer::kPadId);
        batch.loss_mask.emplace_back(length, 0U);
        batch.padding_mask.emplace_back(length, 0U);
        batch.boundary_mask.emplace_back(length, 0U);
        batch.record_ids.emplace_back(length, std::string());
        batch.source_starts.emplace_back(length, 0U);
        batch.source_ends.emplace_back(length, 0U);
        batch.control_categories.emplace_back(length, ControlKind::Pad);
        for (std::size_t index = 0; index < valid; ++index) {
            const auto& token = document.tokens[index];
            batch.input_ids.back()[index] = token.id;
            batch.target_ids.back()[index] = index + 1U < valid ? document.tokens[index + 1U].id : Tokenizer::kPadId;
            batch.loss_mask.back()[index] = index + 1U < valid ? 1U : 0U;
            batch.padding_mask.back()[index] = 1U;
            batch.boundary_mask.back()[index] = token.kind == TokenKind::Control ? 1U : 0U;
            batch.record_ids.back()[index] = token.record_id;
            batch.source_starts.back()[index] = token.source_start;
            batch.source_ends.back()[index] = token.source_end;
            batch.control_categories.back()[index] = token.control;
        }
    }
    validate(batch);
    return batch;
}

void CausalBatchPacker::validate(const PackedBatch& batch) {
    const auto size = batch.input_ids.size();
    if (batch.tokenizer_version.empty() || batch.target_ids.size() != size || batch.loss_mask.size() != size ||
        batch.padding_mask.size() != size || batch.boundary_mask.size() != size || batch.record_ids.size() != size ||
        batch.source_starts.size() != size || batch.source_ends.size() != size || batch.control_categories.size() != size) {
        throw TokenizerError("packed batch vector sizes are inconsistent");
    }
    if (batch.sequence_starts.empty() || batch.sequence_starts.size() != batch.sequence_ends.size()) {
        throw TokenizerError("packed batch sequence boundaries are missing");
    }
    std::size_t cursor = 0;
    for (std::size_t sequence = 0; sequence < batch.sequence_starts.size(); ++sequence) {
        const auto start = batch.sequence_starts[sequence];
        const auto end = batch.sequence_ends[sequence];
        if (start != cursor || start >= end || end > size) throw TokenizerError("packed sequence boundary is invalid");
        for (std::size_t index = start; index < end; ++index) {
            if (batch.padding_mask[index] != 1U || batch.record_ids[index].empty()) throw TokenizerError("packed token is invalid");
            if (batch.source_starts[index] > batch.source_ends[index]) throw TokenizerError("packed source span is reversed");
            if (batch.control_categories[index] == ControlKind::None && batch.source_starts[index] == batch.source_ends[index]) {
                throw TokenizerError("packed content token has an empty source span");
            }
            const bool final = index + 1U == end;
            if (final) {
                if (batch.loss_mask[index] != 0U || batch.target_ids[index] != Tokenizer::kPadId) {
                    throw TokenizerError("packed sequence charges a boundary loss");
                }
            } else if (batch.loss_mask[index] != 1U || batch.target_ids[index] != batch.input_ids[index + 1U]) {
                throw TokenizerError("packed causal target crosses or misses a sequence position");
            }
        }
        cursor = end;
    }
    if (cursor != size) throw TokenizerError("packed batch has unbounded tokens");
}

void CausalBatchPacker::validate(const PaddedBatch& batch) {
    if (batch.tokenizer_version.empty() || batch.input_ids.empty() || batch.sequence_lengths.size() != batch.input_ids.size() ||
        batch.target_ids.size() != batch.input_ids.size() || batch.loss_mask.size() != batch.input_ids.size() ||
        batch.padding_mask.size() != batch.input_ids.size() || batch.boundary_mask.size() != batch.input_ids.size() ||
        batch.record_ids.size() != batch.input_ids.size() || batch.source_starts.size() != batch.input_ids.size() ||
        batch.source_ends.size() != batch.input_ids.size() || batch.control_categories.size() != batch.input_ids.size()) {
        throw TokenizerError("padded batch row counts are inconsistent");
    }
    const auto width = batch.input_ids.front().size();
    if (width == 0U) throw TokenizerError("padded batch width is zero");
    for (std::size_t row = 0; row < batch.input_ids.size(); ++row) {
        if (batch.input_ids[row].size() != width || batch.target_ids[row].size() != width || batch.loss_mask[row].size() != width ||
            batch.padding_mask[row].size() != width || batch.boundary_mask[row].size() != width || batch.record_ids[row].size() != width ||
            batch.source_starts[row].size() != width || batch.source_ends[row].size() != width ||
            batch.control_categories[row].size() != width || batch.sequence_lengths[row] == 0U || batch.sequence_lengths[row] > width) {
            throw TokenizerError("padded batch row width or sequence length is invalid");
        }
        const auto valid = batch.sequence_lengths[row];
        for (std::size_t index = 0; index < width; ++index) {
            const bool active = index < valid;
            if (active) {
                if (batch.padding_mask[row][index] != 1U || batch.record_ids[row][index].empty() ||
                    batch.source_starts[row][index] > batch.source_ends[row][index]) {
                    throw TokenizerError("padded active token is invalid");
                }
                if (batch.control_categories[row][index] == ControlKind::None &&
                    batch.source_starts[row][index] == batch.source_ends[row][index]) {
                    throw TokenizerError("padded content token has an empty source span");
                }
                if (index + 1U == valid) {
                    if (batch.loss_mask[row][index] != 0U || batch.target_ids[row][index] != Tokenizer::kPadId) {
                        throw TokenizerError("padded sequence charges a boundary loss");
                    }
                } else if (batch.loss_mask[row][index] != 1U || batch.target_ids[row][index] != batch.input_ids[row][index + 1U]) {
                    throw TokenizerError("padded causal target is invalid");
                }
            } else if (batch.input_ids[row][index] != Tokenizer::kPadId || batch.target_ids[row][index] != Tokenizer::kPadId ||
                       batch.loss_mask[row][index] != 0U || batch.padding_mask[row][index] != 0U ||
                       batch.control_categories[row][index] != ControlKind::Pad || !batch.record_ids[row][index].empty()) {
                throw TokenizerError("padded position is not masked as padding");
            }
        }
    }
}

}  // namespace cct
