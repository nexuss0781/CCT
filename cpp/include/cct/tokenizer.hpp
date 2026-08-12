#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class TokenizerCandidate : std::uint8_t {
    Byte = 0,
    Subword = 1,
    Hybrid = 2
};

enum class TokenKind : std::uint8_t {
    Content = 0,
    ByteFallback = 1,
    Control = 2
};

enum class ControlKind : std::uint8_t {
    None = 0,
    Pad = 1,
    Bos = 2,
    Eos = 3,
    Unknown = 4,
    Task = 5,
    Schema = 6,
    Citation = 7,
    DocumentBoundary = 8,
    SequenceBoundary = 9
};

using TokenId = std::uint32_t;

class TokenizerError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct TokenizerConfig {
    std::string tokenizer_version = "cct-ase-tokenizer-v1";
    std::uint32_t snapshot_format_version = 1;
    std::string normalization_version = "preserve-bytes-v1";
    TokenizerCandidate candidate = TokenizerCandidate::Byte;
    std::uint64_t seed = 0;
    std::size_t minimum_piece_frequency = 2;
    std::size_t maximum_piece_count = 256;
    std::size_t maximum_piece_bytes = 12;
    bool include_bos_eos = true;
};

struct TokenizerTrainingRecord {
    std::string record_id;
    std::string content;
    bool training_allowed = false;
    bool evaluator_only = false;
};

struct VocabularyEntry {
    TokenId id = 0;
    std::string bytes;
    std::uint64_t frequency = 0;
    TokenKind kind = TokenKind::Content;
    ControlKind control = ControlKind::None;
};

struct Token {
    TokenId id = 0;
    TokenKind kind = TokenKind::Content;
    ControlKind control = ControlKind::None;
    std::string record_id;
    std::size_t source_start = 0;
    std::size_t source_end = 0;
};

struct EncodedDocument {
    std::string record_id;
    std::string source_bytes;
    std::string tokenizer_version;
    std::vector<Token> tokens;
};

struct ThroughputMeasurement {
    std::size_t source_bytes = 0;
    std::size_t token_count = 0;
    std::size_t repetitions = 0;
    double elapsed_seconds = 0.0;
    double bytes_per_second = 0.0;
    double tokens_per_second = 0.0;
    std::size_t estimated_memory_bytes = 0;
    std::size_t resident_memory_bytes = 0;
};

struct PackedBatch {
    std::string tokenizer_version;
    std::vector<TokenId> input_ids;
    std::vector<TokenId> target_ids;
    std::vector<std::uint8_t> loss_mask;
    std::vector<std::uint8_t> padding_mask;
    std::vector<std::uint8_t> boundary_mask;
    std::vector<std::string> record_ids;
    std::vector<std::size_t> source_starts;
    std::vector<std::size_t> source_ends;
    std::vector<ControlKind> control_categories;
    std::vector<std::size_t> sequence_starts;
    std::vector<std::size_t> sequence_ends;
};

struct PaddedBatch {
    std::string tokenizer_version;
    std::vector<std::vector<TokenId>> input_ids;
    std::vector<std::vector<TokenId>> target_ids;
    std::vector<std::vector<std::uint8_t>> loss_mask;
    std::vector<std::vector<std::uint8_t>> padding_mask;
    std::vector<std::vector<std::uint8_t>> boundary_mask;
    std::vector<std::vector<std::string>> record_ids;
    std::vector<std::vector<std::size_t>> source_starts;
    std::vector<std::vector<std::size_t>> source_ends;
    std::vector<std::vector<ControlKind>> control_categories;
    std::vector<std::size_t> sequence_lengths;
};

class Tokenizer {
public:
    static constexpr TokenId kPadId = 0;
    static constexpr TokenId kBosId = 1;
    static constexpr TokenId kEosId = 2;
    static constexpr TokenId kUnkId = 3;
    static constexpr TokenId kTaskId = 4;
    static constexpr TokenId kSchemaId = 5;
    static constexpr TokenId kCitationId = 6;
    static constexpr TokenId kDocumentBoundaryId = 7;
    static constexpr TokenId kSequenceBoundaryId = 8;
    static constexpr TokenId kByteFirstId = 256;
    static constexpr TokenId kLearnedFirstId = 512;
    static constexpr std::uint32_t kSupportedSnapshotFormat = 1;

    static Tokenizer build(const TokenizerConfig& config,
                           const std::vector<TokenizerTrainingRecord>& training_records);
    static Tokenizer from_snapshot(const std::string& snapshot,
                                   const std::string& expected_hash = {});

    const TokenizerConfig& config() const noexcept { return config_; }
    TokenizerCandidate candidate() const noexcept { return config_.candidate; }
    const std::vector<VocabularyEntry>& vocabulary() const noexcept { return vocabulary_; }
    std::string version() const { return config_.tokenizer_version; }
    std::string snapshot_hash() const;
    std::string serialize_snapshot() const;

    EncodedDocument encode(const std::string& content, const std::string& record_id = "anonymous",
                           bool include_controls = true) const;
    std::string decode(const std::vector<TokenId>& ids, bool ignore_controls = true) const;
    std::string decode(const EncodedDocument& document, bool ignore_controls = true) const;

    ThroughputMeasurement measure_throughput(const std::vector<std::string>& documents,
                                             std::size_t repetitions = 3) const;
    std::size_t estimated_memory_bytes() const noexcept;

    static std::string normalize(const std::string& content, const std::string& normalization_version);
    static std::string candidate_name(TokenizerCandidate candidate);
    static std::string token_kind_name(TokenKind kind);
    static std::string control_name(ControlKind control);

private:
    TokenizerConfig config_;
    std::vector<VocabularyEntry> vocabulary_;
    std::vector<std::size_t> piece_order_;
    std::vector<std::string> normalized_training_ids_;

    Tokenizer(TokenizerConfig config, std::vector<VocabularyEntry> vocabulary,
              std::vector<std::size_t> piece_order_, std::vector<std::string> training_ids);

    static std::vector<VocabularyEntry> reserved_vocabulary();
    static ControlKind control_for_id(TokenId id);
    static bool is_reserved_id(TokenId id);
    static bool is_control_id(TokenId id);
    static void validate_config(const TokenizerConfig& config);
    static void validate_vocabulary(const TokenizerConfig& config,
                                   const std::vector<VocabularyEntry>& vocabulary);
    void validate_token_id(TokenId id) const;
    std::vector<Token> encode_content(const std::string& normalized, const std::string& record_id) const;
    const VocabularyEntry& entry_for_id(TokenId id) const;
};

class CausalBatchPacker {
public:
    static PackedBatch pack(const std::vector<EncodedDocument>& documents, std::size_t maximum_tokens = 0);
    static PaddedBatch pad(const std::vector<EncodedDocument>& documents, std::size_t maximum_length = 0);
    static void validate(const PackedBatch& batch);
    static void validate(const PaddedBatch& batch);
};

}  // namespace cct
