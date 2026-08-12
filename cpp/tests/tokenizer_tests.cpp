#include "cct/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

TokenizerTrainingRecord training(const std::string& id, const std::string& content) {
    return {id, content, true, false};
}

Tokenizer make_byte() {
    TokenizerConfig config;
    config.candidate = TokenizerCandidate::Byte;
    config.include_bos_eos = true;
    return Tokenizer::build(config, {training("train-a", "deterministic byte fixture with punctuation")});
}

Tokenizer make_hybrid() {
    TokenizerConfig config;
    config.candidate = TokenizerCandidate::Hybrid;
    config.minimum_piece_frequency = 2;
    config.maximum_piece_count = 128;
    config.maximum_piece_bytes = 10;
    return Tokenizer::build(config,
                            {training("train-a", "preserve user_identifier and JSON delimiters in code"),
                             training("train-b", "preserve user_identifier and JSON delimiters in code"),
                             training("train-c", "preserve user_identifier and JSON delimiters in code")});
}

void test_all_bytes_round_trip_and_offsets() {
    const auto tokenizer = make_byte();
    std::string bytes;
    for (unsigned int value = 0; value < 256U; ++value) bytes.push_back(static_cast<char>(value));
    const auto encoded = tokenizer.encode(bytes, "all-bytes", false);
    require(encoded.tokens.size() == 256U, "byte tokenizer changed the number of byte tokens");
    require(tokenizer.decode(encoded) == bytes, "all byte values did not round-trip");
    for (std::size_t index = 0; index < encoded.tokens.size(); ++index) {
        const auto& token = encoded.tokens[index];
        require(token.kind == TokenKind::ByteFallback && token.source_start == index && token.source_end == index + 1U,
                "byte token offset or kind is incorrect");
        require(token.control == ControlKind::None, "byte token was assigned a control category");
    }
}

void test_unicode_invalid_bytes_and_preserve_bytes_normalization() {
    const auto tokenizer = make_hybrid();
    const std::string content = "Latin café; Ελληνικά; Русский; 中文; 😀; invalid:" +
                                std::string("\xC3\x28\xE2\x28\xA1", 6) + std::string("\0", 1);
    const auto encoded = tokenizer.encode(content, "unicode", false);
    require(Tokenizer::normalize(content, "preserve-bytes-v1") == content, "preserve-bytes normalization changed input");
    require(tokenizer.decode(encoded) == content, "Unicode or malformed bytes were corrupted");
    require(encoded.tokens.back().source_end == content.size(), "Unicode offsets do not reach the source end");
    for (const auto& token : encoded.tokens) {
        require(token.source_start < token.source_end && token.source_end <= content.size(),
                "Unicode content token has an invalid source span");
    }
}

void test_code_identifiers_literals_comments_and_control_collision() {
    const auto tokenizer = make_hybrid();
    const std::string content = "\tauto user_id = read_value(\"<PAD>\\n\");  // user_id\n"
                                "    if (user_id != other_id) { return; }\n";
    const auto encoded = tokenizer.encode(content, "code", true);
    require(tokenizer.decode(encoded) == content, "code fixture did not round-trip");
    require(encoded.tokens.front().control == ControlKind::Bos && encoded.tokens.back().control == ControlKind::Eos,
            "BOS/EOS controls are missing or misplaced");
    require(std::any_of(encoded.tokens.begin(), encoded.tokens.end(), [](const Token& token) {
                return token.kind == TokenKind::Content;
            }),
            "hybrid candidate never emitted a learned content token");
    require(std::any_of(encoded.tokens.begin(), encoded.tokens.end(), [](const Token& token) {
                return token.source_end - token.source_start > 1U;
            }),
            "hybrid candidate did not compress any repeated code span");
    for (const auto& token : encoded.tokens) {
        if (token.kind == TokenKind::Control) {
            require(token.source_start == token.source_end && token.control != ControlKind::None,
                    "control token lacks an explicit category or has a source span");
        } else {
            require(token.source_start < token.source_end, "content token lacks a non-empty source span");
        }
    }
}

void test_reserved_ids_snapshot_and_fail_closed_decode() {
    const auto tokenizer = make_hybrid();
    const auto& vocabulary = tokenizer.vocabulary();
    require(vocabulary.size() >= 265U, "reserved and byte vocabulary is incomplete");
    for (TokenId id = 0; id <= Tokenizer::kSequenceBoundaryId; ++id) {
        require(vocabulary[id].id == id && vocabulary[id].kind == TokenKind::Control,
                "reserved token ID changed");
    }
    for (TokenId id = Tokenizer::kByteFirstId; id < Tokenizer::kByteFirstId + 256U; ++id) {
        const auto& entry = vocabulary[9U + static_cast<std::size_t>(id - Tokenizer::kByteFirstId)];
        require(entry.id == id && entry.kind == TokenKind::ByteFallback && entry.bytes.size() == 1U,
                "byte fallback ID range is not stable");
    }
    const auto snapshot = tokenizer.serialize_snapshot();
    const auto hash = tokenizer.snapshot_hash();
    const auto restored = Tokenizer::from_snapshot(snapshot, hash);
    require(restored.serialize_snapshot() == snapshot && restored.snapshot_hash() == hash,
            "tokenizer snapshot did not round-trip byte-for-byte");
    require(restored.decode(restored.encode("snapshot exactness", "snapshot")) == "snapshot exactness",
            "restored tokenizer cannot encode/decode");
    bool rejected = false;
    try {
        static_cast<void>(restored.decode(std::vector<TokenId>{999999U}));
    } catch (const TokenizerError&) {
        rejected = true;
    }
    require(rejected, "invalid token ID was not rejected");
    rejected = false;
    try {
        static_cast<void>(Tokenizer::from_snapshot(snapshot, std::string(hash.size(), '0')));
    } catch (const TokenizerError&) {
        rejected = true;
    }
    require(rejected, "snapshot hash mismatch was not rejected");
    rejected = false;
    try {
        static_cast<void>(Tokenizer::from_snapshot(snapshot.substr(0, snapshot.size() - 5U), hash));
    } catch (const TokenizerError&) {
        rejected = true;
    }
    require(rejected, "truncated snapshot was not rejected");
}

void test_training_boundary_contamination_and_reproducibility() {
    TokenizerConfig config;
    config.candidate = TokenizerCandidate::Subword;
    config.seed = 42;
    const std::vector<TokenizerTrainingRecord> records{
        training("z-record", "alpha beta gamma alpha beta"),
        training("a-record", "alpha beta gamma alpha beta")};
    const auto first = Tokenizer::build(config, records);
    const auto second = Tokenizer::build(config, {records[0], records[1]});
    require(first.serialize_snapshot() == second.serialize_snapshot(), "same corpus/config did not reproduce snapshot bytes");
    bool rejected = false;
    try {
        static_cast<void>(Tokenizer::build(config, {training("eval", "held out canary"), {"canary", "held out", false, true}}));
    } catch (const TokenizerError&) {
        rejected = true;
    }
    require(rejected, "evaluator-only content entered vocabulary construction");
}

std::pair<std::uint64_t, std::size_t> loss_checksum(const PackedBatch& batch) {
    std::uint64_t sum = 0;
    std::size_t count = 0;
    for (std::size_t index = 0; index < batch.target_ids.size(); ++index) {
        if (batch.loss_mask[index] != 0U) {
            sum += batch.target_ids[index];
            ++count;
        }
    }
    return {sum, count};
}

std::pair<std::uint64_t, std::size_t> loss_checksum(const PaddedBatch& batch) {
    std::uint64_t sum = 0;
    std::size_t count = 0;
    for (std::size_t row = 0; row < batch.target_ids.size(); ++row) {
        for (std::size_t index = 0; index < batch.target_ids[row].size(); ++index) {
            if (batch.loss_mask[row][index] != 0U) {
                sum += batch.target_ids[row][index];
                ++count;
            }
        }
    }
    return {sum, count};
}

void test_packed_padded_masks_and_boundaries() {
    const auto tokenizer = make_hybrid();
    const std::vector<EncodedDocument> documents{
        tokenizer.encode("first document", "doc-a"), tokenizer.encode("second document with more bytes", "doc-b"),
        tokenizer.encode("third", "doc-c")};
    const auto packed = CausalBatchPacker::pack(documents);
    const auto padded = CausalBatchPacker::pad(documents);
    CausalBatchPacker::validate(packed);
    CausalBatchPacker::validate(padded);
    require(loss_checksum(packed) == loss_checksum(padded), "packed and padded loss checksums disagree");
    require(packed.sequence_starts.size() == documents.size() && packed.sequence_ends.size() == documents.size(),
            "packed sequence boundaries are incomplete");
    for (std::size_t sequence = 0; sequence < packed.sequence_starts.size(); ++sequence) {
        const auto end = packed.sequence_ends[sequence];
        require(packed.loss_mask[end - 1U] == 0U && packed.target_ids[end - 1U] == Tokenizer::kPadId,
                "packed document boundary carries loss");
    }
    for (std::size_t row = 0; row < padded.input_ids.size(); ++row) {
        for (std::size_t index = padded.sequence_lengths[row]; index < padded.input_ids[row].size(); ++index) {
            require(padded.padding_mask[row][index] == 0U && padded.loss_mask[row][index] == 0U &&
                        padded.input_ids[row][index] == Tokenizer::kPadId,
                    "padded position is not fully masked");
        }
    }
    bool rejected = false;
    try {
        static_cast<void>(CausalBatchPacker::pack(documents, 2U));
    } catch (const TokenizerError&) {
        rejected = true;
    }
    require(rejected, "over-capacity packed batch was accepted");
}

void test_throughput_report() {
    const auto tokenizer = make_hybrid();
    const auto measurement = tokenizer.measure_throughput({"repeated production tokenization fixture with identifiers and delimiters", "second fixture"}, 4);
    require(measurement.source_bytes > 0U && measurement.token_count > 0U && measurement.repetitions == 4U,
            "throughput report has empty counts");
    require(measurement.elapsed_seconds > 0.0 && measurement.bytes_per_second > 0.0 && measurement.tokens_per_second > 0.0,
            "throughput report has non-positive rates");
    require(measurement.estimated_memory_bytes > 0U, "throughput report lacks estimated memory");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"all_bytes_round_trip_and_offsets", test_all_bytes_round_trip_and_offsets},
        {"unicode_invalid_bytes_and_preserve_bytes_normalization", test_unicode_invalid_bytes_and_preserve_bytes_normalization},
        {"code_identifiers_literals_comments_and_control_collision", test_code_identifiers_literals_comments_and_control_collision},
        {"reserved_ids_snapshot_and_fail_closed_decode", test_reserved_ids_snapshot_and_fail_closed_decode},
        {"training_boundary_contamination_and_reproducibility", test_training_boundary_contamination_and_reproducibility},
        {"packed_padded_masks_and_boundaries", test_packed_padded_masks_and_boundaries},
        {"throughput_report", test_throughput_report}};
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
            ++passed;
        } catch (const std::exception& error) {
            std::cout << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << "SUMMARY " << passed << "/" << tests.size() << " passed\n";
    return passed == tests.size() ? 0 : 1;
}
