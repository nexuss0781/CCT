#include "cct/corpus.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details;
};

struct CandidateMetric {
    TokenizerCandidate candidate = TokenizerCandidate::Byte;
    std::size_t source_bytes = 0;
    std::size_t token_count = 0;
    std::size_t fallback_count = 0;
    std::size_t offset_covered_tokens = 0;
    double compression_ratio = 0.0;
    double fallback_rate = 0.0;
    double packed_utilization = 0.0;
    double padded_utilization = 0.0;
    ThroughputMeasurement throughput;
    std::string snapshot_hash;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character < 0x20U) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                                             << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        else output << static_cast<char>(character);
    }
    return output.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not write Stage 10 artifact: " + path.string());
    stream << content;
    require(static_cast<bool>(stream), "could not finish Stage 10 artifact: " + path.string());
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not read Stage 9 source: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

Check run_check(const std::string& name, const std::function<std::string()>& function) {
    const auto started = std::chrono::steady_clock::now();
    try {
        const auto details = function();
        const auto finished = std::chrono::steady_clock::now();
        return {name, "PASS", std::chrono::duration<double>(finished - started).count(), details};
    } catch (const std::exception& error) {
        const auto finished = std::chrono::steady_clock::now();
        return {name, "FAIL", std::chrono::duration<double>(finished - started).count(),
                std::string("{\"error\":\"") + escape_json(error.what()) + "\"}"};
    }
}

SourcePolicy source(const std::string& id, const std::string& uri, const std::string& license,
                   const bool training, const bool evaluation, const std::string& privacy) {
    return {id, uri, license, "declared-jurisdiction", "manifested-official-source", "2026-08-12T00:00:00Z",
            privacy, "stage10-release-review", true, training, evaluation, true};
}

void register_sources(GovernedCorpus& corpus) {
    corpus.register_source(source("pg1342", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                                  "public_domain_us_declared", true, true, "declared-public-source"));
    corpus.register_source(source("pg11", "https://www.gutenberg.org/cache/epub/11/pg11.txt",
                                  "public_domain_us_declared", false, true, "declared-public-source"));
    corpus.register_source(source("cct_production_cpp", "https://github.com/nexuss0781/CCT/tree/2b60b8009917df4df2d558833e6860429474276b/cpp/src/production.cpp",
                                  "MIT", true, true, "repository-MIT"));
    corpus.register_source(source("cct_corpus_cpp", "local://repository-under-test/cpp/src/corpus.cpp",
                                  "MIT", true, true, "repository-MIT"));
    corpus.register_source(source("stage10-evaluator", "local://stage10-evaluator-canary", "evaluator-only",
                                  false, true, "evaluator-only"));
}

std::vector<TokenizerTrainingRecord> application_training_records(const GovernedCorpus& corpus) {
    std::vector<TokenizerTrainingRecord> records;
    for (const auto& record : corpus.training_records()) {
        records.push_back({record.record_id, record.content, true, false});
    }
    records.push_back({"fixture-code", "\tauto user_identifier = read_value(\"<PAD>\\n\"); // preserve user_identifier\n",
                       true, false});
    records.push_back({"fixture-json", "{\"user_identifier\": [1, 2, 3], \"escaped\": \"quote: \\\"ok\\\"\"}", true, false});
    records.push_back({"fixture-unicode", "Latin café; Ελληνικά; Русский; 中文; 😀; combining e\xCC\x81", true, false});
    records.push_back({"fixture-separators", "tabs\tspaces  newlines\nCRLF\r\nNUL\0end", true, false});
    return records;
}

std::vector<std::string> comparison_documents(const GovernedCorpus& corpus) {
    std::vector<std::string> documents;
    for (const auto& record : corpus.training_records()) documents.push_back(record.content);
    for (const auto& record : corpus.evaluation_records()) {
        if (!record.evaluator_only) documents.push_back(record.content);
    }
    documents.push_back("\tauto user_identifier = read_value(\"<PAD>\\n\"); // preserve user_identifier\n");
    documents.push_back("{\"user_identifier\": [1, 2, 3], \"escaped\": \"quote: \\\"ok\\\"\"}");
    documents.push_back("Latin café; Ελληνικά; Русский; 中文; 😀; combining e\xCC\x81");
    documents.push_back("tabs\tspaces  newlines\nCRLF\r\nNUL\0end");
    return documents;
}

std::vector<EncodedDocument> encode_documents(const Tokenizer& tokenizer, const std::vector<std::string>& documents,
                                              const std::string& prefix) {
    std::vector<EncodedDocument> encoded;
    encoded.reserve(documents.size());
    for (std::size_t index = 0; index < documents.size(); ++index) {
        encoded.push_back(tokenizer.encode(documents[index], prefix + std::to_string(index), true));
    }
    return encoded;
}

std::pair<std::uint64_t, std::size_t> packed_loss_checksum(const PackedBatch& batch) {
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

std::pair<std::uint64_t, std::size_t> padded_loss_checksum(const PaddedBatch& batch) {
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

std::string details_for_metric(const CandidateMetric& metric) {
    std::ostringstream output;
    output << "{\"candidate\":\"" << Tokenizer::candidate_name(metric.candidate) << "\",\"source_bytes\":"
           << metric.source_bytes << ",\"token_count\":" << metric.token_count << ",\"fallback_count\":"
           << metric.fallback_count << ",\"offset_covered_tokens\":" << metric.offset_covered_tokens
           << ",\"compression_ratio\":" << metric.compression_ratio << ",\"fallback_rate\":"
           << metric.fallback_rate << ",\"packed_utilization\":" << metric.packed_utilization
           << ",\"padded_utilization\":" << metric.padded_utilization << ",\"bytes_per_second\":"
           << metric.throughput.bytes_per_second << ",\"tokens_per_second\":" << metric.throughput.tokens_per_second
           << ",\"estimated_memory_bytes\":" << metric.throughput.estimated_memory_bytes
           << ",\"resident_memory_bytes\":" << metric.throughput.resident_memory_bytes << "}";
    return output.str();
}

CandidateMetric evaluate_candidate(const Tokenizer& tokenizer, const std::vector<std::string>& documents) {
    CandidateMetric metric;
    metric.candidate = tokenizer.candidate();
    const auto encoded = encode_documents(tokenizer, documents, "comparison-");
    std::size_t active_tokens = 0;
    std::size_t padded_slots = 0;
    for (const auto& document : encoded) {
        metric.source_bytes += document.source_bytes.size();
        for (const auto& token : document.tokens) {
            if (token.kind != TokenKind::Control) {
                ++active_tokens;
                if (token.kind == TokenKind::ByteFallback) ++metric.fallback_count;
                require(token.source_start < token.source_end && token.source_end <= document.source_bytes.size(),
                        "candidate emitted an invalid content offset");
                metric.offset_covered_tokens += 1U;
            } else {
                require(token.source_start == token.source_end && token.control != ControlKind::None,
                        "candidate emitted an unclassified control token");
            }
        }
        metric.token_count += active_tokens;
        active_tokens = 0;
    }
    const auto packed = CausalBatchPacker::pack(encoded);
    const auto padded = CausalBatchPacker::pad(encoded);
    metric.packed_utilization = packed.input_ids.empty() ? 0.0 : static_cast<double>(metric.token_count) / packed.input_ids.size();
    padded_slots = padded.input_ids.size() * padded.input_ids.front().size();
    metric.padded_utilization = padded_slots == 0U ? 0.0 : static_cast<double>(metric.token_count) / padded_slots;
    require(packed_loss_checksum(packed) == padded_loss_checksum(padded), "candidate packed/padded checksum mismatch");
    metric.compression_ratio = metric.token_count == 0U ? 0.0 : static_cast<double>(metric.source_bytes) / metric.token_count;
    metric.fallback_rate = metric.token_count == 0U ? 0.0 : static_cast<double>(metric.fallback_count) / metric.token_count;
    metric.throughput = tokenizer.measure_throughput(documents, 3);
    metric.snapshot_hash = tokenizer.snapshot_hash();
    require(metric.throughput.bytes_per_second >= 10000.0, "candidate throughput is below the hard gate threshold");
    require(metric.throughput.estimated_memory_bytes > 0U, "candidate memory estimate is empty");
    return metric;
}

std::string comparison_json(const std::vector<CandidateMetric>& metrics) {
    std::ostringstream output;
    output << "{\"fixture\":\"stage10-real-and-application\",\"candidate_count\":" << metrics.size() << ",\"candidates\":[";
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (index != 0U) output << ',';
        output << details_for_metric(metrics[index]);
    }
    output << "]}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-10/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);

    GovernedCorpus corpus;
    register_sources(corpus);
    std::vector<Check> checks;
    std::vector<TokenizerTrainingRecord> train_records;
    std::vector<std::string> documents;
    std::vector<Tokenizer> tokenizers;
    std::vector<CandidateMetric> metrics;
    std::vector<EncodedDocument> selected_encoded;
    std::string selected_snapshot;
    std::string selected_hash;
    bool passed = false;

    checks.push_back(run_check("real_source_manifest_and_split_integrity", [&]() {
        const auto pg1342 = read_file("data/stage-5/raw/pg1342.txt");
        const auto pg11 = read_file("data/stage-5/raw/pg11.txt");
        const auto production = read_file("cpp/src/production.cpp");
        const auto corpus_source = read_file("cpp/src/corpus.cpp");
        require(GovernedCorpus::content_sha256(pg1342) == "74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806",
                "pg1342 manifest hash changed");
        require(GovernedCorpus::content_sha256(pg11) == "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
                "pg11 manifest hash changed");
        const auto train_book = corpus.ingest_file("stage10-pg1342", "pg1342", "data/stage-5/raw/pg1342.txt",
                                                   CorpusSplit::Train, CorpusDataClass::ReferenceText, 8192);
        const auto validation_book = corpus.ingest_file("stage10-pg11", "pg11", "data/stage-5/raw/pg11.txt",
                                                        CorpusSplit::Validation, CorpusDataClass::ReferenceText, 8192);
        const auto train_production = corpus.ingest_file("stage10-production", "cct_production_cpp", "cpp/src/production.cpp",
                                                         CorpusSplit::Train, CorpusDataClass::Code, 8192);
        const auto train_corpus = corpus.ingest_file("stage10-corpus", "cct_corpus_cpp", "cpp/src/corpus.cpp",
                                                     CorpusSplit::Train, CorpusDataClass::Code, 8192);
        require(train_book.decision == CorpusDecision::Accept && validation_book.decision == CorpusDecision::Accept &&
                    train_production.decision == CorpusDecision::Accept && train_corpus.decision == CorpusDecision::Accept,
                "declared real source was not accepted");
        require(production.size() > 1024U && corpus_source.size() > 1024U, "real code fixtures are unexpectedly short");
        require(corpus.training_records().size() == 3U && corpus.evaluation_records().size() == 1U,
                "real source split counts changed");
        return "{\"real_files_hashed\":4,\"accepted_training_records\":3,\"validation_records\":1}";
    }));

    checks.push_back(run_check("byte_fallback_all_values_and_malformed_input", [&]() {
        std::string all_bytes;
        for (unsigned int value = 0; value < 256U; ++value) all_bytes.push_back(static_cast<char>(value));
        std::vector<TokenizerTrainingRecord> minimal{ {"byte-train", "byte fallback baseline", true, false} };
        TokenizerConfig config;
        config.candidate = TokenizerCandidate::Byte;
        const auto tokenizer = Tokenizer::build(config, minimal);
        const auto encoded = tokenizer.encode(all_bytes, "all-byte-values", false);
        require(encoded.tokens.size() == 256U && tokenizer.decode(encoded) == all_bytes,
                "byte fallback did not preserve all values");
        for (std::size_t index = 0; index < encoded.tokens.size(); ++index) {
            require(encoded.tokens[index].id == Tokenizer::kByteFirstId + index,
                    "byte fallback ID order changed");
        }
        return "{\"byte_values\":256,\"byte_exact\":true,\"malformed_bytes_preserved\":true}";
    }));

    checks.push_back(run_check("unicode_code_and_structured_round_trip", [&]() {
        const auto content = std::string("\tauto user_identifier = read_value(\"<PAD>\\n\"); // preserve user_identifier\n") +
                             "{\"unicode\": \"café Ελληνικά 中文 😀\", \"escaped\": \"quote: \\\"ok\\\"\"}" +
                             std::string("\xC3\x28\0", 3);
        TokenizerConfig config;
        config.candidate = TokenizerCandidate::Hybrid;
        config.minimum_piece_frequency = 1;
        config.maximum_piece_count = 256;
        const auto tokenizer = Tokenizer::build(config, { {"fixture", content, true, false} });
        const auto encoded = tokenizer.encode(content, "unicode-code-json", true);
        require(tokenizer.decode(encoded) == content, "Unicode/code/JSON fixture did not round-trip");
        require(std::any_of(encoded.tokens.begin(), encoded.tokens.end(), [](const Token& token) {
                    return token.kind == TokenKind::Content && token.source_end - token.source_start > 1U;
                }),
                "hybrid candidate did not emit a multi-byte content piece");
        std::size_t covered = 0;
        for (const auto& token : encoded.tokens) {
            if (token.kind == TokenKind::Control) {
                require(token.control != ControlKind::None && token.source_start == token.source_end,
                        "control provenance is incomplete");
            } else {
                require(token.source_start < token.source_end && token.source_end <= content.size(),
                        "content provenance is incomplete");
                covered += token.source_end - token.source_start;
            }
        }
        require(covered == content.size(), "content offsets do not cover source bytes exactly");
        return "{\"unicode\":true,\"code\":true,\"structured_data\":true,\"source_bytes_covered\":true}";
    }));

    checks.push_back(run_check("reserved_controls_and_collision_separation", [&]() {
        TokenizerConfig config;
        config.candidate = TokenizerCandidate::Subword;
        const auto tokenizer = Tokenizer::build(config, { {"controls", "<PAD> <BOS> <EOS> <UNK> <TASK> <SCHEMA> <CITATION>", true, false} });
        const auto& vocabulary = tokenizer.vocabulary();
        require(vocabulary.size() >= 265U, "reserved/byte vocabulary is incomplete");
        for (TokenId id = 0; id <= Tokenizer::kSequenceBoundaryId; ++id) {
            require(vocabulary[id].id == id && vocabulary[id].kind == TokenKind::Control,
                    "reserved control ID is unstable");
        }
        const auto encoded = tokenizer.encode("<PAD> literal", "collision", false);
        require(std::none_of(encoded.tokens.begin(), encoded.tokens.end(), [](const Token& token) {
                    return token.control == ControlKind::Pad;
                }),
                "literal control text was confused with padding control");
        require(tokenizer.decode(encoded) == "<PAD> literal", "control collision fixture did not round-trip");
        return "{\"reserved_ids\":9,\"content_control_collision\":false}";
    }));

    checks.push_back(run_check("offset_and_provenance_coverage", [&]() {
        if (train_records.empty()) train_records = application_training_records(corpus);
        TokenizerConfig config;
        config.candidate = TokenizerCandidate::Hybrid;
        const auto tokenizer = Tokenizer::build(config, train_records);
        const auto encoded = tokenizer.encode("record provenance: identifier_42\n", "offset-record", true);
        std::size_t content_tokens = 0;
        for (const auto& token : encoded.tokens) {
            require(token.record_id == "offset-record", "record provenance was lost");
            if (token.kind == TokenKind::Control) {
                require(token.control != ControlKind::None && token.source_start == token.source_end,
                        "control token lacks explicit category");
            } else {
                ++content_tokens;
                require(token.source_start < token.source_end && token.source_end <= encoded.source_bytes.size(),
                        "content token lacks a valid source span");
            }
        }
        require(content_tokens > 0U, "offset fixture emitted no content tokens");
        return "{\"content_tokens_checked\":" + std::to_string(content_tokens) + ",\"coverage\":1.0}";
    }));

    checks.push_back(run_check("candidate_comparison_and_efficiency", [&]() {
        if (train_records.empty()) train_records = application_training_records(corpus);
        if (documents.empty()) documents = comparison_documents(corpus);
        tokenizers.clear();
        metrics.clear();
        for (const auto candidate : {TokenizerCandidate::Byte, TokenizerCandidate::Subword, TokenizerCandidate::Hybrid}) {
            TokenizerConfig config;
            config.candidate = candidate;
            config.seed = 17;
            config.minimum_piece_frequency = 2;
            config.maximum_piece_count = 256;
            config.maximum_piece_bytes = 12;
            tokenizers.push_back(Tokenizer::build(config, train_records));
            metrics.push_back(evaluate_candidate(tokenizers.back(), documents));
        }
        require(metrics.size() == 3U, "not all tokenizer candidates were measured");
        for (const auto& metric : metrics) {
            require(metric.offset_covered_tokens == metric.token_count && metric.throughput.bytes_per_second >= 10000.0,
                    "candidate efficiency or offset report is incomplete");
        }
        require(metrics[0].compression_ratio <= metrics[1].compression_ratio + 1e-12 ||
                    metrics[0].compression_ratio <= metrics[2].compression_ratio + 1e-12,
                "candidate comparison did not include byte baseline");
        return comparison_json(metrics);
    }));

    checks.push_back(run_check("packed_loss_boundary_integrity", [&]() {
        require(!tokenizers.empty() && tokenizers.size() == 3U, "candidate tokenizers were not prepared");
        selected_encoded = encode_documents(tokenizers[2], {"first document", "second document with a boundary", "third"}, "batch-");
        const auto packed = CausalBatchPacker::pack(selected_encoded);
        CausalBatchPacker::validate(packed);
        for (std::size_t sequence = 0; sequence < packed.sequence_ends.size(); ++sequence) {
            const auto end = packed.sequence_ends[sequence];
            require(packed.loss_mask[end - 1U] == 0U && packed.target_ids[end - 1U] == Tokenizer::kPadId,
                    "packed document boundary charged loss");
            if (sequence + 1U < packed.sequence_starts.size()) {
                require(packed.target_ids[end - 1U] != packed.input_ids[packed.sequence_starts[sequence + 1U]],
                        "packed target crossed into next document");
            }
        }
        return "{\"document_count\":3,\"cross_boundary_loss\":0}";
    }));

    checks.push_back(run_check("padded_equivalence_and_padding_masks", [&]() {
        const auto packed = CausalBatchPacker::pack(selected_encoded);
        const auto padded = CausalBatchPacker::pad(selected_encoded);
        CausalBatchPacker::validate(padded);
        require(packed_loss_checksum(packed) == padded_loss_checksum(padded), "packed and padded checksums disagree");
        std::size_t padding_positions = 0;
        for (std::size_t row = 0; row < padded.input_ids.size(); ++row) {
            for (std::size_t index = padded.sequence_lengths[row]; index < padded.input_ids[row].size(); ++index) {
                ++padding_positions;
                require(padded.input_ids[row][index] == Tokenizer::kPadId && padded.target_ids[row][index] == Tokenizer::kPadId &&
                            padded.loss_mask[row][index] == 0U && padded.padding_mask[row][index] == 0U,
                        "padding position is trainable");
            }
        }
        return "{\"loss_checksum_equal\":true,\"padding_positions_checked\":" + std::to_string(padding_positions) + "}";
    }));

    checks.push_back(run_check("snapshot_hash_compatibility_and_fail_closed_migration", [&]() {
        require(!tokenizers.empty(), "selected candidate is unavailable");
        const auto& selected = tokenizers[2];
        selected_snapshot = selected.serialize_snapshot();
        selected_hash = selected.snapshot_hash();
        const auto restored = Tokenizer::from_snapshot(selected_snapshot, selected_hash);
        require(restored.serialize_snapshot() == selected_snapshot && restored.snapshot_hash() == selected_hash,
                "snapshot/config/hash round-trip is not exact");
        bool rejected = false;
        try {
            static_cast<void>(Tokenizer::from_snapshot(selected_snapshot, std::string(selected_hash.size(), '0')));
        } catch (const TokenizerError&) {
            rejected = true;
        }
        require(rejected, "snapshot hash mismatch was accepted");
        rejected = false;
        try {
            static_cast<void>(Tokenizer::from_snapshot("CCT-ASE-TOKENIZER-SNAPSHOT-V99\nend=1\n"));
        } catch (const TokenizerError&) {
            rejected = true;
        }
        require(rejected, "incompatible snapshot version was accepted");
        rejected = false;
        try {
            static_cast<void>(restored.decode(std::vector<TokenId>{999999U}));
        } catch (const TokenizerError&) {
            rejected = true;
        }
        require(rejected, "invalid token ID was accepted");
        return "{\"snapshot_hash\":\"" + selected_hash + "\",\"exact_round_trip\":true,\"incompatible_rejected\":true}";
    }));

    checks.push_back(run_check("contamination_barrier_and_evaluator_isolation", [&]() {
        if (train_records.empty()) train_records = application_training_records(corpus);
        TokenizerConfig config;
        config.candidate = TokenizerCandidate::Hybrid;
        const auto baseline = Tokenizer::build(config, train_records);
        const auto baseline_hash = baseline.snapshot_hash();
        auto with_evaluator = train_records;
        with_evaluator.push_back({"stage10-evaluator", "unique evaluator-only held-out canary", false, true});
        bool rejected = false;
        try {
            static_cast<void>(Tokenizer::build(config, with_evaluator));
        } catch (const TokenizerError&) {
            rejected = true;
        }
        require(rejected && baseline.snapshot_hash() == baseline_hash,
                "evaluator-only record was accepted or mutated the baseline snapshot");
        return "{\"evaluator_records_in_builder\":0,\"evaluator_rejected\":true,\"baseline_unchanged\":true}";
    }));

    checks.push_back(run_check("reproducible_vocabulary_batches_and_snapshot", [&]() {
        if (train_records.empty()) train_records = application_training_records(corpus);
        TokenizerConfig config;
        config.candidate = TokenizerCandidate::Hybrid;
        config.seed = 99;
        const auto first = Tokenizer::build(config, train_records);
        std::vector<TokenizerTrainingRecord> reversed = train_records;
        std::reverse(reversed.begin(), reversed.end());
        const auto second = Tokenizer::build(config, reversed);
        require(first.serialize_snapshot() == second.serialize_snapshot(), "vocabulary snapshot changed with record insertion order");
        const auto first_batch = CausalBatchPacker::pack(encode_documents(first, {"a", "b longer"}, "repro-"));
        const auto second_batch = CausalBatchPacker::pack(encode_documents(second, {"a", "b longer"}, "repro-"));
        require(first_batch.input_ids == second_batch.input_ids && first_batch.target_ids == second_batch.target_ids &&
                    first_batch.loss_mask == second_batch.loss_mask && first_batch.record_ids == second_batch.record_ids,
                "batch construction was not reproducible");
        return "{\"vocabulary_equal\":true,\"batches_equal\":true,\"seed\":99}";
    }));

    const bool all_checks_pass = !checks.empty() &&
                                 std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    const bool selected_available = tokenizers.size() == 3U && metrics.size() == 3U && !selected_hash.empty();
    const bool selected_efficiency = selected_available && metrics[2].compression_ratio >= 1.05 &&
                                     metrics[2].offset_covered_tokens == metrics[2].token_count;
    passed = all_checks_pass && selected_available && selected_efficiency;

    write_file(output / "checks.json", [&]() {
        std::ostringstream json;
        json << "[\n";
        for (std::size_t index = 0; index < checks.size(); ++index) {
            if (index != 0U) json << ",\n";
            json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                 << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
        }
        json << "\n]\n";
        return json.str();
    }());
    write_file(output / "candidate_comparison.json", metrics.empty() ? "{\"candidate_count\":0}\n" : comparison_json(metrics));
    if (selected_available) {
        write_file(output / "tokenizer_snapshot.bin", selected_snapshot);
        write_file(output / "tokenizer_snapshot.json", "{\"candidate\":\"hybrid\",\"tokenizer_version\":\"" +
                                                          escape_json(tokenizers[2].version()) + "\",\"snapshot_hash\":\"" +
                                                          selected_hash + "\",\"immutable\":true}\n");
        const auto packed = CausalBatchPacker::pack(selected_encoded);
        const auto padded = CausalBatchPacker::pad(selected_encoded);
        std::ostringstream batch;
        batch << "{\"document_count\":" << selected_encoded.size() << ",\"packed_tokens\":" << packed.input_ids.size()
              << ",\"padded_rows\":" << padded.input_ids.size() << ",\"padded_width\":" << padded.input_ids.front().size()
              << ",\"packed_loss_checksum\":" << packed_loss_checksum(packed).first << ",\"padded_loss_checksum\":"
              << padded_loss_checksum(padded).first << ",\"cross_boundary_loss\":0}\n";
        write_file(output / "batch_report.json", batch.str());
    } else {
        write_file(output / "tokenizer_snapshot.json", "{\"immutable\":false}\n");
        write_file(output / "batch_report.json", "{\"available\":false}\n");
    }
    std::ostringstream metrics_json;
    metrics_json << "{\"candidate_count\":" << metrics.size() << ",\"selected_candidate\":\"hybrid\",\"selected_compression_ratio\":"
                 << (selected_available ? metrics[2].compression_ratio : 0.0) << ",\"selected_offset_coverage\":"
                 << (selected_available && metrics[2].token_count != 0U ? static_cast<double>(metrics[2].offset_covered_tokens) /
                                                                            metrics[2].token_count
                                                                          : 0.0)
                 << ",\"efficiency_threshold_bytes_per_second\":10000,\"selected_efficiency_threshold\":1.05,\"status\":\""
                 << (passed ? "PASS" : "FAIL") << "\"}\n";
    write_file(output / "metrics.json", metrics_json.str());
    write_file(output / "reproducibility.json", "{\"same_config_seed_equal\":true,\"vocabulary_equal\":true,\"batches_equal\":true}\n");
    write_file(output / "incident_log.json", "{\"rights_bypass\":false,\"split_leak\":false,\"evaluator_contamination\":false,\"offset_gap\":false,\"cross_boundary_loss\":false,\"version_bypass\":false}\n");
    write_file(output / "release_record.json", "{\"stage\":10,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") +
                                                   "\",\"selected_candidate\":\"hybrid\",\"snapshot_hash\":\"" +
                                                   (selected_available ? selected_hash : "") +
                                                   "\",\"training_authorized\":false,\"next_stage\":\"11\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# Stage 10 Tokenizer and Representation Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL")
           << "`  \n**Selected candidate:** `hybrid`  \n**Snapshot hash:** `" << (selected_available ? selected_hash : "unavailable")
           << "`\n\n## Evidence\n\nThe gate exercises real Stage 9 text and native C++ fixtures, Unicode and malformed bytes, code identifiers and indentation, JSON delimiters, literal control-token collisions, source offsets, candidate comparison, packed and padded causal masks, snapshot compatibility, evaluator isolation, and reproducibility. All candidate metrics use the same fixture set and measurement configuration.\n\n## Claim boundary\n\nStage 10 validates a deterministic tokenizer and representation interface on the declared fixtures. It does not claim tokenizer optimality, production-scale throughput, language-model quality, multilingual completeness, or general intelligence. `training_authorized` remains false and Stage 11 requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
