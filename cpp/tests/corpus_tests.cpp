#include "cct/corpus.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

SourcePolicy public_source(const std::string& id, const std::string& uri, const std::string& license, bool training, bool evaluation) {
    return {id, uri, license, "global", "official-download", "2026-08-12T00:00:00Z", "public", "release-review", true, training, evaluation, true};
}

SourcePolicy canary_source() {
    return {"stage9-canary", "local://evaluator-canary", "evaluator-only", "restricted", "test-fixture", "2026-08-12T00:00:00Z", "evaluator", "locked", true, false, true, true};
}

void register_real_sources(GovernedCorpus& corpus) {
    corpus.register_source(public_source("pg1342", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "public_domain_us", true, true));
    corpus.register_source(public_source("pg11", "https://www.gutenberg.org/cache/epub/11/pg11.txt", "public_domain_us", false, true));
    corpus.register_source(public_source("cct_cpp", "https://github.com/nexuss0781/CCT/tree/2b60b8009917df4df2d558833e6860429474276b/cpp", "MIT", true, true));
    corpus.register_source(canary_source());
}

void test_real_source_ingestion_and_labels() {
    GovernedCorpus corpus;
    register_real_sources(corpus);
    const auto book = corpus.ingest_file("pg1342-record", "pg1342", "data/stage-5/raw/pg1342.txt", CorpusSplit::Train, CorpusDataClass::ReferenceText, 8192);
    const auto validation = corpus.ingest_file("pg11-record", "pg11", "data/stage-5/raw/pg11.txt", CorpusSplit::Validation, CorpusDataClass::ReferenceText, 8192);
    const auto code = corpus.ingest_file("cct-code-record", "cct_cpp", "cpp/src/production.cpp", CorpusSplit::Train, CorpusDataClass::Code, 8192);
    require(book.decision == CorpusDecision::Accept && validation.decision == CorpusDecision::Accept && code.decision == CorpusDecision::Accept,
            "real source fixture decisions book=" + std::to_string(static_cast<unsigned int>(book.decision)) +
                " validation=" + std::to_string(static_cast<unsigned int>(validation.decision)) +
                " code=" + std::to_string(static_cast<unsigned int>(code.decision)) +
                " book_reason_count=" + std::to_string(book.reason_codes.size()) +
                " validation_reason_count=" + std::to_string(validation.reason_codes.size()) +
                " code_reason_count=" + std::to_string(code.reason_codes.size()));
    require(book.content_hash.size() == 64 && book.normalized_hash.size() == 64 && !book.transformation_chain.empty(),
            "real source hashes or transformation lineage are incomplete");
    require(book.language_and_domain_labels.size() >= 2 && code.language_and_domain_labels.size() >= 2 &&
                std::find(code.language_and_domain_labels.begin(), code.language_and_domain_labels.end(), "code") != code.language_and_domain_labels.end() &&
                std::find(code.language_and_domain_labels.begin(), code.language_and_domain_labels.end(), "code_candidate") != code.language_and_domain_labels.end(),
            "language/domain labels were not assigned");
    require(corpus.training_records().size() == 2 && corpus.evaluation_records().size() == 1, "split isolation changed real records");
}

void test_rights_privacy_and_quality_fail_closed() {
    GovernedCorpus corpus;
    register_real_sources(corpus);
    corpus.register_source({"unresolved", "https://unresolved.invalid/source", "unknown", "unknown", "unknown", "unknown", "high-risk", "quarantine", false, false, false, false});
    const auto unresolved = corpus.ingest("unresolved-record", "unresolved", "unresolved source text with enough length", CorpusSplit::Train, CorpusDataClass::GeneralText);
    require(unresolved.decision == CorpusDecision::Quarantine && unresolved.quarantined, "unresolved rights were not quarantined");
    const std::string pii_content = "customer email alice@example.com and bank account 1234567890";
    const auto pii = corpus.ingest("pii-record", "pg1342", pii_content, CorpusSplit::Train, CorpusDataClass::GeneralText);
    require(pii.decision == CorpusDecision::Quarantine && pii.pii_detected && pii.redacted && pii.content.empty() &&
                pii.normalized_content == "[redacted]" && pii.content_hash == GovernedCorpus::content_sha256(pii_content) &&
                std::find(pii.language_and_domain_labels.begin(), pii.language_and_domain_labels.end(), "pii_candidate") != pii.language_and_domain_labels.end() &&
                std::find(pii.reason_codes.begin(), pii.reason_codes.end(), "raw_content_purged") != pii.reason_codes.end(),
            "PII record retained raw content or lost its quarantine digest");
    const auto serialized = corpus.serialize();
    require(serialized.find("alice@example.com") == std::string::npos && serialized.find("CCT_GOVERNED_CORPUS_V2") == 0U,
            "serialized corpus retained raw PII or did not publish the privacy schema version");
    const auto restored = GovernedCorpus::deserialize(serialized);
    const auto restored_records = restored.all_records();
    const auto restored_pii = std::find_if(restored_records.begin(), restored_records.end(),
                                           [](const auto& record) { return record.record_id == "pii-record"; });
    require(restored_pii != restored_records.end(), "serialized PII quarantine record was lost");
    require(restored_pii->content.empty() && restored_pii->normalized_content == "[redacted]",
            "deserialized PII quarantine record restored raw content");
    auto legacy = serialized;
    legacy.replace(0U, std::string("CCT_GOVERNED_CORPUS_V2").size(), "CCT_GOVERNED_CORPUS_V1");
    const auto empty_redacted = std::string("\"\" \"[redacted]\"");
    const auto legacy_content_position = legacy.find(empty_redacted);
    require(legacy_content_position != std::string::npos, "could not construct legacy raw-PII fixture");
    legacy.replace(legacy_content_position, empty_redacted.size(), "\"" + pii_content + "\" \"[redacted]\"");
    const auto legacy_restored = GovernedCorpus::deserialize(legacy);
    const auto legacy_records = legacy_restored.all_records();
    const auto legacy_pii = std::find_if(legacy_records.begin(), legacy_records.end(),
                                         [](const auto& record) { return record.record_id == "pii-record"; });
    require(legacy_pii != legacy_records.end() && legacy_pii->content.empty(),
            "legacy corpus deserialization restored raw PII");
    const auto short_record = corpus.ingest("short-record", "pg1342", "tiny", CorpusSplit::Train, CorpusDataClass::GeneralText);
    require(short_record.decision == CorpusDecision::Reject &&
                std::find(short_record.reason_codes.begin(), short_record.reason_codes.end(), "quality_length") != short_record.reason_codes.end(),
            "short quality record was not rejected with a reason");
}

void test_exact_near_duplicate_and_contamination() {
    GovernedCorpus corpus;
    register_real_sources(corpus);
    const std::string base = "the governed corpus records source provenance and quality labels for every accepted training document";
    const auto first = corpus.ingest("base", "pg1342", base, CorpusSplit::Train, CorpusDataClass::GeneralText);
    const auto cross_split = corpus.ingest("cross-split", "pg1342", base, CorpusSplit::Validation, CorpusDataClass::GeneralText);
    require(cross_split.decision == CorpusDecision::Reject &&
                std::find(cross_split.reason_codes.begin(), cross_split.reason_codes.end(), "split_contamination") != cross_split.reason_codes.end(),
            "cross-split contamination was not rejected with an explicit reason");
    require(corpus.detect_contamination(base, CorpusSplit::Validation), "candidate split contamination query missed the training record");
    const auto exact = corpus.ingest("exact", "pg1342", "THE GOVERNED CORPUS records source provenance and quality labels for every accepted training document", CorpusSplit::Train, CorpusDataClass::GeneralText);
    const auto near = corpus.ingest("near", "pg1342", "the governed corpus records source provenance and quality labels for every accepted training document today", CorpusSplit::Train, CorpusDataClass::GeneralText);
    require(first.decision == CorpusDecision::Accept && exact.decision == CorpusDecision::Reject && near.decision == CorpusDecision::Reject,
            "exact or near duplicate was accepted");
    require(std::find(exact.reason_codes.begin(), exact.reason_codes.end(), "exact_duplicate") != exact.reason_codes.end(),
            "exact duplicate reason was not recorded");
    require(std::find(near.reason_codes.begin(), near.reason_codes.end(), "near_duplicate") != near.reason_codes.end(),
            "near duplicate reason was not recorded");
    corpus.add_evaluator_canary("canary", "stage9-canary", "evaluator-only held-out canary phrase must never enter training");
    require(corpus.detect_contamination("evaluator-only held-out canary phrase must never enter training"), "evaluator contamination was not detected");
    for (const auto& record : corpus.training_records()) require(record.record_id != "canary", "evaluator canary leaked into training records");
}

void test_shards_resume_and_deletion() {
    GovernedCorpus corpus;
    register_real_sources(corpus);
    corpus.ingest("one", "pg1342", "invoice reconciliation compares approved totals against the signed purchase order before review", CorpusSplit::Train, CorpusDataClass::GeneralText);
    corpus.ingest("two", "pg1342", "support escalation records outage severity, customer impact, and the responsible service owner", CorpusSplit::Train, CorpusDataClass::GeneralText);
    corpus.ingest("three", "cct_cpp", "static parser validates a manifest entry and returns a deterministic rejection reason", CorpusSplit::Train, CorpusDataClass::Code);
    const auto before = corpus.build_shards(2);
    require(before.size() == 2 && before.front().record_ids.size() == 2, "deterministic shard partition is incorrect");
    const auto restored = GovernedCorpus::deserialize(corpus.serialize());
    require(corpus.equivalent(restored), "corpus replay was not byte-equivalent");
    require(corpus.tombstone("two", "source opt-out"), "tombstone did not apply");
    const auto after = corpus.build_shards(2);
    for (const auto& shard : after) for (const auto& id : shard.record_ids) require(id != "two", "deleted record remained in rebuilt shard");
    require(corpus.audit().size() >= 4, "ingest and deletion lineage is incomplete");
}

void test_utf8_boundary_safe_file_truncation() {
    GovernedCorpus corpus;
    register_real_sources(corpus);
    const auto path = std::filesystem::temp_directory_path() / "cct-corpus-utf8-boundary.txt";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "could not create UTF-8 truncation fixture");
        output << "alpha beta \xE2\x82\xAC omega";
    }
    const auto record = corpus.ingest_file("utf8-boundary", "pg1342", path.string(), CorpusSplit::Train,
                                           CorpusDataClass::GeneralText, 12U);
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    require(record.content == "alpha beta ", "UTF-8 truncation did not stop before a partial code point");
    require(record.content.find(static_cast<char>(0xE2)) == std::string::npos, "truncated UTF-8 lead byte was retained");
}

void test_manifest_and_source_fail_closed() {
    GovernedCorpus corpus;
    bool rejected = false;
    try {
        corpus.ingest("unknown", "not-registered", "record content", CorpusSplit::Train, CorpusDataClass::GeneralText);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "unknown source was accepted");
    rejected = false;
    try {
        corpus.register_source({"bad", "", "", "", "", "", "", "", false, true, false, false});
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "malformed source policy was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"real_source_ingestion_and_labels", test_real_source_ingestion_and_labels},
        {"rights_privacy_and_quality_fail_closed", test_rights_privacy_and_quality_fail_closed},
        {"exact_near_duplicate_and_contamination", test_exact_near_duplicate_and_contamination},
        {"shards_resume_and_deletion", test_shards_resume_and_deletion},
        {"manifest_and_source_fail_closed", test_manifest_and_source_fail_closed},
        {"utf8_boundary_safe_file_truncation", test_utf8_boundary_safe_file_truncation},
    };
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
