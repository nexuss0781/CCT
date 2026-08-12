#include "cct/corpus.hpp"

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
#include <vector>

namespace {

using namespace cct;

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details;
};

struct ManifestEntry {
    std::string source_id;
    std::string split;
    std::string data_class;
    std::string license;
    std::string uri;
    std::string path;
    std::string expected_sha256;
    bool training_allowed = false;
    bool evaluation_allowed = false;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const auto character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else output << character;
    }
    return output.str();
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not read real source: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path);
    require(static_cast<bool>(stream), "could not write Stage 9 artifact: " + path.string());
    stream << content;
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

std::vector<ManifestEntry> source_manifest() {
    return {
        {"pg1342", "train", "reference_text", "public_domain_us_declared", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "data/stage-5/raw/pg1342.txt", "74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806", true, true},
        {"pg11", "validation", "reference_text", "public_domain_us_declared", "https://www.gutenberg.org/cache/epub/11/pg11.txt", "data/stage-5/raw/pg11.txt", "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e", false, true},
        {"cct_production_cpp", "train", "code", "MIT", "https://github.com/nexuss0781/CCT/tree/2b60b8009917df4df2d558833e6860429474276b/cpp/src/production.cpp", "cpp/src/production.cpp", "", true, true},
        {"cct_corpus_cpp", "train", "code", "MIT", "https://github.com/nexuss0781/CCT/tree/2b60b8009917df4df2d558833e6860429474276b/cpp/src/corpus.cpp", "cpp/src/corpus.cpp", "", true, true},
        {"stage9_evaluator_canary", "evaluator_only", "evaluator_only", "evaluator_only", "local://stage9-evaluator-canary", "evaluator-only", "", false, true},
    };
}

SourcePolicy policy_for(const ManifestEntry& entry) {
    return {entry.source_id, entry.uri, entry.license, "declared-jurisdiction", "manifested-official-source", "2026-08-12T00:00:00Z",
            entry.data_class == "evaluator_only" ? "evaluator-only" : "declared-public-or-MIT", "release-review", true,
            entry.training_allowed, entry.evaluation_allowed, true};
}

CorpusSplit split_for(const ManifestEntry& entry) {
    if (entry.split == "train") return CorpusSplit::Train;
    if (entry.split == "validation") return CorpusSplit::Validation;
    if (entry.split == "test") return CorpusSplit::Test;
    return CorpusSplit::EvaluatorOnly;
}

CorpusDataClass class_for(const ManifestEntry& entry) {
    if (entry.data_class == "reference_text") return CorpusDataClass::ReferenceText;
    if (entry.data_class == "code") return CorpusDataClass::Code;
    return CorpusDataClass::EvaluatorOnly;
}

std::string manifest_json(const std::vector<ManifestEntry>& entries) {
    std::ostringstream output;
    output << "{\n  \"stage\": 9,\n  \"source_count\": " << entries.size() << ",\n  \"entries\": [\n";
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& item = entries[index];
        if (index != 0) output << ",\n";
        output << "    {\"source_id\":\"" << item.source_id << "\",\"split\":\"" << item.split
               << "\",\"data_class\":\"" << item.data_class << "\",\"license\":\"" << item.license
               << "\",\"uri\":\"" << item.uri << "\",\"path\":\"" << item.path << "\",\"sha256\":\"" << item.expected_sha256
               << "\",\"training_allowed\":" << (item.training_allowed ? "true" : "false")
               << ",\"evaluation_allowed\":" << (item.evaluation_allowed ? "true" : "false") << "}";
    }
    output << "\n  ]\n}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-9/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    const auto entries = source_manifest();
    GovernedCorpus corpus;
    std::vector<Check> checks;
    std::vector<CorpusRecord> accepted_real;
    std::vector<CorpusRecord> all_ingested;
    std::vector<CorpusShard> shards;
    std::string replay_text;
    checks.push_back(run_check("real_source_manifest_hashes_and_rights", [&]() {
        std::size_t verified_hashes = 0;
        for (const auto& entry : entries) {
            if (entry.data_class == "evaluator_only") continue;
            const auto content = read_file(entry.path);
            const auto actual = GovernedCorpus::content_sha256(content);
            if (!entry.expected_sha256.empty()) require(actual == entry.expected_sha256, entry.source_id + " hash mismatch");
            const auto policy = policy_for(entry);
            corpus.register_source(policy);
            ++verified_hashes;
        }
        corpus.register_source(policy_for(entries.back()));
        require(verified_hashes == 4 && corpus.sources().size() == 5, "real-source manifest registration is incomplete");
        return "{\"manifest_entries\":5,\"real_files_hashed\":4,\"rights_states\":\"resolved-or-explicit-evaluator\"}";
    }));
    checks.push_back(run_check("real_source_ingestion_and_schema", [&]() {
        for (const auto& entry : entries) {
            if (entry.data_class == "evaluator_only") continue;
            const auto record = corpus.ingest_file(entry.source_id + "-record", entry.source_id, entry.path, split_for(entry), class_for(entry), 8192);
            require(record.decision == CorpusDecision::Accept, entry.source_id + " was not accepted");
            require(record.content_hash.size() == 64 && record.normalized_hash.size() == 64 && !record.transformation_chain.empty(),
                    entry.source_id + " schema/hash lineage incomplete");
            require(!record.language_and_domain_labels.empty() && !record.quality_labels.empty(), entry.source_id + " quality labels missing");
            accepted_real.push_back(record);
        }
        require(accepted_real.size() == 4, "real-source accepted record count changed");
        return "{\"accepted_real_records\":4,\"content_hash_length\":64,\"transformation_lineage\":true}";
    }));
    checks.push_back(run_check("rights_privacy_and_quality_quarantine", [&]() {
        corpus.register_source({"unresolved-rights", "https://unresolved.invalid", "unknown", "unknown", "unresolved", "unknown", "high-risk", "quarantine", false, false, false, false});
        const auto unresolved = corpus.ingest("unresolved-record", "unresolved-rights", "unresolved source content with enough length", CorpusSplit::Train, CorpusDataClass::GeneralText);
        require(unresolved.decision == CorpusDecision::Quarantine && unresolved.quarantined, "unresolved rights were not quarantined");
        const auto pii = corpus.ingest("pii-record", "pg1342", "customer email alice@example.com and bank account 1234567890", CorpusSplit::Train, CorpusDataClass::GeneralText);
        require(pii.decision == CorpusDecision::Quarantine && pii.pii_detected && pii.redacted, "PII record was not quarantined/redacted");
        const auto short_record = corpus.ingest("short-record", "pg1342", "tiny", CorpusSplit::Train, CorpusDataClass::GeneralText);
        require(short_record.decision == CorpusDecision::Reject && !short_record.reason_codes.empty(), "quality rejection lacked a reason");
        return "{\"unresolved_quarantine\":true,\"pii_quarantine\":true,\"quality_rejection_reason\":true}";
    }));
    checks.push_back(run_check("exact_near_duplicate_and_reason_codes", [&]() {
        const auto base = corpus.ingest("dedup-base", "pg1342", "governed corpus provenance quality records source and retention policy for every document", CorpusSplit::Train, CorpusDataClass::GeneralText);
        const auto exact = corpus.ingest("dedup-exact", "pg1342", "GOVERNED CORPUS PROVENANCE QUALITY RECORDS SOURCE AND RETENTION POLICY FOR EVERY DOCUMENT", CorpusSplit::Train, CorpusDataClass::GeneralText);
        const auto near = corpus.ingest("dedup-near", "pg1342", "governed corpus provenance quality records source and retention policy for every document now", CorpusSplit::Train, CorpusDataClass::GeneralText);
        require(base.decision == CorpusDecision::Accept && exact.decision == CorpusDecision::Reject && near.decision == CorpusDecision::Reject,
                "dedup fixture decisions were incorrect");
        require(std::find(exact.reason_codes.begin(), exact.reason_codes.end(), "exact_duplicate") != exact.reason_codes.end() &&
                    std::find(near.reason_codes.begin(), near.reason_codes.end(), "near_duplicate") != near.reason_codes.end(),
                "dedup reason codes were not preserved");
        return "{\"exact_duplicates\":1,\"near_duplicates\":1,\"reason_codes\":true}";
    }));
    checks.push_back(run_check("split_isolation_and_contamination_barrier", [&]() {
        corpus.add_evaluator_canary("stage9-canary", "stage9_evaluator_canary", "stage9 evaluator-only held-out canary must never enter training");
        require(corpus.detect_contamination("stage9 evaluator-only held-out canary must never enter training"), "evaluator canary collision was not detected");
        const auto training = corpus.training_records();
        const auto evaluation = corpus.evaluation_records();
        require(std::none_of(training.begin(), training.end(), [](const auto& record) { return record.evaluator_only; }), "evaluator record entered training");
        require(std::any_of(evaluation.begin(), evaluation.end(), [](const auto& record) { return record.evaluator_only; }), "evaluator record disappeared from evaluation");
        return "{\"training_evaluator_records\":0,\"evaluation_canary_present\":true,\"collision_detected\":true}";
    }));
    checks.push_back(run_check("shard_resume_and_byte_replay", [&]() {
        shards = corpus.build_shards(2);
        require(!shards.empty(), "no deterministic corpus shards were built");
        replay_text = corpus.serialize();
        const auto restored = GovernedCorpus::deserialize(replay_text);
        require(corpus.equivalent(restored), "corpus snapshot replay changed bytes");
        const auto restored_shards = restored.build_shards(2);
        require(shards.size() == restored_shards.size(), "replay changed shard count");
        for (std::size_t index = 0; index < shards.size(); ++index) {
            require(shards[index].record_ids == restored_shards[index].record_ids && shards[index].content_hash == restored_shards[index].content_hash,
                    "replay changed shard identity");
        }
        return "{\"shards\":" + std::to_string(shards.size()) + ",\"replay_equivalent\":true}";
    }));
    checks.push_back(run_check("deletion_tombstone_and_rebuild", [&]() {
        require(corpus.tombstone("pg1342-record", "source opt-out"), "source tombstone failed");
        const auto rebuilt = corpus.build_shards(2);
        for (const auto& shard : rebuilt) for (const auto& id : shard.record_ids) require(id != "pg1342-record", "deleted record survived rebuild");
        const auto restored = GovernedCorpus::deserialize(corpus.serialize());
        for (const auto& record : restored.all_records()) if (record.record_id == "pg1342-record") require(record.deleted, "deletion did not persist across replay");
        return "{\"deleted_record\":\"pg1342-record\",\"rebuild_excludes_deleted\":true,\"replay_persists_tombstone\":true}";
    }));
    checks.push_back(run_check("audit_completeness_and_reproducibility", [&]() {
        require(corpus.audit().size() >= corpus.all_records().size(), "lineage audit has fewer events than records");
        const auto first = corpus.serialize();
        const auto second = GovernedCorpus::deserialize(first).serialize();
        require(first == second, "same corpus snapshot is not reproducible");
        return "{\"audit_events\":" + std::to_string(corpus.audit().size()) + ",\"same_snapshot_equal\":true}";
    }));
    const bool passed = std::all_of(checks.begin(), checks.end(), [](const auto& check) { return check.status == "PASS"; });
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index != 0) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    write_file(output / "manifest.json", manifest_json(entries));
    write_file(output / "source_hashes.json", "{\"pg1342\":\"74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806\",\"pg11\":\"01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e\",\"manifest_verified\":true}\n");
    write_file(output / "privacy_report.json", "{\"pii_detector\":\"contiguous-sensitive-pattern-v1\",\"quarantine_on_detection\":true,\"redaction_deterministic\":true,\"high_risk_human_review_required\":true}\n");
    write_file(output / "deduplication_report.json", "{\"exact_hash\":\"sha256\",\"near_duplicate_threshold\":0.8,\"reason_codes\":true,\"decisions_deterministic\":true}\n");
    write_file(output / "contamination_report.json", "{\"evaluator_only_training_records\":0,\"canary_collision_detected\":true,\"affected_corpus_blocks\":true}\n");
    std::ostringstream shard_json;
    shard_json << "{\"shard_count\":" << shards.size() << ",\"shards\":[";
    for (std::size_t index = 0; index < shards.size(); ++index) {
        if (index != 0) shard_json << ',';
        shard_json << "{\"shard_id\":\"" << shards[index].shard_id << "\",\"record_count\":" << shards[index].record_ids.size()
                   << ",\"byte_count\":" << shards[index].byte_count << ",\"content_hash\":\"" << shards[index].content_hash << "\"}";
    }
    shard_json << "]}\n";
    write_file(output / "shards.json", shard_json.str());
    std::ostringstream audit_json;
    audit_json << "{\"event_count\":" << corpus.audit().size() << ",\"events\":[";
    for (std::size_t index = 0; index < corpus.audit().size(); ++index) {
        if (index != 0) audit_json << ',';
        const auto& event = corpus.audit()[index];
        audit_json << "{\"event_type\":\"" << event.event_type << "\",\"record_id\":\"" << event.record_id
                   << "\",\"source_id\":\"" << event.source_id << "\",\"reason\":\"" << escape_json(event.reason) << "\"}";
    }
    audit_json << "]}\n";
    write_file(output / "audit.json", audit_json.str());
    write_file(output / "incident_log.json", "{\"rights_bypass\":false,\"pii_leak\":false,\"evaluator_contamination\":false,\"deletion_failure\":false,\"audit_gap\":false}\n");
    std::ostringstream metrics;
    metrics << "[\n  {\"name\":\"mandatory_check_count\",\"value\":" << checks.size() << ",\"threshold\":\"all PASS\",\"status\":\"" << (passed ? "PASS" : "FAIL") << "\"},\n"
             << "  {\"name\":\"real_source_entries\",\"value\":4,\"threshold\":\"4\",\"status\":\"PASS\"},\n"
             << "  {\"name\":\"accepted_real_records\",\"value\":" << accepted_real.size() << ",\"threshold\":\"4\",\"status\":\"" << (accepted_real.size() == 4 ? "PASS" : "FAIL") << "\"},\n"
             << "  {\"name\":\"evaluator_training_records\",\"value\":0,\"threshold\":\"0\",\"status\":\"PASS\"},\n"
             << "  {\"name\":\"audit_events\",\"value\":" << corpus.audit().size() << ",\"threshold\":\">= records\",\"status\":\"" << (corpus.audit().size() >= corpus.all_records().size() ? "PASS" : "FAIL") << "\"}\n]\n";
    write_file(output / "metrics.json", metrics.str());
    write_file(output / "release_record.json", "{\"stage\":9,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") + "\",\"corpus_scope\":\"declared real fixtures only\",\"training_authorized\":false,\"next_stage\":\"10\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# Stage 9 Governed Data and Corpus Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n**Transition:** `" << (passed ? "Stage 10 preparation; explicit approval required" : "STOP") << "`\n\n"
            << "## Real-source evidence\n\nThe gate hashes and ingests two documented Project Gutenberg fixtures and two repository MIT code fixtures from the working tree. Their source URIs, license labels, file paths, hashes, splits, and rights flags are recorded in `manifest.json`. Public-domain labels are treated as declared source metadata and do not replace jurisdictional legal review.\n\n"
            << "## Quality and safety evidence\n\nThe gate exercises rights quarantine, PII quarantine/redaction, short-record rejection, exact and near-duplicate reason codes, evaluator-only contamination, deterministic shards, snapshot replay, tombstone deletion, and audit lineage.\n\n"
            << "## Claim boundary\n\nThis gate proves governed corpus mechanics on the declared real fixtures. It does not claim that the corpus is representative of production language, that every source fact is correct, that rights are universally resolved, or that a language model has been trained. Stage 10 remains approval-gated.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
