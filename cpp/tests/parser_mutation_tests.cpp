#include "cct/causal.hpp"
#include "cct/knowledge.hpp"
#include "cct/memory.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Parser>
std::size_t mutate_and_check(const std::string& name, const std::string& valid_snapshot, Parser parser) {
    require(!valid_snapshot.empty(), name + " valid snapshot is empty");
    std::vector<std::string> mutations;
    mutations.push_back(valid_snapshot.substr(0, valid_snapshot.size() / 2U));
    mutations.push_back(valid_snapshot + "\nTRAILING_GARBAGE");
    mutations.push_back("CORRUPTED_HEADER\n" + valid_snapshot);
    mutations.push_back(valid_snapshot.substr(0, 1U) + std::string("\0", 1U) + valid_snapshot.substr(1U));
    const auto stride = std::max<std::size_t>(1U, valid_snapshot.size() / 8U);
    for (std::size_t index = 0U; index < valid_snapshot.size(); index += stride) {
        auto mutation = valid_snapshot;
        mutation[index] = static_cast<char>(mutation[index] ^ static_cast<char>(0x20));
        mutations.push_back(std::move(mutation));
    }
    std::size_t rejected = 0U;
    for (const auto& mutation : mutations) {
        try {
            parser(mutation);
        } catch (const std::exception&) {
            ++rejected;
        } catch (...) {
            throw std::runtime_error(name + " parser escaped with a non-standard exception");
        }
    }
    require(rejected * 2U >= mutations.size(), name + " accepted too many bounded mutations");
    return rejected;
}

std::string causal_snapshot() {
    CausalStoreConfig config;
    config.payload_dim = 2U;
    config.coordinate_dim = 2U;
    config.coordinate_min = {0.0, 0.0};
    config.coordinate_max = {1.0, 1.0};
    CausalEventStore store(config);
    CausalEvent event;
    event.id = 1U;
    event.semantic_payload = {1.0, 0.5};
    event.coordinates = {0.5, 0.5};
    event.timestamp = 1;
    event.provenance = ProvenanceKind::Generated;
    event.uncertainty = {UncertaintyKind::Known, 1.0};
    store.insert(event);
    return store.serialize_snapshot();
}

std::string memory_snapshot() {
    MemoryConfig config;
    config.embedding_dim = 2U;
    config.max_active_records = 8U;
    PersistentMemory memory(config);
    MemoryRecord record;
    record.memory_id = 1U;
    record.content = "parser mutation fact";
    record.embedding = {1.0, 0.0};
    record.created_at = 1;
    record.valid_from = 1;
    record.source = {"parser-source", 0U, record.content.size()};
    record.confidence = 0.95;
    memory.write(record);
    return memory.serialize_snapshot();
}

std::string knowledge_snapshot() {
    KnowledgePlane plane;
    KnowledgeRecord record;
    record.knowledge_id = "parser-knowledge";
    record.tenant_id = "tenant-parser";
    record.document_id = "document-parser";
    record.document_version = 1U;
    record.source_uri_or_reference = "https://example.test/parser";
    record.content = "The parser mutation fixture is bounded.";
    record.content_hash = GovernedCorpus::content_sha256(record.content);
    record.embedding_version = "embedding-v1";
    record.lexical_index_version = "lexical-v1";
    record.created_at = 1;
    record.valid_from = 1;
    record.access_policy = {"tenant-parser", {"analyst"}, false};
    record.provenance = "parser-fixture";
    record.citation_spans.push_back({"parser-knowledge#span", 0U, record.content.size(), record.content_hash});
    record.quality = {0.95, 0.95, "normal"};
    plane.ingest(record);
    return plane.serialize_snapshot();
}

std::string tokenizer_snapshot(std::string& expected_hash) {
    TokenizerConfig config;
    config.candidate = TokenizerCandidate::Byte;
    config.include_bos_eos = true;
    const auto tokenizer = Tokenizer::build(config, {TokenizerTrainingRecord{"parser-train", "bounded parser fixture", true, false}});
    expected_hash = tokenizer.snapshot_hash();
    return tokenizer.serialize_snapshot();
}

void test_bounded_parser_mutations() {
    const auto causal = causal_snapshot();
    const auto memory = memory_snapshot();
    const auto knowledge = knowledge_snapshot();
    std::string tokenizer_hash;
    const auto tokenizer = tokenizer_snapshot(tokenizer_hash);
    const auto causal_rejected = mutate_and_check("causal", causal, [](const std::string& value) {
        static_cast<void>(CausalEventStore::deserialize_snapshot(value));
    });
    const auto memory_rejected = mutate_and_check("memory", memory, [](const std::string& value) {
        static_cast<void>(PersistentMemory::deserialize_snapshot(value));
    });
    const auto knowledge_rejected = mutate_and_check("knowledge", knowledge, [](const std::string& value) {
        static_cast<void>(KnowledgePlane::deserialize_snapshot(value));
    });
    const auto tokenizer_rejected = mutate_and_check("tokenizer", tokenizer, [&](const std::string& value) {
        static_cast<void>(Tokenizer::from_snapshot(value, tokenizer_hash));
    });
    require(causal_rejected > 0U && memory_rejected > 0U && knowledge_rejected > 0U && tokenizer_rejected > 0U,
            "parser mutation suite did not reject any malformed inputs");
}

}  // namespace

int main() {
    try {
        test_bounded_parser_mutations();
        std::cout << "PASS bounded_parser_mutations\nSUMMARY 1/1 passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "FAIL bounded_parser_mutations: " << error.what() << "\nSUMMARY 0/1 passed\n";
        return 1;
    }
}
