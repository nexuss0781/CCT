#include "cct/scaling.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cct::Stage5LanguageModel;
using cct::Stage5ModelConfig;
using cct::Stage5ModelKind;
using cct::Stage5TrainConfig;
using cct::Stage5Vocabulary;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string alphabet() {
    return " \nabcdefghijklmnopqrstuvwxyz0123456789.,;:!?(){}[]+-*/=_";
}

std::vector<std::vector<std::vector<double>>> make_inputs(const std::vector<std::size_t>& tokens,
                                                          std::size_t vocabulary) {
    require(tokens.size() >= 2, "token fixture too short");
    std::vector<std::vector<std::vector<double>>> batch(1);
    for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
        std::vector<double> row(vocabulary, 0.0);
        row[tokens[index] % vocabulary] = 1.0;
        batch.front().push_back(std::move(row));
    }
    return batch;
}

std::vector<std::vector<std::vector<double>>> make_targets(const std::vector<std::size_t>& tokens,
                                                           std::size_t vocabulary) {
    require(tokens.size() >= 2, "token fixture too short");
    std::vector<std::vector<std::vector<double>>> batch(1);
    for (std::size_t index = 1; index < tokens.size(); ++index) {
        std::vector<double> row(vocabulary, -1.0);
        row[tokens[index] % vocabulary] = 1.0;
        batch.front().push_back(std::move(row));
    }
    return batch;
}

void test_vocabulary_roundtrip() {
    const std::string text = "Hello, CCT!\n\x01";
    const auto byte_tokens = Stage5Vocabulary::encode_bytes(text, true);
    require(Stage5Vocabulary::decode_bytes(byte_tokens) == text, "byte vocabulary round-trip failed");
    const auto compact_tokens = Stage5Vocabulary::compact_encode("abc z", alphabet(), alphabet().size());
    require(Stage5Vocabulary::compact_decode(compact_tokens, alphabet(), alphabet().size()) == "abc z",
            "compact vocabulary round-trip failed");
    const auto unknown = Stage5Vocabulary::compact_encode("@", alphabet(), alphabet().size());
    require(unknown.size() == 1 && unknown.front() == alphabet().size(), "unknown token fallback failed");
}

void test_training_and_checkpoint() {
    const std::string mini_alphabet = " ab";
    const auto vocabulary = mini_alphabet.size() + 1;
    const auto tokens = Stage5Vocabulary::compact_encode(" a b a b a b a b a b a b a b a b ", mini_alphabet, mini_alphabet.size());
    const auto inputs = make_inputs(tokens, vocabulary);
    const auto targets = make_targets(tokens, vocabulary);
    const std::vector<std::vector<std::uint8_t>> masks(1, std::vector<std::uint8_t>(tokens.size() - 1, 1));
    Stage5LanguageModel model(Stage5ModelConfig{vocabulary, 10, vocabulary, 501, Stage5ModelKind::CCTNoMemory});
    const auto before = model.evaluate(inputs, targets, masks);
    model.train(inputs, targets, masks, Stage5TrainConfig{10, 0.12, 10.0, 0, 12345});
    const auto after = model.evaluate(inputs, targets, masks);
    require(std::isfinite(before.cross_entropy) && std::isfinite(after.cross_entropy) && after.cross_entropy < before.cross_entropy,
            "CCT Stage 5 training did not reduce cross-entropy");
    const auto path = "/tmp/cct_stage5_scaling_checkpoint.chk";
    model.save_checkpoint(path);
    const auto restored = Stage5LanguageModel::load_checkpoint(path);
    require(restored.optimizer_step() == model.optimizer_step() && restored.data_cursor() == model.data_cursor() &&
                restored.manifest_fingerprint() == model.manifest_fingerprint(),
            "Stage 5 checkpoint metadata did not resume");
    const auto restored_eval = restored.evaluate(inputs, targets, masks);
    require(std::abs(restored_eval.cross_entropy - after.cross_entropy) < 1e-12,
            "Stage 5 checkpoint changed evaluation metrics");
}

void test_matched_models_and_memory_context() {
    const auto vocabulary = alphabet().size() + 1;
    const auto tokens = Stage5Vocabulary::compact_encode("a b a b a b a b a b ", alphabet(), alphabet().size());
    const auto inputs = make_inputs(tokens, vocabulary);
    const auto targets = make_targets(tokens, vocabulary);
    const std::vector<std::vector<std::uint8_t>> masks(1, std::vector<std::uint8_t>(tokens.size() - 1, 1));
    for (const auto kind : {Stage5ModelKind::DenseCausalAttention, Stage5ModelKind::GRU,
                            Stage5ModelKind::DiagonalSSM, Stage5ModelKind::CCTNoMemory,
                            Stage5ModelKind::CCTFrozenMemory}) {
        Stage5LanguageModel model(Stage5ModelConfig{vocabulary, 6, vocabulary, 600 + static_cast<unsigned int>(kind), kind});
        const auto before = model.evaluate(inputs, targets, masks);
        model.train(inputs, targets, masks, Stage5TrainConfig{2, 0.06, 8.0, 0, 900});
        const auto after = model.evaluate(inputs, targets, masks);
        require(after.cross_entropy < before.cross_entropy && model.parameter_count() > 0 && model.state_memory_bytes() > 0,
                "matched Stage 5 model failed to train or report resources");
    }
    cct::PersistentMemory memory(cct::MemoryConfig{4, 16, 0.0, 91, true});
    const auto report = cct::evaluate_stage5_memory_augmentation(memory);
    require(report.no_memory_hits == 0 && report.memory_hits == 1 && report.evidence_ids_attributed && report.retrieval_latency_ms >= 0.0,
            "Stage 5 memory attribution report failed");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"vocabulary_roundtrip", test_vocabulary_roundtrip},
        {"training_and_checkpoint", test_training_and_checkpoint},
        {"matched_models_and_memory_context", test_matched_models_and_memory_context},
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
