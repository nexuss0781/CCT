#include "cct/nlp_trainer.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

cct::NlpSequence sequence(const std::vector<cct::TokenId>& ids, const std::string& name) {
    require(ids.size() >= 3U, "test sequence is too short");
    cct::NlpSequence value;
    value.sequence_id = name;
    value.record_id = name;
    value.input_ids = ids;
    value.target_ids.assign(ids.size(), cct::Tokenizer::kPadId);
    value.loss_mask.assign(ids.size(), 0U);
    for (std::size_t index = 0U; index + 1U < ids.size(); ++index) {
        value.target_ids[index] = ids[index + 1U];
        value.loss_mask[index] = 1U;
    }
    return value;
}

void test_preference_optimizer_progresses() {
    cct::NlpModelConfig model{cct::NlpModelKind::Track1CctRecurrence, 512U, 4U, 4U, 8U, 1701U};
    cct::NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.001;
    optimizer.total_steps = 3U;
    optimizer.validation_interval_steps = 3U;
    cct::NlpTrainer trainer(model, optimizer, "tokenizer", "cola-lineage");
    cct::NlpPreferencePair pair{sequence({1U, 4U, 5U, 6U, 7U}, "acceptable"), sequence({1U, 9U, 10U, 11U, 12U}, "unacceptable")};
    const auto before = trainer.model().parameter_vector();
    const auto points = trainer.train_preference_steps({pair}, 3U, 0.05);
    require(points.size() == 3U && trainer.state().optimizer_step == 3U, "preference optimizer did not complete its declared budget");
    require(std::isfinite(points.back().train_loss) && std::isfinite(points.back().gradient_norm), "preference optimizer emitted non-finite evidence");
    require(trainer.model().parameter_vector() != before, "preference optimizer left parameters unchanged");
}

void test_preference_checkpoint_lineage() {
    cct::NlpModelConfig model{cct::NlpModelKind::Track1CctRecurrence, 512U, 4U, 4U, 8U, 1701U};
    cct::NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.001;
    optimizer.total_steps = 1U;
    cct::NlpTrainer trainer(model, optimizer, "tokenizer-v1", "cola-lineage-v1");
    const auto pair = cct::NlpPreferencePair{sequence({1U, 4U, 5U, 6U}, "acceptable"), sequence({1U, 9U, 10U, 11U}, "unacceptable")};
    static_cast<void>(trainer.train_preference_step(pair, 0.05));
    const auto path = std::filesystem::temp_directory_path() / "cct-english-preference-checkpoint.bin";
    trainer.save_checkpoint(path.string());
    const auto restored = cct::NlpTrainer::load_checkpoint(path.string(), "tokenizer-v1", "cola-lineage-v1");
    require(restored.state().optimizer_step == 1U && restored.checkpoint_info().checkpoint_hash == trainer.checkpoint_info().checkpoint_hash,
            "preference checkpoint lineage did not round-trip");
    std::ifstream checkpoint(path, std::ios::binary);
    require(static_cast<bool>(checkpoint), "preference checkpoint could not be reopened for format validation");
    std::string line;
    while (std::getline(checkpoint, line)) {
        require(line.empty() || (line.back() != ' ' && line.back() != '\t'), "preference checkpoint contains trailing whitespace");
    }
    bool rejected = false;
    try {
        static_cast<void>(cct::NlpTrainer::load_checkpoint(path.string(), "tokenizer-v1", "wrong-lineage"));
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "preference checkpoint accepted a mismatched grammar lineage");
    std::filesystem::remove(path);
}

void test_preference_fail_closed() {
    cct::NlpModelConfig model{cct::NlpModelKind::Track1CctRecurrence, 512U, 4U, 4U, 8U, 1701U};
    cct::NlpOptimizerConfig optimizer;
    optimizer.total_steps = 1U;
    cct::NlpTrainer trainer(model, optimizer, "tokenizer", "lineage");
    bool rejected_empty = false;
    try {
        static_cast<void>(trainer.train_preference_steps({}, 1U));
    } catch (const std::exception&) {
        rejected_empty = true;
    }
    require(rejected_empty, "empty preference training set was accepted");
    auto malformed = sequence({1U, 4U, 5U}, "malformed");
    malformed.input_ids.push_back(99U);
    bool rejected_id = false;
    try {
        static_cast<void>(trainer.train_preference_step({malformed, sequence({1U, 9U, 10U}, "valid")}, 0.0));
    } catch (const std::exception&) {
        rejected_id = true;
    }
    require(rejected_id, "out-of-range preference token was accepted");
}

}  // namespace

int main() {
    try {
        test_preference_optimizer_progresses();
        test_preference_checkpoint_lineage();
        test_preference_fail_closed();
        std::cout << "english acquisition tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "english acquisition tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
