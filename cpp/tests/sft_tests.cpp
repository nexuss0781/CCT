#include "cct/sft.hpp"

#include "cct/corpus.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

constexpr const char* kBaseHash = "8ff1f227513d79a840b648bd724823e3fd790ba3bd9e754a086f430ebbd81b62";

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

SftTaskSchema classification_schema() {
    return {"classify", SftTaskKind::Classification, "v1", SftOutputKind::Label, {"positive", "negative"}, 64U, false, true, "classification"};
}

SftTaskSchema extraction_schema() {
    return {"extract", SftTaskKind::StructuredExtraction, "v1", SftOutputKind::Json, {"invoice", "other"}, 256U, false, true, "extraction"};
}

SftInstructionExample example(const std::string& id, const std::string& task, const std::string& input,
                              const std::string& target_label, const bool train, const bool eval) {
    SftInstructionExample item;
    item.example_id = id;
    item.task_id = task;
    item.schema_version = "v1";
    item.input = input;
    item.target = target_label;
    item.target_label = target_label;
    item.input_provenance = "source:" + id;
    item.target_provenance = "annotator:" + id;
    item.policy_class = "bounded";
    item.split = train ? "train" : "validation";
    item.evaluator_owner = "stage13-evaluator";
    item.source_hash = sft_hash(item.input);
    item.target_hash = sft_hash(item.target);
    item.example_hash = sft_hash(item.example_id + "\n" + item.task_id + "\n" + item.schema_version + "\n" +
                               item.input + "\n" + item.target + "\n" + item.target_label + "\n" + item.source_hash + "\n" +
                               item.target_hash + "\n" + item.split);
    item.training_allowed = train;
    item.evaluation_allowed = eval;
    return item;
}

void test_manifest_formatter_and_masks() {
    const auto schema = classification_schema();
    const auto first = example("ex-1", "classify", "positive customer response", "positive", true, false);
    const auto second = example("ex-2", "classify", "negative customer response", "negative", false, true);
    const auto manifest = SftManifest::build({first, second}, {schema});
    require(!manifest.manifest_hash.empty() && manifest.training_examples().size() == 1U && manifest.evaluation_examples().size() == 1U &&
                !manifest.contains_evaluator_training(), "SFT manifest eligibility is incorrect");
    const auto restored = SftManifest::deserialize(manifest.serialize());
    require(restored.serialize() == manifest.serialize(), "SFT manifest replay changed bytes");
    const auto tokenizer = Tokenizer::build({}, {{"formatter", "positive negative response", true, false}});
    const auto formatted = SftFormatter::format(first, schema, tokenizer);
    const auto repeated = SftFormatter::format(first, schema, tokenizer);
    require(formatted.serialized == repeated.serialized && formatted.token_ids == repeated.token_ids &&
                formatted.loss_mask == repeated.loss_mask && formatted.target_token_end > formatted.target_token_start,
            "SFT formatter is not deterministic or target mask is empty");
    bool has_active = false;
    for (const auto value : formatted.loss_mask) if (value != 0U) has_active = true;
    require(has_active, "SFT formatter has no active target positions");
}

void test_full_sft_gradient_and_learning() {
    const auto schema = classification_schema();
    const std::vector<SftInstructionExample> training{
        example("p1", "classify", "positive helpful response", "positive", true, false),
        example("p2", "classify", "positive successful result", "positive", true, false),
        example("n1", "classify", "negative failed response", "negative", true, false),
        example("n2", "classify", "negative rejected result", "negative", true, false)};
    const std::vector<SftInstructionExample> validation{
        example("pv", "classify", "positive helpful result", "positive", false, true),
        example("nv", "classify", "negative failed result", "negative", false, true)};
    SftModel model({kBaseHash, "classify", 8U, 2U, 3U});
    const auto initial = model.evaluate(validation, schema);
    const auto gradient = model.gradients(training.front(), schema);
    require(gradient.size() == model.parameter_count() && std::any_of(gradient.begin(), gradient.end(), [](const double value) { return std::abs(value) > 1e-12; }),
            "full SFT gradient is empty");
    SftOptimizerConfig optimizer;
    optimizer.learning_rate = 0.5;
    optimizer.clip_norm = 5.0;
    for (std::size_t step = 0; step < 80U; ++step) {
        for (const auto& item : training) model.apply_gradient(model.gradients(item, schema), optimizer);
    }
    const auto final = model.evaluate(validation, schema);
    require(final.finite && final.accuracy > initial.accuracy && final.cross_entropy < initial.cross_entropy,
            "full SFT did not improve the held-out classification task");
    std::stringstream stream;
    model.save(stream);
    const auto restored = SftModel::load(stream);
    require(restored.parameter_checksum() == model.parameter_checksum(), "full SFT model serialization changed parameters");
}

void test_adapter_freeze_gradient_and_registry() {
    const auto schema = classification_schema();
    const std::vector<SftInstructionExample> training{
        example("p1", "classify", "positive helpful response", "positive", true, false),
        example("n1", "classify", "negative failed response", "negative", true, false)};
    SftModel base({kBaseHash, "classify", 8U, 2U, 5U});
    const auto base_checksum = base.parameter_checksum();
    SftAdapterSpec spec{"classify-r1", "classify", "support", "adapter-v1", 1U, "output_projection", kBaseHash, "manifest-hash", {"read", "train"}};
    SftAdapter adapter(spec, base.config());
    const auto initial = base.evaluate(training, schema, &adapter);
    SftOptimizerConfig optimizer;
    optimizer.learning_rate = 0.5;
    optimizer.clip_norm = 5.0;
    for (std::size_t step = 0; step < 80U; ++step) {
        for (const auto& item : training) adapter.apply_gradient(adapter.gradients(base, item, schema), optimizer);
    }
    const auto final = base.evaluate(training, schema, &adapter);
    require(adapter.parameter_count() < base.parameter_count() && base.parameter_checksum() == base_checksum &&
                final.cross_entropy < initial.cross_entropy, "adapter did not remain efficient, immutable, or trainable");
    SftAdapterRegistry registry;
    registry.register_adapter(adapter);
    require(registry.authorize("classify-r1", "classify", kBaseHash, "read"), "authorized adapter was denied");
    require(!registry.authorize("classify-r1", "extract", kBaseHash, "read") &&
                !registry.authorize("classify-r1", "classify", "wrong-base", "read") &&
                !registry.authorize("classify-r1", "classify", kBaseHash, "write"),
            "unauthorized adapter access was accepted");
    const auto loaded = registry.load_authorized("classify-r1", "classify", kBaseHash, "read");
    require(loaded.parameter_checksum() == adapter.parameter_checksum(), "adapter registry replay changed identity");
    bool rejected = false;
    try {
        static_cast<void>(registry.load_authorized("classify-r1", "extract", kBaseHash, "read"));
    } catch (const SftError&) {
        rejected = true;
    }
    require(rejected, "unauthorized adapter load did not fail closed");
}

void test_merge_and_structured_decoder() {
    const auto schema = extraction_schema();
    auto item = example("json-1", "extract", "invoice total 42", "invoice", false, true);
    item.citation_id = "source-json-1";
    SftModel base({kBaseHash, "extract", 8U, 2U, 7U});
    SftAdapter adapter({"extract-r1", "extract", "finance", "adapter-v1", 1U, "output_projection", kBaseHash, "manifest", {"read"}}, base.config());
    const auto runtime = base.predict(item, schema, &adapter);
    const auto merged = base.merged(adapter).predict(item, schema, nullptr);
    require(runtime.output == merged.output && runtime.schema_valid && runtime.citation_valid,
            "merged and runtime structured outputs diverged or failed validation");
    const auto invalid = StructuredDecoder::validate({"extract", "invoice", "{bad", 0.9, "", false, false, false}, item, schema);
    require(!invalid.schema_valid, "malformed structured output was accepted");
}

void test_fail_closed_examples_and_finite_metrics() {
    const auto schema = classification_schema();
    auto evaluator = example("eval", "classify", "positive evaluator canary", "positive", true, false);
    evaluator.evaluator_only = true;
    bool rejected = false;
    try {
        static_cast<void>(SftManifest::build({evaluator}, {schema}));
    } catch (const SftError&) {
        rejected = true;
    }
    require(rejected, "evaluator-only SFT training example was accepted");
    SftModel model({kBaseHash, "classify", 8U, 2U, 11U});
    bool malformed_rejected = false;
    try {
        static_cast<void>(SftManifest::deserialize("CCT_SFT_MANIFEST_V999\n"));
    } catch (const SftError&) {
        malformed_rejected = true;
    }
    require(malformed_rejected, "unsupported SFT manifest version was accepted");
    const auto evaluation = model.evaluate({example("eval", "classify", "positive", "positive", false, true)}, schema);
    require(evaluation.finite && std::isfinite(evaluation.cross_entropy), "SFT evaluation is not finite");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"manifest_formatter_and_masks", test_manifest_formatter_and_masks},
        {"full_sft_gradient_and_learning", test_full_sft_gradient_and_learning},
        {"adapter_freeze_gradient_and_registry", test_adapter_freeze_gradient_and_registry},
        {"merge_and_structured_decoder", test_merge_and_structured_decoder},
        {"fail_closed_examples_and_finite_metrics", test_fail_closed_examples_and_finite_metrics}};
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
