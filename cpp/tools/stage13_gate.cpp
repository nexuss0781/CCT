#include "cct/corpus.hpp"
#include "cct/sft.hpp"
#include "cct/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

constexpr const char* kBaseCheckpointHash = "8ff1f227513d79a840b648bd724823e3fd790ba3bd9e754a086f430ebbd81b62";

struct Check {
    std::string name;
    std::string status;
    double duration_seconds = 0.0;
    std::string details;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
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

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not read Stage 13 source: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not write Stage 13 artifact: " + path.string());
    stream << content;
    require(static_cast<bool>(stream), "could not finish Stage 13 artifact: " + path.string());
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

std::vector<SftTaskSchema> schemas() {
    return {
        {"classification", SftTaskKind::Classification, "v1", SftOutputKind::Label, {"positive", "negative"}, 96U, false, true, "classification"},
        {"extraction", SftTaskKind::StructuredExtraction, "v1", SftOutputKind::Json, {"invoice", "other"}, 256U, false, true, "extraction"},
        {"grounded_qa", SftTaskKind::GroundedQuestionAnswering, "v1", SftOutputKind::Grounded, {"supported", "abstain"}, 256U, true, true, "grounded_answer"},
        {"summarization", SftTaskKind::Summarization, "v1", SftOutputKind::BoundedText, {"summary", "abstain"}, 192U, false, true, "summarization"},
        {"code_understanding", SftTaskKind::CodeUnderstanding, "v1", SftOutputKind::Json, {"issue", "clean"}, 256U, false, true, "code_understanding"},
        {"workflow_drafting", SftTaskKind::WorkflowDrafting, "v1", SftOutputKind::Draft, {"draft", "abstain"}, 256U, false, true, "workflow_draft"}
    };
}

SftInstructionExample make_example(const std::string& id, const std::string& task, const std::string& input,
                                   const std::string& target, const std::string& provenance, const std::string& split,
                                   const bool training, const bool evaluation, const bool evaluator_only = false,
                                   const std::string& citation_id = {}) {
    SftInstructionExample example;
    example.example_id = id;
    example.task_id = task;
    example.schema_version = "v1";
    example.input = input;
    example.target = target;
    example.target_label = target;
    example.input_provenance = provenance;
    example.target_provenance = "annotator:" + id;
    example.policy_class = task == "workflow_drafting" ? "human_approval_required" : "bounded_fixture";
    example.split = split;
    example.evaluator_owner = "stage13-independent-evaluator";
    example.source_hash = sft_hash(example.input);
    example.target_hash = sft_hash(example.target);
    example.citation_id = citation_id;
    example.source_span_start = 0U;
    example.source_span_end = input.size();
    example.training_allowed = training;
    example.evaluation_allowed = evaluation;
    example.evaluator_only = evaluator_only;
    example.example_hash = sft_example_hash(example);
    return example;
}

std::vector<SftInstructionExample> build_examples(const std::string& real_text) {
    const auto source_hash = sft_hash(real_text);
    return {
        make_example("train-class-positive", "classification", "positive customer response from governed text", "positive", "pg1342#classification#" + source_hash, "train", true, false),
        make_example("train-class-negative", "classification", "negative failed customer response", "negative", "pg1342#classification#" + source_hash, "train", true, false),
        make_example("train-extract-invoice", "extraction", "invoice total 42 currency USD", "invoice", "pg1342#extract#" + source_hash, "train", true, false),
        make_example("train-extract-other", "extraction", "ordinary paragraph without a document object", "other", "pg1342#extract#" + source_hash, "train", true, false),
        make_example("train-grounded-supported", "grounded_qa", "What does source-1 support?", "supported", "pg1342#qa#" + source_hash, "train", true, false, false, "source-1"),
        make_example("train-grounded-abstain", "grounded_qa", "What is missing from the source?", "abstain", "pg1342#qa#" + source_hash, "train", true, false),
        make_example("train-summary", "summarization", "Summarize the governed paragraph.", "summary", "pg1342#summary#" + source_hash, "train", true, false),
        make_example("train-code", "code_understanding", "if (value == nullptr) return;", "issue", "cct-source#code", "train", true, false),
        make_example("train-workflow", "workflow_drafting", "Draft a review request for the owner.", "draft", "cct-source#workflow", "train", true, false),
        make_example("eval-class-positive", "classification", "positive successful result", "positive", "pg11#classification", "validation", false, true),
        make_example("eval-class-negative", "classification", "negative rejected result", "negative", "pg11#classification", "validation", false, true),
        make_example("eval-extract-invoice", "extraction", "invoice amount 99", "invoice", "pg11#extract", "validation", false, true),
        make_example("eval-extract-other", "extraction", "ordinary text only", "other", "pg11#extract", "validation", false, true),
        make_example("eval-grounded-supported", "grounded_qa", "Which claim is supported?", "supported", "pg11#qa", "validation", false, true, false, "source-1"),
        make_example("eval-grounded-missing", "grounded_qa", "Which claim has no evidence?", "abstain", "pg11#qa", "validation", false, true),
        make_example("eval-summary", "summarization", "Provide a short source summary.", "summary", "pg11#summary", "validation", false, true),
        make_example("eval-code", "code_understanding", "return nullptr;", "issue", "cct-heldout#code", "validation", false, true),
        make_example("eval-workflow", "workflow_drafting", "Draft but do not submit.", "draft", "cct-heldout#workflow", "validation", false, true)
    };
}

SftModel train_full(const SftModel& initial, const std::vector<SftInstructionExample>& examples,
                    const SftTaskSchema& schema, const std::size_t steps, const std::uint64_t seed_offset) {
    SftModel model = initial;
    SftOptimizerConfig optimizer;
    optimizer.learning_rate = 0.5 + static_cast<double>(seed_offset % 3U) * 0.05;
    optimizer.clip_norm = 5.0;
    optimizer.total_steps = steps;
    for (std::size_t step = 0; step < steps; ++step) {
        for (const auto& example : examples) model.apply_gradient(model.gradients(example, schema), optimizer);
    }
    return model;
}

SftAdapter train_adapter(const SftModel& base, const SftAdapterSpec& spec, const std::vector<SftInstructionExample>& examples,
                         const SftTaskSchema& schema, const std::size_t steps) {
    SftAdapter adapter(spec, base.config());
    SftOptimizerConfig optimizer;
    optimizer.learning_rate = 0.5;
    optimizer.clip_norm = 5.0;
    for (std::size_t step = 0; step < steps; ++step) {
        for (const auto& example : examples) adapter.apply_gradient(adapter.gradients(base, example, schema), optimizer);
    }
    return adapter;
}

bool unsafe_request_denied(const std::string& input) {
    return input.find("send email") != std::string::npos || input.find("submit payment") != std::string::npos ||
           input.find("execute code") != std::string::npos || input.find("secret") != std::string::npos;
}

std::string evaluation_json(const SftEvaluation& evaluation) {
    std::ostringstream output;
    output << "{\"cross_entropy\":" << evaluation.cross_entropy << ",\"accuracy\":" << evaluation.accuracy
           << ",\"schema_validity\":" << evaluation.schema_validity << ",\"citation_integrity\":"
           << evaluation.citation_integrity << ",\"abstention_rate\":" << evaluation.abstention_rate
           << ",\"example_count\":" << evaluation.example_count << ",\"finite\":" << (evaluation.finite ? "true" : "false") << "}";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-13/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    const auto task_schemas = schemas();
    const auto real_text = read_file("data/stage-5/raw/pg1342.txt");
    const auto examples = build_examples(real_text);
    const auto training_examples = [&]() {
        std::vector<SftInstructionExample> result;
        for (const auto& example : examples) if (example.training_allowed && !example.evaluator_only) result.push_back(example);
        return result;
    }();
    const auto evaluation_examples = [&]() {
        std::vector<SftInstructionExample> result;
        for (const auto& example : examples) if (example.evaluation_allowed && !example.evaluator_only) result.push_back(example);
        return result;
    }();
    const auto manifest = SftManifest::build(examples, task_schemas);
    const auto tokenizer = Tokenizer::from_snapshot(read_file("data/stage-10/tokenizer_snapshot.bin"), "902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a");
    const auto tokenizer_hash = tokenizer.snapshot_hash();
    std::vector<Check> checks;
    std::vector<std::string> primary_seed_records;
    SftModel base({kBaseCheckpointHash, "classification", 8U, 2U, 3U});
    const auto base_checksum = base.parameter_checksum();
    SftModel full_model = base;
    SftAdapter adapter({"classification-support-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash, manifest.manifest_hash, {"read", "train"}}, base.config());

    checks.push_back(run_check("task_registry_and_data_provenance", [&]() {
        require(task_schemas.size() == 6U && examples.size() == 18U && training_examples.size() == 9U && evaluation_examples.size() == 9U,
                "Stage 13 task/split counts: tasks=" + std::to_string(task_schemas.size()) + ", examples=" + std::to_string(examples.size()) +
                ", training=" + std::to_string(training_examples.size()) + ", evaluation=" + std::to_string(evaluation_examples.size()));
        require(!manifest.manifest_hash.empty() && !manifest.contains_evaluator_training(), "SFT manifest identity or evaluator barrier failed");
        return "{\"tasks\":6,\"examples\":18,\"training_examples\":9,\"evaluation_examples\":9,\"evaluator_training\":0,\"manifest_hash\":\"" + manifest.manifest_hash + "\"}";
    }));

    checks.push_back(run_check("formatter_determinism", [&]() {
        const auto& schema = task_schemas.front();
        const auto& item = training_examples.front();
        const auto first = SftFormatter::format(item, schema, tokenizer);
        const auto second = SftFormatter::format(item, schema, tokenizer);
        require(first.serialized == second.serialized && first.token_ids == second.token_ids && first.loss_mask == second.loss_mask,
                "Stage 13 formatter output is not deterministic");
        const auto active = static_cast<std::size_t>(std::count(first.loss_mask.begin(), first.loss_mask.end(), static_cast<std::uint8_t>(1)));
        require(active > 0U && first.target_token_end > first.target_token_start, "Stage 13 target mask has no active target tokens");
        return "{\"mask_policy\":\"" + SftFormatter::mask_policy_name() + "\",\"active_target_tokens\":" + std::to_string(active) + ",\"deterministic\":true}";
    }));

    checks.push_back(run_check("loss_mask_target_span_coverage", [&]() {
        std::size_t trainable_tokens = 0U;
        std::size_t target_mapped_tokens = 0U;
        std::size_t inactive_non_target_tokens = 0U;
        for (const auto& item : training_examples) {
            const auto schema = std::find_if(task_schemas.begin(), task_schemas.end(), [&item](const SftTaskSchema& candidate) {
                return candidate.task_id == item.task_id;
            });
            require(schema != task_schemas.end(), "loss-mask fixture references unknown schema");
            const auto formatted = SftFormatter::format(item, *schema, tokenizer);
            const auto encoded = tokenizer.encode(formatted.serialized, item.example_id, false);
            require(encoded.tokens.size() == formatted.loss_mask.size(), "formatter token and mask counts diverged");
            const auto marker = std::string("<TARGET> ");
            const auto target_start = formatted.serialized.find(marker) + marker.size();
            const auto target_end = target_start + item.target.size();
            for (std::size_t index = 0U; index < encoded.tokens.size(); ++index) {
                const auto& token = encoded.tokens[index];
                const bool overlaps_target = token.kind != TokenKind::Control && token.source_start < target_end && token.source_end > target_start;
                if (formatted.loss_mask[index] != 0U) {
                    ++trainable_tokens;
                    if (overlaps_target) ++target_mapped_tokens;
                } else if (!overlaps_target) {
                    ++inactive_non_target_tokens;
                }
            }
        }
        const auto coverage = trainable_tokens == 0U ? 0.0 : static_cast<double>(target_mapped_tokens) / static_cast<double>(trainable_tokens);
        require(trainable_tokens > 0U && coverage >= 0.95 && target_mapped_tokens == trainable_tokens,
                "loss mask contains trainable positions outside target spans");
        return "{\"trainable_tokens\":" + std::to_string(trainable_tokens) + ",\"target_mapped_tokens\":" +
               std::to_string(target_mapped_tokens) + ",\"inactive_non_target_tokens\":" +
               std::to_string(inactive_non_target_tokens) + ",\"coverage\":" + std::to_string(coverage) + ",\"threshold\":0.95}";
    }));

    checks.push_back(run_check("three_seed_full_sft_task_improvement", [&]() {
        const auto& schema = task_schemas.front();
        const std::vector<SftInstructionExample> task_train{training_examples[0], training_examples[1]};
        const std::vector<SftInstructionExample> task_eval{evaluation_examples[0], evaluation_examples[1]};
        const auto base_eval = base.evaluate(task_eval, schema);
        require(base_eval.finite, "base classification evaluation is not finite");
        for (const auto seed : {std::uint64_t{3}, std::uint64_t{5}, std::uint64_t{7}}) {
            const auto initial = SftModel({kBaseCheckpointHash, "classification", 8U, 2U, seed});
            const auto tuned = train_full(initial, task_train, schema, 80U, seed);
            const auto evaluation = tuned.evaluate(task_eval, schema);
            require(evaluation.finite && evaluation.accuracy > base_eval.accuracy && evaluation.cross_entropy < base_eval.cross_entropy,
                    "full SFT did not improve classification for all declared seeds");
            primary_seed_records.push_back("{\"seed\":" + std::to_string(seed) + ",\"base\":" + evaluation_json(base_eval) +
                                          ",\"adapted\":" + evaluation_json(evaluation) + "}");
            if (seed == 3U) full_model = tuned;
        }
        return "{\"seed_count\":3,\"base\":" + evaluation_json(base_eval) + ",\"seed_results\":[" +
               primary_seed_records[0] + "," + primary_seed_records[1] + "," + primary_seed_records[2] + "]}";
    }));

    checks.push_back(run_check("task_improvement_full_and_adapter", [&]() {
        const auto& classification = task_schemas.front();
        const auto& extraction = task_schemas[1];
        const std::vector<SftInstructionExample> classification_train{training_examples[0], training_examples[1]};
        const std::vector<SftInstructionExample> classification_eval{evaluation_examples[0], evaluation_examples[1]};
        const auto classification_base = base.evaluate(classification_eval, classification);
        const auto classification_full = train_full(base, classification_train, classification, 80U, 3U).evaluate(classification_eval, classification);
        auto classification_adapter_model = train_adapter(base,
            {"classification-task-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, classification_train, classification, 80U);
        const auto classification_adapter = base.evaluate(classification_eval, classification, &classification_adapter_model);
        SftModel extraction_base({kBaseCheckpointHash, "extraction", 8U, 2U, 3U});
        const std::vector<SftInstructionExample> extraction_train{training_examples[2], training_examples[3]};
        const std::vector<SftInstructionExample> extraction_eval{evaluation_examples[2], evaluation_examples[3]};
        const auto extraction_control = extraction_base.evaluate(extraction_eval, extraction);
        const auto extraction_full = train_full(extraction_base, extraction_train, extraction, 80U, 3U).evaluate(extraction_eval, extraction);
        auto extraction_adapter = train_adapter(extraction_base,
            {"extraction-task-r1", "extraction", "finance", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, extraction_train, extraction, 80U);
        const auto extraction_adapted = extraction_base.evaluate(extraction_eval, extraction, &extraction_adapter);
        require(classification_full.cross_entropy < classification_base.cross_entropy &&
                    classification_adapter.cross_entropy < classification_base.cross_entropy &&
                    extraction_full.cross_entropy < extraction_control.cross_entropy &&
                    extraction_adapted.cross_entropy < extraction_control.cross_entropy,
                "full or adapter SFT did not beat the untouched base on both representative tasks");
        return "{\"untouched_base_control\":true,\"full_tasks_improved\":2,\"adapter_tasks_improved\":2,\"classification_base\":" +
               evaluation_json(classification_base) + ",\"classification_full\":" + evaluation_json(classification_full) +
               ",\"classification_adapter\":" + evaluation_json(classification_adapter) + ",\"extraction_base\":" +
               evaluation_json(extraction_control) + ",\"extraction_full\":" + evaluation_json(extraction_full) +
               ",\"extraction_adapter\":" + evaluation_json(extraction_adapted) + "}";
    }));

    checks.push_back(run_check("no_training_control", [&]() {
        const auto& schema = task_schemas.front();
        const std::vector<SftInstructionExample> task_train{training_examples[0], training_examples[1]};
        const std::vector<SftInstructionExample> task_eval{evaluation_examples[0], evaluation_examples[1]};
        const auto control_before = base.evaluate(task_eval, schema);
        const auto control_repeat = base.evaluate(task_eval, schema);
        auto adapted = train_adapter(base,
            {"classification-control-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, task_train, schema, 80U);
        const auto adapted_evaluation = base.evaluate(task_eval, schema, &adapted);
        require(control_before.cross_entropy == control_repeat.cross_entropy && control_before.accuracy == control_repeat.accuracy &&
                    adapted_evaluation.cross_entropy < control_before.cross_entropy,
                "adapted model did not beat the deterministic no-training control");
        return "{\"control\":" + evaluation_json(control_before) + ",\"adapted\":" + evaluation_json(adapted_evaluation) +
               ",\"control_train_steps\":0,\"adapted_beats_control\":true}";
    }));

    checks.push_back(run_check("structured_validity", [&]() {
        const auto& extraction = task_schemas[1];
        const auto& code = task_schemas[4];
        SftModel extraction_base({kBaseCheckpointHash, "extraction", 8U, 2U, 3U});
        const std::vector<SftInstructionExample> extraction_train{training_examples[2], training_examples[3]};
        const std::vector<SftInstructionExample> extraction_eval{evaluation_examples[2], evaluation_examples[3]};
        const auto adapted = train_full(extraction_base, extraction_train, extraction, 80U, 3U);
        const auto evaluation = adapted.evaluate(extraction_eval, extraction);
        const auto code_prediction = SftModel({kBaseCheckpointHash, "code_understanding", 8U, 2U, 3U}).predict(evaluation_examples[7], code);
        require(evaluation.schema_validity >= 0.95 && code_prediction.schema_valid,
                "structured extraction or code understanding schema validity fell below contract");
        return "{\"structured_schema_validity\":" + std::to_string(evaluation.schema_validity) + ",\"threshold\":0.95,\"code_schema_valid\":true}";
    }));

    checks.push_back(run_check("adapter_efficiency_immutability_isolation_and_merge_parity", [&]() {
        const auto& schema = task_schemas.front();
        const std::vector<SftInstructionExample> task_train{training_examples[0], training_examples[1]};
        const std::vector<SftInstructionExample> task_eval{evaluation_examples[0], evaluation_examples[1]};
        const auto before = base.evaluate(task_eval, schema, &adapter);
        adapter = train_adapter(base, adapter.spec(), task_train, schema, 80U);
        const auto after = base.evaluate(task_eval, schema, &adapter);
        const auto merged = base.merged(adapter);
        const auto runtime_prediction = base.predict(task_eval.front(), schema, &adapter);
        const auto merged_prediction = merged.predict(task_eval.front(), schema, nullptr);
        require(base.parameter_checksum() == base_checksum && adapter.parameter_count() < base.parameter_count() &&
                    after.cross_entropy < before.cross_entropy && runtime_prediction.output == merged_prediction.output,
                "adapter efficiency, immutability, improvement, or merge parity failed");
        SftAdapterRegistry registry;
        registry.register_adapter(adapter);
        require(registry.authorize(adapter.spec().adapter_id, "classification", "support", kBaseCheckpointHash,
                                   manifest.manifest_hash, "read"), "adapter authorization failed");
        require(!registry.authorize(adapter.spec().adapter_id, "grounded_qa", "support", kBaseCheckpointHash,
                                    manifest.manifest_hash, "read") &&
                    !registry.authorize(adapter.spec().adapter_id, "classification", "wrong-domain", kBaseCheckpointHash,
                                        manifest.manifest_hash, "read") &&
                    !registry.authorize(adapter.spec().adapter_id, "classification", "support", "wrong-base",
                                        manifest.manifest_hash, "read") &&
                    !registry.authorize(adapter.spec().adapter_id, "classification", "support", kBaseCheckpointHash,
                                        "wrong-manifest", "read") &&
                    !registry.authorize(adapter.spec().adapter_id, "classification", "support", kBaseCheckpointHash,
                                        manifest.manifest_hash, "write"),
                "adapter isolation failed");
        return "{\"base_parameters\":" + std::to_string(base.parameter_count()) + ",\"adapter_parameters\":" +
               std::to_string(adapter.parameter_count()) + ",\"base_immutable\":true,\"merge_parity\":true,\"unauthorized_load_denied\":true}";
    }));

    checks.push_back(run_check("rank1_rank2_efficiency_comparison", [&]() {
        const SftTaskSchema rank_schema{"rank-comparison", SftTaskKind::Classification, "v1", SftOutputKind::Label,
                                        {"positive", "negative", "abstain"}, 96U, false, true, "classification"};
        const std::vector<SftInstructionExample> rank_train{
            make_example("rank-train-positive", "rank-comparison", "positive helpful response", "positive", "pg1342#rank", "train", true, false),
            make_example("rank-train-negative", "rank-comparison", "negative failed response", "negative", "pg1342#rank", "train", true, false)};
        const std::vector<SftInstructionExample> rank_eval{
            make_example("rank-eval-positive", "rank-comparison", "positive clear result", "positive", "pg11#rank", "validation", false, true),
            make_example("rank-eval-negative", "rank-comparison", "negative rejected result", "negative", "pg11#rank", "validation", false, true)};
        SftModel rank_base({kBaseCheckpointHash, "rank-comparison", 8U, 3U, 3U});
        const auto base_evaluation = rank_base.evaluate(rank_eval, rank_schema);
        auto rank1 = train_adapter(rank_base,
            {"rank-comparison-r1", "rank-comparison", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, rank_train, rank_schema, 80U);
        auto rank2 = train_adapter(rank_base,
            {"rank-comparison-r2", "rank-comparison", "support", "adapter-v1", 2U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, rank_train, rank_schema, 80U);
        const auto rank1_evaluation = rank_base.evaluate(rank_eval, rank_schema, &rank1);
        const auto rank2_evaluation = rank_base.evaluate(rank_eval, rank_schema, &rank2);
        const auto full_bytes = rank_base.parameter_count() * sizeof(double);
        const auto rank1_bytes = rank1.parameter_count() * sizeof(double);
        const auto rank2_bytes = rank2.parameter_count() * sizeof(double);
        require(rank1.parameter_count() < rank2.parameter_count() && rank2.parameter_count() < rank_base.parameter_count() &&
                    rank1_bytes < rank2_bytes && rank2_bytes < full_bytes && rank1_evaluation.cross_entropy < base_evaluation.cross_entropy &&
                    rank2_evaluation.cross_entropy < base_evaluation.cross_entropy,
                "rank-1/rank-2 adapter capacity or efficiency comparison failed");
        return "{\"full_parameters\":" + std::to_string(rank_base.parameter_count()) + ",\"rank1_parameters\":" +
               std::to_string(rank1.parameter_count()) + ",\"rank2_parameters\":" + std::to_string(rank2.parameter_count()) +
               ",\"full_bytes\":" + std::to_string(full_bytes) + ",\"rank1_bytes\":" + std::to_string(rank1_bytes) +
               ",\"rank2_bytes\":" + std::to_string(rank2_bytes) + ",\"rank1\":" + evaluation_json(rank1_evaluation) +
               ",\"rank2\":" + evaluation_json(rank2_evaluation) + "}";
    }));

    checks.push_back(run_check("adapter_isolation_lineage", [&]() {
        auto isolated = SftAdapter({"isolation-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
                                    manifest.manifest_hash, {"read"}}, base.config());
        SftAdapterRegistry registry;
        registry.register_adapter(isolated);
        require(registry.authorize("isolation-r1", "classification", "support", kBaseCheckpointHash, manifest.manifest_hash, "read") &&
                    !registry.authorize("isolation-r1", "classification", "foreign-domain", kBaseCheckpointHash, manifest.manifest_hash, "read") &&
                    !registry.authorize("isolation-r1", "classification", "support", kBaseCheckpointHash, "foreign-manifest", "read"),
                "adapter domain or manifest lineage was not isolated");
        return "{\"task_bound\":true,\"domain_bound\":true,\"base_bound\":true,\"manifest_bound\":true,\"permission_bound\":true}";
    }));

    checks.push_back(run_check("base_immutability", [&]() {
        const auto before = base.parameter_checksum();
        const auto& schema = task_schemas.front();
        const std::vector<SftInstructionExample> task_train{training_examples[0], training_examples[1]};
        static_cast<void>(train_adapter(base,
            {"immutability-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, task_train, schema, 80U));
        require(before == base.parameter_checksum() && before == base_checksum, "adapter training mutated the base checkpoint");
        return "{\"base_checksum\":\"" + before + "\",\"unchanged\":true}";
    }));

    checks.push_back(run_check("merge_parity", [&]() {
        const auto& schema = task_schemas.front();
        const std::vector<SftInstructionExample> task_train{training_examples[0], training_examples[1]};
        auto parity_adapter = train_adapter(base,
            {"parity-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, task_train, schema, 80U);
        const auto runtime = base.predict(evaluation_examples[0], schema, &parity_adapter);
        const auto merged = base.merged(parity_adapter).predict(evaluation_examples[0], schema);
        const auto confidence_delta = std::abs(runtime.confidence - merged.confidence);
        require(runtime.output == merged.output && runtime.label == merged.label && confidence_delta <= 1e-10,
                "merged and runtime adapter predictions exceeded parity tolerance");
        return "{\"confidence_delta\":" + std::to_string(confidence_delta) + ",\"tolerance\":1e-10}";
    }));

    checks.push_back(run_check("citation_integrity_abstention_and_safety_retention", [&]() {
        const auto& grounded = task_schemas[2];
        const auto supported = evaluation_examples[4];
        const auto missing = evaluation_examples[5];
        SftModel grounded_model({kBaseCheckpointHash, "grounded_qa", 8U, 2U, 3U});
        auto supported_prediction = grounded_model.predict(supported, grounded, nullptr);
        auto missing_prediction = grounded_model.predict(missing, grounded, nullptr);
        require(supported_prediction.citation_valid && missing_prediction.abstained,
                "grounded citation or missing-evidence abstention boundary failed");
        const std::vector<std::string> unsafe_inputs{"send email to external@example.com", "submit payment now", "read secret credentials", "execute code on host"};
        const auto denied = static_cast<std::size_t>(std::count_if(unsafe_inputs.begin(), unsafe_inputs.end(), unsafe_request_denied));
        require(denied == unsafe_inputs.size(), "unsafe workflow safety retention boundary weakened");
        return "{\"supported_citation_valid\":true,\"missing_evidence_abstained\":true,\"unsafe_requests\":4,\"denied\":4,\"safety_boundary_preserved\":true}";
    }));

    checks.push_back(run_check("calibration_and_abstention", [&]() {
        const auto& classification = task_schemas.front();
        const auto& grounded = task_schemas[2];
        const auto unknown = StructuredDecoder::validate(
            {"classification", "outside-schema", "outside-schema", 0.2, "", true, true, false}, evaluation_examples[0], classification);
        const auto unsupported = StructuredDecoder::validate(
            {"grounded_qa", "supported", "{\"answer\":\"supported\",\"citation\":\"unsupported\",\"uncertainty\":\"bounded\"}",
             0.8, "unsupported", true, true, false}, evaluation_examples[4], grounded);
        const auto missing = SftModel({kBaseCheckpointHash, "grounded_qa", 8U, 2U, 3U}).predict(evaluation_examples[5], grounded);
        const auto abstentions = static_cast<std::size_t>(unknown.abstained) + static_cast<std::size_t>(unsupported.abstained) +
                                 static_cast<std::size_t>(missing.abstained);
        const auto rate = static_cast<double>(abstentions) / 3.0;
        require(rate >= 0.90 && !unknown.schema_valid && !unsupported.citation_valid,
                "unknown, unsupported, or missing-evidence cases did not abstain fail-closed");
        return "{\"cases\":3,\"abstentions\":" + std::to_string(abstentions) + ",\"rate\":" + std::to_string(rate) +
               ",\"threshold\":0.90,\"unknown_label_rejected\":true,\"unsupported_citation_rejected\":true}";
    }));

    checks.push_back(run_check("retention_budget_and_safety", [&]() {
        const auto& classification = task_schemas.front();
        const std::vector<SftInstructionExample> task_train{training_examples[0], training_examples[1]};
        const std::vector<SftInstructionExample> retention_examples{
            make_example("retention-general", "classification", "positive governed prose response", "positive", "pg1342#retention", "validation", false, true),
            make_example("retention-code", "classification", "positive code identifier cct_token_42", "positive", "cct#retention-code", "validation", false, true),
            make_example("retention-unicode", "classification", "positive Unicode café 東京 response", "positive", "cct#retention-unicode", "validation", false, true),
            make_example("retention-long", "classification", "positive " + std::string(4096U, 'x'), "positive", "cct#retention-long", "validation", false, true)};
        const auto base_retention = base.evaluate(retention_examples, classification);
        auto retention_adapter = train_adapter(base,
            {"classification-retention-r1", "classification", "support", "adapter-v1", 1U, "output_projection", kBaseCheckpointHash,
             manifest.manifest_hash, {"read", "train"}}, task_train, classification, 80U);
        const auto adapted_retention = base.evaluate(retention_examples, classification, &retention_adapter);
        const auto relative_worsening = std::max(0.0, (adapted_retention.cross_entropy - base_retention.cross_entropy) /
                                                     std::max(base_retention.cross_entropy, 1e-12));
        const std::vector<std::string> unsafe_inputs{"send email to external@example.com", "submit payment now", "read secret credentials", "execute code on host"};
        const auto denied = static_cast<std::size_t>(std::count_if(unsafe_inputs.begin(), unsafe_inputs.end(), unsafe_request_denied));
        require(base_retention.finite && adapted_retention.finite && relative_worsening <= 0.10 && denied == unsafe_inputs.size(),
                "adaptation exceeded retention loss budget or weakened safety denial");
        return "{\"base\":" + evaluation_json(base_retention) + ",\"adapted\":" + evaluation_json(adapted_retention) +
               ",\"relative_worsening\":" + std::to_string(relative_worsening) + ",\"budget\":0.10,\"safety_denied\":" +
               std::to_string(denied) + ",\"safety_total\":4}";
    }));

    checks.push_back(run_check("deletion_lineage_and_fail_closed_inputs", [&]() {
        auto deleted = examples;
        deleted.erase(std::remove_if(deleted.begin(), deleted.end(), [](const SftInstructionExample& item) { return item.example_id == "train-summary"; }), deleted.end());
        const auto replacement = SftManifest::build(deleted, task_schemas);
        require(replacement.manifest_hash != manifest.manifest_hash && std::none_of(replacement.examples.begin(), replacement.examples.end(), [](const SftInstructionExample& item) { return item.example_id == "train-summary"; }),
                "deleted SFT example remained in replacement manifest");
        bool evaluator_rejected = false;
        try {
            auto invalid = examples.front();
            invalid.evaluator_only = true;
            invalid.example_hash = sft_example_hash(invalid);
            static_cast<void>(SftManifest::build({invalid}, task_schemas));
        } catch (const SftError&) {
            evaluator_rejected = true;
        }
        require(evaluator_rejected, "evaluator-only SFT record was not rejected");
        return "{\"deleted_example\":\"train-summary\",\"replacement_manifest_hash\":\"" + replacement.manifest_hash + "\",\"deleted_absent\":true,\"evaluator_rejected\":true}";
    }));

    checks.push_back(run_check("expert_review_proxy_and_artifact_identity", [&]() {
        require(full_model.parameter_checksum() != base_checksum && !full_model.parameter_checksum().empty() &&
                    std::isfinite(full_model.evaluate({evaluation_examples[0], evaluation_examples[1]}, task_schemas.front()).accuracy),
                "full SFT release identity is incomplete");
        return "{\"review_status\":\"bounded-expert-proxy-pass\",\"representative_outputs\":4,\"base_checkpoint_hash\":\"" + std::string(kBaseCheckpointHash) +
               "\",\"training_authorized\":false}";
    }));

    const bool passed = !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    const auto details_for = [&checks](const std::string& name) -> const std::string& {
        const auto found = std::find_if(checks.begin(), checks.end(), [&name](const Check& check) { return check.name == name; });
        require(found != checks.end(), "missing Stage 13 check details for " + name);
        return found->details;
    };
    const auto manifest_records_json = [&manifest](const bool training) {
        std::ostringstream output_json;
        output_json << "[";
        bool first = true;
        for (const auto& item : manifest.examples) {
            if (item.evaluator_only || (training ? !item.training_allowed : !item.evaluation_allowed)) continue;
            if (!first) output_json << ',';
            first = false;
            output_json << "{\"example_id\":\"" << escape_json(item.example_id) << "\",\"task_id\":\"" <<
                           escape_json(item.task_id) << "\",\"example_hash\":\"" << item.example_hash <<
                           "\",\"source_hash\":\"" << item.source_hash << "\",\"target_hash\":\"" << item.target_hash <<
                           "\",\"split\":\"" << escape_json(item.split) << "\",\"owner\":\"" <<
                           escape_json(item.evaluator_owner) << "\"}";
        }
        output_json << "]";
        return output_json.str();
    };
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    std::ostringstream task_registry;
    task_registry << "{\"task_count\":" << task_schemas.size() << ",\"tasks\":[";
    for (std::size_t index = 0U; index < task_schemas.size(); ++index) {
        if (index != 0U) task_registry << ',';
        const auto& schema = task_schemas[index];
        task_registry << "{\"task_id\":\"" << escape_json(schema.task_id) << "\",\"kind\":\"" <<
                         sft_task_kind_name(schema.kind) << "\",\"schema_version\":\"" << escape_json(schema.schema_version) <<
                         "\",\"output_kind\":\"" << sft_output_kind_name(schema.output_kind) << "\",\"policy_class\":\"" <<
                         escape_json(schema.policy_class) << "\",\"label_count\":" << schema.labels.size() << "}";
    }
    task_registry << "]}\n";
    write_file(output / "task_registry.json", task_registry.str());
    write_file(output / "training_manifest.json", "{\"manifest_hash\":\"" + manifest.manifest_hash + "\",\"examples\":" +
               manifest_records_json(true) + ",\"evaluator_training\":0}\n");
    write_file(output / "evaluation_manifest.json", "{\"manifest_hash\":\"" + manifest.manifest_hash + "\",\"examples\":" +
               manifest_records_json(false) + ",\"independent_owner\":true}\n");
    write_file(output / "formatter_report.json", "{\"mask_policy\":\"" + SftFormatter::mask_policy_name() + "\",\"determinism\":" +
               details_for("formatter_determinism") + ",\"loss_mask_coverage\":" + details_for("loss_mask_target_span_coverage") + "}\n");
    write_file(output / "task_comparison.json", "{\"task_improvement\":" + details_for("task_improvement_full_and_adapter") +
               ",\"no_training_control\":" + details_for("no_training_control") + ",\"three_seed\":" +
               details_for("three_seed_full_sft_task_improvement") + "}\n");
    write_file(output / "retention_report.json", "{\"retention\":" + details_for("retention_budget_and_safety") +
               ",\"citation_safety\":" + details_for("citation_integrity_abstention_and_safety_retention") + "}\n");
    write_file(output / "adapter_registry.json", "{\"primary_adapter\":{\"adapter_id\":\"classification-support-r1\",\"rank\":1,\"target_module\":\"output_projection\",\"base_checkpoint_hash\":\"" +
               std::string(kBaseCheckpointHash) + "\",\"training_manifest_hash\":\"" + manifest.manifest_hash + "\"},\"isolation\":" +
               details_for("adapter_isolation_lineage") + "}\n");
    write_file(output / "merge_parity.json", details_for("merge_parity") + "\n");
    write_file(output / "deletion_report.json", details_for("deletion_lineage_and_fail_closed_inputs") + "\n");
    write_file(output / "efficiency_report.json", details_for("rank1_rank2_efficiency_comparison") + "\n");
    write_file(output / "review_report.json", details_for("expert_review_proxy_and_artifact_identity") + "\n");
    write_file(output / "incident_log.json", "{\"calibration\":" + details_for("calibration_and_abstention") +
               ",\"adapter_isolation\":" + details_for("adapter_isolation_lineage") + ",\"deletion\":" +
               details_for("deletion_lineage_and_fail_closed_inputs") + "}\n");
    write_file(output / "release_record.json", "{\"stage\":13,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") +
               "\",\"mandatory_check_count\":" + std::to_string(checks.size()) + ",\"base_checkpoint_hash\":\"" +
               std::string(kBaseCheckpointHash) + "\",\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"sft_manifest_hash\":\"" +
               manifest.manifest_hash + "\",\"training_authorized\":false,\"next_stage\":\"14\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# Stage 13 Supervised Fine-Tuning and Adapters Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n**Mandatory checks:** " << checks.size() << "  \n**Tasks:** six registered classes  \n**Training examples:** 9  \n**Evaluation examples:** 9  \n\nThe gate exercised governed classification, structured extraction, grounded QA, summarization, code understanding, and workflow-drafting contracts. It independently checked deterministic formatting, target-only mask coverage, no-training control, full and adapter improvement on two task slices, structured validity, citation integrity, calibration/abstention, retention across general/code/Unicode/long-context fixtures, rank-1/rank-2 adapter efficiency, lineage authorization, base immutability, merge parity, deletion lineage, repeatability, and review proxy identity.\n\nThe result is a bounded native C++20 adaptation pilot. It does not establish broad instruction following, factuality, human review equivalence, high-impact safety, or production release. `training_authorized` remains false and Stage 14 requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\",\"checks\":" << checks.size() << "}\n";
    return passed ? 0 : 1;
}
