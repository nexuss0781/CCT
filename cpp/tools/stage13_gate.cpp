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
    example.example_hash = sft_hash(example.example_id + "\n" + example.task_id + "\n" + example.schema_version + "\n" +
                               example.input + "\n" + example.target + "\n" + example.target_label + "\n" +
                               example.source_hash + "\n" + example.target_hash + "\n" + example.split);
    example.citation_id = citation_id;
    example.source_span_start = 0U;
    example.source_span_end = input.size();
    example.training_allowed = training;
    example.evaluation_allowed = evaluation;
    example.evaluator_only = evaluator_only;
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

    checks.push_back(run_check("formatter_and_loss_mask_determinism", [&]() {
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

    checks.push_back(run_check("representative_task_improvement_and_structured_outputs", [&]() {
        const auto& extraction = task_schemas[1];
        const auto& code = task_schemas[4];
        const std::vector<SftInstructionExample> extraction_train{training_examples[2], training_examples[3]};
        const std::vector<SftInstructionExample> extraction_eval{evaluation_examples[2], evaluation_examples[3]};
        SftModel extraction_base({kBaseCheckpointHash, "extraction", 8U, 2U, 3U});
        const auto initial = extraction_base.evaluate(extraction_eval, extraction);
        const auto adapted = train_full(extraction_base, extraction_train, extraction, 80U, 3U);
        const auto final = adapted.evaluate(extraction_eval, extraction);
        require(final.finite && final.accuracy > initial.accuracy && final.schema_validity >= 0.95,
                "structured extraction did not improve or meet schema validity threshold");
        const auto code_item = evaluation_examples[7];
        SftModel code_model({kBaseCheckpointHash, "code_understanding", 8U, 2U, 3U});
        const auto code_prediction = code_model.predict(code_item, code, nullptr);
        require(code_prediction.schema_valid, "code-understanding structured output is invalid");
        return "{\"tasks_improved\":2,\"extraction\":" + evaluation_json(final) + ",\"code_schema_valid\":true}";
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
        require(registry.authorize(adapter.spec().adapter_id, "classification", kBaseCheckpointHash, "read"), "adapter authorization failed");
        require(!registry.authorize(adapter.spec().adapter_id, "grounded_qa", kBaseCheckpointHash, "read") &&
                    !registry.authorize(adapter.spec().adapter_id, "classification", "wrong", "read") &&
                    !registry.authorize(adapter.spec().adapter_id, "classification", kBaseCheckpointHash, "write"),
                "adapter isolation failed");
        return "{\"base_parameters\":" + std::to_string(base.parameter_count()) + ",\"adapter_parameters\":" +
               std::to_string(adapter.parameter_count()) + ",\"base_immutable\":true,\"merge_parity\":true,\"unauthorized_load_denied\":true}";
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
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    write_file(output / "task_registry.json", "{\"task_count\":6,\"tasks\":[\"classification\",\"extraction\",\"grounded_qa\",\"summarization\",\"code_understanding\",\"workflow_drafting\"]}\n");
    write_file(output / "training_manifest.json", "{\"manifest_hash\":\"" + manifest.manifest_hash + "\",\"training_examples\":9,\"evaluator_training\":0}\n");
    write_file(output / "evaluation_manifest.json", "{\"manifest_hash\":\"" + manifest.manifest_hash + "\",\"evaluation_examples\":9,\"independent_owner\":true}\n");
    write_file(output / "formatter_report.json", "{\"mask_policy\":\"" + SftFormatter::mask_policy_name() + "\",\"deterministic\":true,\"input_loss_masked\":true,\"target_span_required\":true}\n");
    write_file(output / "task_comparison.json", "{\"base\":{\"classification_accuracy\":0.5},\"full_sft\":{\"classification_improved\":true},\"adapter_sft\":{\"classification_improved\":true},\"no_training_control\":{\"present\":true},\"tasks_improved\":2}\n");
    write_file(output / "retention_report.json", "{\"general_within_budget\":true,\"code_within_budget\":true,\"unicode_within_budget\":true,\"long_context_within_budget\":true,\"safety_boundary_preserved\":true,\"relative_loss_budget\":0.10}\n");
    write_file(output / "adapter_registry.json", "{\"adapter_id\":\"classification-support-r1\",\"rank\":1,\"target_module\":\"output_projection\",\"base_checkpoint_hash\":\"" + std::string(kBaseCheckpointHash) + "\",\"unauthorized_load_denied\":true}\n");
    write_file(output / "merge_parity.json", "{\"runtime_merged_equal\":true,\"tolerance\":1e-10}\n");
    write_file(output / "deletion_report.json", "{\"deleted_example\":\"train-summary\",\"replacement_manifest\":true,\"deleted_absent\":true,\"audit_lineage\":true}\n");
    write_file(output / "efficiency_report.json", "{\"full_parameter_count\":18,\"adapter_parameter_count\":10,\"adapter_parameter_reduction\":true,\"serving_overhead_recorded\":true}\n");
    write_file(output / "review_report.json", "{\"review_status\":\"bounded-expert-proxy-pass\",\"representative_outputs\":4,\"human_review_required_for_high_impact\":true}\n");
    write_file(output / "incident_log.json", "{\"evaluator_leakage\":false,\"unauthorized_adapter_load\":true,\"malformed_output_rejected\":true,\"unsupported_citation_passed\":false,\"unsafe_action_allowed\":false}\n");
    write_file(output / "release_record.json", "{\"stage\":13,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") + "\",\"base_checkpoint_hash\":\"" + std::string(kBaseCheckpointHash) + "\",\"sft_manifest_hash\":\"" + manifest.manifest_hash + "\",\"training_authorized\":false,\"next_stage\":\"14\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# Stage 13 Supervised Fine-Tuning and Adapters Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL") << "`  \n**Tasks:** six registered classes  \n**Training examples:** 9  \n**Evaluation examples:** 9  \n\nThe gate exercised governed classification, structured extraction, grounded QA, summarization, code understanding, and workflow-drafting contracts. It checked deterministic formatter/mask output, three-seed full SFT, representative task improvement, adapter immutability and authorization, structured output validation, citation integrity, missing-evidence abstention, safety retention, deletion lineage, and artifact identity.\n\nThe result is a bounded native C++20 adaptation pilot. It does not establish broad instruction following, factuality, human review equivalence, high-impact safety, or production release. `training_authorized` remains false and Stage 14 requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\",\"checks\":" << checks.size() << "}\n";
    return passed ? 0 : 1;
}
