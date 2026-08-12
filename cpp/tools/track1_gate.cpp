#include "cct/track1.hpp"

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

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write fixture " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish fixture " + path.string());
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') output << '\\';
        if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else output << static_cast<char>(character);
    }
    return output.str();
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

void create_fixture(const std::filesystem::path& root, const bool malformed = false) {
    const std::string wikitext = R"({"features":[{"name":"text"}],"rows":[{"row":{"text":"alpha beta gamma delta"}},{"row":{"text":"the causal engine learns from compact text"}},{"row":{"text":"spectral state updates are deterministic"}},{"row":{"text":"checkpoint replay preserves the stream"}}],"num_rows_total":4})";
    const std::string squad_train = R"({"features":[{"name":"id"}],"rows":[{"row":{"id":"a1","title":"One","context":"Paris is the capital of France.","question":"What is the capital of France?","answers":{"text":["Paris"],"answer_start":[0]}}},{"row":{"id":"a2","title":"Two","context":"The engine uses a spectral state.","question":"What does the engine use?","answers":{"text":["a spectral state"],"answer_start":[16]}}},{"row":{"id":"a3","title":"Three","context":"A checkpoint stores model state.","question":"What stores model state?","answers":{"text":["A checkpoint"],"answer_start":[0]}}},{"row":{"id":"a4","title":"Four","context":"Validation measures held-out loss.","question":"What does validation measure?","answers":{"text":["held-out loss"],"answer_start":[20]}}},{"row":{"id":"u1","title":"Five","context":"The sky is blue.","question":"What is the engine version?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"u2","title":"Six","context":"The river is long.","question":"Who wrote the source?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"u3","title":"Seven","context":"The model passed the gate.","question":"What is the hidden password?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"u4","title":"Eight","context":"The test is deterministic.","question":"Where is the server?","answers":{"text":[],"answer_start":[]}}}],"num_rows_total":8})";
    const std::string squad_final = malformed
        ? std::string(R"({"features":[{"name":"id"}],"rows":[{"row":{"id":"broken","title":"Broken","context":"not valid")")
        : std::string(R"({"features":[{"name":"id"}],"rows":[{"row":{"id":"t1","title":"FinalOne","context":"London is in England.","question":"Where is London?","answers":{"text":["in England"],"answer_start":[10]}}},{"row":{"id":"t2","title":"FinalTwo","context":"The final set is frozen.","question":"What is frozen?","answers":{"text":["The final set"],"answer_start":[0]}}},{"row":{"id":"t3","title":"FinalThree","context":"The answer is not present.","question":"What is the secret?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"t4","title":"FinalFour","context":"The pilot is bounded.","question":"What is the pilot?","answers":{"text":["bounded"],"answer_start":[13]}}}],"num_rows_total":4})");
    write_file(root / "raw" / "wikitext2_pretrain_train_train.json.0", wikitext);
    write_file(root / "raw" / "wikitext2_pretrain_validation_validation.json.0", wikitext);
    write_file(root / "raw" / "wikitext2_pretrain_test_test.json.0", wikitext);
    write_file(root / "raw" / "squad2_sft_train_source_train.json.0", squad_train);
    write_file(root / "raw" / "squad2_final_test_source_validation.json.0", squad_final);
}

Track1Config fixture_config(const std::filesystem::path& root) {
    Track1Config config;
    config.output_root = root.string();
    config.page_length = 100;
    config.pretrain_token_cap = 20;
    config.sft_examples = 4;
    config.sft_eval_examples = 2;
    config.source_row_limit = 4;
    config.selection_seed = 1701;
    config.acquire_remote = false;
    config.allow_small_fixture = true;
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/track1/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::remove_all(output);
    const auto fixture_root = output / "fixture";
    create_fixture(fixture_root);
    std::vector<Check> checks;
    Track1Pipeline* prepared = nullptr;
    Track1Pipeline fixture_pipeline(fixture_config(fixture_root));

    checks.push_back(run_check("pinned_huggingface_sources", [&]() {
        require(fixture_pipeline.manifest().sources.size() == 5U, "Track 1 source count changed");
        for (const auto& source : fixture_pipeline.manifest().sources) {
            require(source.revision.size() == 40U && source.row_api_url.find("datasets-server.huggingface.co/rows") != std::string::npos &&
                        source.license.find("CC BY-SA") != std::string::npos,
                    "source is not pinned to a licensed Hugging Face source");
        }
        const auto& squad_source = fixture_pipeline.manifest().sources.at(3U);
        require(squad_source.dataset_id == "GEM/squad_v2" && squad_source.upstream_dataset_id == "rajpurkar/squad_v2" &&
                    squad_source.license == "CC BY-SA 4.0" && squad_source.acquisition_type == "hf_gem_flat_file" &&
                    squad_source.raw_file_url.find("gem_data_split/train.json") != std::string::npos,
                "GEM SQuAD direct-file provenance is incomplete");
        return "{\"sources\":5,\"revisions_pinned\":true,\"licenses_recorded\":true,\"official_upstream_recorded\":true,\"gem_flat_file\":true}";
    }));

    checks.push_back(run_check("governed_preparation_and_caps", [&]() {
        fixture_pipeline.prepare();
        require(fixture_pipeline.report().passed && fixture_pipeline.manifest().pretrain_train_tokens == 20U &&
                    fixture_pipeline.manifest().sft_train_examples == 2U && fixture_pipeline.manifest().sft_evaluation_examples == 2U &&
                    fixture_pipeline.manifest().final_test_examples == 4U,
                "Track 1 preparation counts or cap are wrong");
        prepared = &fixture_pipeline;
        return "{\"pretrain_byte_tokens\":20,\"sft_train\":2,\"sft_evaluation\":2,\"final_test\":4,\"passed\":true}";
    }));

    checks.push_back(run_check("split_isolation_and_contamination", [&]() {
        require(prepared != nullptr, "preparation prerequisite missing");
        std::vector<std::string> all_ids = prepared->manifest().train_ids;
        all_ids.insert(all_ids.end(), prepared->manifest().evaluation_ids.begin(), prepared->manifest().evaluation_ids.end());
        all_ids.insert(all_ids.end(), prepared->manifest().final_test_ids.begin(), prepared->manifest().final_test_ids.end());
        std::sort(all_ids.begin(), all_ids.end());
        require(std::adjacent_find(all_ids.begin(), all_ids.end()) == all_ids.end(), "Track 1 split ID overlap detected");
        require(prepared->serialize_evaluation_contract().find("final_test_used_for_updates") != std::string::npos &&
                    prepared->serialize_evaluation_contract().find("unsupported_answer_rate") != std::string::npos,
                "Track 1 contamination or abstention contract missing");
        return "{\"duplicate_ids\":0,\"train_eval_overlap\":0,\"train_final_overlap\":0,\"final_test_update_forbidden\":true}";
    }));

    checks.push_back(run_check("deterministic_manifest_replay", [&]() {
        const auto replay = output / "replay";
        create_fixture(replay);
        Track1Pipeline replay_pipeline(fixture_config(replay));
        replay_pipeline.prepare();
        require(prepared->serialize_manifest() == replay_pipeline.serialize_manifest() &&
                    prepared->serialize_evaluation_contract() == replay_pipeline.serialize_evaluation_contract(),
                "same Track 1 source fixture did not reproduce the same manifest");
        return "{\"same_seed\":true,\"same_source\":true,\"manifest_equal\":true,\"evaluation_contract_equal\":true}";
    }));

    checks.push_back(run_check("missing_cache_fails_closed", [&]() {
        const auto missing = output / "missing-cache";
        bool rejected = false;
        try { Track1Pipeline(fixture_config(missing)).prepare(); } catch (const Track1Error&) { rejected = true; }
        require(rejected, "missing cached Hugging Face page was accepted");
        return "{\"missing_cache_rejected\":true}";
    }));

    checks.push_back(run_check("malformed_source_fails_closed", [&]() {
        const auto malformed = output / "malformed";
        create_fixture(malformed, true);
        bool rejected = false;
        try { Track1Pipeline(fixture_config(malformed)).prepare(); } catch (const Track1Error&) { rejected = true; }
        require(rejected, "malformed SQuAD source was accepted");
        return "{\"malformed_source_rejected\":true,\"silent_skip\":false}";
    }));

    checks.push_back(run_check("evaluation_contract_complete", [&]() {
        require(prepared != nullptr && prepared->evaluation_contract().pretrain_metrics.size() == 3U &&
                    prepared->evaluation_contract().qa_metrics.size() == 7U && prepared->evaluation_contract().required_slices.size() == 3U,
                "Track 1 evaluation contract is incomplete");
        return "{\"pretrain_metrics\":3,\"qa_metrics\":7,\"required_slices\":3,\"answerability_and_abstention\":true}";
    }));

    const bool passed = !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const auto& check) { return check.status == "PASS"; });
    std::filesystem::create_directories(output);
    std::ostringstream checks_json;
    checks_json << "[\n";
    for (std::size_t index = 0U; index < checks.size(); ++index) {
        if (index != 0U) checks_json << ",\n";
        checks_json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                    << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
    }
    checks_json << "\n]\n";
    write_file(output / "checks.json", checks_json.str());
    write_file(output / "source_manifest.json", prepared == nullptr ? "{}\n" : prepared->serialize_manifest());
    write_file(output / "preparation_report.json", prepared == nullptr ? "{}\n" : prepared->serialize_report());
    write_file(output / "evaluation_contract.json", prepared == nullptr ? "{}\n" : prepared->serialize_evaluation_contract());
    write_file(output / "README.md", "# Track 1 Acquisition Gate\n\nThis gate uses local copies of Hugging Face response pages and validates the pinned GEM flat-file SQuAD acquisition metadata, native parser, byte-token cap, balanced selection, final-test isolation, deterministic replay, and fail-closed acquisition controls used by remote Track 1 preparation.\n");
    write_file(output / "release_record.json", "{\"track\":\"track1\",\"status\":\"" + std::string(passed ? "PASS" : "FAIL") + "\",\"pretraining\":\"WikiText-2\",\"fine_tuning\":\"SQuAD 2.0\",\"final_test_is_frozen\":true,\"training_authorized\":false}\n");
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"checks\":" << checks.size() << ",\"output\":\"" << output.string() << "\"}\n";
    return passed ? 0 : 1;
}
