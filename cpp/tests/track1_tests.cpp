#include "cct/track1.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot create fixture " + path.string());
    output << content;
}

void create_fixture(const std::filesystem::path& root, const bool malformed = false) {
    const std::string wikitext = R"({"features":[{"name":"text"}],"rows":[{"row":{"text":"alpha beta gamma delta"}},{"row":{"text":"the causal engine learns from compact text"}},{"row":{"text":"spectral state updates are deterministic"}},{"row":{"text":"checkpoint replay preserves the stream"}}],"num_rows_total":4})";
    const std::string squad_train = R"({"features":[{"name":"id"}],"rows":[{"row":{"id":"a1","title":"One","context":"Paris is the capital of France.","question":"What is the capital of France?","answers":{"text":["Paris"],"answer_start":[0]}}},{"row":{"id":"a2","title":"Two","context":"The engine uses a spectral state.","question":"What does the engine use?","answers":{"text":["a spectral state"],"answer_start":[16]}}},{"row":{"id":"a3","title":"Three","context":"A checkpoint stores model state.","question":"What stores model state?","answers":{"text":["A checkpoint"],"answer_start":[0]}}},{"row":{"id":"a4","title":"Four","context":"Validation measures held-out loss.","question":"What does validation measure?","answers":{"text":["held-out loss"],"answer_start":[20]}}},{"row":{"id":"u1","title":"Five","context":"The sky is blue.","question":"What is the engine version?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"u2","title":"Six","context":"The river is long.","question":"Who wrote the source?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"u3","title":"Seven","context":"The model passed the gate.","question":"What is the hidden password?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"u4","title":"Eight","context":"The test is deterministic.","question":"Where is the server?","answers":{"text":[],"answer_start":[]}}}],"num_rows_total":8})";
    const auto final_json = malformed
        ? std::string(R"({"features":[{"name":"id"}],"rows":[{"row":{"id":"broken","title":"Broken","context":"not valid")")
        : std::string(R"({"features":[{"name":"id"}],"rows":[{"row":{"id":"t1","title":"FinalOne","context":"London is in England.","question":"Where is London?","answers":{"text":["in England"],"answer_start":[10]}}},{"row":{"id":"t2","title":"FinalTwo","context":"The final set is frozen.","question":"What is frozen?","answers":{"text":["The final set"],"answer_start":[0]}}},{"row":{"id":"t3","title":"FinalThree","context":"The answer is not present.","question":"What is the secret?","answers":{"text":[],"answer_start":[]}}},{"row":{"id":"t4","title":"FinalFour","context":"The pilot is bounded.","question":"What is the pilot?","answers":{"text":["bounded"],"answer_start":[13]}}}],"num_rows_total":4})");
    const std::vector<std::pair<std::string, std::string>> files{
        {"wikitext2_pretrain_train_train.json.0", wikitext},
        {"wikitext2_pretrain_validation_validation.json.0", wikitext},
        {"wikitext2_pretrain_test_test.json.0", wikitext},
        {"squad2_sft_train_source_train.json.0", squad_train},
        {"squad2_final_test_source_validation.json.0", final_json}};
    for (const auto& [name, content] : files) write(root / "raw" / name, content);
}

void create_flat_fixture(const std::filesystem::path& root) {
    const std::string wikitext = R"({"features":[{"name":"text"}],"rows":[{"row":{"text":"alpha beta gamma delta"}},{"row":{"text":"the causal engine learns from compact text"}},{"row":{"text":"spectral state updates are deterministic"}},{"row":{"text":"checkpoint replay preserves the stream"}}],"num_rows_total":4})";
    const std::string train = R"({"data":[{"id":"a1","title":"One","context":"Paris is the capital of France.","question":"What is the capital of France?","answers":{"text":["Paris"],"answer_start":[0]}},{"id":"a2","title":"Two","context":"The engine uses a spectral state.","question":"What does the engine use?","answers":{"text":["a spectral state"],"answer_start":[16]}},{"id":"u1","title":"Five","context":"The sky is blue.","question":"What is the engine version?","answers":{"text":[],"answer_start":[]}},{"id":"u2","title":"Six","context":"The river is long.","question":"Who wrote the source?","answers":{"text":[],"answer_start":[]}}]})";
    const std::string final = R"({"data":[{"id":"t1","title":"FinalOne","context":"London is in England.","question":"Where is London?","answers":{"text":["in England"],"answer_start":[10]}},{"id":"t2","title":"FinalTwo","context":"The final set is frozen.","question":"What is frozen?","answers":{"text":["The final set"],"answer_start":[0]}},{"id":"t3","title":"FinalThree","context":"The answer is not present.","question":"What is the secret?","answers":{"text":[],"answer_start":[]}},{"id":"t4","title":"FinalFour","context":"The pilot is bounded.","question":"What is the pilot?","answers":{"text":["bounded"],"answer_start":[13]}}]})";
    write(root / "raw" / "wikitext2_pretrain_train_train.json.0", wikitext);
    write(root / "raw" / "wikitext2_pretrain_validation_validation.json.0", wikitext);
    write(root / "raw" / "wikitext2_pretrain_test_test.json.0", wikitext);
    write(root / "raw" / "squad2_sft_train_source_train.json", train);
    write(root / "raw" / "squad2_final_test_source_validation.json", final);
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

void test_pinned_sources_and_contract() {
    Track1Pipeline pipeline(fixture_config("/tmp/cct-track1-source-contract"));
    require(pipeline.manifest().sources.size() == 5U, "Track 1 source count changed");
    for (const auto& source : pipeline.manifest().sources) {
        require(source.revision.size() == 40U && source.row_api_url.find("datasets-server.huggingface.co/rows") != std::string::npos &&
                    source.license.find("CC BY-SA") != std::string::npos,
                "Track 1 source is not pinned to a licensed Hugging Face source");
    }
    const auto& squad_source = pipeline.manifest().sources.at(3U);
    require(squad_source.dataset_id == "GEM/squad_v2" && squad_source.upstream_dataset_id == "rajpurkar/squad_v2" &&
                squad_source.license == "CC BY-SA 4.0" && squad_source.acquisition_type == "hf_gem_flat_file" &&
                squad_source.raw_file_url.find("gem_data_split/train.json") != std::string::npos,
            "GEM SQuAD direct-file provenance is incomplete");
    require(pipeline.evaluation_contract().qa_metrics.size() == 7U &&
                pipeline.evaluation_contract().forbidden_behaviors.front() == "final_test_used_for_updates",
            "Track 1 evaluation contract is incomplete");
}

void test_prepare_counts_caps_and_isolation() {
    const std::filesystem::path root = "/tmp/cct-track1-unit";
    std::filesystem::remove_all(root);
    create_fixture(root);
    Track1Pipeline pipeline(fixture_config(root));
    pipeline.prepare();
    require(pipeline.report().passed && pipeline.manifest().pretrain_train_tokens == 20U &&
                pipeline.manifest().sft_train_examples == 2U && pipeline.manifest().sft_evaluation_examples == 2U &&
                pipeline.manifest().final_test_examples == 4U,
            "Track 1 preparation counts or cap are wrong");
    const std::set<std::string> train(pipeline.manifest().train_ids.begin(), pipeline.manifest().train_ids.end());
    for (const auto& id : pipeline.manifest().evaluation_ids) require(!train.contains(id), "SFT evaluation overlaps training");
    for (const auto& id : pipeline.manifest().final_test_ids) require(!train.contains(id), "final test overlaps training");
    require(pipeline.manifest().manifest_digest.size() == 64U && pipeline.serialize_manifest().find("wikitext2_pretrain_train") != std::string::npos,
            "Track 1 manifest identity or source provenance is missing");
}

void test_deterministic_replay() {
    const std::filesystem::path first = "/tmp/cct-track1-replay-a";
    const std::filesystem::path second = "/tmp/cct-track1-replay-b";
    std::filesystem::remove_all(first);
    std::filesystem::remove_all(second);
    create_fixture(first);
    std::filesystem::create_directories(second);
    std::filesystem::copy(first / "raw", second / "raw", std::filesystem::copy_options::recursive);
    auto first_config = fixture_config(first);
    auto second_config = fixture_config(second);
    Track1Pipeline first_pipeline(first_config);
    Track1Pipeline second_pipeline(second_config);
    first_pipeline.prepare();
    second_pipeline.prepare();
    require(first_pipeline.serialize_manifest() == second_pipeline.serialize_manifest(), "Track 1 manifest replay is not deterministic");
    require(first_pipeline.serialize_evaluation_contract() == second_pipeline.serialize_evaluation_contract(), "Track 1 evaluation contract replay changed");
}

void test_direct_flat_file_parser() {
    const auto root = std::filesystem::path("/tmp/cct-track1-flat");
    std::filesystem::remove_all(root);
    create_flat_fixture(root);
    auto config = fixture_config(root);
    Track1Pipeline pipeline(config);
    pipeline.prepare();
    require(pipeline.report().passed && pipeline.report().malformed_rows == 0U && pipeline.report().unanswerable_rows == 2U &&
                pipeline.manifest().sft_train_examples == 2U && pipeline.manifest().final_test_examples == 4U,
            "GEM flat-file preparation did not preserve balanced categories and final-test rows");
}

void test_missing_cache_fails_closed() {
    const auto root = std::filesystem::path("/tmp/cct-track1-missing");
    std::filesystem::remove_all(root);
    bool rejected = false;
    try { Track1Pipeline(fixture_config(root)).prepare(); } catch (const Track1Error&) { rejected = true; }
    require(rejected, "missing cached Hugging Face source was not rejected");
}

void test_malformed_source_fails_closed() {
    const auto root = std::filesystem::path("/tmp/cct-track1-malformed");
    std::filesystem::remove_all(root);
    create_fixture(root, true);
    bool rejected = false;
    try { Track1Pipeline(fixture_config(root)).prepare(); } catch (const Track1Error&) { rejected = true; }
    require(rejected, "malformed SQuAD source was not rejected");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"pinned_sources_and_contract", test_pinned_sources_and_contract},
        {"prepare_counts_caps_and_isolation", test_prepare_counts_caps_and_isolation},
        {"deterministic_replay", test_deterministic_replay},
        {"direct_flat_file_parser", test_direct_flat_file_parser},
        {"missing_cache_fails_closed", test_missing_cache_fails_closed},
        {"malformed_source_fails_closed", test_malformed_source_fails_closed}};
    std::size_t passed = 0U;
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
