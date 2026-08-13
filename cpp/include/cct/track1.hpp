#pragma once

#include "cct/corpus.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cct {

enum class Track1Split : std::uint8_t {
    PretrainTrain = 0,
    PretrainValidation = 1,
    PretrainTest = 2,
    SftTrain = 3,
    SftEvaluation = 4,
    FinalTest = 5
};

std::string track1_split_name(Track1Split split);

struct Track1Source {
    std::string source_id;
    std::string dataset_id;
    std::string config;
    std::string split;
    std::string revision;
    std::string license;
    std::string row_api_url;
    std::size_t total_rows = 0;
    std::string raw_digest;
    std::string upstream_dataset_id;
    std::string acquisition_type = "hf_rows";
    std::string raw_file_url;
    std::string archive_member;
    std::string attestation_digest;
};

struct Track1Config {
    std::string output_root = "artifacts/track1";
    std::size_t page_length = 100;
    std::size_t pretrain_token_cap = 2000000;
    std::size_t sft_examples = 8000;
    std::size_t sft_eval_examples = 800;
    std::uint64_t selection_seed = 1701;
    std::size_t source_row_limit = 0;
    std::size_t squad_train_row_offset = 0;
    std::size_t squad_final_test_row_offset = 0;
    bool acquire_remote = true;
    bool allow_small_fixture = false;
};

struct Track1Example {
    std::string id;
    std::string title;
    std::string context;
    std::string question;
    std::string answer;
    std::size_t answer_start = 0;
    std::size_t source_answer_start = 0;
    bool answerable = false;
    std::string source_id;
    Track1Split split = Track1Split::SftTrain;
    std::string content_digest;
};

struct Track1Manifest {
    std::string manifest_version = "track1-v1";
    std::string tokenizer_snapshot = "data/stage-10/tokenizer_snapshot.bin";
    std::string selection_policy = "stable-question-id-hash-v1";
    std::uint64_t selection_seed = 1701;
    std::size_t pretrain_train_tokens = 0;
    std::size_t pretrain_validation_tokens = 0;
    std::size_t pretrain_test_tokens = 0;
    std::size_t sft_train_examples = 0;
    std::size_t sft_evaluation_examples = 0;
    std::size_t final_test_examples = 0;
    std::vector<Track1Source> sources;
    std::vector<std::string> train_ids;
    std::vector<std::string> evaluation_ids;
    std::vector<std::string> final_test_ids;
    std::string manifest_digest;
};

struct Track1PreparationReport {
    bool passed = false;
    std::size_t source_pages = 0;
    std::size_t source_rows = 0;
    std::size_t pretrain_rows = 0;
    std::size_t pretrain_tokens = 0;
    std::size_t sft_rows = 0;
    std::size_t sft_train_rows = 0;
    std::size_t sft_evaluation_rows = 0;
    std::size_t final_test_rows = 0;
    std::size_t duplicate_ids = 0;
    std::size_t overlap_ids = 0;
    std::size_t malformed_rows = 0;
    std::size_t unanswerable_rows = 0;
    std::string manifest_path;
    std::string report_path;
};

struct Track1EvaluationContract {
    std::string metric_version = "track1-eval-v1";
    std::vector<std::string> pretrain_metrics{"loss", "perplexity", "tokens_per_second"};
    std::vector<std::string> qa_metrics{"exact_match", "token_f1", "answerability_auroc", "answerability_auprc", "abstention_precision", "abstention_recall", "unsupported_answer_rate"};
    std::vector<std::string> required_slices{"all", "answerable", "unanswerable"};
    std::vector<std::string> forbidden_behaviors{"final_test_used_for_updates", "split_id_overlap", "missing_provenance", "nonfinite_metric"};
};

class Track1Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Track1Pipeline {
public:
    explicit Track1Pipeline(Track1Config config = {});

    const Track1Config& config() const noexcept;
    const Track1Manifest& manifest() const noexcept;
    const Track1PreparationReport& report() const noexcept;
    const Track1EvaluationContract& evaluation_contract() const noexcept;

    void prepare();
    void validate_existing() const;
    std::string serialize_manifest() const;
    std::string serialize_report() const;
    std::string serialize_evaluation_contract() const;

private:
    Track1Config config_;
    Track1Manifest manifest_;
    Track1PreparationReport report_;
    Track1EvaluationContract evaluation_contract_;

    void prepare_wikitext();
    void prepare_squad();
    void write_artifacts() const;
    void validate_manifest() const;
};

}  // namespace cct
