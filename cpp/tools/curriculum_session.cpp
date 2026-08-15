#include "cct/nlp_trainer.hpp"

#include "cct/corpus.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cct;

void require(const bool condition, const std::string& message) {
    if (!condition) throw NlpTrainingError(message);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read curriculum file " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

std::vector<std::string> read_records(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read curriculum record file " + path.string());
    std::vector<std::string> records;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) records.push_back(line);
    }
    require(!records.empty(), "curriculum record file is empty: " + path.string());
    return records;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned int>(character)
                                               << std::dec << std::setfill(' ');
                else output << static_cast<char>(character);
        }
    }
    return output.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    require(static_cast<bool>(output), "cannot write curriculum report " + path.string());
    output << content;
    require(static_cast<bool>(output), "cannot finish curriculum report " + path.string());
}

std::size_t number(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0U;
        const auto parsed = std::stoull(value, &consumed);
        require(consumed == value.size(), "invalid numeric value for " + name);
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw NlpTrainingError("invalid numeric value for " + name);
    }
}

struct Options {
    std::filesystem::path input = "artifacts/curriculum/current";
    std::filesystem::path output = "artifacts/curriculum/session";
    std::filesystem::path tokenizer = "data/stage-10/tokenizer_snapshot.bin";
    std::filesystem::path parent_checkpoint;
    std::string session_id = "level-0-session-0";
    std::size_t level = 0U;
    std::size_t pretrain_steps = 100U;
    std::size_t sft_steps = 50U;
    std::size_t context_length = 128U;
    std::size_t embedding_dim = 16U;
    std::size_t hidden_dim = 16U;
    std::size_t batch_size = 8U;
    std::uint64_t seed = 1701U;
};

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        const auto value = [&]() {
            require(index + 1 < argc, "missing value for " + key);
            return std::string(argv[++index]);
        };
        if (key == "--input") options.input = value();
        else if (key == "--output") options.output = value();
        else if (key == "--tokenizer") options.tokenizer = value();
        else if (key == "--parent-checkpoint") options.parent_checkpoint = value();
        else if (key == "--session-id") options.session_id = value();
        else if (key == "--level") options.level = number(value(), key);
        else if (key == "--pretrain-steps") options.pretrain_steps = number(value(), key);
        else if (key == "--sft-steps") options.sft_steps = number(value(), key);
        else if (key == "--context") options.context_length = number(value(), key);
        else if (key == "--embedding") options.embedding_dim = number(value(), key);
        else if (key == "--hidden") options.hidden_dim = number(value(), key);
        else if (key == "--batch") options.batch_size = number(value(), key);
        else if (key == "--seed") options.seed = static_cast<std::uint64_t>(number(value(), key));
        else if (key == "--help") {
            std::cout << "cct_curriculum_session --input PATH --output PATH --tokenizer PATH [--parent-checkpoint PATH] --session-id ID --level N "
                         "--pretrain-steps N --sft-steps N --context N --embedding N --hidden N --batch N --seed N\n";
            std::exit(0);
        } else throw NlpTrainingError("unknown argument " + key);
    }
    require(!options.session_id.empty() && options.pretrain_steps > 0U && options.sft_steps > 0U && options.context_length >= 2U &&
                options.embedding_dim > 0U && options.hidden_dim > 0U && options.batch_size > 0U,
            "curriculum session configuration is invalid");
    return options;
}

std::vector<EncodedDocument> encode_records(const Tokenizer& tokenizer, const std::vector<std::string>& records,
                                             const std::string& prefix, const bool training) {
    std::vector<EncodedDocument> documents;
    documents.reserve(records.size());
    for (std::size_t index = 0U; index < records.size(); ++index) {
        auto document = tokenizer.encode(records[index], prefix + ":" + std::to_string(index), false);
        require(document.tokens.size() >= 2U, "curriculum record tokenized to fewer than two tokens");
        document.training_allowed = training;
        document.evaluation_allowed = !training;
        document.evaluator_only = false;
        documents.push_back(std::move(document));
    }
    return documents;
}

NlpDataset make_dataset(const Tokenizer& tokenizer, const std::filesystem::path& train_path,
                       const std::filesystem::path& validation_path, const std::string& prefix,
                       const std::string& tokenizer_hash, const std::size_t context_length) {
    const auto train_records = read_records(train_path);
    const auto validation_records = read_records(validation_path);
    return NlpDataset::build(encode_records(tokenizer, train_records, prefix + "-train", true),
                             encode_records(tokenizer, validation_records, prefix + "-validation", false), tokenizer_hash, context_length);
}

NlpOptimizerConfig optimizer_config(const std::size_t steps, const std::size_t batch_size) {
    NlpOptimizerConfig optimizer;
    optimizer.learning_rate = 0.001;
    optimizer.beta1 = 0.9;
    optimizer.beta2 = 0.999;
    optimizer.epsilon = 1e-8;
    optimizer.weight_decay = 1e-4;
    optimizer.clip_norm = 1.0;
    optimizer.warmup_steps = std::min<std::size_t>(10U, steps);
    optimizer.batch_size = batch_size;
    optimizer.total_steps = steps;
    optimizer.validation_interval_steps = 1U;
    return optimizer;
}

std::string evaluation_json(const NlpEvaluation& evaluation) {
    std::ostringstream output;
    output << std::setprecision(12) << "{\"cross_entropy\":" << evaluation.cross_entropy << ",\"perplexity\":" << evaluation.perplexity
           << ",\"token_accuracy\":" << evaluation.token_accuracy << ",\"token_count\":" << evaluation.token_count
           << ",\"tokens_per_second\":" << evaluation.tokens_per_second << ",\"finite\":" << (evaluation.finite ? "true" : "false") << "}";
    return output.str();
}

struct PhaseReport {
    NlpEvaluation before;
    NlpEvaluation after;
    std::size_t steps = 0U;
};

PhaseReport train_phase(NlpTrainer& trainer, const NlpDataset& dataset, const std::size_t steps) {
    PhaseReport result;
    result.before = trainer.evaluate(dataset.validation);
    static_cast<void>(trainer.train_steps(dataset, steps));
    result.after = trainer.evaluate(dataset.validation);
    result.steps = steps;
    require(result.before.finite && result.after.finite && result.after.cross_entropy <= result.before.cross_entropy,
            "curriculum phase did not produce finite or non-increasing held-out loss");
    return result;
}

void require_same_model(const NextTokenModel& left, const NextTokenModel& right) {
    const auto left_parameters = left.parameter_vector();
    const auto right_parameters = right.parameter_vector();
    require(left_parameters == right_parameters, "curriculum checkpoint reload changed model parameters");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto tokenizer_content = read_file(options.tokenizer);
        const auto tokenizer_hash = GovernedCorpus::content_sha256(tokenizer_content);
        const auto tokenizer = Tokenizer::from_snapshot(tokenizer_content, tokenizer_hash);
        const auto token_id_limit = tokenizer.vocabulary().back().id;
        const auto vocabulary_size = 2U + static_cast<std::size_t>(token_id_limit - Tokenizer::kByteFirstId);
        const auto pretrain_test_path = options.input / "pretrain_test.txt";
        const auto pretrain_test_content = read_file(pretrain_test_path);
        require(!pretrain_test_content.empty(), "curriculum held-out test file is empty");
        const auto pretrain_test_hash = GovernedCorpus::content_sha256(pretrain_test_content);
        const auto pretrain = make_dataset(tokenizer, options.input / "pretrain_train.txt", options.input / "pretrain_validation.txt",
                                           "fineweb", tokenizer_hash, options.context_length);
        const auto sft = make_dataset(tokenizer, options.input / "sft_train.txt", options.input / "sft_validation.txt",
                                      "oasst", tokenizer_hash, options.context_length);
        const NlpModelConfig model_config{NlpModelKind::Track1CctRecurrence, vocabulary_size, options.embedding_dim, options.hidden_dim,
                                          options.context_length, options.seed, true, token_id_limit};
        const auto started = std::chrono::steady_clock::now();
        const auto session_start_parent = options.parent_checkpoint.empty() ? std::string("GENESIS") : std::string();
        NlpTrainer trainer = options.parent_checkpoint.empty()
                                 ? NlpTrainer(model_config, optimizer_config(options.pretrain_steps, options.batch_size), tokenizer_hash, pretrain.dataset_hash)
                                 : NlpTrainer::load_checkpoint(options.parent_checkpoint.string(), tokenizer_hash);
        const auto parent_checkpoint_hash = options.parent_checkpoint.empty() ? std::string("GENESIS") : trainer.checkpoint_info().checkpoint_hash;
        if (!options.parent_checkpoint.empty()) {
            require(trainer.model().config().kind == model_config.kind && trainer.model().config().vocabulary_size == model_config.vocabulary_size &&
                        trainer.model().config().embedding_dim == model_config.embedding_dim && trainer.model().config().hidden_dim == model_config.hidden_dim &&
                        trainer.model().config().context_length == model_config.context_length && trainer.model().config().compact_vocabulary,
                    "parent checkpoint model configuration does not match curriculum session");
        }
        static_cast<void>(session_start_parent);
        trainer.begin_continuation(pretrain.dataset_hash, options.session_id + "-pretrain", parent_checkpoint_hash, options.pretrain_steps);
        const auto pretrain_phase = train_phase(trainer, pretrain, options.pretrain_steps);
        std::filesystem::create_directories(options.output);
        const auto pretrain_checkpoint_path = options.output / "pretrain_checkpoint.bin";
        trainer.save_checkpoint(pretrain_checkpoint_path.string());
        const auto pretrain_checkpoint_hash = trainer.checkpoint_info().checkpoint_hash;
        auto sft_trainer = NlpTrainer::load_checkpoint(pretrain_checkpoint_path.string(), tokenizer_hash, pretrain.dataset_hash);
        sft_trainer.begin_continuation(sft.dataset_hash, options.session_id + "-sft", pretrain_checkpoint_hash, options.sft_steps);
        const auto sft_phase = train_phase(sft_trainer, sft, options.sft_steps);
        const auto checkpoint_path = options.output / "checkpoint.bin";
        sft_trainer.save_checkpoint(checkpoint_path.string());
        const auto final_checkpoint_hash = sft_trainer.checkpoint_info().checkpoint_hash;
        const auto reloaded = NlpTrainer::load_checkpoint(checkpoint_path.string(), tokenizer_hash, sft.dataset_hash);
        require(reloaded.checkpoint_info().session_id == options.session_id + "-sft" &&
                    reloaded.checkpoint_info().parent_checkpoint_hash == pretrain_checkpoint_hash &&
                    reloaded.checkpoint_info().checkpoint_hash == final_checkpoint_hash,
                "final curriculum checkpoint lineage did not reload exactly");
        require_same_model(reloaded.model(), sft_trainer.model());
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::ostringstream report;
        report << std::setprecision(12) << "{\"status\":\"PASS\",\"session_id\":\"" << json_escape(options.session_id)
               << "\",\"level\":" << options.level << ",\"tokenizer_hash\":\"" << tokenizer_hash << "\",\"parent_checkpoint_hash\":\""
               << parent_checkpoint_hash << "\",\"pretrain_dataset_hash\":\"" << pretrain.dataset_hash << "\",\"sft_dataset_hash\":\""
               << sft.dataset_hash << "\",\"pretrain_test_sha256\":\"" << pretrain_test_hash
               << "\",\"human_test_file\":\"pretrain_test.txt\",\"human_test_automatic\":false,\"pretrain_checkpoint_hash\":\""
               << pretrain_checkpoint_hash << "\",\"checkpoint_hash\":\"" << final_checkpoint_hash << "\",\"pretrain\":{\"before\":" << evaluation_json(pretrain_phase.before) << ",\"after\":"
               << evaluation_json(pretrain_phase.after) << ",\"steps\":" << pretrain_phase.steps << "},\"sft\":{\"before\":"
               << evaluation_json(sft_phase.before) << ",\"after\":" << evaluation_json(sft_phase.after) << ",\"steps\":" << sft_phase.steps
               << "},\"pretrain_checkpoint\":\"pretrain_checkpoint.bin\",\"checkpoint\":\"checkpoint.bin\",\"elapsed_seconds\":" << elapsed
               << ",\"human_validation_required\":true,\"training_authorized\":false}\n";
        write_file(options.output / "session_report.json", report.str());
        std::cout << report.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"FAIL\",\"error\":\"" << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
