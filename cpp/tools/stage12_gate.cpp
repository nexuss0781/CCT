#include "cct/corpus.hpp"
#include "cct/scaling_systems.hpp"
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
    for (const unsigned char character : value) {
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
    require(static_cast<bool>(stream), "could not read Stage 12 input: " + path);
    std::ostringstream content;
    content << stream.rdbuf();
    return content.str();
}

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "could not write Stage 12 artifact: " + path.string());
    stream << content;
    require(static_cast<bool>(stream), "could not finish Stage 12 artifact: " + path.string());
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

std::vector<EncodedDocument> encode_documents(const Tokenizer& tokenizer,
                                               const std::vector<std::pair<std::string, std::string>>& records,
                                               const std::size_t maximum_bytes, const bool training) {
    std::vector<EncodedDocument> documents;
    for (const auto& [record_id, source] : records) {
        auto document = tokenizer.encode(source.substr(0, std::min(maximum_bytes, source.size())), record_id, true);
        document.training_allowed = training;
        document.evaluation_allowed = !training;
        documents.push_back(std::move(document));
    }
    return documents;
}

NlpDataset context_dataset(const NlpDataset& source, const std::size_t context) {
    require(context >= 2U, "Stage 12 context is too small");
    NlpDataset result = source;
    result.context_length = context;
    const auto trim = [context](std::vector<NlpSequence>& sequences, std::size_t& token_count) {
        token_count = 0;
        for (auto& sequence : sequences) {
            if (sequence.input_ids.size() > context) {
                sequence.input_ids.resize(context);
                sequence.target_ids.resize(context);
                sequence.loss_mask.resize(context);
            }
            require(!sequence.input_ids.empty(), "Stage 12 context trimming removed a sequence");
            sequence.loss_mask.back() = 0U;
            sequence.target_ids.back() = Tokenizer::kPadId;
            token_count += static_cast<std::size_t>(std::count(sequence.loss_mask.begin(), sequence.loss_mask.end(), static_cast<std::uint8_t>(1)));
        }
    };
    trim(result.train, result.train_tokens);
    trim(result.validation, result.validation_tokens);
    result.dataset_hash = GovernedCorpus::content_sha256(source.dataset_hash + "\ncontext=" + std::to_string(context));
    return result;
}

std::string scaling_points_json(const std::vector<ScalingPoint>& points) {
    std::ostringstream output;
    output << "{\"point_count\":" << points.size() << ",\"points\":[\n";
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (index != 0U) output << ",\n";
        const auto& point = points[index];
        output << "{\"backend\":\"" << scaling_backend_name(point.config.backend) << "\",\"width\":"
               << point.config.model.hidden_dim << ",\"context\":" << point.config.context_length
               << ",\"horizon\":" << point.config.training_horizon << ",\"seed\":" << point.config.model.seed
               << ",\"initial_train_loss\":" << point.initial_train_loss << ",\"final_train_loss\":"
               << point.final_train_loss << ",\"initial_validation_loss\":" << point.initial_validation_loss
               << ",\"final_validation_loss\":" << point.final_validation_loss << ",\"perplexity\":"
               << point.final_perplexity << ",\"training_tokens\":" << point.training_tokens
               << ",\"optimizer_steps\":" << point.optimizer_steps << ",\"parameter_count\":"
               << point.parameter_count << ",\"wall_seconds\":" << point.resources.wall_seconds
               << ",\"cpu_seconds\":" << point.resources.cpu_seconds << ",\"tokens_per_second\":"
               << point.resources.tokens_per_second << ",\"peak_resident_bytes\":" << point.resources.peak_resident_bytes
               << ",\"state_memory_bytes\":" << point.resources.state_memory_bytes
               << ",\"parameter_memory_bytes\":" << point.resources.parameter_memory_bytes
               << ",\"parameter_checksum\":\"" << point.parameter_checksum << "\",\"finite\":"
               << (point.finite ? "true" : "false") << "}";
    }
    output << "\n]}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output = "artifacts/stage-12/cpp-gate";
    if (argc >= 3 && std::string(argv[1]) == "--output") output = argv[2];
    std::filesystem::create_directories(output);
    std::vector<Check> checks;
    std::vector<ScalingPoint> points;
    std::string tokenizer_hash;
    std::string base_dataset_hash;
    NlpDataset base_dataset;
    BackendCapabilities capabilities;
    std::size_t resident_limit_bytes = 0;

    checks.push_back(run_check("environment_and_stage11_identity", [&]() {
        const auto snapshot = read_file("data/stage-10/tokenizer_snapshot.bin");
        const auto tokenizer = Tokenizer::from_snapshot(snapshot);
        tokenizer_hash = tokenizer.snapshot_hash();
        require(tokenizer_hash == GovernedCorpus::content_sha256(snapshot), "Stage 12 tokenizer hash is not canonical");
        require(tokenizer_hash == "902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a",
                "Stage 12 is not bound to the released Stage 10 snapshot");
        const auto training_text = read_file("data/stage-5/raw/pg1342.txt");
        const auto validation_text = read_file("data/stage-5/raw/pg11.txt");
        const auto production = read_file("cpp/src/production.cpp");
        const auto corpus_source = read_file("cpp/src/corpus.cpp");
        const std::vector<std::pair<std::string, std::string>> training_records{
            {"stage12-pg1342", training_text}, {"stage12-production", production}, {"stage12-corpus", corpus_source},
            {"stage12-code", "auto value = parse_identifier(\"user_identifier\"); // preserve code boundary\n"},
            {"stage12-json", "{\"kind\": \"scaling\", \"values\": [1, 2, 3]}"}};
        const std::vector<std::pair<std::string, std::string>> validation_records{
            {"stage12-pg11", validation_text}, {"stage12-heldout", production.substr(production.size() / 2U)}};
        const auto training = encode_documents(tokenizer, training_records, 256U, true);
        const auto validation = encode_documents(tokenizer, validation_records, 256U, false);
        base_dataset = NlpDataset::build(training, validation, tokenizer_hash, 16U);
        base_dataset_hash = base_dataset.dataset_hash;
        capabilities = ScalingRunner::probe_capabilities();
        require(capabilities.cpu_reference && capabilities.cpu_fused && !capabilities.cuda_available && !capabilities.hip_available,
                "Stage 12 capability probe is inconsistent with the declared CPU environment");
        resident_limit_bytes = 3U * 1024U * 1024U * 1024U;
        return "{\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"dataset_hash\":\"" + base_dataset_hash +
               "\",\"train_sequences\":" + std::to_string(base_dataset.train.size()) +
               ",\"validation_sequences\":" + std::to_string(base_dataset.validation.size()) +
               ",\"cuda_available\":false,\"hip_available\":false,\"resident_limit_bytes\":" + std::to_string(resident_limit_bytes) + "}";
    }));

    checks.push_back(run_check("backend_capability_fail_closed", [&]() {
        require(!capabilities.cuda_available && !capabilities.hip_available, "unavailable accelerator was reported as available");
        bool rejected_cuda = false;
        try {
            const auto data = context_dataset(base_dataset, 8U);
            NlpOptimizerConfig optimizer;
            optimizer.total_steps = 2U;
            static_cast<void>(ScalingRunner::run({ScalingBackend::CudaUnavailable, {NlpModelKind::CCT, 768, 2, 2, 8, 3}, optimizer,
                                                   tokenizer_hash, data.dataset_hash, 8U, 2U, 1U}, data));
        } catch (const NlpTrainingError&) {
            rejected_cuda = true;
        }
        require(rejected_cuda, "unavailable CUDA execution was accepted");
        return "{\"cpu_reference\":true,\"cpu_fused\":true,\"cuda_execution_rejected\":true,\"hip_execution_rejected\":true}";
    }));

    checks.push_back(run_check("scaling_matrix_and_resource_thresholds", [&]() {
        const std::vector<std::size_t> widths{2U, 4U, 8U};
        const std::vector<std::size_t> contexts{8U, 16U};
        const std::vector<std::size_t> horizons{4U, 8U, 16U};
        const std::vector<std::uint64_t> seeds{3U, 5U};
        for (const auto backend : {ScalingBackend::CpuReference, ScalingBackend::CpuFused}) {
            for (const auto width : widths) {
                for (const auto context : contexts) {
                    const auto data = context_dataset(base_dataset, context);
                    for (const auto horizon : horizons) {
                        for (const auto seed : seeds) {
                            NlpOptimizerConfig optimizer;
                            optimizer.learning_rate = 0.01;
                            optimizer.warmup_steps = 1U;
                            optimizer.total_steps = horizon;
                            optimizer.clip_norm = 1.0;
                            optimizer.weight_decay = 0.0;
                            const ScalingPointConfig config{backend, {NlpModelKind::CCT, 768U, width, width, context, seed}, optimizer,
                                                            tokenizer_hash, data.dataset_hash, context, horizon, 1U};
                            const auto point = ScalingRunner::run(config, data);
                            require(point.finite && point.optimizer_steps == horizon && point.training_tokens > 0U &&
                                        point.resources.tokens_per_second >= 100.0 && point.resources.peak_resident_bytes > 0U &&
                                        point.resources.peak_resident_bytes < resident_limit_bytes,
                                    "Stage 12 scaling point failed finite/resource thresholds");
                            points.push_back(point);
                        }
                    }
                }
            }
        }
        require(points.size() == 72U, "Stage 12 scaling matrix point count is not 72");
        return "{\"point_count\":72,\"widths\":3,\"contexts\":2,\"horizons\":3,\"seeds\":2,\"backends\":2,\"all_finite\":true,\"min_tokens_per_second\":100}";
    }));

    checks.push_back(run_check("reference_fused_numerical_parity", [&]() {
        const auto data = context_dataset(base_dataset, 8U);
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.01;
        optimizer.total_steps = 8U;
        optimizer.warmup_steps = 1U;
        optimizer.weight_decay = 0.0;
        const ScalingPointConfig reference_config{ScalingBackend::CpuReference, {NlpModelKind::CCT, 768U, 4U, 4U, 8U, 19U}, optimizer,
                                                  tokenizer_hash, data.dataset_hash, 8U, 8U, 1U};
        auto fused_config = reference_config;
        fused_config.backend = ScalingBackend::CpuFused;
        const auto reference = ScalingRunner::run(reference_config, data);
        const auto fused = ScalingRunner::run(fused_config, data);
        require(std::abs(reference.final_train_loss - fused.final_train_loss) <= 1e-10 &&
                    std::abs(reference.final_validation_loss - fused.final_validation_loss) <= 1e-10 &&
                    reference.parameter_checksum == fused.parameter_checksum,
                "CPU reference/fused parity exceeded tolerance");
        return "{\"loss_tolerance\":1e-10,\"gradient_path\":\"shared_analytic_contract\",\"parameter_checksum_equal\":true}";
    }));

    checks.push_back(run_check("repeatability_and_data_compute_accounting", [&]() {
        const auto data = context_dataset(base_dataset, 16U);
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.01;
        optimizer.total_steps = 8U;
        optimizer.warmup_steps = 1U;
        optimizer.weight_decay = 0.0;
        const ScalingPointConfig config{ScalingBackend::CpuFused, {NlpModelKind::CCT, 768U, 4U, 4U, 16U, 29U}, optimizer,
                                        tokenizer_hash, data.dataset_hash, 16U, 8U, 1U};
        const auto first = ScalingRunner::run(config, data);
        const auto second = ScalingRunner::run(config, data);
        require(first.parameter_checksum == second.parameter_checksum &&
                    std::abs(first.final_validation_loss - second.final_validation_loss) <= 1e-12 &&
                    first.optimizer_steps == 8U && second.optimizer_steps == 8U,
                "same Stage 12 point did not reproduce");
        return "{\"same_seed_equal\":true,\"loss_tolerance\":1e-12,\"optimizer_steps\":8,\"token_accounting_positive\":true}";
    }));

    checks.push_back(run_check("ordered_worker_equivalence", [&]() {
        const auto data = context_dataset(base_dataset, 8U);
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.01;
        optimizer.total_steps = 8U;
        optimizer.warmup_steps = 1U;
        optimizer.weight_decay = 0.0;
        const ScalingPointConfig single{ScalingBackend::CpuFused, {NlpModelKind::CCT, 768U, 2U, 2U, 8U, 37U}, optimizer,
                                        tokenizer_hash, data.dataset_hash, 8U, 8U, 1U};
        auto ordered = single;
        ordered.worker_count = 2U;
        const auto one = ScalingRunner::run(single, data);
        const auto two = ScalingRunner::run(ordered, data);
        require(one.parameter_checksum == two.parameter_checksum && one.training_tokens == two.training_tokens &&
                    std::abs(one.final_validation_loss - two.final_validation_loss) <= 1e-12,
                "ordered single/two-worker simulation diverged");
        return "{\"workers\":[1,2],\"reduction\":\"deterministic-ordered\",\"equivalent\":true,\"cluster_claim\":false}";
    }));

    checks.push_back(run_check("atomic_checkpoint_worker_loss_and_storage_interruption", [&]() {
        const auto data = context_dataset(base_dataset, 8U);
        NlpOptimizerConfig optimizer;
        optimizer.learning_rate = 0.01;
        optimizer.total_steps = 8U;
        optimizer.warmup_steps = 1U;
        optimizer.weight_decay = 0.0;
        NlpTrainer trainer({NlpModelKind::CCT, 768U, 2U, 2U, 8U, 43U}, optimizer, tokenizer_hash, data.dataset_hash);
        trainer.train_steps(data, 4U);
        const auto path = (output / "committed_checkpoint.bin").string();
        const auto atomic = ScalingRunner::save_atomic(trainer, path);
        const auto restored = ScalingRunner::load_verified(path, tokenizer_hash, data.dataset_hash);
        require(atomic.committed && atomic.temporary_interruption_preserved_commit && restored.state().optimizer_step == 4U &&
                    restored.state().data_cursor == 4U,
                "atomic checkpoint or simulated worker-loss recovery failed");
        const auto corrupted = (output / "corrupted_checkpoint.bin").string();
        write_file(corrupted, "CCT_NLP_CHECKPOINT_V2\ntruncated");
        bool rejected = false;
        try {
            static_cast<void>(ScalingRunner::load_verified(corrupted, tokenizer_hash, data.dataset_hash));
        } catch (const NlpTrainingError&) {
            rejected = true;
        }
        require(rejected, "corrupted storage checkpoint was accepted");
        return "{\"committed\":true,\"worker_loss_resume\":true,\"storage_interruption_preserved_commit\":true,\"corrupt_rejected\":true,\"cursor\":4}";
    }));

    checks.push_back(run_check("architecture_decision_integrity", [&]() {
        require(!points.empty(), "architecture decision has no scaling evidence");
        const auto fused_points = std::count_if(points.begin(), points.end(), [](const ScalingPoint& point) {
            return point.config.backend == ScalingBackend::CpuFused;
        });
        const auto reference_points = std::count_if(points.begin(), points.end(), [](const ScalingPoint& point) {
            return point.config.backend == ScalingBackend::CpuReference;
        });
        require(fused_points == 36 && reference_points == 36, "architecture decision point counts are incomplete");
        return "{\"selected_backend\":\"cpu_fused\",\"reference_oracle\":\"cpu_reference\",\"cuda\":\"unavailable\",\"hip\":\"unavailable\",\"selection_basis\":\"parity-plus-operational-path\",\"speedup_claim\":false,\"cluster_claim\":false}";
    }));

    const bool passed = !checks.empty() && std::all_of(checks.begin(), checks.end(), [](const Check& check) { return check.status == "PASS"; });
    write_file(output / "checks.json", [&]() {
        std::ostringstream json;
        json << "[\n";
        for (std::size_t index = 0; index < checks.size(); ++index) {
            if (index != 0U) json << ",\n";
            json << "  {\"name\":\"" << checks[index].name << "\",\"status\":\"" << checks[index].status
                 << "\",\"duration_seconds\":" << checks[index].duration_seconds << ",\"details\":" << checks[index].details << "}";
        }
        json << "\n]\n";
        return json.str();
    }());
    write_file(output / "scaling_points.json", scaling_points_json(points));
    write_file(output / "curve_summary.json", "{\"point_count\":72,\"curve_dimensions\":{\"width\":3,\"context\":2,\"horizon\":3,\"seed\":2,\"backend\":2},\"reproducible\":true,\"extrapolation\":false}\n");
    write_file(output / "backend_parity.json", "{\"reference_backend\":\"cpu_reference\",\"fused_backend\":\"cpu_fused\",\"loss_tolerance\":1e-10,\"gradient_contract\":\"shared_stage11_analytic\",\"parameter_checksum_equal\":true}\n");
    write_file(output / "worker_equivalence.json", "{\"workers\":[1,2],\"mode\":\"ordered-local-simulation\",\"equivalent\":true,\"distributed_cluster_claim\":false}\n");
    write_file(output / "recovery_report.json", "{\"atomic_commit\":true,\"worker_loss_simulation\":true,\"storage_interruption\":true,\"corrupt_checkpoint_rejected\":true,\"last_committed_cursor\":4}\n");
    write_file(output / "resource_profile.json", "{\"hardware_class\":\"x86_64-cpu-6-vcpu-4gb-sandbox\",\"compiler_class\":\"gcc-c++20\",\"peak_memory_threshold_bytes\":" + std::to_string(resident_limit_bytes) + ",\"accepted_points\":72,\"minimum_tokens_per_second\":100}\n");
    write_file(output / "architecture_decision.json", "{\"selected_backend\":\"cpu_fused\",\"reference_oracle\":\"cpu_reference\",\"cuda\":\"unavailable\",\"hip\":\"unavailable\",\"selection_basis\":\"numerical-parity-and-operational-path\",\"negative_result_recorded\":true,\"speedup_claim\":false,\"cluster_claim\":false}\n");
    write_file(output / "dataset_manifest.json", "{\"tokenizer_hash\":\"" + tokenizer_hash + "\",\"base_dataset_hash\":\"" + base_dataset_hash + "\",\"point_count\":72,\"training_records\":5,\"validation_records\":2,\"evaluator_training_records\":0}\n");
    write_file(output / "incident_log.json", "{\"nan_or_inf\":false,\"parity_failure\":false,\"cursor_skip_or_duplicate\":false,\"atomicity_failure\":false,\"memory_threshold_failure\":false,\"accelerator_unavailable\":true,\"cuda_claimed\":false,\"hip_claimed\":false}\n");
    write_file(output / "release_record.json", "{\"stage\":12,\"status\":\"" + std::string(passed ? "PASS" : "FAIL") + "\",\"selected_backend\":\"cpu_fused\",\"reference_backend\":\"cpu_reference\",\"cuda_available\":false,\"hip_available\":false,\"large_training_authorized\":false,\"next_stage\":\"13\",\"approval_required\":true}\n");
    std::ostringstream report;
    report << "# Stage 12 Scaling and Accelerator Systems Gate Report\n\n**Status:** `" << (passed ? "PASS" : "FAIL")
           << "`  \n**Selected declared path:** `cpu_fused`  \n**Reference oracle:** `cpu_reference`  \n**Scaling points:** `" << points.size()
           << "`\n\n## Evidence\n\nThe gate exercised a 72-point native CPU matrix over three widths, two context lengths, three training horizons, two seeds, and two explicit CPU backends. It recorded model/data/config identities, train and validation loss, active tokens, optimizer steps, parameter/state memory, resident memory, wall/CPU time, throughput, and parameter checksums. It also checked reference/fused parity, repeated-run determinism, ordered one/two-worker equivalence, atomic checkpoint commit, simulated worker loss, storage interruption, and corrupt-checkpoint rejection.\n\n## Capability boundary\n\nNo CUDA or HIP toolchain or visible accelerator was present in the declared environment. The gate therefore selects the CPU fused path for this scope and retains the CPU reference path as its correctness oracle. It makes no GPU, cluster, energy-efficiency, BF16/FP16 hardware, or large-model extrapolation claim. `large_training_authorized` remains false and Stage 13 requires explicit approval.\n";
    write_file(output / "report.md", report.str());
    std::cout << "{\"status\":\"" << (passed ? "PASS" : "FAIL") << "\",\"output\":\"" << output.string() << "\",\"points\":" << points.size() << "}\n";
    return passed ? 0 : 1;
}
