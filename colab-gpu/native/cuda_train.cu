#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kStreamMagic = 0x314750544343ULL;
constexpr std::uint64_t kCheckpointMagic = 0x314B504354434ULL;
constexpr int kVocab = 512;
constexpr int kMaxContext = 128;
constexpr int kMaxEmbedding = 64;
constexpr int kMaxHidden = 64;

struct Config {
    int context = 64;
    int embedding = 32;
    int hidden = 32;
    int batch = 16;
    int steps = 1000;
    int checkpoint_every = 100;
    std::uint64_t seed = 7;
    float learning_rate = 0.0025F;
    float weight_decay = 1.0e-5F;
    float clip_norm = 1.0F;
};

struct Dataset {
    std::vector<std::uint32_t> sequences;
    std::size_t sequence_count = 0;
    std::uint64_t token_count = 0;
    std::uint64_t file_hash = 0;
};

struct HostModel {
    std::vector<float> parameters;
    std::vector<float> first;
    std::vector<float> second;
    std::uint64_t step = 0;
};

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t fnv1a(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) { hash ^= byte; hash *= 1099511628211ULL; }
    return hash;
}

std::uint64_t fnv1a_tokens(const std::vector<std::uint32_t>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto value : values) {
        hash ^= value & 0xffU; hash *= 1099511628211ULL;
        hash ^= (value >> 8U) & 0xffU; hash *= 1099511628211ULL;
        hash ^= (value >> 16U) & 0xffU; hash *= 1099511628211ULL;
        hash ^= (value >> 24U) & 0xffU; hash *= 1099511628211ULL;
    }
    return hash;
}

std::size_t parameter_count(const Config& config) {
    const auto embedding = static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(config.embedding);
    const auto input = static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(config.embedding);
    const auto recurrent = static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(config.hidden);
    const auto hidden_bias = static_cast<std::size_t>(config.hidden);
    const auto output = static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(config.hidden);
    return embedding + input + recurrent + hidden_bias + output + static_cast<std::size_t>(kVocab);
}

std::size_t embedding_offset(const Config&) { return 0U; }
std::size_t input_offset(const Config& config) { return static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(config.embedding); }
std::size_t recurrent_offset(const Config& config) { return input_offset(config) + static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(config.embedding); }
std::size_t hidden_bias_offset(const Config& config) { return recurrent_offset(config) + static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(config.hidden); }
std::size_t output_offset(const Config& config) { return hidden_bias_offset(config) + static_cast<std::size_t>(config.hidden); }
std::size_t output_bias_offset(const Config& config) { return output_offset(config) + static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(config.hidden); }

Dataset read_dataset(const std::string& path, const Config& config, const std::uint64_t max_tokens) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot open token stream " + path);
    std::uint64_t magic = 0;
    std::uint64_t declared = 0;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    stream.read(reinterpret_cast<char*>(&declared), sizeof(declared));
    require(magic == kStreamMagic, "invalid token stream magic: " + path);
    const auto bounded = std::min<std::uint64_t>(declared, max_tokens);
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(bounded));
    stream.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(std::uint32_t)));
    require(stream || stream.eof(), "truncated token stream " + path);
    Dataset dataset;
    dataset.token_count = tokens.size();
    dataset.file_hash = fnv1a_tokens(tokens);
    require(tokens.size() > static_cast<std::size_t>(config.context + 1), "not enough tokens in " + path);
    dataset.sequence_count = (tokens.size() - 1U) / static_cast<std::size_t>(config.context);
    dataset.sequences.resize(dataset.sequence_count * static_cast<std::size_t>(config.context + 1));
    for (std::size_t sequence = 0; sequence < dataset.sequence_count; ++sequence) {
        const auto source = sequence * static_cast<std::size_t>(config.context);
        const auto destination = sequence * static_cast<std::size_t>(config.context + 1);
        std::copy_n(tokens.begin() + static_cast<std::ptrdiff_t>(source), config.context + 1, dataset.sequences.begin() + static_cast<std::ptrdiff_t>(destination));
        for (int index = 0; index <= config.context; ++index) require(dataset.sequences[destination + static_cast<std::size_t>(index)] < kVocab, "token ID exceeds CUDA vocabulary");
    }
    return dataset;
}

__device__ inline float device_exp(const float value) { return expf(value); }

__global__ void zero_kernel(float* values, const std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = 0.0F;
}

__global__ void train_kernel(const std::uint32_t* sequences, const std::size_t count, const int context,
                            const int embedding, const int hidden, const float* parameters, float* gradients) {
    const auto sequence_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence_index >= count) return;
    float states[kMaxContext][kMaxHidden];
    const auto* sequence = sequences + sequence_index * static_cast<std::size_t>(context + 1);
    const auto input_offset = kVocab * embedding;
    const auto recurrent_offset = input_offset + hidden * embedding;
    const auto hidden_bias_offset = recurrent_offset + hidden * hidden;
    const auto output_offset = hidden_bias_offset + hidden;
    const auto output_bias_offset = output_offset + kVocab * hidden;
    for (int time = 0; time < context; ++time) {
        for (int unit = 0; unit < hidden; ++unit) {
            float value = parameters[hidden_bias_offset + unit];
            for (int feature = 0; feature < embedding; ++feature) value += parameters[input_offset + unit * embedding + feature] * parameters[sequence[time] * embedding + feature];
            if (time > 0) for (int prior = 0; prior < hidden; ++prior) value += parameters[recurrent_offset + unit * hidden + prior] * states[time - 1][prior];
            states[time][unit] = tanhf(value);
        }
    }
    float dh_next[kMaxHidden] = {0.0F};
    float errors[kVocab];
    for (int time = context - 1; time >= 0; --time) {
        float maximum = -CUDART_INF_F;
        for (int label = 0; label < kVocab; ++label) {
            float logit = parameters[output_bias_offset + label];
            for (int unit = 0; unit < hidden; ++unit) logit += parameters[output_offset + label * hidden + unit] * states[time][unit];
            errors[label] = logit;
            maximum = fmaxf(maximum, logit);
        }
        float total = 0.0F;
        for (int label = 0; label < kVocab; ++label) { errors[label] = device_exp(errors[label] - maximum); total += errors[label]; }
        const auto target = sequence[time + 1];
        for (int label = 0; label < kVocab; ++label) {
            errors[label] /= total;
            errors[label] -= label == static_cast<int>(target) ? 1.0F : 0.0F;
            atomicAdd(&gradients[output_offset + label * hidden], errors[label] * states[time][0]);
            for (int unit = 1; unit < hidden; ++unit) atomicAdd(&gradients[output_offset + label * hidden + unit], errors[label] * states[time][unit]);
            atomicAdd(&gradients[output_bias_offset + label], errors[label]);
        }
        float delta[kMaxHidden];
        for (int unit = 0; unit < hidden; ++unit) {
            float value = dh_next[unit];
            for (int label = 0; label < kVocab; ++label) value += parameters[output_offset + label * hidden + unit] * errors[label];
            delta[unit] = value * (1.0F - states[time][unit] * states[time][unit]);
            atomicAdd(&gradients[hidden_bias_offset + unit], delta[unit]);
            for (int feature = 0; feature < embedding; ++feature) {
                atomicAdd(&gradients[input_offset + unit * embedding + feature], delta[unit] * parameters[sequence[time] * embedding + feature]);
                atomicAdd(&gradients[sequence[time] * embedding + feature], delta[unit] * parameters[input_offset + unit * embedding + feature]);
            }
            if (time > 0) for (int prior = 0; prior < hidden; ++prior) atomicAdd(&gradients[recurrent_offset + unit * hidden + prior], delta[unit] * states[time - 1][prior]);
        }
        for (int prior = 0; prior < hidden; ++prior) {
            float value = 0.0F;
            for (int unit = 0; unit < hidden; ++unit) value += parameters[recurrent_offset + unit * hidden + prior] * delta[unit];
            dh_next[prior] = value;
        }
    }
}

__global__ void update_kernel(float* parameters, float* first, float* second, const float* gradients, const std::size_t count,
                              const float learning_rate, const float beta1, const float beta2, const float epsilon,
                              const float weight_decay, const float clip_scale, const float first_correction, const float second_correction) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto gradient = gradients[index] * clip_scale;
    first[index] = beta1 * first[index] + (1.0F - beta1) * gradient;
    second[index] = beta2 * second[index] + (1.0F - beta2) * gradient * gradient;
    const auto normalized_first = first[index] / first_correction;
    const auto normalized_second = second[index] / second_correction;
    parameters[index] -= learning_rate * (normalized_first / (sqrtf(normalized_second) + epsilon) + weight_decay * parameters[index]);
}

__global__ void score_kernel(const std::uint32_t* sequences, const std::size_t count, const int context,
                            const int embedding, const int hidden, const float* parameters, float* loss, unsigned long long* correct) {
    const auto sequence_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sequence_index >= count) return;
    float state[kMaxHidden] = {0.0F};
    const auto* sequence = sequences + sequence_index * static_cast<std::size_t>(context + 1);
    const auto input_offset = kVocab * embedding;
    const auto recurrent_offset = input_offset + hidden * embedding;
    const auto hidden_bias_offset = recurrent_offset + hidden * hidden;
    const auto output_offset = hidden_bias_offset + hidden;
    const auto output_bias_offset = output_offset + kVocab * hidden;
    float total_loss = 0.0F;
    unsigned long long local_correct = 0ULL;
    for (int time = 0; time < context; ++time) {
        float next_state[kMaxHidden];
        for (int unit = 0; unit < hidden; ++unit) {
            float value = parameters[hidden_bias_offset + unit];
            for (int feature = 0; feature < embedding; ++feature) value += parameters[input_offset + unit * embedding + feature] * parameters[sequence[time] * embedding + feature];
            for (int prior = 0; prior < hidden; ++prior) value += parameters[recurrent_offset + unit * hidden + prior] * state[prior];
            next_state[unit] = tanhf(value);
        }
        for (int unit = 0; unit < hidden; ++unit) state[unit] = next_state[unit];
        float maximum = -CUDART_INF_F;
        float logits[kVocab];
        for (int label = 0; label < kVocab; ++label) {
            logits[label] = parameters[output_bias_offset + label];
            for (int unit = 0; unit < hidden; ++unit) logits[label] += parameters[output_offset + label * hidden + unit] * state[unit];
            maximum = fmaxf(maximum, logits[label]);
        }
        float total = 0.0F;
        for (int label = 0; label < kVocab; ++label) { logits[label] = device_exp(logits[label] - maximum); total += logits[label]; }
        const auto target = sequence[time + 1];
        float best = -1.0F;
        std::uint32_t predicted = 0U;
        for (int label = 0; label < kVocab; ++label) {
            logits[label] /= total;
            if (logits[label] > best) { best = logits[label]; predicted = static_cast<std::uint32_t>(label); }
        }
        total_loss += -logf(fmaxf(logits[target], 1.0e-20F));
        if (predicted == target) ++local_correct;
    }
    atomicAdd(loss, total_loss);
    atomicAdd(correct, local_correct);
}

std::vector<float> initial_parameters(const Config& config) {
    std::vector<float> parameters(parameter_count(config));
    std::uint64_t state = config.seed + 0x9e3779b97f4a7c15ULL;
    for (auto& value : parameters) {
        state ^= state >> 12U; state ^= state << 25U; state ^= state >> 27U;
        const auto random = static_cast<double>((state * 2685821657736338717ULL) % 1000000ULL) / 1000000.0 - 0.5;
        value = static_cast<float>(random * 0.04);
    }
    return parameters;
}

void save_checkpoint(const std::string& path, const Config& config, const HostModel& model, const std::uint64_t dataset_hash,
                     const std::uint64_t tokenizer_hash, const std::uint64_t train_tokens, const std::uint64_t validation_tokens) {
    const auto temporary = path + ".tmp";
    std::ofstream stream(temporary, std::ios::binary);
    require(static_cast<bool>(stream), "cannot write checkpoint");
    stream.write(reinterpret_cast<const char*>(&kCheckpointMagic), sizeof(kCheckpointMagic));
    stream.write(reinterpret_cast<const char*>(&config), sizeof(config));
    stream.write(reinterpret_cast<const char*>(&model.step), sizeof(model.step));
    stream.write(reinterpret_cast<const char*>(&dataset_hash), sizeof(dataset_hash));
    stream.write(reinterpret_cast<const char*>(&tokenizer_hash), sizeof(tokenizer_hash));
    stream.write(reinterpret_cast<const char*>(&train_tokens), sizeof(train_tokens));
    stream.write(reinterpret_cast<const char*>(&validation_tokens), sizeof(validation_tokens));
    const auto size = static_cast<std::uint64_t>(model.parameters.size());
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(reinterpret_cast<const char*>(model.parameters.data()), static_cast<std::streamsize>(model.parameters.size() * sizeof(float)));
    stream.write(reinterpret_cast<const char*>(model.first.data()), static_cast<std::streamsize>(model.first.size() * sizeof(float)));
    stream.write(reinterpret_cast<const char*>(model.second.data()), static_cast<std::streamsize>(model.second.size() * sizeof(float)));
    stream.close();
    require(static_cast<bool>(stream), "checkpoint write failed");
    std::filesystem::rename(temporary, path);
}

HostModel load_checkpoint(const std::string& path, const Config& expected, const std::uint64_t expected_dataset,
                          const std::uint64_t expected_tokenizer) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot open resume checkpoint");
    std::uint64_t magic = 0;
    Config config{};
    std::uint64_t step = 0;
    std::uint64_t dataset = 0;
    std::uint64_t tokenizer = 0;
    std::uint64_t train_tokens = 0;
    std::uint64_t validation_tokens = 0;
    std::uint64_t size = 0;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    stream.read(reinterpret_cast<char*>(&config), sizeof(config));
    stream.read(reinterpret_cast<char*>(&step), sizeof(step));
    stream.read(reinterpret_cast<char*>(&dataset), sizeof(dataset));
    stream.read(reinterpret_cast<char*>(&tokenizer), sizeof(tokenizer));
    stream.read(reinterpret_cast<char*>(&train_tokens), sizeof(train_tokens));
    stream.read(reinterpret_cast<char*>(&validation_tokens), sizeof(validation_tokens));
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    require(magic == kCheckpointMagic && (expected_dataset == 0ULL || dataset == expected_dataset) && tokenizer == expected_tokenizer &&
                config.context == expected.context && config.embedding == expected.embedding && config.hidden == expected.hidden,
            "checkpoint identity or configuration mismatch");
    require(size == parameter_count(expected), "checkpoint parameter count mismatch");
    HostModel model;
    model.step = step;
    model.parameters.resize(size); model.first.resize(size); model.second.resize(size);
    stream.read(reinterpret_cast<char*>(model.parameters.data()), static_cast<std::streamsize>(size * sizeof(float)));
    stream.read(reinterpret_cast<char*>(model.first.data()), static_cast<std::streamsize>(size * sizeof(float)));
    stream.read(reinterpret_cast<char*>(model.second.data()), static_cast<std::streamsize>(size * sizeof(float)));
    require(static_cast<bool>(stream), "truncated or corrupt checkpoint");
    require(std::all_of(model.parameters.begin(), model.parameters.end(), [](const float value) { return std::isfinite(value); }), "checkpoint has non-finite parameters");
    return model;
}

struct DeviceModel {
    float* parameters = nullptr;
    float* first = nullptr;
    float* second = nullptr;
    float* gradients = nullptr;
    ~DeviceModel() {
        cudaFree(parameters); cudaFree(first); cudaFree(second); cudaFree(gradients);
    }
    void allocate(const std::size_t size) {
        check_cuda(cudaMalloc(&parameters, size * sizeof(float)), "cudaMalloc parameters");
        check_cuda(cudaMalloc(&first, size * sizeof(float)), "cudaMalloc first moments");
        check_cuda(cudaMalloc(&second, size * sizeof(float)), "cudaMalloc second moments");
        check_cuda(cudaMalloc(&gradients, size * sizeof(float)), "cudaMalloc gradients");
    }
    void upload(const HostModel& model) {
        check_cuda(cudaMemcpy(parameters, model.parameters.data(), model.parameters.size() * sizeof(float), cudaMemcpyHostToDevice), "upload parameters");
        check_cuda(cudaMemcpy(first, model.first.data(), model.first.size() * sizeof(float), cudaMemcpyHostToDevice), "upload first moments");
        check_cuda(cudaMemcpy(second, model.second.data(), model.second.size() * sizeof(float), cudaMemcpyHostToDevice), "upload second moments");
    }
    void download(HostModel& model) {
        check_cuda(cudaMemcpy(model.parameters.data(), parameters, model.parameters.size() * sizeof(float), cudaMemcpyDeviceToHost), "download parameters");
        check_cuda(cudaMemcpy(model.first.data(), first, model.first.size() * sizeof(float), cudaMemcpyDeviceToHost), "download first moments");
        check_cuda(cudaMemcpy(model.second.data(), second, model.second.size() * sizeof(float), cudaMemcpyDeviceToHost), "download second moments");
    }
};

struct DeviceDataset {
    std::uint32_t* values = nullptr;
    std::size_t size = 0;
    ~DeviceDataset() { cudaFree(values); }
    void upload(const Dataset& dataset) {
        size = dataset.sequences.size();
        check_cuda(cudaMalloc(&values, size * sizeof(std::uint32_t)), "cudaMalloc dataset");
        check_cuda(cudaMemcpy(values, dataset.sequences.data(), size * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "upload dataset");
    }
};

struct Score { double loss = 0.0; double accuracy = 0.0; double seconds = 0.0; std::uint64_t tokens = 0; };

Score score(const Dataset& dataset, const Config& config, const DeviceModel& model, DeviceDataset& device_dataset) {
    float* loss_device = nullptr;
    unsigned long long* correct_device = nullptr;
    check_cuda(cudaMalloc(&loss_device, sizeof(float)), "cudaMalloc loss");
    check_cuda(cudaMalloc(&correct_device, sizeof(unsigned long long)), "cudaMalloc correct");
    check_cuda(cudaMemset(loss_device, 0, sizeof(float)), "cudaMemset loss");
    check_cuda(cudaMemset(correct_device, 0, sizeof(unsigned long long)), "cudaMemset correct");
    const auto started = std::chrono::steady_clock::now();
    constexpr int threads = 64;
    score_kernel<<<static_cast<unsigned int>((dataset.sequence_count + threads - 1U) / threads), threads>>>(device_dataset.values, dataset.sequence_count, config.context, config.embedding, config.hidden, model.parameters, loss_device, correct_device);
    check_cuda(cudaGetLastError(), "score kernel");
    check_cuda(cudaDeviceSynchronize(), "score synchronize");
    float loss = 0.0F;
    unsigned long long correct = 0ULL;
    check_cuda(cudaMemcpy(&loss, loss_device, sizeof(loss), cudaMemcpyDeviceToHost), "download loss");
    check_cuda(cudaMemcpy(&correct, correct_device, sizeof(correct), cudaMemcpyDeviceToHost), "download correct");
    cudaFree(loss_device); cudaFree(correct_device);
    const auto tokens = static_cast<std::uint64_t>(dataset.sequence_count) * static_cast<std::uint64_t>(config.context);
    Score result;
    result.loss = static_cast<double>(loss) / static_cast<double>(tokens);
    result.accuracy = static_cast<double>(correct) / static_cast<double>(tokens);
    result.tokens = tokens;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

void train(const Dataset& dataset, const Config& config, DeviceModel& device_model, DeviceDataset& device_dataset,
           HostModel& host_model, const std::string& checkpoint, const std::uint64_t dataset_hash, const std::uint64_t tokenizer_hash,
           const std::uint64_t validation_tokens, const Dataset& validation, DeviceDataset& device_validation) {
    const std::size_t size = host_model.parameters.size();
    constexpr int threads = 64;
    const auto blocks = static_cast<unsigned int>((size + threads - 1U) / threads);
    const auto sequence_blocks = static_cast<unsigned int>((static_cast<std::size_t>(config.batch) + threads - 1U) / threads);
    for (int local_step = 0; local_step < config.steps; ++local_step) {
        const auto start = static_cast<std::size_t>(local_step) * static_cast<std::size_t>(config.batch) % dataset.sequence_count;
        zero_kernel<<<blocks, threads>>>(device_model.gradients, size);
        train_kernel<<<sequence_blocks, threads>>>(device_dataset.values + start * static_cast<std::size_t>(config.context + 1),
                                                   std::min<std::size_t>(config.batch, dataset.sequence_count - start), config.context,
                                                   config.embedding, config.hidden, device_model.parameters, device_model.gradients);
        check_cuda(cudaGetLastError(), "train kernel");
        check_cuda(cudaDeviceSynchronize(), "train synchronize");
        check_cuda(cudaMemcpy(host_model.first.data(), device_model.gradients, size * sizeof(float), cudaMemcpyDeviceToHost), "download gradients");
        double norm_squared = 0.0;
        for (const auto value : host_model.first) norm_squared += static_cast<double>(value) * static_cast<double>(value);
        const auto norm = std::sqrt(norm_squared);
        const auto clip_scale = static_cast<float>(norm > static_cast<double>(config.clip_norm) ? static_cast<double>(config.clip_norm) / norm : 1.0);
        ++host_model.step;
        const auto first_correction = 1.0F - std::pow(0.9F, static_cast<float>(host_model.step));
        const auto second_correction = 1.0F - std::pow(0.999F, static_cast<float>(host_model.step));
        update_kernel<<<blocks, threads>>>(device_model.parameters, device_model.first, device_model.second, device_model.gradients, size,
                                            config.learning_rate, 0.9F, 0.999F, 1.0e-8F, config.weight_decay, clip_scale,
                                            first_correction, second_correction);
        check_cuda(cudaGetLastError(), "update kernel");
        check_cuda(cudaDeviceSynchronize(), "update synchronize");
        if (host_model.step % static_cast<std::uint64_t>(config.checkpoint_every) == 0U || local_step + 1 == config.steps) {
            device_model.download(host_model);
            save_checkpoint(checkpoint, config, host_model, dataset_hash, tokenizer_hash, dataset.token_count, validation_tokens);
            const auto validation_score = score(validation, config, device_model, device_validation);
            std::cout << "{\"step\":" << host_model.step << ",\"validation_loss\":" << validation_score.loss
                      << ",\"validation_accuracy\":" << validation_score.accuracy << ",\"validation_perplexity\":"
                      << std::exp(std::min(20.0, validation_score.loss)) << "}\n";
        }
    }
}

struct Arguments {
    std::string train;
    std::string validation;
    std::string test;
    std::string checkpoint = "checkpoint.bin";
    std::string resume;
    std::string resume_base;
    std::uint64_t max_train_tokens = 16000000ULL;
    std::uint64_t max_validation_tokens = 2000000ULL;
    std::uint64_t max_test_tokens = 2000000ULL;
    Config config;
};

Arguments parse(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto value = [&]() -> std::string { require(index + 1 < argc, "missing value for " + key); return argv[++index]; };
        if (key == "--train") arguments.train = value();
        else if (key == "--validation") arguments.validation = value();
        else if (key == "--test") arguments.test = value();
        else if (key == "--checkpoint") arguments.checkpoint = value();
        else if (key == "--resume") arguments.resume = value();
        else if (key == "--resume-base") arguments.resume_base = value();
        else if (key == "--steps") arguments.config.steps = std::stoi(value());
        else if (key == "--batch") arguments.config.batch = std::stoi(value());
        else if (key == "--context") arguments.config.context = std::stoi(value());
        else if (key == "--hidden") arguments.config.hidden = std::stoi(value());
        else if (key == "--embedding") arguments.config.embedding = std::stoi(value());
        else if (key == "--seed") arguments.config.seed = std::stoull(value());
        else if (key == "--max-train-tokens") arguments.max_train_tokens = std::stoull(value());
        else if (key == "--max-validation-tokens") arguments.max_validation_tokens = std::stoull(value());
        else if (key == "--max-test-tokens") arguments.max_test_tokens = std::stoull(value());
        else throw std::runtime_error("unknown argument " + key);
    }
    require(!arguments.train.empty() && !arguments.validation.empty() && !arguments.test.empty(), "train, validation, and test paths are required");
    require(arguments.config.context > 0 && arguments.config.context <= kMaxContext && arguments.config.embedding > 0 && arguments.config.embedding <= kMaxEmbedding &&
                arguments.config.hidden > 0 && arguments.config.hidden <= kMaxHidden && arguments.config.batch > 0 && arguments.config.steps > 0,
            "invalid CUDA CCT configuration");
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse(argc, argv);
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "no CUDA device available; enable a Colab GPU runtime");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        const auto train_data = read_dataset(arguments.train, arguments.config, arguments.max_train_tokens);
        const auto validation = read_dataset(arguments.validation, arguments.config, arguments.max_validation_tokens);
        const auto test = read_dataset(arguments.test, arguments.config, arguments.max_test_tokens);
        const auto dataset_hash = train_data.file_hash ^ (validation.file_hash << 1U) ^ (test.file_hash << 2U);
        const auto tokenizer_hash = fnv1a("stage10-byte-fallback-v1|vocab=512|boundary=8");
        HostModel host;
        const auto resume_path = !arguments.resume.empty() ? arguments.resume : arguments.resume_base;
        if (resume_path.empty()) {
            host.parameters = initial_parameters(arguments.config);
            host.first.assign(host.parameters.size(), 0.0F);
            host.second.assign(host.parameters.size(), 0.0F);
        } else {
            host = load_checkpoint(resume_path, arguments.config, arguments.resume.empty() ? 0ULL : dataset_hash, tokenizer_hash);
            if (!arguments.resume.empty() || !arguments.resume_base.empty()) {
                host.first.assign(host.parameters.size(), 0.0F);
                host.second.assign(host.parameters.size(), 0.0F);
                host.step = 0;
            }
        }
        DeviceModel device_model;
        device_model.allocate(host.parameters.size());
        device_model.upload(host);
        DeviceDataset device_train; device_train.upload(train_data);
        DeviceDataset device_validation; device_validation.upload(validation);
        DeviceDataset device_test; device_test.upload(test);
        const auto before = score(validation, arguments.config, device_model, device_validation);
        train(train_data, arguments.config, device_model, device_train, host, arguments.checkpoint, dataset_hash, tokenizer_hash, validation.token_count, validation, device_validation);
        device_model.download(host);
        const auto after = score(validation, arguments.config, device_model, device_validation);
        const auto test_score = score(test, arguments.config, device_model, device_test);
        save_checkpoint(arguments.checkpoint, arguments.config, host, dataset_hash, tokenizer_hash, train_data.token_count, validation.token_count);
        std::cout << std::setprecision(10)
                  << "{\"status\":\"PASS\",\"backend\":\"cuda\",\"device\":\"" << properties.name
                  << "\",\"compute_capability\":" << properties.major << "." << properties.minor
                  << ",\"vocab_size\":" << kVocab << ",\"embedding_dim\":" << arguments.config.embedding
                  << ",\"hidden_dim\":" << arguments.config.hidden << ",\"context\":" << arguments.config.context
                  << ",\"steps\":" << host.step << ",\"train_tokens\":" << train_data.token_count
                  << ",\"validation_tokens\":" << validation.token_count << ",\"test_tokens\":" << test.token_count
                  << ",\"validation_before_loss\":" << before.loss << ",\"validation_after_loss\":" << after.loss
                  << ",\"validation_after_perplexity\":" << std::exp(std::min(20.0, after.loss))
                  << ",\"validation_accuracy\":" << after.accuracy << ",\"test_loss\":" << test_score.loss
                  << ",\"test_perplexity\":" << std::exp(std::min(20.0, test_score.loss)) << ",\"test_accuracy\":" << test_score.accuracy
                  << ",\"dataset_hash\":" << dataset_hash << ",\"tokenizer_hash\":" << tokenizer_hash
                  << ",\"checkpoint\":\"" << arguments.checkpoint << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_train error: " << error.what() << '\n';
        return 2;
    }
}
