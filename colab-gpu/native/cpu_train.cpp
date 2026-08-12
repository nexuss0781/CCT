#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

static_assert(sizeof(Config) == 48U, "checkpoint Config layout must match the CUDA trainer");

struct Dataset {
    std::vector<std::uint32_t> sequences;
    std::size_t sequence_count = 0U;
    std::uint64_t token_count = 0ULL;
    std::uint64_t file_hash = 0ULL;
};

struct Model {
    std::vector<float> parameters;
    std::vector<float> first;
    std::vector<float> second;
    std::uint64_t step = 0ULL;
};

struct Score {
    double loss = 0.0;
    double accuracy = 0.0;
    double seconds = 0.0;
    std::uint64_t tokens = 0ULL;
};

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t fnv1a(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t fnv1a_tokens(const std::vector<std::uint32_t>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto value : values) {
        hash ^= value & 0xffU;
        hash *= 1099511628211ULL;
        hash ^= (value >> 8U) & 0xffU;
        hash *= 1099511628211ULL;
        hash ^= (value >> 16U) & 0xffU;
        hash *= 1099511628211ULL;
        hash ^= (value >> 24U) & 0xffU;
        hash *= 1099511628211ULL;
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

std::size_t input_offset(const Config& config) {
    return static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(config.embedding);
}

std::size_t recurrent_offset(const Config& config) {
    return input_offset(config) + static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(config.embedding);
}

std::size_t hidden_bias_offset(const Config& config) {
    return recurrent_offset(config) + static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(config.hidden);
}

std::size_t output_offset(const Config& config) {
    return hidden_bias_offset(config) + static_cast<std::size_t>(config.hidden);
}

std::size_t output_bias_offset(const Config& config) {
    return output_offset(config) + static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(config.hidden);
}

Dataset read_dataset(const std::string& path, const Config& config, const std::uint64_t max_tokens) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot open token stream " + path);
    std::uint64_t magic = 0ULL;
    std::uint64_t declared = 0ULL;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    stream.read(reinterpret_cast<char*>(&declared), sizeof(declared));
    require(magic == kStreamMagic, "invalid token stream magic: " + path);
    const auto bounded = std::min(declared, max_tokens);
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(bounded));
    stream.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(std::uint32_t)));
    require(stream || stream.eof(), "truncated token stream " + path);
    require(tokens.size() > static_cast<std::size_t>(config.context + 1), "not enough tokens in " + path);

    Dataset dataset;
    dataset.token_count = tokens.size();
    dataset.file_hash = fnv1a_tokens(tokens);
    dataset.sequence_count = (tokens.size() - 1U) / static_cast<std::size_t>(config.context);
    dataset.sequences.resize(dataset.sequence_count * static_cast<std::size_t>(config.context + 1));
    for (std::size_t sequence = 0U; sequence < dataset.sequence_count; ++sequence) {
        const auto source = sequence * static_cast<std::size_t>(config.context);
        const auto destination = sequence * static_cast<std::size_t>(config.context + 1);
        std::copy_n(tokens.begin() + static_cast<std::ptrdiff_t>(source), config.context + 1,
                    dataset.sequences.begin() + static_cast<std::ptrdiff_t>(destination));
        for (int index = 0; index <= config.context; ++index) {
            require(dataset.sequences[destination + static_cast<std::size_t>(index)] < kVocab, "token ID exceeds CPU vocabulary");
        }
    }
    return dataset;
}

std::vector<float> initial_parameters(const Config& config) {
    std::vector<float> parameters(parameter_count(config));
    std::uint64_t state = config.seed + 0x9e3779b97f4a7c15ULL;
    for (auto& value : parameters) {
        state ^= state >> 12U;
        state ^= state << 25U;
        state ^= state >> 27U;
        const auto random = static_cast<double>((state * 2685821657736338717ULL) % 1000000ULL) / 1000000.0 - 0.5;
        value = static_cast<float>(random * 0.04);
    }
    return parameters;
}

void save_checkpoint(const std::string& path, const Config& config, const Model& model, const std::uint64_t dataset_hash,
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

Model load_checkpoint(const std::string& path, const Config& expected, const std::uint64_t expected_dataset,
                      const std::uint64_t expected_tokenizer) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot open resume checkpoint");
    std::uint64_t magic = 0ULL;
    Config config{};
    std::uint64_t step = 0ULL;
    std::uint64_t dataset = 0ULL;
    std::uint64_t tokenizer = 0ULL;
    std::uint64_t train_tokens = 0ULL;
    std::uint64_t validation_tokens = 0ULL;
    std::uint64_t size = 0ULL;
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
    Model model;
    model.step = step;
    model.parameters.resize(size);
    model.first.resize(size);
    model.second.resize(size);
    stream.read(reinterpret_cast<char*>(model.parameters.data()), static_cast<std::streamsize>(size * sizeof(float)));
    stream.read(reinterpret_cast<char*>(model.first.data()), static_cast<std::streamsize>(size * sizeof(float)));
    stream.read(reinterpret_cast<char*>(model.second.data()), static_cast<std::streamsize>(size * sizeof(float)));
    require(static_cast<bool>(stream), "truncated or corrupt checkpoint");
    require(std::all_of(model.parameters.begin(), model.parameters.end(), [](const float value) { return std::isfinite(value); }),
            "checkpoint has non-finite parameters");
    return model;
}

void forward(const std::uint32_t* sequence, const Config& config, const std::vector<float>& parameters,
             std::vector<float>& states) {
    const auto input = input_offset(config);
    const auto recurrent = recurrent_offset(config);
    const auto hidden_bias = hidden_bias_offset(config);
    std::fill(states.begin(), states.end(), 0.0F);
    for (int time = 0; time < config.context; ++time) {
        for (int unit = 0; unit < config.hidden; ++unit) {
            float value = parameters[hidden_bias + static_cast<std::size_t>(unit)];
            for (int feature = 0; feature < config.embedding; ++feature) {
                value += parameters[input + static_cast<std::size_t>(unit * config.embedding + feature)] *
                         parameters[static_cast<std::size_t>(sequence[time]) * static_cast<std::size_t>(config.embedding) +
                                    static_cast<std::size_t>(feature)];
            }
            if (time > 0) {
                for (int prior = 0; prior < config.hidden; ++prior) {
                    value += parameters[recurrent + static_cast<std::size_t>(unit * config.hidden + prior)] *
                             states[static_cast<std::size_t>(time - 1) * static_cast<std::size_t>(config.hidden) +
                                    static_cast<std::size_t>(prior)];
                }
            }
            states[static_cast<std::size_t>(time) * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(unit)] = std::tanh(value);
        }
    }
}

void accumulate_sequence(const std::uint32_t* sequence, const Config& config, const std::vector<float>& parameters,
                         std::vector<float>& gradients, std::vector<float>& states, std::vector<float>& dh_next,
                         std::vector<float>& errors) {
    const auto input = input_offset(config);
    const auto recurrent = recurrent_offset(config);
    const auto hidden_bias = hidden_bias_offset(config);
    const auto output = output_offset(config);
    const auto output_bias = output_bias_offset(config);
    forward(sequence, config, parameters, states);
    std::fill(dh_next.begin(), dh_next.end(), 0.0F);
    for (int time = config.context - 1; time >= 0; --time) {
        float maximum = -std::numeric_limits<float>::max();
        for (int label = 0; label < kVocab; ++label) {
            float logit = parameters[output_bias + static_cast<std::size_t>(label)];
            for (int unit = 0; unit < config.hidden; ++unit) {
                logit += parameters[output + static_cast<std::size_t>(label * config.hidden + unit)] *
                         states[static_cast<std::size_t>(time) * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(unit)];
            }
            errors[static_cast<std::size_t>(label)] = logit;
            maximum = std::max(maximum, logit);
        }
        float total = 0.0F;
        for (int label = 0; label < kVocab; ++label) {
            errors[static_cast<std::size_t>(label)] = std::exp(errors[static_cast<std::size_t>(label)] - maximum);
            total += errors[static_cast<std::size_t>(label)];
        }
        const auto target = sequence[time + 1];
        for (int label = 0; label < kVocab; ++label) {
            auto& error = errors[static_cast<std::size_t>(label)];
            error /= total;
            error -= label == static_cast<int>(target) ? 1.0F : 0.0F;
            for (int unit = 0; unit < config.hidden; ++unit) {
                gradients[output + static_cast<std::size_t>(label * config.hidden + unit)] +=
                    error * states[static_cast<std::size_t>(time) * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(unit)];
            }
            gradients[output_bias + static_cast<std::size_t>(label)] += error;
        }
        std::vector<float> delta(static_cast<std::size_t>(config.hidden), 0.0F);
        for (int unit = 0; unit < config.hidden; ++unit) {
            float value = dh_next[static_cast<std::size_t>(unit)];
            for (int label = 0; label < kVocab; ++label) {
                value += parameters[output + static_cast<std::size_t>(label * config.hidden + unit)] *
                         errors[static_cast<std::size_t>(label)];
            }
            const auto state_index = static_cast<std::size_t>(time) * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(unit);
            delta[static_cast<std::size_t>(unit)] = value * (1.0F - states[state_index] * states[state_index]);
            gradients[hidden_bias + static_cast<std::size_t>(unit)] += delta[static_cast<std::size_t>(unit)];
            for (int feature = 0; feature < config.embedding; ++feature) {
                const auto embedding_index = static_cast<std::size_t>(sequence[time]) * static_cast<std::size_t>(config.embedding) +
                                             static_cast<std::size_t>(feature);
                const auto input_index = input + static_cast<std::size_t>(unit * config.embedding + feature);
                gradients[input_index] += delta[static_cast<std::size_t>(unit)] * parameters[embedding_index];
                gradients[embedding_index] += delta[static_cast<std::size_t>(unit)] * parameters[input_index];
            }
            if (time > 0) {
                for (int prior = 0; prior < config.hidden; ++prior) {
                    gradients[recurrent + static_cast<std::size_t>(unit * config.hidden + prior)] +=
                        delta[static_cast<std::size_t>(unit)] *
                        states[static_cast<std::size_t>(time - 1) * static_cast<std::size_t>(config.hidden) + static_cast<std::size_t>(prior)];
                }
            }
        }
        for (int prior = 0; prior < config.hidden; ++prior) {
            float value = 0.0F;
            for (int unit = 0; unit < config.hidden; ++unit) {
                value += parameters[recurrent + static_cast<std::size_t>(unit * config.hidden + prior)] * delta[static_cast<std::size_t>(unit)];
            }
            dh_next[static_cast<std::size_t>(prior)] = value;
        }
    }
}

Score score(const Dataset& dataset, const Config& config, const std::vector<float>& parameters) {
    const auto started = std::chrono::steady_clock::now();
    const auto output = output_offset(config);
    const auto output_bias = output_bias_offset(config);
    std::vector<float> state(static_cast<std::size_t>(config.hidden));
    std::vector<float> next_state(static_cast<std::size_t>(config.hidden));
    std::vector<float> logits(static_cast<std::size_t>(kVocab));
    double total_loss = 0.0;
    std::uint64_t correct = 0ULL;
    for (std::size_t sequence_index = 0U; sequence_index < dataset.sequence_count; ++sequence_index) {
        const auto* sequence = dataset.sequences.data() + sequence_index * static_cast<std::size_t>(config.context + 1);
        std::fill(state.begin(), state.end(), 0.0F);
        for (int time = 0; time < config.context; ++time) {
            for (int unit = 0; unit < config.hidden; ++unit) {
                float value = 0.0F;
                const auto hidden_bias = hidden_bias_offset(config);
                const auto input = input_offset(config);
                const auto recurrent = recurrent_offset(config);
                value = parameters[hidden_bias + static_cast<std::size_t>(unit)];
                for (int feature = 0; feature < config.embedding; ++feature) {
                    value += parameters[input + static_cast<std::size_t>(unit * config.embedding + feature)] *
                             parameters[static_cast<std::size_t>(sequence[time]) * static_cast<std::size_t>(config.embedding) +
                                        static_cast<std::size_t>(feature)];
                }
                for (int prior = 0; prior < config.hidden; ++prior) {
                    value += parameters[recurrent + static_cast<std::size_t>(unit * config.hidden + prior)] * state[static_cast<std::size_t>(prior)];
                }
                next_state[static_cast<std::size_t>(unit)] = std::tanh(value);
            }
            state = next_state;
            float maximum = -std::numeric_limits<float>::max();
            for (int label = 0; label < kVocab; ++label) {
                float logit = parameters[output_bias + static_cast<std::size_t>(label)];
                for (int unit = 0; unit < config.hidden; ++unit) {
                    logit += parameters[output + static_cast<std::size_t>(label * config.hidden + unit)] * state[static_cast<std::size_t>(unit)];
                }
                logits[static_cast<std::size_t>(label)] = logit;
                maximum = std::max(maximum, logit);
            }
            float total = 0.0F;
            for (int label = 0; label < kVocab; ++label) {
                logits[static_cast<std::size_t>(label)] = std::exp(logits[static_cast<std::size_t>(label)] - maximum);
                total += logits[static_cast<std::size_t>(label)];
            }
            const auto target = sequence[time + 1];
            float best = -1.0F;
            std::uint32_t predicted = 0U;
            for (int label = 0; label < kVocab; ++label) {
                logits[static_cast<std::size_t>(label)] /= total;
                if (logits[static_cast<std::size_t>(label)] > best) {
                    best = logits[static_cast<std::size_t>(label)];
                    predicted = static_cast<std::uint32_t>(label);
                }
            }
            total_loss += -std::log(std::max(logits[static_cast<std::size_t>(target)], 1.0e-20F));
            if (predicted == target) ++correct;
        }
    }
    const auto tokens = static_cast<std::uint64_t>(dataset.sequence_count) * static_cast<std::uint64_t>(config.context);
    require(tokens > 0ULL, "cannot score an empty token stream");
    Score result;
    result.loss = total_loss / static_cast<double>(tokens);
    result.accuracy = static_cast<double>(correct) / static_cast<double>(tokens);
    result.tokens = tokens;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

void train(const Dataset& dataset, const Config& config, Model& model, const std::string& checkpoint,
           const std::uint64_t dataset_hash, const std::uint64_t tokenizer_hash, const std::uint64_t validation_tokens,
           const Dataset& validation) {
    const auto size = model.parameters.size();
    const auto state_size = static_cast<std::size_t>(config.context) * static_cast<std::size_t>(config.hidden);
    std::vector<float> gradients(size);
    std::vector<float> states(state_size);
    std::vector<float> dh_next(static_cast<std::size_t>(config.hidden));
    std::vector<float> errors(static_cast<std::size_t>(kVocab));
    for (int local_step = 0; local_step < config.steps; ++local_step) {
        std::fill(gradients.begin(), gradients.end(), 0.0F);
        const auto start = static_cast<std::size_t>(local_step) * static_cast<std::size_t>(config.batch) % dataset.sequence_count;
        const auto count = std::min<std::size_t>(static_cast<std::size_t>(config.batch), dataset.sequence_count - start);
        for (std::size_t sequence_index = 0U; sequence_index < count; ++sequence_index) {
            const auto* sequence = dataset.sequences.data() + (start + sequence_index) * static_cast<std::size_t>(config.context + 1);
            accumulate_sequence(sequence, config, model.parameters, gradients, states, dh_next, errors);
        }
        double norm_squared = 0.0;
        for (const auto value : gradients) norm_squared += static_cast<double>(value) * static_cast<double>(value);
        const auto norm = std::sqrt(norm_squared);
        const auto clip_scale = static_cast<float>(norm > static_cast<double>(config.clip_norm) ? static_cast<double>(config.clip_norm) / norm : 1.0);
        ++model.step;
        const auto first_correction = 1.0F - std::pow(0.9F, static_cast<float>(model.step));
        const auto second_correction = 1.0F - std::pow(0.999F, static_cast<float>(model.step));
        require(first_correction > 0.0F && second_correction > 0.0F, "invalid optimizer bias correction");
        for (std::size_t index = 0U; index < size; ++index) {
            const auto gradient = gradients[index] * clip_scale;
            model.first[index] = 0.9F * model.first[index] + 0.1F * gradient;
            model.second[index] = 0.999F * model.second[index] + 0.001F * gradient * gradient;
            const auto normalized_first = model.first[index] / first_correction;
            const auto normalized_second = model.second[index] / second_correction;
            model.parameters[index] -= config.learning_rate *
                                       (normalized_first / (std::sqrt(normalized_second) + 1.0e-8F) +
                                        config.weight_decay * model.parameters[index]);
        }
        require(std::all_of(model.parameters.begin(), model.parameters.end(), [](const float value) { return std::isfinite(value); }),
                "training produced non-finite parameters");
        if (model.step % static_cast<std::uint64_t>(config.checkpoint_every) == 0ULL || local_step + 1 == config.steps) {
            save_checkpoint(checkpoint, config, model, dataset_hash, tokenizer_hash, dataset.token_count, validation_tokens);
            const auto validation_score = score(validation, config, model.parameters);
            std::cout << "{\"step\":" << model.step << ",\"validation_loss\":" << validation_score.loss
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
        auto value = [&]() -> std::string {
            require(index + 1 < argc, "missing value for " + key);
            return argv[++index];
        };
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
        else if (key == "--checkpoint-every") arguments.config.checkpoint_every = std::stoi(value());
        else if (key == "--seed") arguments.config.seed = std::stoull(value());
        else if (key == "--max-train-tokens") arguments.max_train_tokens = std::stoull(value());
        else if (key == "--max-validation-tokens") arguments.max_validation_tokens = std::stoull(value());
        else if (key == "--max-test-tokens") arguments.max_test_tokens = std::stoull(value());
        else throw std::runtime_error("unknown argument " + key);
    }
    require(!arguments.train.empty() && !arguments.validation.empty() && !arguments.test.empty(),
            "train, validation, and test paths are required");
    require(arguments.config.context > 0 && arguments.config.context <= kMaxContext && arguments.config.embedding > 0 &&
                arguments.config.embedding <= kMaxEmbedding && arguments.config.hidden > 0 && arguments.config.hidden <= kMaxHidden &&
                arguments.config.batch > 0 && arguments.config.steps > 0 && arguments.config.checkpoint_every > 0,
            "invalid CPU CCT configuration");
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse(argc, argv);
        const auto train_data = read_dataset(arguments.train, arguments.config, arguments.max_train_tokens);
        const auto validation = read_dataset(arguments.validation, arguments.config, arguments.max_validation_tokens);
        const auto test = read_dataset(arguments.test, arguments.config, arguments.max_test_tokens);
        const auto dataset_hash = train_data.file_hash ^ (validation.file_hash << 1U) ^ (test.file_hash << 2U);
        const auto tokenizer_hash = fnv1a("stage10-byte-fallback-v1|vocab=512|boundary=8");
        Model model;
        const auto resume_path = !arguments.resume.empty() ? arguments.resume : arguments.resume_base;
        if (resume_path.empty()) {
            model.parameters = initial_parameters(arguments.config);
            model.first.assign(model.parameters.size(), 0.0F);
            model.second.assign(model.parameters.size(), 0.0F);
        } else {
            model = load_checkpoint(resume_path, arguments.config, arguments.resume.empty() ? 0ULL : dataset_hash, tokenizer_hash);
            if (!arguments.resume.empty() || !arguments.resume_base.empty()) {
                model.first.assign(model.parameters.size(), 0.0F);
                model.second.assign(model.parameters.size(), 0.0F);
                model.step = 0ULL;
            }
        }
        const auto before = score(validation, arguments.config, model.parameters);
        train(train_data, arguments.config, model, arguments.checkpoint, dataset_hash, tokenizer_hash, validation.token_count, validation);
        const auto after = score(validation, arguments.config, model.parameters);
        const auto test_score = score(test, arguments.config, model.parameters);
        save_checkpoint(arguments.checkpoint, arguments.config, model, dataset_hash, tokenizer_hash, train_data.token_count, validation.token_count);
        std::cout << std::setprecision(10)
                  << "{\"status\":\"PASS\",\"backend\":\"native-c++20-cpu\",\"device\":\"host\""
                  << ",\"vocab_size\":" << kVocab << ",\"embedding_dim\":" << arguments.config.embedding
                  << ",\"hidden_dim\":" << arguments.config.hidden << ",\"context\":" << arguments.config.context
                  << ",\"steps\":" << model.step << ",\"train_tokens\":" << train_data.token_count
                  << ",\"validation_tokens\":" << validation.token_count << ",\"test_tokens\":" << test.token_count
                  << ",\"validation_before_loss\":" << before.loss << ",\"validation_after_loss\":" << after.loss
                  << ",\"validation_after_perplexity\":" << std::exp(std::min(20.0, after.loss))
                  << ",\"validation_accuracy\":" << after.accuracy << ",\"test_loss\":" << test_score.loss
                  << ",\"test_perplexity\":" << std::exp(std::min(20.0, test_score.loss)) << ",\"test_accuracy\":" << test_score.accuracy
                  << ",\"dataset_hash\":" << dataset_hash << ",\"tokenizer_hash\":" << tokenizer_hash
                  << ",\"checkpoint\":\"" << arguments.checkpoint << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu_train error: " << error.what() << '\n';
        return 2;
    }
}
