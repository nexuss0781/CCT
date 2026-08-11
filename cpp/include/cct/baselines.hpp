#pragma once

#include "cct/sequence.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cct {

enum class BaselineKind { DenseCausalAttention, GRU, DiagonalSSM };

struct BaselineConfig {
    std::size_t input_dim = 1;
    std::size_t hidden_dim = 8;
    std::size_t output_dim = 1;
    std::uint64_t seed = 0;
};

class MatchedBaseline {
public:
    MatchedBaseline(BaselineKind kind, BaselineConfig config);

    BaselineKind kind() const noexcept { return kind_; }
    const BaselineConfig& config() const noexcept { return config_; }
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) const;
    double loss(const std::vector<std::vector<double>>& inputs,
                const std::vector<std::vector<double>>& targets,
                const std::vector<std::uint8_t>& mask = {}) const;
    void train_finite_difference(const std::vector<std::vector<std::vector<double>>>& input_batch,
                                 const std::vector<std::vector<std::vector<double>>>& target_batch,
                                 const std::vector<std::vector<std::uint8_t>>& masks,
                                 std::size_t epochs, double learning_rate,
                                 double clip_norm = 1.0);

    std::size_t parameter_count() const noexcept;
    std::size_t state_memory_bytes(std::size_t sequence_length) const noexcept;
    std::vector<double> parameter_vector() const;
    void set_parameter_vector(const std::vector<double>& values);
    std::string name() const;

private:
    BaselineKind kind_;
    BaselineConfig config_;
    std::vector<double> parameters_;

    void initialize();
    std::vector<double> matvec(const std::vector<double>& matrix, std::size_t rows,
                               std::size_t columns, const std::vector<double>& vector) const;
    std::vector<double> dense_forward_step(const std::vector<std::vector<double>>& inputs,
                                           std::size_t time) const;
    std::vector<double> gru_forward_step(const std::vector<double>& input,
                                         std::vector<double>& hidden) const;
    std::vector<double> ssm_forward_step(const std::vector<double>& input,
                                         std::vector<double>& hidden) const;
    double batch_loss(const std::vector<std::vector<std::vector<double>>>& input_batch,
                      const std::vector<std::vector<std::vector<double>>>& target_batch,
                      const std::vector<std::vector<std::uint8_t>>& masks) const;
};

}  // namespace cct
