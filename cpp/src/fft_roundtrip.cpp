#include "cct/field.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <fftw3.h>

namespace cct {

namespace {
std::size_t count_for(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
}
}

double fft_round_trip_error(const std::vector<double>& field,
                            const std::vector<std::size_t>& shape) {
    if (shape.empty() || shape.size() > 3) {
        throw NumericalError("FFTW round-trip supports one to three dimensions");
    }
    const auto count = count_for(shape);
    if (field.size() != count) {
        throw NumericalError("round-trip field has an unexpected size");
    }
    std::vector<int> dimensions;
    for (const auto value : shape) dimensions.push_back(static_cast<int>(value));
    const auto compact_count = (count / shape.back()) * (shape.back() / 2 + 1);
    auto* input = static_cast<double*>(fftw_malloc(sizeof(double) * count));
    auto* spectrum = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * compact_count));
    auto* output = static_cast<double*>(fftw_malloc(sizeof(double) * count));
    if (!input || !spectrum || !output) {
        if (input) fftw_free(input);
        if (spectrum) fftw_free(spectrum);
        if (output) fftw_free(output);
        throw NumericalError("FFTW allocation failed");
    }
    std::copy(field.begin(), field.end(), input);
    const auto forward = fftw_plan_dft_r2c(static_cast<int>(shape.size()), dimensions.data(), input, spectrum, FFTW_ESTIMATE);
    const auto inverse = fftw_plan_dft_c2r(static_cast<int>(shape.size()), dimensions.data(), spectrum, output, FFTW_ESTIMATE);
    if (!forward || !inverse) {
        if (forward) fftw_destroy_plan(forward);
        if (inverse) fftw_destroy_plan(inverse);
        fftw_free(input);
        fftw_free(spectrum);
        fftw_free(output);
        throw NumericalError("FFTW plan creation failed");
    }
    fftw_execute(forward);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(count);
    double maximum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        maximum = std::max(maximum, std::abs(output[index] * scale - field[index]));
    }
    fftw_destroy_plan(forward);
    fftw_destroy_plan(inverse);
    fftw_free(input);
    fftw_free(spectrum);
    fftw_free(output);
    return maximum;
}

}  // namespace cct
