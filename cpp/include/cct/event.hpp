#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cct {

struct Event {
    std::vector<float> semantic_vector;
    std::vector<std::int64_t> temporal_tensor;
    std::vector<float> causal_potential_vector;
};

class Manifold {
public:
    explicit Manifold(std::vector<std::size_t> dimensions);

    void place_event(Event event);
    const Event* get_event(const std::vector<std::int64_t>& coordinates) const;
    std::vector<Event> events() const;
    std::size_t filled_cells() const noexcept;
    const std::vector<std::size_t>& dimensions() const noexcept { return dimensions_; }
    std::string repr() const;

private:
    std::vector<std::size_t> checked_coordinates(
        const std::vector<std::int64_t>& coordinates) const;
    std::size_t flat_index(const std::vector<std::size_t>& coordinates) const;

    std::vector<std::size_t> dimensions_;
    std::unordered_map<std::size_t, Event> events_;
};

}  // namespace cct
