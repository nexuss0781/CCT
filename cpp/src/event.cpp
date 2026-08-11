#include "cct/event.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cct {

Manifold::Manifold(std::vector<std::size_t> dimensions)
    : dimensions_(std::move(dimensions)) {
    if (dimensions_.empty() || std::any_of(dimensions_.begin(), dimensions_.end(), [](std::size_t value) {
            return value == 0;
        })) {
        throw std::invalid_argument("dimensions must contain positive axes");
    }
}

std::vector<std::size_t> Manifold::checked_coordinates(
    const std::vector<std::int64_t>& coordinates) const {
    if (coordinates.size() != dimensions_.size()) {
        throw std::invalid_argument("coordinates have wrong dimensionality");
    }
    std::vector<std::size_t> checked;
    checked.reserve(coordinates.size());
    for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
        const auto coordinate = coordinates[axis];
        if (coordinate < 0 || static_cast<std::size_t>(coordinate) >= dimensions_[axis]) {
            throw std::out_of_range("coordinates are out of bounds");
        }
        checked.push_back(static_cast<std::size_t>(coordinate));
    }
    return checked;
}

std::size_t Manifold::flat_index(const std::vector<std::size_t>& coordinates) const {
    std::size_t index = 0;
    for (std::size_t axis = 0; axis < dimensions_.size(); ++axis) {
        index = index * dimensions_[axis] + coordinates[axis];
    }
    return index;
}

void Manifold::place_event(Event event) {
    const auto coordinates = checked_coordinates(event.temporal_tensor);
    const auto index = flat_index(coordinates);
    if (events_.contains(index)) {
        throw std::invalid_argument("cell is already occupied");
    }
    events_.emplace(index, std::move(event));
}

const Event* Manifold::get_event(const std::vector<std::int64_t>& coordinates) const {
    const auto checked = checked_coordinates(coordinates);
    const auto iterator = events_.find(flat_index(checked));
    return iterator == events_.end() ? nullptr : &iterator->second;
}

std::vector<Event> Manifold::events() const {
    std::vector<std::pair<std::size_t, Event>> ordered;
    ordered.reserve(events_.size());
    for (const auto& [index, event] : events_) {
        ordered.emplace_back(index, event);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    std::vector<Event> result;
    result.reserve(ordered.size());
    for (auto& [index, event] : ordered) {
        (void)index;
        result.push_back(std::move(event));
    }
    return result;
}

std::size_t Manifold::filled_cells() const noexcept {
    return events_.size();
}

std::string Manifold::repr() const {
    std::ostringstream stream;
    stream << "Manifold(dimensions: [";
    for (std::size_t axis = 0; axis < dimensions_.size(); ++axis) {
        if (axis > 0) {
            stream << ", ";
        }
        stream << dimensions_[axis];
    }
    stream << "], filled_cells: " << filled_cells() << ")";
    return stream.str();
}

}  // namespace cct
