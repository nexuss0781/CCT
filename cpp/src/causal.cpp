#include "cct/causal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <unistd.h>

namespace cct {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw CausalGraphError(message);
}

bool finite(double value) {
    return std::isfinite(value);
}

bool contains_id(const std::vector<EventId>& values, EventId value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::vector<EventId> sorted_unique(std::vector<EventId> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::uint64_t mix_hash(std::uint64_t state, std::uint64_t value) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    return state;
}

std::uint64_t double_bits(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

void atomic_write_snapshot(const std::string& path, const std::string& content) {
    const std::filesystem::path target(path);
    const auto parent = target.parent_path().empty() ? std::filesystem::path(".") : target.parent_path();
    std::error_code directory_error;
    std::filesystem::create_directories(parent, directory_error);
    require(!directory_error, "could not create causal snapshot parent directory");
    const auto template_path = (parent / (target.filename().string() + ".tmp.XXXXXX")).string();
    std::vector<char> temporary_template(template_path.begin(), template_path.end());
    temporary_template.push_back('\0');
    const auto descriptor = ::mkstemp(temporary_template.data());
    require(descriptor >= 0, "could not create causal snapshot temporary file");
    const auto temporary_path = std::string(temporary_template.data());
    std::size_t written = 0U;
    while (written < content.size()) {
        const auto count = ::write(descriptor, content.data() + written, content.size() - written);
        if (count <= 0) {
            ::close(descriptor);
            static_cast<void>(::unlink(temporary_path.c_str()));
            throw CausalGraphError("could not write causal snapshot temporary file");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
        static_cast<void>(::unlink(temporary_path.c_str()));
        throw CausalGraphError("could not durably flush causal snapshot temporary file");
    }
    if (::rename(temporary_path.c_str(), target.c_str()) != 0) {
        static_cast<void>(::unlink(temporary_path.c_str()));
        throw CausalGraphError("could not atomically publish causal snapshot");
    }
    const auto directory_descriptor = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(directory_descriptor >= 0, "could not open causal snapshot parent directory");
    const auto directory_sync = ::fsync(directory_descriptor);
    const auto directory_close = ::close(directory_descriptor);
    require(directory_sync == 0 && directory_close == 0, "could not durably publish causal snapshot directory entry");
}

}  // namespace

CausalEventStore::CausalEventStore(CausalStoreConfig config) : config_(std::move(config)) {
    require(config_.payload_dim > 0, "payload dimension must be positive");
    require(config_.coordinate_dim > 0, "coordinate dimension must be positive");
    require(config_.coordinate_min.size() == config_.coordinate_dim &&
                config_.coordinate_max.size() == config_.coordinate_dim,
            "coordinate bounds dimension mismatch");
    for (std::size_t index = 0; index < config_.coordinate_dim; ++index) {
        require(finite(config_.coordinate_min[index]) && finite(config_.coordinate_max[index]) &&
                    config_.coordinate_min[index] <= config_.coordinate_max[index],
                "invalid coordinate bounds");
    }
}

bool CausalEventStore::contains(EventId id) const noexcept {
    return std::any_of(ordered_events_.begin(), ordered_events_.end(),
                       [id](const CausalEvent& item) { return item.id == id; });
}

std::size_t CausalEventStore::size() const noexcept { return ordered_events_.size(); }

const CausalEvent& CausalEventStore::event(EventId id) const {
    const auto iterator = std::find_if(ordered_events_.begin(), ordered_events_.end(),
                                       [id](const CausalEvent& item) { return item.id == id; });
    if (iterator == ordered_events_.end()) throw CausalGraphError("event ID not found");
    return *iterator;
}

std::vector<EventId> CausalEventStore::existing_parent_ids(const CausalEvent& item) const {
    std::vector<EventId> result;
    for (const auto parent : item.causal_parents) {
        if (contains(parent)) result.push_back(parent);
    }
    return sorted_unique(std::move(result));
}

void CausalEventStore::validate_event(const CausalEvent& item) const {
    require(item.schema_version == CausalEvent::kSchemaVersion, "unsupported event schema version");
    require(item.id != 0, "event ID must be nonzero");
    require(item.semantic_payload.size() == config_.payload_dim, "event payload dimension mismatch");
    require(item.coordinates.size() == config_.coordinate_dim, "event coordinate dimension mismatch");
    for (std::size_t index = 0; index < item.coordinates.size(); ++index) {
        require(finite(item.coordinates[index]) && item.coordinates[index] >= config_.coordinate_min[index] &&
                    item.coordinates[index] <= config_.coordinate_max[index],
                "event coordinate outside declared range");
    }
    for (const auto value : item.semantic_payload) require(finite(value), "event payload contains non-finite value");
    require(std::is_sorted(item.causal_parents.begin(), item.causal_parents.end()),
            "causal parents must be sorted");
    require(std::adjacent_find(item.causal_parents.begin(), item.causal_parents.end()) == item.causal_parents.end(),
            "causal parents must be unique");
    require(!contains_id(item.causal_parents, item.id), "event cannot be its own parent");
    require(std::is_sorted(item.unresolved_parent_ids.begin(), item.unresolved_parent_ids.end()),
            "unresolved parents must be sorted");
    require(std::adjacent_find(item.unresolved_parent_ids.begin(), item.unresolved_parent_ids.end()) ==
                item.unresolved_parent_ids.end(),
            "unresolved parents must be unique");
    for (const auto parent : item.causal_parents) {
        const bool exists = contains(parent);
        const bool unresolved = contains_id(item.unresolved_parent_ids, parent);
        require(exists != unresolved, "parent must exist or be explicitly unresolved, but not both");
        require(exists || config_.allow_unresolved_parents, "missing parent is not allowed by store configuration");
        if (exists) {
            const auto parent_timestamp = event(parent).timestamp;
            const bool valid_timestamp = config_.temporal_policy == TemporalCausalityPolicy::AllowSameTimestamp
                                              ? parent_timestamp <= item.timestamp
                                              : parent_timestamp < item.timestamp;
            require(valid_timestamp, "causal parent timestamp violates the configured temporal policy");
        }
    }
    for (const auto unresolved : item.unresolved_parent_ids) {
        require(unresolved != item.id && !contains(unresolved), "unresolved parent is already present or self-referential");
    }
    require(finite(item.uncertainty.confidence) && item.uncertainty.confidence >= 0.0 &&
                item.uncertainty.confidence <= 1.0,
            "uncertainty confidence must be in [0,1]");
    if (item.intervention.has_value()) {
        require(item.intervention->variable < config_.payload_dim, "intervention variable out of range");
        require(finite(item.intervention->value), "intervention value is non-finite");
        require(item.intervention->mode != EventMode::Observed, "observed mode must not be stored as intervention");
    }
}

void CausalEventStore::insert(const CausalEvent& item) {
    require(!contains(item.id), "duplicate event ID");
    validate_event(item);
    ordered_events_.push_back(item);
    try {
        validate_acyclic();
    } catch (...) {
        ordered_events_.pop_back();
        throw;
    }
}

std::vector<EventId> CausalEventStore::sorted_ids() const {
    std::vector<EventId> result;
    result.reserve(ordered_events_.size());
    for (const auto& item : ordered_events_) result.push_back(item.id);
    return sorted_unique(std::move(result));
}

std::vector<EventId> CausalEventStore::parents_of(EventId id) const {
    const auto& item = event(id);
    return sorted_unique(item.causal_parents);
}

std::vector<EventId> CausalEventStore::children_of(EventId id) const {
    require(contains(id), "event ID not found");
    std::vector<EventId> result;
    for (const auto& item : ordered_events_) {
        if (contains_id(item.causal_parents, id)) result.push_back(item.id);
    }
    return sorted_unique(std::move(result));
}

std::vector<EventId> CausalEventStore::causal_past(EventId id) const {
    require(contains(id), "event ID not found");
    std::set<EventId> visited;
    std::vector<EventId> stack = parents_of(id);
    while (!stack.empty()) {
        const auto current = stack.back();
        stack.pop_back();
        if (!visited.insert(current).second) continue;
        const auto ancestors = parents_of(current);
        stack.insert(stack.end(), ancestors.begin(), ancestors.end());
    }
    return {visited.begin(), visited.end()};
}

std::vector<EventId> CausalEventStore::causal_future(EventId id) const {
    require(contains(id), "event ID not found");
    std::set<EventId> visited;
    std::vector<EventId> stack = children_of(id);
    while (!stack.empty()) {
        const auto current = stack.back();
        stack.pop_back();
        if (!visited.insert(current).second) continue;
        const auto descendants = children_of(current);
        stack.insert(stack.end(), descendants.begin(), descendants.end());
    }
    return {visited.begin(), visited.end()};
}

bool CausalEventStore::has_cycle() const {
    enum class Mark : std::uint8_t { None, Active, Done };
    std::map<EventId, Mark> marks;
    for (const auto id : sorted_ids()) marks[id] = Mark::None;
    std::function<bool(EventId)> visit = [&](EventId id) {
        auto& mark = marks[id];
        if (mark == Mark::Active) return true;
        if (mark == Mark::Done) return false;
        mark = Mark::Active;
        for (const auto parent : parents_of(id)) {
            if (contains(parent) && visit(parent)) return true;
        }
        mark = Mark::Done;
        return false;
    };
    for (const auto id : sorted_ids()) {
        if (marks[id] == Mark::None && visit(id)) return true;
    }
    return false;
}

void CausalEventStore::validate_acyclic() const {
    require(!has_cycle(), "causal-parent edges must form a DAG");
}

std::vector<EventId> CausalEventStore::topological_order() const {
    std::map<EventId, std::size_t> indegree;
    std::map<EventId, std::vector<EventId>> children;
    for (const auto id : sorted_ids()) indegree[id] = 0;
    for (const auto& item : ordered_events_) {
        for (const auto parent : item.causal_parents) {
            if (!contains(parent)) continue;
            ++indegree[item.id];
            children[parent].push_back(item.id);
        }
    }
    std::priority_queue<EventId, std::vector<EventId>, std::greater<EventId>> ready;
    for (const auto& [id, degree] : indegree) {
        if (degree == 0) ready.push(id);
    }
    std::vector<EventId> result;
    result.reserve(ordered_events_.size());
    while (!ready.empty()) {
        const auto id = ready.top();
        ready.pop();
        result.push_back(id);
        auto& next = children[id];
        std::sort(next.begin(), next.end());
        for (const auto child : next) {
            if (--indegree[child] == 0) ready.push(child);
        }
    }
    require(result.size() == ordered_events_.size(), "causal graph has a cycle");
    return result;
}

std::string CausalEventStore::serialize_snapshot() const {
    std::ostringstream output;
    output << "CCT_CAUSAL_EVENT_SNAPSHOT_V2\n" << std::setprecision(17);
    output << "CONFIG " << config_.payload_dim << ' ' << config_.coordinate_dim << ' '
           << (config_.allow_unresolved_parents ? 1 : 0);
    for (const auto value : config_.coordinate_min) output << ' ' << value;
    for (const auto value : config_.coordinate_max) output << ' ' << value;
    output << '\n' << "POLICY " << static_cast<unsigned int>(config_.temporal_policy) << '\n';
    std::vector<const CausalEvent*> events;
    for (const auto& item : ordered_events_) events.push_back(&item);
    std::sort(events.begin(), events.end(), [](const auto* left, const auto* right) { return left->id < right->id; });
    for (const auto* item : events) {
        output << "EVENT " << item->id << ' ' << item->schema_version << ' ' << item->timestamp << ' '
               << static_cast<unsigned int>(item->provenance) << ' '
               << static_cast<unsigned int>(item->uncertainty.kind) << ' ' << item->uncertainty.confidence << ' '
               << (item->intervention.has_value() ? 1 : 0);
        if (item->intervention.has_value()) {
            output << ' ' << item->intervention->variable << ' ' << item->intervention->value << ' '
                   << static_cast<unsigned int>(item->intervention->mode);
        }
        output << ' ' << item->semantic_payload.size();
        for (const auto value : item->semantic_payload) output << ' ' << value;
        output << ' ' << item->coordinates.size();
        for (const auto value : item->coordinates) output << ' ' << value;
        output << ' ' << item->causal_parents.size();
        for (const auto value : item->causal_parents) output << ' ' << value;
        output << ' ' << item->unresolved_parent_ids.size();
        for (const auto value : item->unresolved_parent_ids) output << ' ' << value;
        output << ' ' << item->provenance_links.size();
        for (const auto value : item->provenance_links) output << ' ' << value;
        output << '\n';
    }
    return output.str();
}

std::string CausalEventStore::deterministic_export() const { return serialize_snapshot(); }

void CausalEventStore::save_snapshot(const std::string& path) const {
    atomic_write_snapshot(path, serialize_snapshot());
}

CausalEventStore CausalEventStore::deserialize_snapshot(const std::string& snapshot) {
    constexpr std::size_t maximum_snapshot_bytes = 64U * 1024U * 1024U;
    constexpr std::size_t maximum_dimension = 4096U;
    constexpr std::size_t maximum_events = 1'000'000U;
    constexpr std::size_t maximum_vector_count = 1'000'000U;
    require(snapshot.size() <= maximum_snapshot_bytes, "causal snapshot exceeds byte budget");
    std::istringstream input(snapshot);
    std::string header;
    std::getline(input, header);
    const bool version_one = header == "CCT_CAUSAL_EVENT_SNAPSHOT_V1";
    const bool version_two = header == "CCT_CAUSAL_EVENT_SNAPSHOT_V2";
    require(version_one || version_two, "invalid causal graph snapshot header");
    std::string token;
    std::size_t payload_dim = 0;
    std::size_t coordinate_dim = 0;
    int allow_unresolved = 0;
    input >> token >> payload_dim >> coordinate_dim >> allow_unresolved;
    require(static_cast<bool>(input) && token == "CONFIG", "missing causal graph configuration");
    require(payload_dim > 0U && payload_dim <= maximum_dimension && coordinate_dim > 0U && coordinate_dim <= maximum_dimension,
            "causal graph dimensions exceed snapshot budget");
    CausalStoreConfig config;
    config.payload_dim = payload_dim;
    config.coordinate_dim = coordinate_dim;
    config.allow_unresolved_parents = allow_unresolved != 0;
    config.coordinate_min.resize(coordinate_dim);
    config.coordinate_max.resize(coordinate_dim);
    for (auto& value : config.coordinate_min) input >> value;
    for (auto& value : config.coordinate_max) input >> value;
    require(static_cast<bool>(input), "causal graph coordinate bounds are truncated");
    if (version_two) {
        unsigned int policy = 0U;
        input >> token >> policy;
        require(static_cast<bool>(input) && token == "POLICY" && policy <= static_cast<unsigned int>(TemporalCausalityPolicy::AllowSameTimestamp),
                "causal temporal policy is invalid");
        config.temporal_policy = static_cast<TemporalCausalityPolicy>(policy);
    }
    CausalEventStore store(config);
    std::size_t event_count = 0U;
    while (input >> token) {
        require(++event_count <= maximum_events, "causal graph event count exceeds snapshot budget");
        require(token == "EVENT", "invalid event record in causal graph snapshot");
        CausalEvent item;
        unsigned int provenance = 0;
        unsigned int uncertainty = 0;
        int has_intervention = 0;
        input >> item.id >> item.schema_version >> item.timestamp >> provenance >> uncertainty >> item.uncertainty.confidence >>
            has_intervention;
        require(static_cast<bool>(input), "causal event header is truncated");
        item.provenance = static_cast<ProvenanceKind>(provenance);
        item.uncertainty.kind = static_cast<UncertaintyKind>(uncertainty);
        if (has_intervention != 0) {
            Intervention intervention;
            unsigned int mode = 0;
            input >> intervention.variable >> intervention.value >> mode;
            intervention.mode = static_cast<EventMode>(mode);
            item.intervention = intervention;
        }
        std::size_t count = 0;
        input >> count;
        require(static_cast<bool>(input) && count <= maximum_vector_count, "causal semantic payload count exceeds snapshot budget");
        item.semantic_payload.resize(count);
        for (auto& value : item.semantic_payload) input >> value;
        input >> count;
        require(static_cast<bool>(input) && count <= maximum_vector_count, "causal coordinate count exceeds snapshot budget");
        item.coordinates.resize(count);
        for (auto& value : item.coordinates) input >> value;
        input >> count;
        require(static_cast<bool>(input) && count <= maximum_vector_count, "causal parent count exceeds snapshot budget");
        item.causal_parents.resize(count);
        for (auto& value : item.causal_parents) input >> value;
        input >> count;
        require(static_cast<bool>(input) && count <= maximum_vector_count, "causal unresolved-parent count exceeds snapshot budget");
        item.unresolved_parent_ids.resize(count);
        for (auto& value : item.unresolved_parent_ids) input >> value;
        input >> count;
        require(static_cast<bool>(input) && count <= maximum_vector_count, "causal provenance-link count exceeds snapshot budget");
        item.provenance_links.resize(count);
        for (auto& value : item.provenance_links) input >> value;
        require(!store.contains(item.id), "duplicate event ID in snapshot");
        store.ordered_events_.push_back(std::move(item));
    }
    for (const auto& item : store.ordered_events_) store.validate_event(item);
    store.validate_acyclic();
    return store;
}

CausalEventStore CausalEventStore::load_snapshot(const std::string& path) {
    constexpr std::uintmax_t maximum_snapshot_bytes = 64U * 1024U * 1024U;
    std::error_code size_error;
    const auto size = std::filesystem::file_size(path, size_error);
    require(!size_error && size <= maximum_snapshot_bytes, "causal snapshot file exceeds byte budget");
    std::ifstream stream(path);
    require(static_cast<bool>(stream), "could not read causal graph snapshot");
    std::ostringstream content;
    content << stream.rdbuf();
    return deserialize_snapshot(content.str());
}

CausalEventEncoder::CausalEventEncoder(CausalEncodingConfig config) : config_(std::move(config)) {
    require(config_.payload_dim > 0, "causal encoder payload dimension must be positive");
    require(config_.coordinate_dim > 0, "causal encoder coordinate dimension must be positive");
}

std::size_t CausalEventEncoder::encoded_dim() const noexcept {
    return config_.payload_dim * (config_.include_causal_edges ? 2U : 1U) +
           (config_.include_coordinates ? config_.coordinate_dim : 0U) + 1U +
           (config_.include_intervention_marker ? 3U : 0U) + (config_.include_uncertainty ? 2U : 0U) +
           (config_.include_provenance ? 5U : 0U) + (config_.include_causal_edges ? 2U : 0U);
}

EncodedCausalSequence CausalEventEncoder::encode(const std::vector<CausalEvent>& events) const {
    EncodedCausalSequence result;
    result.inputs.reserve(events.size());
    result.mask.assign(events.size(), 1);
    result.event_ids.reserve(events.size());
    for (const auto& item : events) {
        require(item.schema_version == CausalEvent::kSchemaVersion, "encoder received unsupported event schema");
        require(item.semantic_payload.size() == config_.payload_dim, "encoder payload dimension mismatch");
        require(item.coordinates.size() == config_.coordinate_dim, "encoder coordinate dimension mismatch");
        result.event_ids.push_back(item.id);
    }
    for (std::size_t index = 0; index < events.size(); ++index) {
        const auto& item = events[index];
        std::vector<double> encoded;
        encoded.reserve(encoded_dim());
        encoded.insert(encoded.end(), item.semantic_payload.begin(), item.semantic_payload.end());
        std::vector<double> parent_mean(config_.payload_dim, 0.0);
        std::size_t available_parent_count = 0;
        for (const auto parent_id : item.causal_parents) {
            const auto iterator = std::find_if(events.begin(), events.end(), [parent_id](const CausalEvent& candidate) {
                return candidate.id == parent_id;
            });
            const bool available = iterator != events.end() &&
                                   (!config_.prevent_future_leakage || iterator->timestamp < item.timestamp);
            if (!available) {
                ++result.excluded_future_parent_count;
                continue;
            }
            for (std::size_t feature = 0; feature < config_.payload_dim; ++feature) {
                parent_mean[feature] += iterator->semantic_payload[feature];
            }
            ++available_parent_count;
        }
        if (config_.include_causal_edges) {
            if (available_parent_count > 0) {
                for (auto& value : parent_mean) value /= static_cast<double>(available_parent_count);
            }
            encoded.insert(encoded.end(), parent_mean.begin(), parent_mean.end());
        }
        if (config_.include_coordinates) encoded.insert(encoded.end(), item.coordinates.begin(), item.coordinates.end());
        encoded.push_back(static_cast<double>(item.timestamp) / 1000.0);
        if (config_.include_intervention_marker) {
            const auto mode = item.intervention.has_value() ? item.intervention->mode : EventMode::Observed;
            encoded.push_back(mode == EventMode::DoIntervention ? 1.0 : 0.0);
            encoded.push_back(mode == EventMode::Counterfactual ? 1.0 : 0.0);
            encoded.push_back(item.intervention.has_value() ? item.intervention->value : 0.0);
        }
        if (config_.include_uncertainty) {
            encoded.push_back(item.uncertainty.confidence);
            encoded.push_back(static_cast<double>(item.uncertainty.kind) / 3.0);
        }
        if (config_.include_provenance) {
            for (unsigned int kind = 0; kind < 5; ++kind) {
                encoded.push_back(static_cast<unsigned int>(item.provenance) == kind ? 1.0 : 0.0);
            }
        }
        if (config_.include_causal_edges) {
            encoded.push_back(static_cast<double>(item.causal_parents.size()));
            encoded.push_back(static_cast<double>(available_parent_count));
        }
        require(encoded.size() == encoded_dim(), "causal encoder produced an invalid feature width");
        result.inputs.push_back(std::move(encoded));
    }
    return result;
}

GraphConditionedSequence::GraphConditionedSequence(GraphConditionedConfig config)
    : config_(std::move(config)), encoder_(config_.encoding), core_(config_.sequence) {
    require(config_.sequence.input_dim == encoder_.encoded_dim(),
            "graph-conditioned sequence input_dim must equal causal encoder width");
}

EncodedCausalSequence GraphConditionedSequence::encode(const std::vector<CausalEvent>& events) const {
    return encoder_.encode(events);
}

SequenceOutput GraphConditionedSequence::forward(const std::vector<CausalEvent>& events) const {
    const auto encoded = encode(events);
    return core_.forward(encoded.inputs, encoded.mask);
}

SequenceOutput GraphConditionedSequence::forward_scan(const std::vector<CausalEvent>& events) const {
    const auto encoded = encode(events);
    return core_.forward_scan(encoded.inputs, encoded.mask);
}

namespace {

std::vector<double> evaluate_truth(const StructuralModelTruth& truth, const std::vector<double>& noise,
                                   std::optional<Intervention> intervention) {
    std::vector<double> values(truth.variable_count, 0.0);
    for (std::size_t variable = 0; variable < truth.variable_count; ++variable) {
        if (intervention.has_value() && intervention->variable == variable) {
            values[variable] = intervention->value;
            continue;
        }
        double value = truth.intercepts[variable] + noise[variable];
        for (std::size_t parent = 0; parent < truth.variable_count; ++parent) {
            value += truth.coefficients[variable][parent] * values[parent] +
                     truth.nonlinear_coefficients[variable][parent] * (1000.0 * (std::tanh(values[parent]) - values[parent]));
        }
        values[variable] = value;
    }
    return values;
}

std::vector<std::size_t> all_prior_variables(std::size_t child) {
    std::vector<std::size_t> result;
    for (std::size_t parent = 0; parent < child; ++parent) result.push_back(parent);
    return result;
}

}  // namespace

CausalDataset SyntheticCausalGenerator::generate(const SyntheticCausalConfig& config) {
    require(config.variable_count >= 4, "synthetic causal generator requires at least four variables");
    require(config.training_samples > 8 && config.test_samples > 4, "synthetic causal sample counts are too small");
    std::mt19937_64 random(config.seed);
    std::normal_distribution<double> noise_distribution(0.0, 0.025);
    StructuralModelTruth truth;
    truth.variable_count = config.variable_count;
    truth.intercepts.assign(config.variable_count, 0.0);
    truth.coefficients.assign(config.variable_count, std::vector<double>(config.variable_count, 0.0));
    truth.nonlinear_coefficients.assign(config.variable_count, std::vector<double>(config.variable_count, 0.0));
    truth.parents.resize(config.variable_count);
    for (std::size_t child = 1; child < config.variable_count; ++child) {
        truth.parents[child] = all_prior_variables(child);
        if (child > 2) truth.parents[child].erase(truth.parents[child].begin());
        for (const auto parent : truth.parents[child]) {
            truth.coefficients[child][parent] = 0.34 + 0.06 * static_cast<double>((parent + child) % 4);
        }
        truth.intercepts[child] = -0.08 + 0.04 * static_cast<double>(child);
    }
    const auto nonlinear_child = config.variable_count - 1;
    if (!truth.parents[nonlinear_child].empty()) {
        const auto nonlinear_parent = truth.parents[nonlinear_child].back();
        truth.nonlinear_coefficients[nonlinear_child][nonlinear_parent] = config.seed % 2 == 0 ? 0.22 : 0.0;
    }
    const auto make_noise = [&](std::size_t sample_index) {
        std::vector<double> noise(config.variable_count, 0.0);
        const auto confounder = config.confounded_observations ?
                                    0.025 * std::sin(0.31 * static_cast<double>(sample_index + 1)) : 0.0;
        for (std::size_t variable = 0; variable < config.variable_count; ++variable) {
            noise[variable] = noise_distribution(random) + ((variable == 1 || variable == 2) ? confounder : 0.0);
        }
        return noise;
    };
    const auto make_sample = [&](std::size_t sample_index, bool allow_intervention) {
        const auto noise = make_noise(sample_index);
        StructuralSample sample;
        if (allow_intervention && sample_index % 3 == 0) {
            const auto variable = 1U + (sample_index % (config.variable_count - 1));
            const auto value = -0.72 + 0.17 * static_cast<double>(sample_index % 9);
            sample.interventions.push_back({variable, value, EventMode::DoIntervention});
        }
        sample.values = evaluate_truth(truth, noise,
                                       sample.interventions.empty() ? std::nullopt : std::optional<Intervention>(sample.interventions.front()));
        return sample;
    };
    CausalDataset dataset;
    dataset.evaluator_truth = truth;
    for (std::size_t index = 0; index < config.training_samples; ++index) {
        dataset.training_samples.push_back(make_sample(index, true));
    }
    for (std::size_t index = 0; index < config.test_samples; ++index) {
        dataset.test_samples.push_back(make_sample(index + 1000, false));
    }
    for (std::size_t index = 0; index < config.test_samples; ++index) {
        const auto noise = make_noise(index + 2000);
        const auto factual = evaluate_truth(truth, noise, std::nullopt);
        const Intervention intervention{1U + (index % (config.variable_count - 1)), 0.91 - 0.13 * static_cast<double>(index % 7), EventMode::DoIntervention};
        const auto intervened = evaluate_truth(truth, noise, intervention);
        const auto target = config.variable_count - 1;
        dataset.intervention_cases.push_back({factual, intervention, target, intervened[target]});
        const Intervention counterfactual_intervention{0, -0.83 + 0.11 * static_cast<double>(index % 8), EventMode::Counterfactual};
        const auto counterfactual = evaluate_truth(truth, noise, counterfactual_intervention);
        dataset.counterfactual_cases.push_back({factual, counterfactual_intervention, target, counterfactual});
    }
    const auto first_values = dataset.test_samples.front().values;
    for (std::size_t variable = 0; variable < config.variable_count; ++variable) {
        CausalEvent item;
        item.id = 1000 + variable;
        item.semantic_payload = {first_values[variable]};
        item.coordinates = {static_cast<double>(variable) / static_cast<double>(config.variable_count - 1),
                            static_cast<double>(variable % 2)};
        item.timestamp = static_cast<std::int64_t>(variable);
        for (const auto parent : truth.parents[variable]) item.causal_parents.push_back(1000 + parent);
        item.provenance = ProvenanceKind::Generated;
        item.uncertainty = {UncertaintyKind::Known, 1.0};
        dataset.visible_events.push_back(std::move(item));
    }
    if (dataset.visible_events.size() >= 4) {
        dataset.visible_events[1].causal_parents.push_back(dataset.visible_events[3].id);
        std::sort(dataset.visible_events[1].causal_parents.begin(), dataset.visible_events[1].causal_parents.end());
        dataset.invalid_fixture = true;
    }
    std::uint64_t fingerprint = 1469598103934665603ULL;
    fingerprint = mix_hash(fingerprint, config.seed);
    for (const auto& sample : dataset.test_samples) {
        for (const auto value : sample.values) fingerprint = mix_hash(fingerprint, double_bits(value));
    }
    dataset.dataset_fingerprint = fingerprint;
    return dataset;
}

CausalEventLearner::CausalEventLearner(std::size_t variable_count) : variable_count_(variable_count) {
    require(variable_count_ >= 2, "causal learner requires at least two variables");
}

void CausalEventLearner::validate_sample(const StructuralSample& sample) const {
    require(sample.values.size() == variable_count_, "causal sample dimension mismatch");
    for (const auto value : sample.values) require(finite(value), "causal sample contains non-finite value");
    for (const auto& intervention : sample.interventions) {
        require(intervention.variable < variable_count_ && finite(intervention.value), "invalid sample intervention");
    }
}

std::vector<double> CausalEventLearner::solve_ridge(const std::vector<std::vector<double>>& matrix,
                                                    const std::vector<double>& vector) const {
    require(!matrix.empty() && matrix.size() == vector.size(), "invalid regression system");
    const auto width = matrix.front().size();
    constexpr std::size_t maximum_features = 512U;
    require(width > 0U && width <= maximum_features, "causal regression feature dimension exceeds budget");
    require(matrix.size() <= 1'000'000U, "causal regression sample count exceeds budget");
    double maximum_abs = 0.0;
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        require(matrix[row].size() == width && std::isfinite(vector[row]), "inconsistent or non-finite regression row");
        for (const auto value : matrix[row]) {
            require(std::isfinite(value), "causal regression feature is non-finite");
            maximum_abs = std::max(maximum_abs, std::abs(value));
        }
    }
    const auto regularization = 1e-8 * std::max(1.0, maximum_abs * maximum_abs);
    const auto rows = matrix.size() + width;
    std::vector<std::vector<double>> qr(rows, std::vector<double>(width, 0.0));
    std::vector<double> rhs(rows, 0.0);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        qr[row] = matrix[row];
        rhs[row] = vector[row];
    }
    const auto regularization_sqrt = std::sqrt(regularization);
    for (std::size_t index = 0; index < width; ++index) qr[matrix.size() + index][index] = regularization_sqrt;
    for (std::size_t column = 0; column < width; ++column) {
        double norm_squared = 0.0;
        for (std::size_t row = column; row < rows; ++row) norm_squared += qr[row][column] * qr[row][column];
        const auto norm = std::sqrt(norm_squared);
        require(std::isfinite(norm) && norm > 1e-14 * std::max(1.0, maximum_abs), "ill-conditioned causal regression system");
        const auto alpha = qr[column][column] >= 0.0 ? -norm : norm;
        std::vector<double> reflector(rows - column, 0.0);
        for (std::size_t row = column; row < rows; ++row) reflector[row - column] = qr[row][column];
        reflector.front() -= alpha;
        double reflector_norm_squared = 0.0;
        for (const auto value : reflector) reflector_norm_squared += value * value;
        require(std::isfinite(reflector_norm_squared) && reflector_norm_squared > 0.0, "causal QR reflector is degenerate");
        for (std::size_t target = column; target < width; ++target) {
            double projection = 0.0;
            for (std::size_t row = column; row < rows; ++row) projection += reflector[row - column] * qr[row][target];
            projection *= 2.0 / reflector_norm_squared;
            for (std::size_t row = column; row < rows; ++row) qr[row][target] -= projection * reflector[row - column];
        }
        double rhs_projection = 0.0;
        for (std::size_t row = column; row < rows; ++row) rhs_projection += reflector[row - column] * rhs[row];
        rhs_projection *= 2.0 / reflector_norm_squared;
        for (std::size_t row = column; row < rows; ++row) rhs[row] -= rhs_projection * reflector[row - column];
        qr[column][column] = alpha;
        for (std::size_t row = column + 1U; row < rows; ++row) qr[row][column] = 0.0;
    }
    double minimum_diagonal = std::numeric_limits<double>::infinity();
    double maximum_diagonal = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
        const auto magnitude = std::abs(qr[index][index]);
        require(std::isfinite(magnitude), "causal QR diagonal is non-finite");
        minimum_diagonal = std::min(minimum_diagonal, magnitude);
        maximum_diagonal = std::max(maximum_diagonal, magnitude);
    }
    require(minimum_diagonal > maximum_diagonal * 1e-12, "ill-conditioned causal regression system");
    std::vector<double> result(width, 0.0);
    for (std::size_t reverse = width; reverse-- > 0U;) {
        double residual = rhs[reverse];
        for (std::size_t column = reverse + 1U; column < width; ++column) residual -= qr[reverse][column] * result[column];
        result[reverse] = residual / qr[reverse][reverse];
        require(std::isfinite(result[reverse]), "causal regression coefficient is non-finite");
    }
    return result;
}

void CausalEventLearner::fit(const std::vector<StructuralSample>& samples,
                             const std::vector<std::vector<std::size_t>>& parent_hypotheses,
                             bool intervention_aware) {
    require(parent_hypotheses.size() == variable_count_, "causal parent hypothesis count mismatch");
    require(samples.size() >= variable_count_ + 2, "causal learner requires more samples");
    parents_ = parent_hypotheses;
    coefficients_.assign(variable_count_, {});
    nonlinear_coefficients_.assign(variable_count_, {});
    intercepts_.assign(variable_count_, 0.0);
    confidences_.assign(variable_count_, 0.0);
    for (std::size_t child = 0; child < variable_count_; ++child) {
        for (const auto parent : parents_[child]) require(parent < variable_count_ && parent != child, "invalid parent hypothesis");
        std::vector<std::vector<double>> design;
        std::vector<std::vector<double>> linear_design;
        std::vector<double> target;
        for (const auto& sample : samples) {
            validate_sample(sample);
            if (!intervention_aware && !sample.interventions.empty()) continue;
            const bool target_intervened = std::any_of(sample.interventions.begin(), sample.interventions.end(),
                                                       [child](const Intervention& intervention) {
                                                           return intervention.variable == child;
                                                       });
            if (target_intervened) continue;
            std::vector<double> row{1.0};
            std::vector<double> linear_row{1.0};
            for (const auto parent : parents_[child]) {
                row.push_back(sample.values[parent]);
                linear_row.push_back(sample.values[parent]);
            }
            for (const auto parent : parents_[child]) row.push_back(1000.0 * (std::tanh(sample.values[parent]) - sample.values[parent]));
            design.push_back(std::move(row));
            linear_design.push_back(std::move(linear_row));
            target.push_back(sample.values[child]);
        }
        require(design.size() >= design.front().size(), "not enough usable samples for causal regression");
        const auto solution = solve_ridge(design, target);
        const auto linear_solution = solve_ridge(linear_design, target);
        double extended_error = 0.0;
        double linear_error = 0.0;
        for (std::size_t row = 0; row < design.size(); ++row) {
            double extended_prediction = 0.0;
            double linear_prediction = 0.0;
            for (std::size_t feature = 0; feature < design[row].size(); ++feature) extended_prediction += design[row][feature] * solution[feature];
            for (std::size_t feature = 0; feature < linear_design[row].size(); ++feature) linear_prediction += linear_design[row][feature] * linear_solution[feature];
            extended_error += (extended_prediction - target[row]) * (extended_prediction - target[row]);
            linear_error += (linear_prediction - target[row]) * (linear_prediction - target[row]);
        }
        const bool use_nonlinear = extended_error < linear_error * 0.90;
        intercepts_[child] = use_nonlinear ? solution[0] : linear_solution[0];
        coefficients_[child].resize(parents_[child].size(), 0.0);
        nonlinear_coefficients_[child].resize(parents_[child].size(), 0.0);
        double signal = 0.0;
        for (std::size_t index = 0; index < parents_[child].size(); ++index) {
            coefficients_[child][index] = use_nonlinear ? solution[1 + index] : linear_solution[1 + index];
            nonlinear_coefficients_[child][index] = use_nonlinear ? solution[1 + parents_[child].size() + index] : 0.0;
            signal += std::abs(coefficients_[child][index]) + std::abs(nonlinear_coefficients_[child][index]);
        }
        confidences_[child] = std::min(1.0, signal / (0.5 + static_cast<double>(parents_[child].size())));
    }
    fitted_ = true;
}

std::vector<EdgePrediction> CausalEventLearner::edge_predictions(double threshold) const {
    require(fitted_, "causal learner is not fitted");
    std::vector<EdgePrediction> result;
    for (std::size_t child = 0; child < variable_count_; ++child) {
        for (std::size_t index = 0; index < parents_[child].size(); ++index) {
            const auto signal = std::abs(coefficients_[child][index]) + std::abs(nonlinear_coefficients_[child][index]);
            result.push_back({parents_[child][index], child, std::min(1.0, signal), signal >= threshold});
        }
    }
    return result;
}

std::vector<double> CausalEventLearner::evaluate_world(const std::vector<double>& residuals,
                                                       std::optional<Intervention> intervention) const {
    std::vector<double> values(variable_count_, 0.0);
    for (std::size_t child = 0; child < variable_count_; ++child) {
        if (intervention.has_value() && intervention->variable == child) {
            values[child] = intervention->value;
            continue;
        }
        double value = intercepts_[child] + residuals[child];
        for (std::size_t index = 0; index < parents_[child].size(); ++index) {
            const auto parent = parents_[child][index];
            value += coefficients_[child][index] * values[parent] +
                     nonlinear_coefficients_[child][index] * (1000.0 * (std::tanh(values[parent]) - values[parent]));
        }
        values[child] = value;
    }
    return values;
}

CausalPrediction CausalEventLearner::predict_observation(const std::vector<double>& context, std::size_t target) const {
    require(fitted_, "causal learner is not fitted");
    require(context.size() == variable_count_ && target < variable_count_, "invalid causal observation query");
    return {context[target], 1.0, false};
}

CausalPrediction CausalEventLearner::predict_intervention(const std::vector<double>& context,
                                                          std::size_t variable,
                                                          double value,
                                                          std::size_t target) const {
    require(fitted_, "causal learner is not fitted");
    require(context.size() == variable_count_ && variable < variable_count_ && target < variable_count_,
            "invalid intervention query");
    if (graph_incomplete_ || graph_conflicting_) return {0.0, 0.0, true};
    std::vector<double> residuals(variable_count_, 0.0);
    for (std::size_t child = 0; child < variable_count_; ++child) {
        double expected = intercepts_[child];
        for (std::size_t index = 0; index < parents_[child].size(); ++index) {
            const auto parent = parents_[child][index];
            expected += coefficients_[child][index] * context[parent] +
                        nonlinear_coefficients_[child][index] * (1000.0 * (std::tanh(context[parent]) - context[parent]));
        }
        residuals[child] = context[child] - expected;
    }
    const auto values = evaluate_world(residuals, Intervention{variable, value, EventMode::DoIntervention});
    return {values[target], confidences_[target], false};
}

CausalPrediction CausalEventLearner::predict_counterfactual(const std::vector<double>& factual,
                                                             const Intervention& intervention,
                                                             std::size_t target) const {
    require(fitted_, "causal learner is not fitted");
    require(factual.size() == variable_count_ && intervention.variable < variable_count_ && target < variable_count_,
            "invalid counterfactual query");
    if (graph_incomplete_ || graph_conflicting_) return {0.0, 0.0, true};
    std::vector<double> residuals(variable_count_, 0.0);
    for (std::size_t child = 0; child < variable_count_; ++child) {
        double expected = intercepts_[child];
        for (std::size_t index = 0; index < parents_[child].size(); ++index) {
            const auto parent = parents_[child][index];
            expected += coefficients_[child][index] * factual[parent] +
                        nonlinear_coefficients_[child][index] * (1000.0 * (std::tanh(factual[parent]) - factual[parent]));
        }
        residuals[child] = factual[child] - expected;
    }
    const auto values = evaluate_world(residuals, intervention);
    return {values[target], confidences_[target], false};
}

void CausalEventLearner::set_graph_quality(bool incomplete, bool conflicting) noexcept {
    graph_incomplete_ = incomplete;
    graph_conflicting_ = conflicting;
}

}  // namespace cct
