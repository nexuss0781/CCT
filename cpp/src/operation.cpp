#include "cct/operation.hpp"

#include "cct/sft.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cct {
namespace {

constexpr std::size_t kMaximumOperationSchemas = 64U;
constexpr std::size_t kMaximumOperationFields = 32U;
constexpr std::size_t kMaximumOperationArguments = 64U;
constexpr std::size_t kMaximumOperationEvidence = 32U;
constexpr std::size_t kMaximumOperationBytes = 1U << 20U;
constexpr std::size_t kMaximumOperationStringBytes = 8192U;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_size(const std::string& value, const std::string& name) {
    require(!value.empty() && value.size() <= kMaximumOperationStringBytes, name + " is empty or exceeds the operation byte budget");
}

void require_finite(const double value, const std::string& name) {
    require(std::isfinite(value), name + " must be finite");
}

std::string quote(const std::string& value) {
    std::ostringstream output;
    output << std::quoted(value);
    return output.str();
}

std::string value_text(const OperationValue& value) {
    return value.canonical();
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

unsigned int enum_value(const OperationValueKind value) { return static_cast<unsigned int>(value); }
unsigned int enum_value(const OperationAuthorizationClass value) { return static_cast<unsigned int>(value); }
unsigned int enum_value(const OperationDecision value) { return static_cast<unsigned int>(value); }
unsigned int enum_value(const OperationErrorClass value) { return static_cast<unsigned int>(value); }

OperationValue parse_value(const OperationValueKind kind, const std::string& text) {
    if (kind == OperationValueKind::String) return OperationValue(text);
    if (kind == OperationValueKind::Boolean) {
        require(text == "true" || text == "false", "invalid boolean operation argument");
        return OperationValue(text == "true");
    }
    if (kind == OperationValueKind::Integer) {
        std::istringstream input(text);
        std::int64_t value = 0;
        input >> value;
        require(static_cast<bool>(input), "invalid integer operation argument");
        input >> std::ws;
        require(input.peek() == std::char_traits<char>::eof(), "invalid integer operation argument");
        return OperationValue(value);
    }
    std::istringstream input(text);
    double value = 0.0;
    input >> value;
    require(static_cast<bool>(input) && std::isfinite(value), "invalid numeric operation argument");
    input >> std::ws;
    require(input.peek() == std::char_traits<char>::eof(), "invalid numeric operation argument");
    return OperationValue(value);
}

void validate_value_against_field(const OperationValue& value, const OperationFieldSchema& field) {
    require(value.kind() == field.kind, "operation argument type mismatch for " + field.name);
    require(value.canonical().size() <= field.maximum_bytes, "operation argument exceeds field bound for " + field.name);
    if (field.kind == OperationValueKind::Integer) {
        const auto integer = std::get<std::int64_t>(value.value);
        require(integer >= field.minimum_integer && integer <= field.maximum_integer, "operation integer bound violation for " + field.name);
    } else if (field.kind == OperationValueKind::Number) {
        const auto number = std::get<double>(value.value);
        require_finite(number, "operation number");
        require(number >= field.minimum_number && number <= field.maximum_number, "operation number bound violation for " + field.name);
    }
    if (!field.enum_values.empty()) require(contains(field.enum_values, value.canonical()), "operation enum violation for " + field.name);
}

void validate_field(const OperationFieldSchema& field) {
    require_size(field.name, "operation field name");
    require_size(field.description, "operation field description");
    require(field.maximum_bytes > 0U && field.maximum_bytes <= kMaximumOperationStringBytes, "operation field byte bound is invalid");
    if (field.kind == OperationValueKind::Integer) require(field.minimum_integer <= field.maximum_integer, "operation integer bounds are inverted");
    if (field.kind == OperationValueKind::Number) {
        require_finite(field.minimum_number, "operation minimum number");
        require_finite(field.maximum_number, "operation maximum number");
        require(field.minimum_number <= field.maximum_number, "operation number bounds are inverted");
    }
    std::set<std::string> enums;
    for (const auto& option : field.enum_values) require_size(option, "operation enum value");
    for (const auto& option : field.enum_values) require(enums.insert(option).second, "duplicate operation enum value");
    if (field.has_default) validate_value_against_field(field.default_value, field);
}

void validate_schema(const OperationSchema& schema) {
    require_size(schema.operation_id, "operation ID");
    require_size(schema.schema_version, "operation schema version");
    require_size(schema.description, "operation schema description");
    require(schema.schema_version == "cct-operation-v1", "unsupported operation schema version");
    require(schema.side_effect_free, "operation schema cannot authorize side effects in Level 1");
    require(!schema.fields.empty() && schema.fields.size() <= kMaximumOperationFields, "operation field count is outside bounds");
    std::set<std::string> names;
    for (const auto& field : schema.fields) {
        validate_field(field);
        require(names.insert(field.name).second, "duplicate operation field name");
    }
}

std::string canonical_field(const OperationFieldSchema& field) {
    std::ostringstream output;
    output << field.name << '|' << field.description << '|' << enum_value(field.kind) << '|' << field.required << '|'
           << field.maximum_bytes << '|' << field.has_default << '|';
    if (field.has_default) output << value_text(field.default_value);
    output << '|' << field.minimum_integer << '|' << field.maximum_integer << '|' << std::setprecision(17)
           << field.minimum_number << '|' << field.maximum_number << '|';
    for (const auto& option : field.enum_values) output << option.size() << ':' << option;
    return output.str();
}

std::string canonical_schema(const OperationSchema& schema) {
    std::ostringstream output;
    output << "operation-schema-v1|" << schema.operation_id << '|' << schema.schema_version << '|' << schema.description << '|'
           << enum_value(schema.authorization) << '|' << schema.allows_ambiguity << '|' << schema.requires_evidence << '|'
           << schema.side_effect_free << '|';
    for (const auto& field : schema.fields) output << canonical_field(field) << ';';
    return output.str();
}

std::string canonical_call_without_identity(const OperationCall& call) {
    std::ostringstream output;
    output << "operation-call-v1|" << call.schema_version << '|' << call.request_id << '|' << call.tenant_id << '|' << call.user_id << '|'
           << call.role << '|' << call.operation_id << '|' << call.ambiguous << '|' << call.requests_external_action << '|'
           << call.operation_schema_hash << '|';
    for (const auto& argument : call.arguments) output << argument.name << '=' << enum_value(argument.value.kind()) << ':' << argument.value.canonical() << ';';
    output << '|';
    for (const auto& evidence : call.evidence) output << evidence.source_id << ':' << evidence.span << ':' << std::setprecision(17) << evidence.confidence << ';';
    return output.str();
}

std::string canonical_demonstration_without_hash(const OperationDemonstration& demonstration) {
    std::ostringstream output;
    output << "operation-demonstration-v1|" << demonstration.demonstration_id << '|' << demonstration.operation_id << '|'
           << demonstration.source_id << '|' << demonstration.source_span << '|' << demonstration.split << '|' << demonstration.evaluator_only << '|'
           << canonical_call_without_identity(demonstration.call) << '|' << enum_value(demonstration.expected_decision) << '|'
           << enum_value(demonstration.expected_error) << '|' << demonstration.expected_output << '|' << demonstration.expected_explanation << '|'
           << demonstration.correction << '|' << demonstration.source_hash;
    return output.str();
}

std::string canonical_manifest_without_hash(const OperationManifest& manifest) {
    std::ostringstream output;
    output << "operation-manifest-v1|" << manifest.manifest_version << '|';
    for (const auto& demonstration : manifest.demonstrations) output << demonstration.demonstration_hash << ';';
    return output.str();
}

void validate_call_shape(const OperationCall& call) {
    require(call.schema_version == "cct-operation-call-v1", "unsupported operation call schema version");
    require_size(call.request_id, "operation request ID");
    require_size(call.tenant_id, "operation tenant ID");
    require_size(call.user_id, "operation user ID");
    require_size(call.role, "operation role");
    require_size(call.operation_id, "operation ID");
    require(call.arguments.size() <= kMaximumOperationArguments, "operation argument count exceeds budget");
    require(call.evidence.size() <= kMaximumOperationEvidence, "operation evidence count exceeds budget");
    for (const auto& argument : call.arguments) {
        require_size(argument.name, "operation argument name");
        if (argument.value.kind() == OperationValueKind::String) require(argument.value.canonical().size() <= kMaximumOperationStringBytes, "operation string argument exceeds budget");
        if (argument.value.kind() == OperationValueKind::Number) require_finite(std::get<double>(argument.value.value), "operation numeric argument");
    }
    for (const auto& evidence : call.evidence) {
        require_size(evidence.source_id, "operation evidence source");
        require_size(evidence.span, "operation evidence span");
        require_finite(evidence.confidence, "operation evidence confidence");
        require(evidence.confidence >= 0.0 && evidence.confidence <= 1.0, "operation evidence confidence is outside bounds");
    }
}

std::string operation_call_text(const OperationCall& call) {
    return call.serialize();
}

}  // namespace

OperationValueKind OperationValue::kind() const noexcept {
    return static_cast<OperationValueKind>(value.index());
}

std::string OperationValue::canonical() const {
    std::ostringstream output;
    if (std::holds_alternative<std::string>(value)) return std::get<std::string>(value);
    if (std::holds_alternative<std::int64_t>(value)) return std::to_string(std::get<std::int64_t>(value));
    if (std::holds_alternative<double>(value)) {
        output << std::setprecision(17) << std::get<double>(value);
        return output.str();
    }
    return std::get<bool>(value) ? "true" : "false";
}

std::string OperationSchema::identity_hash() const { return sft_hash(canonical_schema(*this)); }

std::string OperationCall::serialize() const {
    validate_call_shape(*this);
    std::ostringstream output;
    output << "CCT_OPERATION_CALL_V1\n" << quote(schema_version) << ' ' << quote(request_id) << ' ' << quote(tenant_id) << ' '
           << quote(user_id) << ' ' << quote(role) << ' ' << quote(operation_id) << ' ' << quote(operation_schema_hash) << ' '
           << quote(operation_manifest_hash) << ' ' << quote(checkpoint_identity_hash) << ' ' << ambiguous << ' ' << requests_external_action << ' '
           << arguments.size();
    for (const auto& argument : arguments) output << ' ' << quote(argument.name) << ' ' << enum_value(argument.value.kind()) << ' ' << quote(argument.value.canonical());
    output << ' ' << evidence.size();
    for (const auto& evidence_item : evidence) output << ' ' << quote(evidence_item.source_id) << ' ' << quote(evidence_item.span) << ' ' << std::setprecision(17) << evidence_item.confidence;
    output << '\n';
    require(output.str().size() <= kMaximumOperationBytes, "serialized operation call exceeds budget");
    return output.str();
}

OperationCall OperationCall::deserialize(const std::string& serialized) {
    require(serialized.size() <= kMaximumOperationBytes, "serialized operation call exceeds budget");
    std::istringstream input(serialized);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_OPERATION_CALL_V1", "invalid operation call header");
    OperationCall call;
    std::size_t argument_count = 0U;
    input >> std::quoted(call.schema_version) >> std::quoted(call.request_id) >> std::quoted(call.tenant_id) >> std::quoted(call.user_id) >>
        std::quoted(call.role) >> std::quoted(call.operation_id) >> std::quoted(call.operation_schema_hash) >>
        std::quoted(call.operation_manifest_hash) >> std::quoted(call.checkpoint_identity_hash) >> call.ambiguous >> call.requests_external_action >> argument_count;
    require(static_cast<bool>(input) && argument_count <= kMaximumOperationArguments, "invalid operation argument count");
    for (std::size_t index = 0U; index < argument_count; ++index) {
        OperationArgument argument;
        unsigned int kind = 0U;
        std::string text;
        input >> std::quoted(argument.name) >> kind >> std::quoted(text);
        require(static_cast<bool>(input) && kind <= enum_value(OperationValueKind::Boolean), "invalid operation argument kind");
        argument.value = parse_value(static_cast<OperationValueKind>(kind), text);
        call.arguments.push_back(std::move(argument));
    }
    std::size_t evidence_count = 0U;
    input >> evidence_count;
    require(static_cast<bool>(input) && evidence_count <= kMaximumOperationEvidence, "invalid operation evidence count");
    for (std::size_t index = 0U; index < evidence_count; ++index) {
        OperationEvidence evidence;
        input >> std::quoted(evidence.source_id) >> std::quoted(evidence.span) >> evidence.confidence;
        require(static_cast<bool>(input), "truncated operation evidence");
        call.evidence.push_back(std::move(evidence));
    }
    input >> std::ws;
    require(input.peek() == std::char_traits<char>::eof(), "operation call has trailing data");
    validate_call_shape(call);
    return call;
}

std::string operation_demonstration_hash(const OperationDemonstration& demonstration) {
    require_size(demonstration.demonstration_id, "operation demonstration ID");
    require_size(demonstration.operation_id, "operation demonstration operation ID");
    require_size(demonstration.source_id, "operation demonstration source ID");
    require_size(demonstration.source_span, "operation demonstration source span");
    require_size(demonstration.split, "operation demonstration split");
    require_size(demonstration.source_hash, "operation demonstration source hash");
    validate_call_shape(demonstration.call);
    require(demonstration.call.operation_id == demonstration.operation_id, "operation demonstration call identity mismatch");
    return sft_hash(canonical_demonstration_without_hash(demonstration));
}

void OperationManifest::finalize() {
    require(manifest_version == "cct-operation-manifest-v1", "unsupported operation manifest version");
    require(!demonstrations.empty() && demonstrations.size() <= kMaximumOperationSchemas * kMaximumOperationFields,
            "operation demonstration count is outside bounds");
    std::set<std::string> ids;
    for (auto& demonstration : demonstrations) {
        require(ids.insert(demonstration.demonstration_id).second, "duplicate operation demonstration ID");
        demonstration.demonstration_hash = operation_demonstration_hash(demonstration);
    }
    manifest_hash = sft_hash(canonical_manifest_without_hash(*this));
}

std::string OperationManifest::serialize() const {
    require(manifest_version == "cct-operation-manifest-v1" && !manifest_hash.empty(), "operation manifest is not finalized");
    OperationManifest copy = *this;
    copy.finalize();
    require(copy.manifest_hash == manifest_hash, "operation manifest self-integrity mismatch");
    std::ostringstream output;
    output << "CCT_OPERATION_MANIFEST_V1\n" << quote(manifest_version) << ' ' << quote(manifest_hash) << ' ' << demonstrations.size() << '\n';
    for (const auto& demonstration : demonstrations) {
        output << quote(demonstration.demonstration_id) << ' ' << quote(demonstration.operation_id) << ' ' << quote(demonstration.source_id) << ' '
               << quote(demonstration.source_span) << ' ' << quote(demonstration.split) << ' ' << demonstration.evaluator_only << ' '
               << quote(demonstration.source_hash) << ' ' << enum_value(demonstration.expected_decision) << ' ' << enum_value(demonstration.expected_error) << ' '
               << quote(demonstration.expected_output) << ' ' << quote(demonstration.expected_explanation) << ' ' << quote(demonstration.correction) << ' '
               << quote(demonstration.demonstration_hash) << ' ' << quote(operation_call_text(demonstration.call));
        output << '\n';
    }
    require(output.str().size() <= kMaximumOperationBytes, "serialized operation manifest exceeds budget");
    return output.str();
}

OperationManifest OperationManifest::deserialize(const std::string& serialized) {
    require(serialized.size() <= kMaximumOperationBytes, "serialized operation manifest exceeds budget");
    std::istringstream input(serialized);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_OPERATION_MANIFEST_V1", "invalid operation manifest header");
    OperationManifest manifest;
    std::size_t count = 0U;
    input >> std::quoted(manifest.manifest_version) >> std::quoted(manifest.manifest_hash) >> count;
    require(static_cast<bool>(input) && count > 0U && count <= kMaximumOperationSchemas * kMaximumOperationFields, "invalid operation demonstration count");
    for (std::size_t index = 0U; index < count; ++index) {
        OperationDemonstration demonstration;
        unsigned int decision = 0U;
        unsigned int error = 0U;
        std::string call_text;
        input >> std::quoted(demonstration.demonstration_id) >> std::quoted(demonstration.operation_id) >> std::quoted(demonstration.source_id) >>
            std::quoted(demonstration.source_span) >> std::quoted(demonstration.split) >> demonstration.evaluator_only >> std::quoted(demonstration.source_hash) >>
            decision >> error >> std::quoted(demonstration.expected_output) >> std::quoted(demonstration.expected_explanation) >> std::quoted(demonstration.correction) >>
            std::quoted(demonstration.demonstration_hash) >> std::quoted(call_text);
        require(static_cast<bool>(input) && decision <= enum_value(OperationDecision::Abstained) && error <= enum_value(OperationErrorClass::IdentityMismatch),
                "invalid operation demonstration header");
        demonstration.expected_decision = static_cast<OperationDecision>(decision);
        demonstration.expected_error = static_cast<OperationErrorClass>(error);
        demonstration.call = OperationCall::deserialize(call_text);
        manifest.demonstrations.push_back(std::move(demonstration));
    }
    input >> std::ws;
    require(input.peek() == std::char_traits<char>::eof(), "operation manifest has trailing data");
    for (const auto& demonstration : manifest.demonstrations) require(operation_demonstration_hash(demonstration) == demonstration.demonstration_hash,
                                                                       "operation demonstration hash mismatch");
    const auto recorded_hash = manifest.manifest_hash;
    manifest.finalize();
    require(manifest.manifest_hash == recorded_hash, "operation manifest hash mismatch");
    return manifest;
}

bool OperationManifest::contains_evaluator_training() const {
    return std::any_of(demonstrations.begin(), demonstrations.end(), [](const OperationDemonstration& demonstration) {
        return demonstration.evaluator_only && demonstration.split == "train";
    });
}

void OperationCheckpointIdentity::finalize() {
    require_size(model_checkpoint_hash, "operation checkpoint model hash");
    require_size(tokenizer_hash, "operation checkpoint tokenizer hash");
    require_size(operation_schema_registry_hash, "operation schema registry hash");
    require_size(operation_manifest_hash, "operation manifest hash");
    require_size(training_config_hash, "operation training config hash");
    identity_hash = sft_hash(model_checkpoint_hash + "|" + tokenizer_hash + "|" + operation_schema_registry_hash + "|" + operation_manifest_hash + "|" + training_config_hash);
}

std::string OperationCheckpointIdentity::serialize() const {
    OperationCheckpointIdentity copy = *this;
    copy.finalize();
    require(copy.identity_hash == identity_hash, "operation checkpoint identity mismatch");
    std::ostringstream output;
    output << "CCT_OPERATION_CHECKPOINT_ID_V1\n" << quote(model_checkpoint_hash) << ' ' << quote(tokenizer_hash) << ' '
           << quote(operation_schema_registry_hash) << ' ' << quote(operation_manifest_hash) << ' ' << quote(training_config_hash) << ' ' << quote(identity_hash) << '\n';
    return output.str();
}

OperationCheckpointIdentity OperationCheckpointIdentity::deserialize(const std::string& serialized) {
    require(serialized.size() <= kMaximumOperationBytes, "serialized operation identity exceeds budget");
    std::istringstream input(serialized);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_OPERATION_CHECKPOINT_ID_V1", "invalid operation checkpoint identity header");
    OperationCheckpointIdentity identity;
    input >> std::quoted(identity.model_checkpoint_hash) >> std::quoted(identity.tokenizer_hash) >>
        std::quoted(identity.operation_schema_registry_hash) >> std::quoted(identity.operation_manifest_hash) >>
        std::quoted(identity.training_config_hash) >> std::quoted(identity.identity_hash) >> std::ws;
    require(static_cast<bool>(input) && input.peek() == std::char_traits<char>::eof(), "invalid operation checkpoint identity serialization");
    const auto recorded_hash = identity.identity_hash;
    identity.finalize();
    require(identity.identity_hash == recorded_hash, "operation checkpoint identity hash mismatch");
    return identity;
}

void OperationRegistry::register_schema(const OperationSchema& schema_value) {
    validate_schema(schema_value);
    require(schemas_.size() < kMaximumOperationSchemas, "operation schema registry exceeds budget");
    require(std::none_of(schemas_.begin(), schemas_.end(), [&](const OperationSchema& schema) { return schema.operation_id == schema_value.operation_id; }),
            "duplicate operation schema ID");
    schemas_.push_back(schema_value);
}

const OperationSchema& OperationRegistry::schema(const std::string& operation_id) const {
    const auto found = std::find_if(schemas_.begin(), schemas_.end(), [&](const OperationSchema& item) { return item.operation_id == operation_id; });
    require(found != schemas_.end(), "unknown operation schema");
    return *found;
}

std::string OperationRegistry::identity_hash() const {
    require(!schemas_.empty(), "operation schema registry is empty");
    std::vector<std::string> hashes;
    hashes.reserve(schemas_.size());
    for (const auto& schema_value : schemas_) hashes.push_back(schema_value.identity_hash());
    std::sort(hashes.begin(), hashes.end());
    std::ostringstream output;
    output << "operation-registry-v1|";
    for (const auto& hash : hashes) output << hash << ';';
    return sft_hash(output.str());
}

std::string OperationRegistry::explain(const std::string& operation_id) const {
    const auto& schema_value = schema(operation_id);
    std::ostringstream output;
    output << schema_value.operation_id << " (" << schema_value.schema_version << "): " << schema_value.description << ". Fields: ";
    for (std::size_t index = 0U; index < schema_value.fields.size(); ++index) {
        if (index != 0U) output << ", ";
        const auto& field = schema_value.fields[index];
        output << field.name << (field.required ? " required" : " optional") << " [" << operation_value_kind_name(field.kind) << "]";
        if (field.has_default) output << " default=" << field.default_value.canonical();
    }
    return output.str();
}

std::string OperationRegistry::serialize() const {
    require(schemas_.size() > 0U && schemas_.size() <= kMaximumOperationSchemas, "operation schema registry is outside bounds");
    std::ostringstream output;
    output << "CCT_OPERATION_REGISTRY_V1\n" << schemas_.size() << '\n';
    for (const auto& schema_value : schemas_) {
        validate_schema(schema_value);
        output << quote(schema_value.operation_id) << ' ' << quote(schema_value.schema_version) << ' ' << quote(schema_value.description) << ' '
               << enum_value(schema_value.authorization) << ' ' << schema_value.allows_ambiguity << ' ' << schema_value.requires_evidence << ' '
               << schema_value.side_effect_free << ' ' << schema_value.fields.size() << '\n';
        for (const auto& field : schema_value.fields) {
            output << quote(field.name) << ' ' << quote(field.description) << ' ' << enum_value(field.kind) << ' ' << field.required << ' '
                   << field.maximum_bytes << ' ' << field.has_default << ' ' << quote(field.default_value.canonical()) << ' '
                   << field.minimum_integer << ' ' << field.maximum_integer << ' ' << std::setprecision(17) << field.minimum_number << ' '
                   << field.maximum_number << ' ' << field.enum_values.size();
            for (const auto& option : field.enum_values) output << ' ' << quote(option);
            output << '\n';
        }
    }
    require(output.str().size() <= kMaximumOperationBytes, "serialized operation registry exceeds budget");
    return output.str();
}

OperationRegistry OperationRegistry::deserialize(const std::string& serialized) {
    require(serialized.size() <= kMaximumOperationBytes, "serialized operation registry exceeds budget");
    std::istringstream input(serialized);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_OPERATION_REGISTRY_V1", "invalid operation registry header");
    std::size_t schema_count = 0U;
    input >> schema_count;
    require(static_cast<bool>(input) && schema_count > 0U && schema_count <= kMaximumOperationSchemas, "invalid operation schema count");
    OperationRegistry registry;
    for (std::size_t schema_index = 0U; schema_index < schema_count; ++schema_index) {
        OperationSchema schema_value;
        unsigned int authorization = 0U;
        std::size_t field_count = 0U;
        input >> std::quoted(schema_value.operation_id) >> std::quoted(schema_value.schema_version) >> std::quoted(schema_value.description) >>
            authorization >> schema_value.allows_ambiguity >> schema_value.requires_evidence >> schema_value.side_effect_free >> field_count;
        require(static_cast<bool>(input) && authorization <= enum_value(OperationAuthorizationClass::Admin) && field_count > 0U && field_count <= kMaximumOperationFields,
                "invalid operation schema record");
        schema_value.authorization = static_cast<OperationAuthorizationClass>(authorization);
        for (std::size_t field_index = 0U; field_index < field_count; ++field_index) {
            OperationFieldSchema field;
            unsigned int kind = 0U;
            std::size_t enum_count = 0U;
            std::string default_text;
            input >> std::quoted(field.name) >> std::quoted(field.description) >> kind >> field.required >> field.maximum_bytes >> field.has_default >>
                std::quoted(default_text) >> field.minimum_integer >> field.maximum_integer >> field.minimum_number >> field.maximum_number >> enum_count;
            require(static_cast<bool>(input) && kind <= enum_value(OperationValueKind::Boolean) && enum_count <= kMaximumOperationFields,
                    "invalid operation field record");
            field.kind = static_cast<OperationValueKind>(kind);
            if (field.has_default) field.default_value = parse_value(field.kind, default_text);
            for (std::size_t option_index = 0U; option_index < enum_count; ++option_index) {
                std::string option;
                input >> std::quoted(option);
                require(static_cast<bool>(input), "truncated operation enum record");
                field.enum_values.push_back(std::move(option));
            }
            schema_value.fields.push_back(std::move(field));
        }
        registry.register_schema(schema_value);
    }
    input >> std::ws;
    require(input.peek() == std::char_traits<char>::eof(), "operation registry has trailing data");
    return registry;
}

OperationTeacher::OperationTeacher(OperationRegistry registry, OperationManifest manifest, OperationCheckpointIdentity identity)
    : registry_(std::move(registry)), manifest_(std::move(manifest)), identity_(std::move(identity)) {
    OperationCheckpointIdentity identity_check = identity_;
    identity_check.finalize();
    require(identity_check.identity_hash == identity_.identity_hash, "operation checkpoint identity is stale");
    OperationManifest manifest_check = manifest_;
    manifest_check.finalize();
    require(manifest_check.manifest_hash == manifest_.manifest_hash, "operation manifest identity is stale");
    require(!manifest_.contains_evaluator_training(), "operation manifest contains evaluator-only training data");
    require(registry_.identity_hash() == identity_.operation_schema_registry_hash, "operation registry identity does not match checkpoint");
    require(manifest_.manifest_hash == identity_.operation_manifest_hash, "operation manifest identity does not match checkpoint");
    for (const auto& demonstration : manifest_.demonstrations) {
        const auto& schema_value = registry_.schema(demonstration.operation_id);
        require(demonstration.call.operation_schema_hash == schema_value.identity_hash(), "operation demonstration schema lineage mismatch");
        require(demonstration.call.operation_manifest_hash == manifest_.manifest_hash, "operation demonstration manifest lineage mismatch");
        require(demonstration.call.checkpoint_identity_hash == identity_.identity_hash, "operation demonstration checkpoint lineage mismatch");
        require(demonstration.demonstration_hash == operation_demonstration_hash(demonstration), "operation demonstration identity is stale");
    }
}

OperationResponse OperationTeacher::reject(const OperationCall& call, const OperationErrorClass error_class, const std::string& detail,
                                           const OperationDecision decision) const {
    OperationResponse response;
    response.decision = decision;
    response.error_class = error_class;
    response.error_code = error_code(error_class);
    response.operation_id = call.operation_id;
    response.explanation = detail;
    response.correction = "Correct the request according to the declared operation schema; no external side effect was performed.";
    response.side_effect_performed = false;
    try {
        response.serialized_call = call.serialize();
        response.audit_digest = sft_hash(response.serialized_call + "|" + response.error_code + "|" + detail);
    } catch (const std::exception&) {
        response.audit_digest = sft_hash(call.request_id + "|" + response.error_code + "|serialization-error");
    }
    return response;
}

OperationResponse OperationTeacher::respond(const OperationCall& call, const OperationAuthContext& auth) const {
    if (call.schema_version != "cct-operation-call-v1") return reject(call, OperationErrorClass::SchemaVersionMismatch, "operation call schema version is unsupported");
    try {
        validate_call_shape(call);
    } catch (const std::exception& error) {
        return reject(call, OperationErrorClass::SerializationError, error.what());
    }
    if (!auth.authenticated || auth.tenant_id.empty() || auth.user_id.empty() || call.tenant_id != auth.tenant_id || call.user_id != auth.user_id) {
        return reject(call, OperationErrorClass::IdentityMissing, "authenticated tenant and user identity must match the operation call");
    }
    const OperationSchema* schema_value = nullptr;
    try {
        schema_value = &registry_.schema(call.operation_id);
    } catch (const std::exception&) {
        return reject(call, OperationErrorClass::UnknownOperation, "operation is not declared in the frozen schema registry");
    }
    if (call.operation_schema_hash != schema_value->identity_hash() || call.operation_manifest_hash != identity_.operation_manifest_hash ||
        call.checkpoint_identity_hash != identity_.identity_hash) {
        return reject(call, OperationErrorClass::IdentityMismatch, "operation schema, manifest, or checkpoint identity does not match the teacher");
    }
    const auto role_allowed = [&]() {
        if (schema_value->authorization == OperationAuthorizationClass::PublicRead) return true;
        if (schema_value->authorization == OperationAuthorizationClass::TenantMember) return contains(auth.roles, "member") || contains(auth.roles, "operator") || contains(auth.roles, "reviewer") || contains(auth.roles, "admin");
        if (schema_value->authorization == OperationAuthorizationClass::Reviewer) return contains(auth.roles, "reviewer") || contains(auth.roles, "admin");
        return contains(auth.roles, "admin");
    }();
    if (!role_allowed || (!auth.allowed_operations.empty() && !contains(auth.allowed_operations, call.operation_id))) {
        return reject(call, OperationErrorClass::AuthorizationDenied, "caller is not authorized for this operation");
    }
    if (call.requests_external_action || !schema_value->side_effect_free) return reject(call, OperationErrorClass::SideEffectDenied, "Level 1 operation teacher is side-effect free");
    if (call.ambiguous) return reject(call, OperationErrorClass::AmbiguousRequest, "request is ambiguous; provide one declared operation and complete arguments", OperationDecision::Abstained);
    if (schema_value->requires_evidence && call.evidence.empty()) return reject(call, OperationErrorClass::EvidenceMissing, "declared evidence is required before operation validation", OperationDecision::Abstained);

    std::map<std::string, OperationValue> supplied;
    for (const auto& argument : call.arguments) {
        if (supplied.find(argument.name) != supplied.end()) return reject(call, OperationErrorClass::DuplicateArgument, "operation argument is duplicated: " + argument.name);
        const auto field = std::find_if(schema_value->fields.begin(), schema_value->fields.end(), [&](const OperationFieldSchema& item) { return item.name == argument.name; });
        if (field == schema_value->fields.end()) return reject(call, OperationErrorClass::UnknownField, "operation argument is not declared: " + argument.name);
        try {
            validate_value_against_field(argument.value, *field);
        } catch (const std::exception& error) {
            const auto text = std::string(error.what());
            const auto type_error = text.find("type mismatch") != std::string::npos;
            const auto enum_error = text.find("enum violation") != std::string::npos;
            const auto bounds_error = text.find("bound") != std::string::npos || text.find("exceeds") != std::string::npos;
            return reject(call, type_error ? OperationErrorClass::TypeMismatch : (enum_error ? OperationErrorClass::EnumViolation :
                                      (bounds_error ? OperationErrorClass::BoundsViolation : OperationErrorClass::SerializationError)), text);
        }
        supplied.emplace(argument.name, argument.value);
    }
    std::vector<OperationArgument> normalized;
    for (const auto& field : schema_value->fields) {
        const auto found = supplied.find(field.name);
        if (found != supplied.end()) {
            normalized.push_back({field.name, found->second});
        } else if (field.required) {
            return reject(call, OperationErrorClass::RequiredFieldMissing, "required operation field is missing: " + field.name);
        } else if (field.has_default) {
            normalized.push_back({field.name, field.default_value});
        }
    }
    OperationCall normalized_call = call;
    normalized_call.arguments = std::move(normalized);
    OperationResponse response;
    response.decision = OperationDecision::Accepted;
    response.error_class = OperationErrorClass::None;
    response.error_code = error_code(OperationErrorClass::None);
    response.operation_id = call.operation_id;
    response.schema_hash = schema_value->identity_hash();
    response.normalized_arguments = normalized_call.arguments;
    response.serialized_call = normalized_call.serialize();
    response.explanation = schema_value->description;
    response.output = "validated operation " + schema_value->operation_id;
    response.audit_digest = sft_hash(response.serialized_call + "|accepted|" + response.output);
    response.side_effect_performed = false;
    return response;
}

OperationResponse OperationTeacher::correct(const OperationCall& call, const OperationAuthContext& auth) const {
    auto response = respond(call, auth);
    if (response.decision != OperationDecision::Accepted) {
        try {
            response.correction = explain(call.operation_id);
        } catch (const std::exception&) {
            response.correction = "Select a declared operation and provide its required fields.";
        }
    }
    return response;
}

std::string OperationTeacher::explain(const std::string& operation_id) const { return registry_.explain(operation_id); }

std::string OperationTeacher::serialize() const {
    std::ostringstream output;
    output << "CCT_OPERATION_TEACHER_V1\n" << quote(registry_.serialize()) << ' ' << quote(manifest_.serialize()) << ' ' << quote(identity_.serialize()) << '\n';
    require(output.str().size() <= kMaximumOperationBytes * 2U, "serialized operation teacher exceeds budget");
    return output.str();
}

OperationTeacher OperationTeacher::deserialize(const std::string& serialized) {
    require(serialized.size() <= kMaximumOperationBytes * 2U, "serialized operation teacher exceeds budget");
    std::istringstream input(serialized);
    std::string header;
    std::getline(input, header);
    require(header == "CCT_OPERATION_TEACHER_V1", "invalid operation teacher header");
    std::string registry_text;
    std::string manifest_text;
    std::string identity_text;
    input >> std::quoted(registry_text) >> std::quoted(manifest_text) >> std::quoted(identity_text) >> std::ws;
    require(static_cast<bool>(input) && input.peek() == std::char_traits<char>::eof(), "invalid operation teacher serialization");
    return OperationTeacher(OperationRegistry::deserialize(registry_text), OperationManifest::deserialize(manifest_text),
                            OperationCheckpointIdentity::deserialize(identity_text));
}

std::string OperationTeacher::error_code(const OperationErrorClass error_class) {
    switch (error_class) {
        case OperationErrorClass::None: return "OK";
        case OperationErrorClass::SchemaVersionMismatch: return "OP_SCHEMA_VERSION_MISMATCH";
        case OperationErrorClass::IdentityMissing: return "OP_IDENTITY_MISSING";
        case OperationErrorClass::UnknownOperation: return "OP_UNKNOWN";
        case OperationErrorClass::RequiredFieldMissing: return "OP_REQUIRED_FIELD_MISSING";
        case OperationErrorClass::UnknownField: return "OP_UNKNOWN_FIELD";
        case OperationErrorClass::TypeMismatch: return "OP_TYPE_MISMATCH";
        case OperationErrorClass::BoundsViolation: return "OP_BOUNDS";
        case OperationErrorClass::EnumViolation: return "OP_ENUM";
        case OperationErrorClass::AuthorizationDenied: return "OP_AUTH_DENIED";
        case OperationErrorClass::AmbiguousRequest: return "OP_AMBIGUOUS";
        case OperationErrorClass::EvidenceMissing: return "OP_EVIDENCE_MISSING";
        case OperationErrorClass::SideEffectDenied: return "OP_SIDE_EFFECT_DENIED";
        case OperationErrorClass::DuplicateArgument: return "OP_DUPLICATE_ARGUMENT";
        case OperationErrorClass::SerializationError: return "OP_SERIALIZATION_ERROR";
        case OperationErrorClass::IdentityMismatch: return "OP_IDENTITY_MISMATCH";
    }
    return "OP_SERIALIZATION_ERROR";
}

std::string operation_value_kind_name(const OperationValueKind kind) {
    switch (kind) {
        case OperationValueKind::String: return "string";
        case OperationValueKind::Integer: return "integer";
        case OperationValueKind::Number: return "number";
        case OperationValueKind::Boolean: return "boolean";
    }
    return "unknown";
}

std::string operation_authorization_name(const OperationAuthorizationClass authorization) {
    switch (authorization) {
        case OperationAuthorizationClass::PublicRead: return "public_read";
        case OperationAuthorizationClass::TenantMember: return "tenant_member";
        case OperationAuthorizationClass::Reviewer: return "reviewer";
        case OperationAuthorizationClass::Admin: return "admin";
    }
    return "unknown";
}

std::string operation_decision_name(const OperationDecision decision) {
    switch (decision) {
        case OperationDecision::Accepted: return "accepted";
        case OperationDecision::Rejected: return "rejected";
        case OperationDecision::Abstained: return "abstained";
    }
    return "unknown";
}

std::string operation_error_name(const OperationErrorClass error_class) {
    switch (error_class) {
        case OperationErrorClass::None: return "none";
        case OperationErrorClass::SchemaVersionMismatch: return "schema_version_mismatch";
        case OperationErrorClass::IdentityMissing: return "identity_missing";
        case OperationErrorClass::UnknownOperation: return "unknown_operation";
        case OperationErrorClass::RequiredFieldMissing: return "required_field_missing";
        case OperationErrorClass::UnknownField: return "unknown_field";
        case OperationErrorClass::TypeMismatch: return "type_mismatch";
        case OperationErrorClass::BoundsViolation: return "bounds_violation";
        case OperationErrorClass::EnumViolation: return "enum_violation";
        case OperationErrorClass::AuthorizationDenied: return "authorization_denied";
        case OperationErrorClass::AmbiguousRequest: return "ambiguous_request";
        case OperationErrorClass::EvidenceMissing: return "evidence_missing";
        case OperationErrorClass::SideEffectDenied: return "side_effect_denied";
        case OperationErrorClass::DuplicateArgument: return "duplicate_argument";
        case OperationErrorClass::SerializationError: return "serialization_error";
        case OperationErrorClass::IdentityMismatch: return "identity_mismatch";
    }
    return "unknown";
}

}  // namespace cct
