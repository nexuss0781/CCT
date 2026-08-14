#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <utility>

namespace cct {

enum class OperationValueKind : std::uint8_t { String = 0, Integer = 1, Number = 2, Boolean = 3 };
enum class OperationAuthorizationClass : std::uint8_t { PublicRead = 0, TenantMember = 1, Reviewer = 2, Admin = 3 };
enum class OperationDecision : std::uint8_t { Accepted = 0, Rejected = 1, Abstained = 2 };
enum class OperationErrorClass : std::uint8_t {
    None = 0,
    SchemaVersionMismatch = 1,
    IdentityMissing = 2,
    UnknownOperation = 3,
    RequiredFieldMissing = 4,
    UnknownField = 5,
    TypeMismatch = 6,
    BoundsViolation = 7,
    EnumViolation = 8,
    AuthorizationDenied = 9,
    AmbiguousRequest = 10,
    EvidenceMissing = 11,
    SideEffectDenied = 12,
    DuplicateArgument = 13,
    SerializationError = 14,
    IdentityMismatch = 15
};

struct OperationValue {
    std::variant<std::string, std::int64_t, double, bool> value;

    OperationValue() : value(std::string{}) {}
    explicit OperationValue(std::string text) : value(std::move(text)) {}
    explicit OperationValue(const char* text) : value(std::string(text)) {}
    explicit OperationValue(std::int64_t integer) : value(integer) {}
    explicit OperationValue(double number) : value(number) {}
    explicit OperationValue(bool boolean) : value(boolean) {}

    OperationValueKind kind() const noexcept;
    std::string canonical() const;
    bool operator==(const OperationValue& other) const noexcept { return value == other.value; }
};

struct OperationFieldSchema {
    std::string name;
    std::string description;
    OperationValueKind kind = OperationValueKind::String;
    bool required = false;
    std::size_t maximum_bytes = 0;
    bool has_default = false;
    OperationValue default_value;
    std::int64_t minimum_integer = 0;
    std::int64_t maximum_integer = 0;
    double minimum_number = 0.0;
    double maximum_number = 0.0;
    std::vector<std::string> enum_values;
};

struct OperationSchema {
    static constexpr std::uint32_t kSchemaVersion = 1;
    std::string operation_id;
    std::string schema_version = "cct-operation-v1";
    std::string description;
    OperationAuthorizationClass authorization = OperationAuthorizationClass::TenantMember;
    std::vector<OperationFieldSchema> fields;
    bool allows_ambiguity = false;
    bool requires_evidence = false;
    bool side_effect_free = true;

    std::string identity_hash() const;
};

struct OperationArgument {
    std::string name;
    OperationValue value;
};

struct OperationEvidence {
    std::string source_id;
    std::string span;
    double confidence = 0.0;
};

struct OperationCall {
    static constexpr std::uint32_t kSchemaVersion = 1;
    std::string schema_version = "cct-operation-call-v1";
    std::string request_id;
    std::string tenant_id;
    std::string user_id;
    std::string role;
    std::string operation_id;
    std::vector<OperationArgument> arguments;
    std::vector<OperationEvidence> evidence;
    std::string operation_schema_hash;
    std::string operation_manifest_hash;
    std::string checkpoint_identity_hash;
    bool ambiguous = false;
    bool requests_external_action = false;

    std::string serialize() const;
    static OperationCall deserialize(const std::string& serialized);
};

struct OperationAuthContext {
    bool authenticated = false;
    std::string tenant_id;
    std::string user_id;
    std::vector<std::string> roles;
    std::vector<std::string> allowed_operations;
};

struct OperationResponse {
    OperationDecision decision = OperationDecision::Rejected;
    OperationErrorClass error_class = OperationErrorClass::SerializationError;
    std::string error_code;
    std::string operation_id;
    std::string schema_hash;
    std::vector<OperationArgument> normalized_arguments;
    std::string serialized_call;
    std::string explanation;
    std::string correction;
    std::string output;
    std::string audit_digest;
    bool side_effect_performed = false;
};

struct OperationDemonstration {
    std::string demonstration_id;
    std::string operation_id;
    std::string source_id;
    std::string source_span;
    std::string split;
    bool evaluator_only = false;
    OperationCall call;
    OperationDecision expected_decision = OperationDecision::Accepted;
    OperationErrorClass expected_error = OperationErrorClass::None;
    std::string expected_output;
    std::string expected_explanation;
    std::string correction;
    std::string source_hash;
    std::string demonstration_hash;
};

struct OperationManifest {
    static constexpr std::uint32_t kSchemaVersion = 1;
    std::string manifest_version = "cct-operation-manifest-v1";
    std::vector<OperationDemonstration> demonstrations;
    std::string manifest_hash;

    void finalize();
    std::string serialize() const;
    static OperationManifest deserialize(const std::string& serialized);
    bool contains_evaluator_training() const;
};

struct OperationCheckpointIdentity {
    static constexpr std::uint32_t kSchemaVersion = 1;
    std::string model_checkpoint_hash;
    std::string tokenizer_hash;
    std::string operation_schema_registry_hash;
    std::string operation_manifest_hash;
    std::string training_config_hash;
    std::string identity_hash;

    void finalize();
    std::string serialize() const;
    static OperationCheckpointIdentity deserialize(const std::string& serialized);
};

class OperationRegistry {
public:
    void register_schema(const OperationSchema& schema);
    const OperationSchema& schema(const std::string& operation_id) const;
    const std::vector<OperationSchema>& schemas() const noexcept { return schemas_; }
    std::string identity_hash() const;
    std::string explain(const std::string& operation_id) const;
    std::string serialize() const;
    static OperationRegistry deserialize(const std::string& serialized);

private:
    std::vector<OperationSchema> schemas_;
};

class OperationTeacher {
public:
    OperationTeacher(OperationRegistry registry, OperationManifest manifest, OperationCheckpointIdentity identity);

    const OperationRegistry& registry() const noexcept { return registry_; }
    const OperationManifest& manifest() const noexcept { return manifest_; }
    const OperationCheckpointIdentity& identity() const noexcept { return identity_; }

    OperationResponse respond(const OperationCall& call, const OperationAuthContext& auth) const;
    OperationResponse correct(const OperationCall& call, const OperationAuthContext& auth) const;
    std::string explain(const std::string& operation_id) const;
    std::string serialize() const;
    static OperationTeacher deserialize(const std::string& serialized);

private:
    OperationRegistry registry_;
    OperationManifest manifest_;
    OperationCheckpointIdentity identity_;

    OperationResponse reject(const OperationCall& call, OperationErrorClass error_class, const std::string& detail,
                             OperationDecision decision = OperationDecision::Rejected) const;
    static std::string error_code(OperationErrorClass error_class);
};

std::string operation_demonstration_hash(const OperationDemonstration& demonstration);
std::string operation_value_kind_name(OperationValueKind kind);
std::string operation_authorization_name(OperationAuthorizationClass authorization);
std::string operation_decision_name(OperationDecision decision);
std::string operation_error_name(OperationErrorClass error_class);

}  // namespace cct
