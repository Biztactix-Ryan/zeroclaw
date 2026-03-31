//! Verify acceptance criterion for story US-ZCL-39:
//!
//! > Unit tests validate serialization format
//!
//! Checks that the C# SDK test project contains xunit tests covering
//! ToolCallRequest and ToolCallResponse serialization, ensuring the
//! wire format matches the Rust SDK's snake_case conventions.

use std::path::Path;

fn read_test_source() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("sdks/csharp/tests/ToolsSerializationTests.cs");
    assert!(
        path.is_file(),
        "ToolsSerializationTests.cs not found at {}",
        path.display()
    );
    std::fs::read_to_string(&path).expect("failed to read ToolsSerializationTests.cs")
}

// ---------------------------------------------------------------------------
// Test file exists
// ---------------------------------------------------------------------------

#[test]
fn csharp_tools_serialization_test_file_exists() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("sdks/csharp/tests/ToolsSerializationTests.cs");
    assert!(
        path.is_file(),
        "ToolsSerializationTests.cs not found at {}",
        path.display()
    );
}

// ---------------------------------------------------------------------------
// Uses xunit framework
// ---------------------------------------------------------------------------

#[test]
fn csharp_tools_serialization_uses_xunit() {
    let src = read_test_source();

    assert!(
        src.contains("using Xunit;"),
        "Unit tests must use xunit framework"
    );
    assert!(
        src.contains("[Fact]"),
        "Unit tests must have [Fact]-annotated test methods"
    );
}

// ---------------------------------------------------------------------------
// Tests ToolCallRequest serialization to snake_case
// ---------------------------------------------------------------------------

#[test]
fn csharp_tools_serialization_tests_request_snake_case() {
    let src = read_test_source();

    assert!(
        src.contains("\"tool_name\""),
        "Tests must verify tool_name field is serialized in snake_case"
    );
    assert!(
        src.contains("DoesNotContain(\"\\\"ToolName\\\"\"")
            || src.contains("DoesNotContain(\"\\\"ToolName\\\""),
        "Tests must verify PascalCase ToolName is NOT in serialized output"
    );
}

// ---------------------------------------------------------------------------
// Tests ToolCallResponse deserialization
// ---------------------------------------------------------------------------

#[test]
fn csharp_tools_serialization_tests_response_deserialization() {
    let src = read_test_source();

    assert!(
        src.contains("Deserialize<ToolCallResponse>")
            || src.contains("Deserializes_Success"),
        "Tests must verify ToolCallResponse deserialization"
    );
    assert!(
        src.contains("\"success\"") && src.contains("\"output\""),
        "Tests must verify success and output fields in deserialized response"
    );
}

// ---------------------------------------------------------------------------
// Tests cover error responses
// ---------------------------------------------------------------------------

#[test]
fn csharp_tools_serialization_tests_error_response() {
    let src = read_test_source();

    assert!(
        src.contains("Deserializes_Error") || src.contains("\"error\""),
        "Tests must cover error field deserialization"
    );
}

// ---------------------------------------------------------------------------
// Tests use matching JsonOptions (SnakeCaseLower)
// ---------------------------------------------------------------------------

#[test]
fn csharp_tools_serialization_tests_use_snake_case_options() {
    let src = read_test_source();

    assert!(
        src.contains("JsonNamingPolicy.SnakeCaseLower"),
        "Tests must use SnakeCaseLower naming policy matching Tools.cs"
    );
}
