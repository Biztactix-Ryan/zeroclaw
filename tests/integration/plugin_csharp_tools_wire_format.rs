//! Verify acceptance criterion for story US-ZCL-39:
//!
//! > Arguments serialized as JSON matching Rust SDK wire format
//!
//! Reads the C# SDK Tools.cs source and asserts that the JSON
//! serialization configuration produces snake_case field names
//! matching the Rust SDK's `ToolCallRequest` and `ToolCallResponse`
//! structs (tool_name, arguments, success, output, error).

use std::path::Path;

fn read_csharp_tools_source() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("sdks/csharp/src/Tools.cs");
    assert!(
        path.is_file(),
        "C# SDK Tools.cs not found at {}",
        path.display()
    );
    std::fs::read_to_string(&path).expect("failed to read C# Tools.cs")
}

// ---------------------------------------------------------------------------
// JSON options use snake_case naming policy
// ---------------------------------------------------------------------------

#[test]
fn json_options_use_snake_case_lower() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("JsonNamingPolicy.SnakeCaseLower"),
        "JsonSerializerOptions must use SnakeCaseLower naming policy to match Rust serde defaults"
    );
}

// ---------------------------------------------------------------------------
// ToolCallRequest has ToolName property (serializes to tool_name)
// ---------------------------------------------------------------------------

#[test]
fn request_has_tool_name_property() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public string ToolName { get; set; }"),
        "ToolCallRequest must have ToolName property (serializes to tool_name via SnakeCaseLower)"
    );
}

// ---------------------------------------------------------------------------
// ToolCallRequest has Arguments property (serializes to arguments)
// ---------------------------------------------------------------------------

#[test]
fn request_has_arguments_property() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public object Arguments { get; set; }"),
        "ToolCallRequest must have Arguments property (serializes to arguments via SnakeCaseLower)"
    );
}

// ---------------------------------------------------------------------------
// ToolCallResponse has Success property (matches Rust success field)
// ---------------------------------------------------------------------------

#[test]
fn response_has_success_property() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public bool Success { get; set; }"),
        "ToolCallResponse must have Success property (matches Rust wire format success field)"
    );
}

// ---------------------------------------------------------------------------
// ToolCallResponse has Output property (matches Rust output field)
// ---------------------------------------------------------------------------

#[test]
fn response_has_output_property() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public string Output { get; set; }"),
        "ToolCallResponse must have Output property (matches Rust wire format output field)"
    );
}

// ---------------------------------------------------------------------------
// ToolCallResponse has Error property (matches Rust error field)
// ---------------------------------------------------------------------------

#[test]
fn response_has_error_property() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public string? Error { get; set; }"),
        "ToolCallResponse must have Error property (matches Rust wire format optional error field)"
    );
}

// ---------------------------------------------------------------------------
// Uses System.Text.Json for serialization
// ---------------------------------------------------------------------------

#[test]
fn uses_system_text_json() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("using System.Text.Json;"),
        "Tools.cs must use System.Text.Json for JSON serialization"
    );
}

// ---------------------------------------------------------------------------
// Serializes request with JsonSerializer.SerializeToUtf8Bytes
// ---------------------------------------------------------------------------

#[test]
fn serializes_request_to_utf8_bytes() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("JsonSerializer.SerializeToUtf8Bytes(request, JsonOptions)"),
        "Must serialize the request with JsonOptions (SnakeCaseLower) to produce Rust-compatible wire format"
    );
}

// ---------------------------------------------------------------------------
// Deserializes response with JsonSerializer.Deserialize and JsonOptions
// ---------------------------------------------------------------------------

#[test]
fn deserializes_response_with_json_options() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("JsonSerializer.Deserialize<TResponse>(outputBytes, JsonOptions)"),
        "Must deserialize the response with JsonOptions to correctly parse snake_case Rust wire format"
    );
}

// ---------------------------------------------------------------------------
// PropertyNameCaseInsensitive is true for robust deserialization
// ---------------------------------------------------------------------------

#[test]
fn json_options_case_insensitive() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("PropertyNameCaseInsensitive = true"),
        "JsonOptions should set PropertyNameCaseInsensitive = true for robust wire format parsing"
    );
}
