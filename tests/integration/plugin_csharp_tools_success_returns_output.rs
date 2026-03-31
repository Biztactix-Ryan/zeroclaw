//! Verify acceptance criterion for story US-ZCL-39:
//!
//! > Successful responses return output string
//!
//! Reads the C# SDK Tools.cs source and asserts that the ToolCall method
//! returns the `Output` property from the deserialized response when the
//! call succeeds, rather than a bool or the raw bytes.

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
// ToolCall return type is string (the output)
// ---------------------------------------------------------------------------

#[test]
fn tool_call_returns_string() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public static string ToolCall("),
        "ToolCall must return string so callers receive the output directly"
    );
}

// ---------------------------------------------------------------------------
// Success path returns response.Output
// ---------------------------------------------------------------------------

#[test]
fn success_path_returns_response_output() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("return response.Output"),
        "On success, ToolCall must return response.Output (the tool's output string)"
    );
}

// ---------------------------------------------------------------------------
// Response type has Output property as string
// ---------------------------------------------------------------------------

#[test]
fn response_output_is_string_type() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public string Output { get; set; }"),
        "ToolCallResponse.Output must be a string to carry the tool's output"
    );
}

// ---------------------------------------------------------------------------
// Success is checked before returning output
// ---------------------------------------------------------------------------

#[test]
fn checks_success_before_returning_output() {
    let src = read_csharp_tools_source();

    // The success check (!response.Success) must appear before the return
    let success_check = src.find("!response.Success");
    let return_output = src.find("return response.Output");

    assert!(
        success_check.is_some(),
        "ToolCall must check response.Success before returning output"
    );
    assert!(
        return_output.is_some(),
        "ToolCall must return response.Output on success"
    );
    assert!(
        success_check.unwrap() < return_output.unwrap(),
        "Success check must precede the return of response.Output"
    );
}

// ---------------------------------------------------------------------------
// Error is checked before returning output
// ---------------------------------------------------------------------------

#[test]
fn checks_error_before_returning_output() {
    let src = read_csharp_tools_source();

    // The error check must appear before the return
    let error_check = src.find("response.Error is not null");
    let return_output = src.find("return response.Output");

    assert!(
        error_check.is_some(),
        "ToolCall must check response.Error before returning output"
    );
    assert!(
        return_output.is_some(),
        "ToolCall must return response.Output on success"
    );
    assert!(
        error_check.unwrap() < return_output.unwrap(),
        "Error check must precede the return of response.Output"
    );
}

// ---------------------------------------------------------------------------
// Response is deserialized from host function output
// ---------------------------------------------------------------------------

#[test]
fn response_deserialized_from_host_output() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("CallHostFunction<ToolCallRequest, ToolCallResponse>"),
        "ToolCall must deserialize the host response as ToolCallResponse to extract Output"
    );
}
