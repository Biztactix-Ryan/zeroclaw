//! Verify acceptance criterion for story US-ZCL-39:
//!
//! > Failed responses throw PluginException with error message
//!
//! Reads the C# SDK Tools.cs source and asserts that the ToolCall method
//! throws a PluginException containing the error message when the host
//! returns a failure response.

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
// Response type has Error property
// ---------------------------------------------------------------------------

#[test]
fn response_has_error_property() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public string? Error { get; set; }"),
        "ToolCallResponse must have a nullable Error property to carry the error message"
    );
}

// ---------------------------------------------------------------------------
// Error response throws PluginException with the error message
// ---------------------------------------------------------------------------

#[test]
fn error_response_throws_plugin_exception_with_message() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("throw new PluginException(response.Error)"),
        "When response.Error is set, ToolCall must throw PluginException with the error message"
    );
}

// ---------------------------------------------------------------------------
// Error check uses PluginException (not a generic exception)
// ---------------------------------------------------------------------------

#[test]
fn throws_plugin_exception_not_generic() {
    let src = read_csharp_tools_source();

    // Every throw should use PluginException specifically
    for line in src.lines() {
        if line.contains("throw new") {
            assert!(
                line.contains("PluginException"),
                "All exceptions in Tools.cs must be PluginException, found: {}",
                line.trim()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Error is checked before success check (fail-fast on explicit errors)
// ---------------------------------------------------------------------------

#[test]
fn error_checked_before_success() {
    let src = read_csharp_tools_source();

    let error_check = src.find("response.Error is not null");
    let success_check = src.find("!response.Success");

    assert!(
        error_check.is_some(),
        "ToolCall must check response.Error for explicit error messages"
    );
    assert!(
        success_check.is_some(),
        "ToolCall must check response.Success as a fallback"
    );
    assert!(
        error_check.unwrap() < success_check.unwrap(),
        "Error check must precede success check (fail-fast on explicit errors)"
    );
}

// ---------------------------------------------------------------------------
// success=false without error message still throws
// ---------------------------------------------------------------------------

#[test]
fn success_false_without_error_still_throws() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("throw new PluginException(\"tool call returned success=false\")"),
        "When success=false but no error message, ToolCall must still throw PluginException"
    );
}
