//! Verify acceptance criterion for story US-ZCL-39:
//!
//! > ZeroClaw.PluginSdk.Tools class exposes ToolCall(toolName arguments)
//!
//! Reads the C# SDK Tools.cs source and asserts that the public static
//! class `Tools` exposes a `ToolCall(string toolName, object arguments)`
//! method with the correct signature.

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
// Tools class exists and is public static
// ---------------------------------------------------------------------------

#[test]
fn tools_class_is_public_static() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public static class Tools"),
        "Tools.cs must define a public static class named Tools"
    );
}

// ---------------------------------------------------------------------------
// Tools class lives in ZeroClaw.PluginSdk namespace
// ---------------------------------------------------------------------------

#[test]
fn tools_class_in_correct_namespace() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("namespace ZeroClaw.PluginSdk"),
        "Tools class must be in the ZeroClaw.PluginSdk namespace"
    );
}

// ---------------------------------------------------------------------------
// ToolCall method exists with correct signature
// ---------------------------------------------------------------------------

#[test]
fn tool_call_method_has_correct_signature() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public static string ToolCall(string toolName, object arguments)"),
        "Tools must expose: public static string ToolCall(string toolName, object arguments)"
    );
}

// ---------------------------------------------------------------------------
// ToolCall accepts toolName as first parameter
// ---------------------------------------------------------------------------

#[test]
fn tool_call_accepts_tool_name_parameter() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("string toolName"),
        "ToolCall must accept a string toolName parameter"
    );
}

// ---------------------------------------------------------------------------
// ToolCall accepts arguments as second parameter
// ---------------------------------------------------------------------------

#[test]
fn tool_call_accepts_arguments_parameter() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("object arguments"),
        "ToolCall must accept an object arguments parameter"
    );
}

// ---------------------------------------------------------------------------
// ToolCall returns a string
// ---------------------------------------------------------------------------

#[test]
fn tool_call_returns_string() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("public static string ToolCall("),
        "ToolCall must return a string"
    );
}
