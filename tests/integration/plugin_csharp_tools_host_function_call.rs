//! Verify acceptance criterion for story US-ZCL-39:
//!
//! > Calls zeroclaw_tool_call host function via Extism .NET PDK
//!
//! Reads the C# SDK Tools.cs source and asserts that the implementation
//! imports and invokes the `zeroclaw_tool_call` host function through
//! the Extism PDK's DllImport mechanism.

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
// Host function is imported via DllImport from "extism"
// ---------------------------------------------------------------------------

#[test]
fn zeroclaw_tool_call_imported_via_dllimport() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains(r#"DllImport("extism""#),
        "Tools.cs must use DllImport(\"extism\") to import the host function via Extism PDK"
    );
}

// ---------------------------------------------------------------------------
// DllImport entry point is zeroclaw_tool_call
// ---------------------------------------------------------------------------

#[test]
fn dllimport_entry_point_is_zeroclaw_tool_call() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains(r#"EntryPoint = "zeroclaw_tool_call""#),
        "DllImport must specify EntryPoint = \"zeroclaw_tool_call\""
    );
}

// ---------------------------------------------------------------------------
// Extern method declaration for the host function exists
// ---------------------------------------------------------------------------

#[test]
fn extern_method_declared_for_host_function() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("static extern ulong zeroclaw_tool_call(ulong input)"),
        "Must declare: static extern ulong zeroclaw_tool_call(ulong input)"
    );
}

// ---------------------------------------------------------------------------
// ToolCall invokes the host function via CallHostFunction helper
// ---------------------------------------------------------------------------

#[test]
fn tool_call_invokes_host_function() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("zeroclaw_tool_call, request"),
        "ToolCall must pass zeroclaw_tool_call as the host function to invoke"
    );
}

// ---------------------------------------------------------------------------
// Uses Extism PDK types (Pdk.Allocate, MemoryBlock)
// ---------------------------------------------------------------------------

#[test]
fn uses_extism_pdk_allocate() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("Pdk.Allocate("),
        "Must use Pdk.Allocate to write input into Extism shared memory"
    );
}

#[test]
fn uses_extism_memory_block() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("MemoryBlock.Find("),
        "Must use MemoryBlock.Find to read the host function's output from shared memory"
    );
}

// ---------------------------------------------------------------------------
// Imports the Extism namespace
// ---------------------------------------------------------------------------

#[test]
fn imports_extism_namespace() {
    let src = read_csharp_tools_source();

    assert!(
        src.contains("using Extism;"),
        "Tools.cs must import the Extism namespace for PDK types"
    );
}
