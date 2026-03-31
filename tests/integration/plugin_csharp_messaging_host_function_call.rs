//! Verify acceptance criterion for story US-ZCL-40:
//!
//! > Calls zeroclaw_send_message and zeroclaw_get_channels host functions
//!
//! Reads the C# SDK Messaging.cs source and asserts that the implementation
//! imports and invokes the `zeroclaw_send_message` and `zeroclaw_get_channels`
//! host functions through the Extism PDK's DllImport mechanism.

use std::path::Path;

fn read_csharp_messaging_source() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("sdks/csharp/src/Messaging.cs");
    assert!(
        path.is_file(),
        "C# SDK Messaging.cs not found at {}",
        path.display()
    );
    std::fs::read_to_string(&path).expect("failed to read C# Messaging.cs")
}

// ---------------------------------------------------------------------------
// Host functions are imported via DllImport from "extism"
// ---------------------------------------------------------------------------

#[test]
fn zeroclaw_send_message_imported_via_dllimport() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains(r#"DllImport("extism", EntryPoint = "zeroclaw_send_message")"#),
        "Messaging.cs must use DllImport(\"extism\") to import zeroclaw_send_message"
    );
}

#[test]
fn zeroclaw_get_channels_imported_via_dllimport() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains(r#"DllImport("extism", EntryPoint = "zeroclaw_get_channels")"#),
        "Messaging.cs must use DllImport(\"extism\") to import zeroclaw_get_channels"
    );
}

// ---------------------------------------------------------------------------
// Extern method declarations for both host functions exist
// ---------------------------------------------------------------------------

#[test]
fn extern_method_declared_for_send_message() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains("static extern ulong zeroclaw_send_message(ulong input)"),
        "Must declare: static extern ulong zeroclaw_send_message(ulong input)"
    );
}

#[test]
fn extern_method_declared_for_get_channels() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains("static extern ulong zeroclaw_get_channels(ulong input)"),
        "Must declare: static extern ulong zeroclaw_get_channels(ulong input)"
    );
}

// ---------------------------------------------------------------------------
// Send invokes the zeroclaw_send_message host function
// ---------------------------------------------------------------------------

#[test]
fn send_invokes_send_message_host_function() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains("zeroclaw_send_message, request"),
        "Send must pass zeroclaw_send_message as the host function to invoke"
    );
}

// ---------------------------------------------------------------------------
// GetChannels invokes the zeroclaw_get_channels host function
// ---------------------------------------------------------------------------

#[test]
fn get_channels_invokes_get_channels_host_function() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains("zeroclaw_get_channels,"),
        "GetChannels must pass zeroclaw_get_channels as the host function to invoke"
    );
}

// ---------------------------------------------------------------------------
// Uses Extism PDK types (Pdk.Allocate, MemoryBlock)
// ---------------------------------------------------------------------------

#[test]
fn uses_extism_pdk_allocate() {
    let src = read_csharp_messaging_source();

    assert!(
        src.contains("Pdk.Allocate("),
        "Must use Pdk.Allocate to write input into Extism shared memory"
    );
}

#[test]
fn uses_extism_memory_block() {
    let src = read_csharp_messaging_source();

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
    let src = read_csharp_messaging_source();

    assert!(
        src.contains("using Extism;"),
        "Messaging.cs must import the Extism namespace for PDK types"
    );
}
