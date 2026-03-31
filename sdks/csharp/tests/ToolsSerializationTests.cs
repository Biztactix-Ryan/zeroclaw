using System.Text.Json;
using Xunit;

namespace ZeroClaw.PluginSdk.Tests;

/// <summary>
/// Validates that Tools request/response JSON serialization matches the
/// Rust SDK wire format (snake_case keys, same field names and structure).
/// These tests exercise the serialization logic without requiring a WASM host.
/// </summary>
public class ToolsSerializationTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    // -- ToolCallRequest mirror (matches Tools.cs private class) ---------------

    public sealed class ToolCallRequest
    {
        public string ToolName { get; set; } = "";
        public object Arguments { get; set; } = new object();
    }

    // -- ToolCallResponse mirror (matches Tools.cs private class) --------------

    public sealed class ToolCallResponse
    {
        public bool Success { get; set; }
        public string Output { get; set; } = "";
        public string? Error { get; set; }
    }

    // -- ToolCallRequest serialization ----------------------------------------

    [Fact]
    public void ToolCallRequest_Serializes_SnakeCase()
    {
        var req = new ToolCallRequest
        {
            ToolName = "search",
            Arguments = new { query = "hello" },
        };
        var json = JsonSerializer.Serialize(req, JsonOptions);

        Assert.Contains("\"tool_name\"", json);
        Assert.Contains("\"arguments\"", json);
        Assert.Contains("\"search\"", json);
        // Must NOT contain PascalCase
        Assert.DoesNotContain("\"ToolName\"", json);
        Assert.DoesNotContain("\"Arguments\"", json);
    }

    [Fact]
    public void ToolCallRequest_MatchesRustWireFormat()
    {
        var req = new ToolCallRequest
        {
            ToolName = "echo",
            Arguments = new { message = "ping" },
        };
        var json = JsonSerializer.Serialize(req, JsonOptions);
        var parsed = JsonDocument.Parse(json);
        var root = parsed.RootElement;

        Assert.Equal("echo", root.GetProperty("tool_name").GetString());
        Assert.Equal(JsonValueKind.Object, root.GetProperty("arguments").ValueKind);
        Assert.Equal("ping", root.GetProperty("arguments").GetProperty("message").GetString());
        // Exactly two top-level properties: tool_name, arguments
        Assert.Equal(2, root.EnumerateObject().Count());
    }

    [Fact]
    public void ToolCallRequest_SerializesNestedArguments()
    {
        var req = new ToolCallRequest
        {
            ToolName = "create_item",
            Arguments = new { name = "test", count = 42, enabled = true },
        };
        var json = JsonSerializer.Serialize(req, JsonOptions);
        var parsed = JsonDocument.Parse(json);
        var args = parsed.RootElement.GetProperty("arguments");

        Assert.Equal("test", args.GetProperty("name").GetString());
        Assert.Equal(42, args.GetProperty("count").GetInt32());
        Assert.True(args.GetProperty("enabled").GetBoolean());
    }

    [Fact]
    public void ToolCallRequest_SerializesEmptyArguments()
    {
        var req = new ToolCallRequest
        {
            ToolName = "no_args",
            Arguments = new { },
        };
        var json = JsonSerializer.Serialize(req, JsonOptions);
        var parsed = JsonDocument.Parse(json);
        var root = parsed.RootElement;

        Assert.Equal("no_args", root.GetProperty("tool_name").GetString());
        Assert.Equal(JsonValueKind.Object, root.GetProperty("arguments").ValueKind);
    }

    // -- ToolCallResponse deserialization -------------------------------------

    [Fact]
    public void ToolCallResponse_Deserializes_Success()
    {
        var json = """{"success": true, "output": "result data"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<ToolCallResponse>(json, JsonOptions)!;

        Assert.True(resp.Success);
        Assert.Equal("result data", resp.Output);
        Assert.Null(resp.Error);
    }

    [Fact]
    public void ToolCallResponse_Deserializes_Error()
    {
        var json = """{"success": false, "output": "", "error": "tool not found"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<ToolCallResponse>(json, JsonOptions)!;

        Assert.False(resp.Success);
        Assert.Equal("", resp.Output);
        Assert.Equal("tool not found", resp.Error);
    }

    [Fact]
    public void ToolCallResponse_Deserializes_ErrorOnly()
    {
        var json = """{"error": "permission denied"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<ToolCallResponse>(json, JsonOptions)!;

        Assert.False(resp.Success);
        Assert.Equal("permission denied", resp.Error);
    }

    [Fact]
    public void ToolCallResponse_Deserializes_SuccessWithoutError()
    {
        var json = """{"success": true, "output": "ok"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<ToolCallResponse>(json, JsonOptions)!;

        Assert.True(resp.Success);
        Assert.Equal("ok", resp.Output);
        Assert.Null(resp.Error);
    }

    [Fact]
    public void ToolCallResponse_Deserializes_EmptyOutput()
    {
        var json = """{"success": true, "output": ""}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<ToolCallResponse>(json, JsonOptions)!;

        Assert.True(resp.Success);
        Assert.Equal("", resp.Output);
        Assert.Null(resp.Error);
    }

    // -- Round-trip serialization/deserialization ------------------------------

    [Fact]
    public void ToolCallRequest_Roundtrip_PreservesFields()
    {
        var original = new ToolCallRequest
        {
            ToolName = "lookup",
            Arguments = new { id = 99 },
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(original, JsonOptions);
        var deserialized = JsonSerializer.Deserialize<ToolCallRequest>(json, JsonOptions)!;

        Assert.Equal("lookup", deserialized.ToolName);
    }

    [Fact]
    public void ToolCallResponse_Roundtrip_PreservesFields()
    {
        var original = new ToolCallResponse
        {
            Success = true,
            Output = "some output",
            Error = null,
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(original, JsonOptions);
        var deserialized = JsonSerializer.Deserialize<ToolCallResponse>(json, JsonOptions)!;

        Assert.True(deserialized.Success);
        Assert.Equal("some output", deserialized.Output);
        Assert.Null(deserialized.Error);
    }
}
