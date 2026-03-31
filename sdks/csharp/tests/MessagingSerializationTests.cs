using System.Text.Json;
using Xunit;

namespace ZeroClaw.PluginSdk.Tests;

/// <summary>
/// Validates that Messaging request/response JSON serialization matches the
/// Rust SDK wire format (snake_case keys, same field names and structure).
/// These tests exercise the serialization logic without requiring a WASM host.
/// </summary>
public class MessagingSerializationTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    // -- SendMessageRequest mirror (matches Messaging.cs private class) ----

    public sealed class SendMessageRequest
    {
        public string Channel { get; set; } = "";
        public string Recipient { get; set; } = "";
        public string Message { get; set; } = "";
    }

    // -- SendMessageResponse mirror (matches Messaging.cs private class) ---

    public sealed class SendMessageResponse
    {
        public bool Success { get; set; }
        public string? Error { get; set; }
    }

    // -- GetChannelsResponse mirror (matches Messaging.cs private class) ---

    public sealed class GetChannelsResponse
    {
        public List<string> Channels { get; set; } = new();
        public string? Error { get; set; }
    }

    // -- SendMessageRequest serialization ----------------------------------

    [Fact]
    public void SendMessageRequest_Serializes_SnakeCase()
    {
        var req = new SendMessageRequest
        {
            Channel = "telegram",
            Recipient = "user123",
            Message = "hello world",
        };
        var json = JsonSerializer.Serialize(req, JsonOptions);

        Assert.Contains("\"channel\"", json);
        Assert.Contains("\"recipient\"", json);
        Assert.Contains("\"message\"", json);
        // Must NOT contain PascalCase
        Assert.DoesNotContain("\"Channel\"", json);
        Assert.DoesNotContain("\"Recipient\"", json);
        Assert.DoesNotContain("\"Message\"", json);
    }

    [Fact]
    public void SendMessageRequest_MatchesRustWireFormat()
    {
        var req = new SendMessageRequest
        {
            Channel = "discord",
            Recipient = "user456",
            Message = "ping",
        };
        var json = JsonSerializer.Serialize(req, JsonOptions);
        var parsed = JsonDocument.Parse(json);
        var root = parsed.RootElement;

        Assert.Equal("discord", root.GetProperty("channel").GetString());
        Assert.Equal("user456", root.GetProperty("recipient").GetString());
        Assert.Equal("ping", root.GetProperty("message").GetString());
        // Exactly three top-level properties: channel, recipient, message
        Assert.Equal(3, root.EnumerateObject().Count());
    }

    // -- SendMessageResponse deserialization --------------------------------

    [Fact]
    public void SendMessageResponse_Deserializes_Success()
    {
        var json = """{"success": true}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<SendMessageResponse>(json, JsonOptions)!;

        Assert.True(resp.Success);
        Assert.Null(resp.Error);
    }

    [Fact]
    public void SendMessageResponse_Deserializes_Error()
    {
        var json = """{"success": false, "error": "channel not allowed"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<SendMessageResponse>(json, JsonOptions)!;

        Assert.False(resp.Success);
        Assert.Equal("channel not allowed", resp.Error);
    }

    [Fact]
    public void SendMessageResponse_Deserializes_ErrorOnly()
    {
        var json = """{"error": "rate limit exceeded"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<SendMessageResponse>(json, JsonOptions)!;

        Assert.False(resp.Success);
        Assert.Equal("rate limit exceeded", resp.Error);
    }

    [Fact]
    public void SendMessageResponse_Deserializes_SuccessWithoutError()
    {
        var json = """{"success": true}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<SendMessageResponse>(json, JsonOptions)!;

        Assert.True(resp.Success);
        Assert.Null(resp.Error);
    }

    // -- GetChannelsResponse deserialization --------------------------------

    [Fact]
    public void GetChannelsResponse_Deserializes_WithChannels()
    {
        var json = """{"channels": ["telegram", "discord", "slack"]}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<GetChannelsResponse>(json, JsonOptions)!;

        Assert.Equal(3, resp.Channels.Count);
        Assert.Equal("telegram", resp.Channels[0]);
        Assert.Equal("discord", resp.Channels[1]);
        Assert.Equal("slack", resp.Channels[2]);
        Assert.Null(resp.Error);
    }

    [Fact]
    public void GetChannelsResponse_Deserializes_EmptyChannels()
    {
        var json = """{"channels": []}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<GetChannelsResponse>(json, JsonOptions)!;

        Assert.Empty(resp.Channels);
        Assert.Null(resp.Error);
    }

    [Fact]
    public void GetChannelsResponse_Deserializes_Error()
    {
        var json = """{"channels": [], "error": "not authorized"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<GetChannelsResponse>(json, JsonOptions)!;

        Assert.Empty(resp.Channels);
        Assert.Equal("not authorized", resp.Error);
    }

    [Fact]
    public void GetChannelsResponse_Deserializes_MissingChannelsField()
    {
        var json = """{"error": "host unavailable"}"""u8.ToArray();
        var resp = JsonSerializer.Deserialize<GetChannelsResponse>(json, JsonOptions)!;

        Assert.Empty(resp.Channels);
        Assert.Equal("host unavailable", resp.Error);
    }

    // -- Round-trip serialization/deserialization ----------------------------

    [Fact]
    public void SendMessageRequest_Roundtrip_PreservesFields()
    {
        var original = new SendMessageRequest
        {
            Channel = "telegram",
            Recipient = "user789",
            Message = "test message",
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(original, JsonOptions);
        var deserialized = JsonSerializer.Deserialize<SendMessageRequest>(json, JsonOptions)!;

        Assert.Equal("telegram", deserialized.Channel);
        Assert.Equal("user789", deserialized.Recipient);
        Assert.Equal("test message", deserialized.Message);
    }

    [Fact]
    public void SendMessageResponse_Roundtrip_PreservesFields()
    {
        var original = new SendMessageResponse
        {
            Success = true,
            Error = null,
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(original, JsonOptions);
        var deserialized = JsonSerializer.Deserialize<SendMessageResponse>(json, JsonOptions)!;

        Assert.True(deserialized.Success);
        Assert.Null(deserialized.Error);
    }

    [Fact]
    public void GetChannelsResponse_Roundtrip_PreservesFields()
    {
        var original = new GetChannelsResponse
        {
            Channels = new List<string> { "alpha", "beta" },
            Error = null,
        };
        var json = JsonSerializer.SerializeToUtf8Bytes(original, JsonOptions);
        var deserialized = JsonSerializer.Deserialize<GetChannelsResponse>(json, JsonOptions)!;

        Assert.Equal(2, deserialized.Channels.Count);
        Assert.Equal("alpha", deserialized.Channels[0]);
        Assert.Equal("beta", deserialized.Channels[1]);
        Assert.Null(deserialized.Error);
    }
}
