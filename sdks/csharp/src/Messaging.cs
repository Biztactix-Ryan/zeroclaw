using System.Runtime.InteropServices;
using System.Text.Json;
using Extism;

namespace ZeroClaw.PluginSdk;

/// <summary>
/// Provides access to ZeroClaw's messaging subsystem from plugin code.
/// Calls the zeroclaw_send_message and zeroclaw_get_channels host functions
/// via Extism shared memory, marshalling JSON with System.Text.Json matching
/// the Rust SDK wire format.
/// </summary>
public static class Messaging
{
    // -----------------------------------------------------------------------
    // Host function imports (rewritten by Extism MSBuild at compile time)
    // -----------------------------------------------------------------------

    [DllImport("extism", EntryPoint = "zeroclaw_send_message")]
    private static extern ulong zeroclaw_send_message(ulong input);

    [DllImport("extism", EntryPoint = "zeroclaw_get_channels")]
    private static extern ulong zeroclaw_get_channels(ulong input);

    // -----------------------------------------------------------------------
    // JSON options — snake_case to match Rust SDK wire format
    // -----------------------------------------------------------------------

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    // -----------------------------------------------------------------------
    // Request / response types (mirror the Rust SDK structs)
    // -----------------------------------------------------------------------

    private sealed class SendMessageRequest
    {
        public string Channel { get; set; } = "";
        public string Recipient { get; set; } = "";
        public string Message { get; set; } = "";
    }

    private sealed class SendMessageResponse
    {
        public bool Success { get; set; }
        public string? Error { get; set; }
    }

    private sealed class GetChannelsResponse
    {
        public List<string> Channels { get; set; } = new();
        public string? Error { get; set; }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// <summary>
    /// Send a message to a recipient on the given channel.
    /// </summary>
    /// <param name="channel">Channel name to send through.</param>
    /// <param name="recipient">Recipient identifier.</param>
    /// <param name="message">Message text to send.</param>
    /// <exception cref="PluginException">Thrown when the host reports an error.</exception>
    public static void Send(string channel, string recipient, string message)
    {
        var request = new SendMessageRequest
        {
            Channel = channel,
            Recipient = recipient,
            Message = message,
        };
        var response = CallHostFunction<SendMessageRequest, SendMessageResponse>(
            zeroclaw_send_message, request);

        if (response.Error is not null)
            throw new PluginException(response.Error);
        if (!response.Success)
            throw new PluginException("send_message returned success=false");
    }

    /// <summary>
    /// Get the list of available channel names.
    /// </summary>
    /// <returns>List of channel names the plugin is allowed to use.</returns>
    /// <exception cref="PluginException">Thrown when the host reports an error.</exception>
    public static List<string> GetChannels()
    {
        var response = CallHostFunction<object, GetChannelsResponse>(
            zeroclaw_get_channels, new { });

        if (response.Error is not null)
            throw new PluginException(response.Error);

        return response.Channels;
    }

    // -----------------------------------------------------------------------
    // Shared host-call helper
    // -----------------------------------------------------------------------

    private static TResponse CallHostFunction<TRequest, TResponse>(
        Func<ulong, ulong> hostFn, TRequest request)
    {
        var inputBytes = JsonSerializer.SerializeToUtf8Bytes(request, JsonOptions);
        using var inputBlock = Pdk.Allocate(inputBytes);

        var outputOffset = hostFn(inputBlock.Offset);

        using var outputBlock = MemoryBlock.Find(outputOffset);
        if (outputBlock.IsEmpty)
            throw new PluginException("host function returned empty response");

        var outputBytes = outputBlock.ReadBytes();
        return JsonSerializer.Deserialize<TResponse>(outputBytes, JsonOptions)
            ?? throw new PluginException("failed to deserialize host response");
    }
}
