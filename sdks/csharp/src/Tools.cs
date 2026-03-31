using System.Runtime.InteropServices;
using System.Text.Json;
using Extism;

namespace ZeroClaw.PluginSdk;

/// <summary>
/// Provides access to ZeroClaw's tool delegation from plugin code.
/// Calls the zeroclaw_tool_call host function via Extism shared memory,
/// marshalling JSON with System.Text.Json matching the Rust SDK wire format.
/// </summary>
public static class Tools
{
    // -----------------------------------------------------------------------
    // Host function imports (rewritten by Extism MSBuild at compile time)
    // -----------------------------------------------------------------------

    [DllImport("extism", EntryPoint = "zeroclaw_tool_call")]
    private static extern ulong zeroclaw_tool_call(ulong input);

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

    private sealed class ToolCallRequest
    {
        public string ToolName { get; set; } = "";
        public object Arguments { get; set; } = new object();
    }

    private sealed class ToolCallResponse
    {
        public bool Success { get; set; }
        public string Output { get; set; } = "";
        public string? Error { get; set; }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// <summary>
    /// Call a registered tool by name with the given arguments.
    /// </summary>
    /// <param name="toolName">Name of the tool to invoke.</param>
    /// <param name="arguments">Arguments to pass to the tool (serialized as JSON).</param>
    /// <returns>Output string from the tool on success.</returns>
    /// <exception cref="PluginException">Thrown when the host reports an error or the call fails.</exception>
    public static string ToolCall(string toolName, object arguments)
    {
        var request = new ToolCallRequest { ToolName = toolName, Arguments = arguments };
        var response = CallHostFunction<ToolCallRequest, ToolCallResponse>(
            zeroclaw_tool_call, request);

        if (response.Error is not null)
            throw new PluginException(response.Error);
        if (!response.Success)
            throw new PluginException("tool call returned success=false");

        return response.Output;
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
