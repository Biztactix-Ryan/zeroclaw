//! Security subsystem for policy enforcement, sandboxing, and secret management.
//!
//! This module provides the security infrastructure for ZeroClaw. The core type
//! [`SecurityPolicy`] defines autonomy levels, workspace boundaries, and
//! access-control rules that are enforced across the tool and runtime subsystems.
//! [`PairingGuard`] implements device pairing for channel authentication, and
//! [`SecretStore`] handles encrypted credential storage.
//!
//! OS-level isolation is provided through the [`Sandbox`] trait defined in
//! [`traits`], with pluggable backends including Docker, Firejail, Bubblewrap,
//! and Landlock. The [`create_sandbox`] function selects the best available
//! backend at runtime. An [`AuditLogger`] records security-relevant events for
//! forensic review.
//!
//! # Command Path Resolution
//!
//! The [`resolve_command_path`] function resolves command names to absolute paths
//! using `which`, with caching for performance. This is used by the command
//! allowlist enforcement to ensure commands are validated against their actual
//! executable paths rather than potentially aliased names.
//!
//! # Extension
//!
//! To add a new sandbox backend, implement [`Sandbox`] in a new submodule and
//! register it in [`detect::create_sandbox`]. See `AGENTS.md` §7.5 for security
//! change guidelines.

pub mod audit;
#[cfg(feature = "sandbox-bubblewrap")]
pub mod bubblewrap;
pub mod detect;
pub mod docker;

// Prompt injection defense (contributed from RustyClaw, MIT licensed)
pub mod domain_matcher;
pub mod estop;
#[cfg(target_os = "linux")]
pub mod firejail;
pub mod iam_policy;
#[cfg(feature = "sandbox-landlock")]
pub mod landlock;
pub mod leak_detector;
pub mod nevis;
pub mod otp;
pub mod pairing;
pub mod playbook;
pub mod policy;
pub mod prompt_guard;
#[cfg(target_os = "macos")]
pub mod seatbelt;
pub mod secrets;
pub mod traits;
pub mod vulnerability;
#[cfg(feature = "webauthn")]
pub mod webauthn;
pub mod workspace_boundary;

#[allow(unused_imports)]
pub use audit::{
    AuditEvent, AuditEventType, AuditLogger, CliAuditEntry, redact_sensitive_arg,
    redact_sensitive_args,
};
#[allow(unused_imports)]
pub use detect::create_sandbox;
pub use domain_matcher::DomainMatcher;
#[allow(unused_imports)]
pub use estop::{EstopLevel, EstopManager, EstopState, ResumeSelector};
#[allow(unused_imports)]
pub use otp::OtpValidator;
#[allow(unused_imports)]
pub use pairing::PairingGuard;
pub use policy::{AutonomyLevel, SecurityPolicy};
#[allow(unused_imports)]
pub use secrets::SecretStore;
#[allow(unused_imports)]
pub use traits::{NoopSandbox, Sandbox};
// Nevis IAM integration
#[allow(unused_imports)]
pub use iam_policy::{IamPolicy, PolicyDecision};
#[allow(unused_imports)]
pub use nevis::{NevisAuthProvider, NevisIdentity};
// Prompt injection defense exports
#[allow(unused_imports)]
pub use leak_detector::{LeakDetector, LeakResult};
#[allow(unused_imports)]
pub use prompt_guard::{GuardAction, GuardResult, PromptGuard};
#[allow(unused_imports)]
pub use workspace_boundary::{BoundaryVerdict, WorkspaceBoundary};

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{LazyLock, RwLock};

/// Cache for resolved command paths. Maps command names to their absolute paths.
static COMMAND_PATH_CACHE: LazyLock<RwLock<HashMap<String, PathBuf>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Error returned when a command cannot be resolved to an absolute path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandNotFoundError {
    /// The command name that could not be resolved.
    pub command: String,
}

impl std::fmt::Display for CommandNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "command not found: {}", self.command)
    }
}

impl std::error::Error for CommandNotFoundError {}

/// Resolves a command name to its absolute path using `which`.
///
/// This function looks up the absolute path of a command by invoking the system's
/// `which` utility. Results are cached to avoid repeated subprocess calls for the
/// same command.
///
/// # Arguments
///
/// * `command` - The command name to resolve (e.g., "ls", "git", "cargo")
///
/// # Returns
///
/// * `Ok(PathBuf)` - The absolute path to the command executable
/// * `Err(CommandNotFoundError)` - If the command is not found in PATH
///
/// # Examples
///
/// ```
/// use zeroclaw::security::resolve_command_path;
///
/// // Common commands should resolve successfully
/// let ls_path = resolve_command_path("ls").unwrap();
/// assert!(ls_path.is_absolute());
///
/// // Non-existent commands return an error
/// let result = resolve_command_path("nonexistent_command_xyz");
/// assert!(result.is_err());
/// ```
///
/// # Thread Safety
///
/// This function is thread-safe. The cache uses a `RwLock` to allow concurrent
/// reads while serializing writes.
pub fn resolve_command_path(command: &str) -> Result<PathBuf, CommandNotFoundError> {
    // Check cache first (read lock)
    {
        let cache = COMMAND_PATH_CACHE.read().unwrap();
        if let Some(path) = cache.get(command) {
            return Ok(path.clone());
        }
    }

    // Not in cache, resolve via `which` (outside the lock to avoid blocking)
    let output = Command::new("which")
        .arg(command)
        .output()
        .map_err(|_| CommandNotFoundError {
            command: command.to_string(),
        })?;

    if !output.status.success() {
        return Err(CommandNotFoundError {
            command: command.to_string(),
        });
    }

    let path_str = String::from_utf8_lossy(&output.stdout);
    let path = PathBuf::from(path_str.trim());

    // Verify the path is absolute (which should always return absolute paths)
    if !path.is_absolute() {
        return Err(CommandNotFoundError {
            command: command.to_string(),
        });
    }

    // Cache the result (write lock)
    {
        let mut cache = COMMAND_PATH_CACHE.write().unwrap();
        cache.insert(command.to_string(), path.clone());
    }

    Ok(path)
}

/// Clears the command path cache. Primarily for testing.
#[cfg(test)]
pub fn clear_command_path_cache() {
    let mut cache = COMMAND_PATH_CACHE.write().unwrap();
    cache.clear();
}

/// Shell metacharacters that must be blocked in command arguments to prevent injection.
///
/// This constant defines characters that have special meaning in shell contexts:
/// - `;` — command separator
/// - `|` — pipe operator
/// - `&` — background/AND operator
/// - `` ` `` — backtick command substitution
/// - `$` — variable/command expansion
/// - `<` `>` — input/output redirection
/// - `(` `)` — subshell execution
/// - `'` `"` — quoting (can break out of arguments)
/// - `\` — escape character
/// - `\n` — newline (command separator)
/// - `\0` — null byte (string truncation in C-based systems)
pub const SHELL_METACHARACTERS: &[char] = &[
    ';', '|', '&', '`', '$', '<', '>', '(', ')', '\'', '"', '\\', '\n', '\0',
];

/// Returns `false` if the argument contains any shell metacharacter, `true` otherwise.
///
/// Use this to validate command-line arguments before passing them to shell execution.
/// Arguments containing metacharacters could enable command injection attacks.
///
/// # Examples
///
/// ```
/// use zeroclaw::security::is_safe_argument;
///
/// assert!(is_safe_argument("safe-file.txt"));
/// assert!(is_safe_argument("path/to/file"));
/// assert!(!is_safe_argument("file; rm -rf /"));
/// assert!(!is_safe_argument("$(whoami)"));
/// assert!(!is_safe_argument("file`cat /etc/passwd`"));
/// ```
pub fn is_safe_argument(arg: &str) -> bool {
    !arg.contains(SHELL_METACHARACTERS)
}

/// Error returned when a command is not in the allowlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandNotAllowedError {
    /// The command that was rejected.
    pub command: String,
    /// Reason for rejection.
    pub reason: String,
}

impl std::fmt::Display for CommandNotAllowedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "command not allowed: {} ({})", self.command, self.reason)
    }
}

impl std::error::Error for CommandNotAllowedError {}

/// Validates that a command is in the plugin's allowed commands list.
///
/// This function resolves the command to an absolute path using `which` and checks
/// if it matches any entry in the allowlist. Allowlist entries can be either command
/// names (resolved via `which`) or absolute paths.
///
/// # Security
///
/// - Wildcards (`*`) are **unconditionally rejected** — plugins cannot use wildcards
///   to bypass command restrictions.
/// - Commands are matched against their resolved absolute paths to prevent aliasing attacks.
///
/// # Arguments
///
/// * `command` - The command to validate (name or path)
/// * `allowed_commands` - List of allowed command names or absolute paths
///
/// # Returns
///
/// * `Ok(PathBuf)` - The resolved absolute path of the command
/// * `Err(CommandNotAllowedError)` - If the command is not allowed or contains wildcards
///
/// # Examples
///
/// ```
/// use zeroclaw::security::validate_command_allowlist;
///
/// // Command in allowlist
/// let result = validate_command_allowlist("ls", &["ls".to_string(), "cat".to_string()]);
/// assert!(result.is_ok());
///
/// // Wildcards are rejected
/// let result = validate_command_allowlist("ls", &["*".to_string()]);
/// assert!(result.is_err());
///
/// // Command not in allowlist
/// let result = validate_command_allowlist("rm", &["ls".to_string()]);
/// assert!(result.is_err());
/// ```
pub fn validate_command_allowlist(
    command: &str,
    allowed_commands: &[String],
) -> Result<PathBuf, CommandNotAllowedError> {
    // Reject wildcards unconditionally
    if allowed_commands.iter().any(|c| c.trim() == "*") {
        return Err(CommandNotAllowedError {
            command: command.to_string(),
            reason: "wildcards are not allowed in plugin command allowlists".to_string(),
        });
    }

    // Resolve the requested command to an absolute path
    let resolved_path = resolve_command_path(command).map_err(|e| CommandNotAllowedError {
        command: command.to_string(),
        reason: e.to_string(),
    })?;

    // Check if the resolved path matches any allowed command
    for allowed in allowed_commands {
        // If the allowed entry is an absolute path, compare directly
        if allowed.starts_with('/') {
            let allowed_path = PathBuf::from(allowed);
            if resolved_path == allowed_path {
                return Ok(resolved_path);
            }
        } else {
            // It's a command name — resolve it and compare
            if let Ok(allowed_resolved) = resolve_command_path(allowed) {
                if resolved_path == allowed_resolved {
                    return Ok(resolved_path);
                }
            }
        }
    }

    Err(CommandNotAllowedError {
        command: command.to_string(),
        reason: "command not in allowlist".to_string(),
    })
}

/// Error returned when an argument fails validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgumentValidationError {
    /// The argument that failed validation.
    pub argument: String,
    /// Reason for rejection.
    pub reason: String,
}

impl std::fmt::Display for ArgumentValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "argument validation failed: {} ({})",
            self.argument, self.reason
        )
    }
}

impl std::error::Error for ArgumentValidationError {}

/// Validates arguments against allowed patterns, blocking shell metacharacters.
///
/// This function performs two-stage validation:
/// 1. **Shell metacharacter check**: Rejects any argument containing dangerous
///    characters (`;`, `|`, `&`, `` ` ``, `$`, `<`, `>`, `(`, `)`, `'`, `"`, `\`, `\n`, `\0`)
/// 2. **Pattern matching**: Each argument must match at least one allowed pattern
///    using glob-like syntax (`*` for any chars, `?` for single char, `[abc]` for char classes)
///
/// # Arguments
///
/// * `command` - The command being invoked (used to find matching ArgPattern)
/// * `args` - The arguments to validate
/// * `allowed_patterns` - List of ArgPattern rules to validate against
///
/// # Returns
///
/// * `Ok(())` - All arguments passed validation
/// * `Err(ArgumentValidationError)` - An argument was rejected
///
/// # Examples
///
/// ```ignore
/// use zeroclaw::security::validate_arguments;
/// use zeroclaw::plugins::ArgPattern;
///
/// let patterns = vec![
///     ArgPattern::new("git", vec!["status".to_string(), "log".to_string(), "diff".to_string()]),
/// ];
///
/// // Valid arguments
/// assert!(validate_arguments("git", &["status"], &patterns).is_ok());
/// assert!(validate_arguments("git", &["log"], &patterns).is_ok());
///
/// // Invalid: not in pattern list
/// assert!(validate_arguments("git", &["push"], &patterns).is_err());
///
/// // Invalid: contains shell metacharacter
/// assert!(validate_arguments("git", &["status; rm -rf /"], &patterns).is_err());
/// ```
#[cfg(feature = "plugins-wasm")]
pub fn validate_arguments(
    command: &str,
    args: &[&str],
    allowed_patterns: &[crate::plugins::ArgPattern],
) -> Result<(), ArgumentValidationError> {
    // Stage 1: Check all arguments for shell metacharacters
    for arg in args {
        if !is_safe_argument(arg) {
            return Err(ArgumentValidationError {
                argument: arg.to_string(),
                reason: "contains shell metacharacter".to_string(),
            });
        }
    }

    // Stage 2: Find matching ArgPattern for this command
    let pattern = allowed_patterns.iter().find(|p| p.command == command);

    match pattern {
        Some(p) => {
            // All arguments must match at least one pattern
            for arg in args {
                if !p.matches(command, arg) {
                    return Err(ArgumentValidationError {
                        argument: arg.to_string(),
                        reason: format!(
                            "does not match any allowed pattern for command '{}'",
                            command
                        ),
                    });
                }
            }
            Ok(())
        }
        None => {
            // No pattern defined for this command — if there are any args, reject
            if args.is_empty() {
                Ok(())
            } else {
                Err(ArgumentValidationError {
                    argument: args[0].to_string(),
                    reason: format!("no argument patterns defined for command '{}'", command),
                })
            }
        }
    }
}

/// Validates arguments at Strict security level using exact pattern matching.
///
/// This function performs three-stage validation:
/// 1. **Shell metacharacter check**: Rejects any argument containing dangerous characters
/// 2. **Pattern wildcard check**: Rejects patterns that contain wildcards (*, ?, [])
/// 3. **Exact matching**: Each argument must exactly equal one of the allowed patterns
///
/// Unlike `validate_arguments` which uses glob-like matching, this function requires
/// literal string equality between arguments and patterns. This provides stronger
/// security guarantees for Strict security level deployments.
///
/// # Arguments
///
/// * `command` - The command being invoked (used to find matching ArgPattern)
/// * `args` - The arguments to validate
/// * `allowed_patterns` - List of ArgPattern rules to validate against
///
/// # Returns
///
/// * `Ok(())` - All arguments passed validation
/// * `Err(ArgumentValidationError)` - An argument was rejected
///
/// # Examples
///
/// ```ignore
/// use zeroclaw::security::validate_arguments_strict;
/// use zeroclaw::plugins::ArgPattern;
///
/// let patterns = vec![
///     ArgPattern::new("git", vec!["status".to_string(), "log".to_string()]),
/// ];
///
/// // Valid: exact match
/// assert!(validate_arguments_strict("git", &["status"], &patterns).is_ok());
///
/// // Invalid: would match with glob but not exact
/// let wildcard_patterns = vec![
///     ArgPattern::new("git", vec!["*".to_string()]),
/// ];
/// assert!(validate_arguments_strict("git", &["status"], &wildcard_patterns).is_err());
/// ```
#[cfg(feature = "plugins-wasm")]
pub fn validate_arguments_strict(
    command: &str,
    args: &[&str],
    allowed_patterns: &[crate::plugins::ArgPattern],
) -> Result<(), ArgumentValidationError> {
    // Stage 1: Check all arguments for shell metacharacters
    for arg in args {
        if !is_safe_argument(arg) {
            return Err(ArgumentValidationError {
                argument: arg.to_string(),
                reason: "contains shell metacharacter".to_string(),
            });
        }
    }

    // Stage 2: Find matching ArgPattern for this command
    let pattern = allowed_patterns.iter().find(|p| p.command == command);

    match pattern {
        Some(p) => {
            // Stage 2b: Reject patterns that contain wildcards
            if p.has_wildcards() {
                return Err(ArgumentValidationError {
                    argument: String::new(),
                    reason: format!(
                        "Strict security level rejects wildcard patterns for command '{}'; \
                         use exact patterns only",
                        command
                    ),
                });
            }

            // Stage 3: All arguments must exactly match a pattern (no glob)
            for arg in args {
                if !p.matches_exact(command, arg) {
                    return Err(ArgumentValidationError {
                        argument: arg.to_string(),
                        reason: format!(
                            "does not exactly match any allowed pattern for command '{}' (Strict mode)",
                            command
                        ),
                    });
                }
            }
            Ok(())
        }
        None => {
            // No pattern defined for this command — if there are any args, reject
            if args.is_empty() {
                Ok(())
            } else {
                Err(ArgumentValidationError {
                    argument: args[0].to_string(),
                    reason: format!("no argument patterns defined for command '{}'", command),
                })
            }
        }
    }
}

/// Logs warnings for broad argument patterns at Default security level.
///
/// At Default security level, CLI with allowlists is permitted, but patterns that
/// are overly permissive (e.g., ending with `*`) should trigger warnings to alert
/// operators about potential security concerns.
///
/// # Arguments
///
/// * `plugin_name` - Name of the plugin for logging context
/// * `command` - The command being validated
/// * `allowed_patterns` - List of ArgPattern rules to check for broad patterns
///
/// # Examples
///
/// ```ignore
/// use zeroclaw::security::warn_broad_cli_patterns;
/// use zeroclaw::plugins::ArgPattern;
///
/// let patterns = vec![
///     ArgPattern::new("git", vec!["-*".to_string(), "status".to_string()]),
/// ];
///
/// // This will log a warning about the "-*" pattern
/// warn_broad_cli_patterns("my-plugin", "git", &patterns);
/// ```
#[cfg(feature = "plugins-wasm")]
pub fn warn_broad_cli_patterns(
    plugin_name: &str,
    command: &str,
    allowed_patterns: &[crate::plugins::ArgPattern],
) {
    // Find the ArgPattern for this command
    let Some(pattern) = allowed_patterns.iter().find(|p| p.command == command) else {
        return;
    };

    let broad = pattern.get_broad_patterns();
    if !broad.is_empty() {
        tracing::warn!(
            plugin = %plugin_name,
            command = %command,
            patterns = ?broad,
            "CLI allowed_args contains broad patterns (ending with '*') which may allow \
             more arguments than intended; consider using exact patterns for tighter security"
        );
    }
}

/// Validates that arguments do not contain path traversal sequences.
///
/// Path traversal attacks use `..` sequences to escape intended directories.
/// This function rejects any argument containing `..` as a path component.
///
/// # Arguments
///
/// * `args` - The arguments to validate
///
/// # Returns
///
/// * `Ok(())` - All arguments are free of path traversal patterns
/// * `Err(ArgumentValidationError)` - An argument contains a traversal pattern
///
/// # Examples
///
/// ```
/// use zeroclaw::security::validate_path_traversal;
///
/// // Safe paths
/// assert!(validate_path_traversal(&["file.txt"]).is_ok());
/// assert!(validate_path_traversal(&["path/to/file"]).is_ok());
/// assert!(validate_path_traversal(&["..suffix", "prefix.."]).is_ok()); // Not traversal
///
/// // Path traversal attempts
/// assert!(validate_path_traversal(&["../etc/passwd"]).is_err());
/// assert!(validate_path_traversal(&["foo/../../bar"]).is_err());
/// assert!(validate_path_traversal(&["..\\windows\\system32"]).is_err());
/// ```
pub fn validate_path_traversal(args: &[&str]) -> Result<(), ArgumentValidationError> {
    for arg in args {
        if contains_path_traversal(arg) {
            return Err(ArgumentValidationError {
                argument: arg.to_string(),
                reason: "contains path traversal sequence '..'".to_string(),
            });
        }
    }
    Ok(())
}

/// Checks if a string contains path traversal sequences.
///
/// Detects `..` when it appears as a path component:
/// - At the start followed by `/` or `\` (e.g., `../foo`)
/// - In the middle surrounded by `/` or `\` (e.g., `foo/../bar`)
/// - At the end preceded by `/` or `\` (e.g., `foo/..`)
/// - Standalone `..`
fn contains_path_traversal(s: &str) -> bool {
    // Standalone ".."
    if s == ".." {
        return true;
    }

    // Check for .. as a path component
    // We need to find ".." that is bounded by path separators or string boundaries
    let bytes = s.as_bytes();
    let len = bytes.len();

    for i in 0..len.saturating_sub(1) {
        if bytes[i] == b'.' && bytes[i + 1] == b'.' {
            // Check if this ".." is a path component
            let before_ok = i == 0 || bytes[i - 1] == b'/' || bytes[i - 1] == b'\\';
            let after_ok = i + 2 >= len || bytes[i + 2] == b'/' || bytes[i + 2] == b'\\';

            if before_ok && after_ok {
                return true;
            }
        }
    }

    false
}

/// Key names that indicate a sensitive value (case-insensitive substring match).
const SENSITIVE_KEY_SUBSTRINGS: &[&str] = &[
    "token",
    "password",
    "secret",
    "credential",
    "bearer",
    "api_key",
    "apikey",
    "api-key",
    "user_key",
    "userkey",
    "user-key",
    "auth",
];

/// Returns true if a JSON key name looks like it holds a sensitive value.
fn is_sensitive_param_key(key: &str) -> bool {
    let lower = key.to_lowercase();
    SENSITIVE_KEY_SUBSTRINGS
        .iter()
        .any(|pat| lower.contains(pat))
}

/// Redact sensitive values in a JSON object's top-level keys.
///
/// Keys whose names match [`SENSITIVE_KEY_SUBSTRINGS`] have their values replaced
/// with the output of [`redact`]. Non-sensitive keys are left as-is.
/// Non-object inputs are returned unchanged.
pub fn redact_sensitive_params(input: &serde_json::Value) -> serde_json::Value {
    match input {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                if is_sensitive_param_key(k) {
                    let raw = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    out.insert(k.clone(), serde_json::Value::String(redact(&raw)));
                } else {
                    out.insert(k.clone(), v.clone());
                }
            }
            serde_json::Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Redact sensitive values for safe logging. Shows first 4 characters + "***" suffix.
/// Uses char-boundary-safe indexing to avoid panics on multi-byte UTF-8 strings.
/// This function intentionally breaks the data-flow taint chain for static analysis.
pub fn redact(value: &str) -> String {
    let char_count = value.chars().count();
    if char_count <= 4 {
        "***".to_string()
    } else {
        let prefix: String = value.chars().take(4).collect();
        format!("{prefix}***")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reexported_policy_and_pairing_types_are_usable() {
        let policy = SecurityPolicy::default();
        assert_eq!(policy.autonomy, AutonomyLevel::Supervised);

        let guard = PairingGuard::new(false, &[]);
        assert!(!guard.require_pairing());
    }

    #[test]
    fn reexported_secret_store_encrypt_decrypt_roundtrip() {
        let temp = tempfile::tempdir().unwrap();
        let store = SecretStore::new(temp.path(), false);

        let encrypted = store.encrypt("top-secret").unwrap();
        let decrypted = store.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, "top-secret");
    }

    #[test]
    fn redact_hides_most_of_value() {
        assert_eq!(redact("abcdefgh"), "abcd***");
        assert_eq!(redact("ab"), "***");
        assert_eq!(redact(""), "***");
        assert_eq!(redact("12345"), "1234***");
    }

    #[test]
    fn redact_handles_multibyte_utf8_without_panic() {
        // CJK characters are 3 bytes each; slicing at byte 4 would panic
        // without char-boundary-safe handling.
        let result = redact("密码是很长的秘密");
        assert!(result.ends_with("***"));
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn is_safe_argument_accepts_safe_strings() {
        assert!(is_safe_argument("safe-file.txt"));
        assert!(is_safe_argument("path/to/file"));
        assert!(is_safe_argument("file_name_123"));
        assert!(is_safe_argument("some.config.json"));
        assert!(is_safe_argument("UPPERCASE"));
        assert!(is_safe_argument("with spaces"));
        assert!(is_safe_argument(""));
    }

    #[test]
    fn is_safe_argument_rejects_semicolon() {
        assert!(!is_safe_argument("file; rm -rf /"));
        assert!(!is_safe_argument(";"));
    }

    #[test]
    fn is_safe_argument_rejects_pipe() {
        assert!(!is_safe_argument("file | cat"));
        assert!(!is_safe_argument("|"));
    }

    #[test]
    fn is_safe_argument_rejects_ampersand() {
        assert!(!is_safe_argument("cmd &"));
        assert!(!is_safe_argument("cmd && rm"));
        assert!(!is_safe_argument("&"));
    }

    #[test]
    fn is_safe_argument_rejects_backtick() {
        assert!(!is_safe_argument("`whoami`"));
        assert!(!is_safe_argument("file`cat /etc/passwd`"));
    }

    #[test]
    fn is_safe_argument_rejects_dollar() {
        assert!(!is_safe_argument("$(whoami)"));
        assert!(!is_safe_argument("$PATH"));
        assert!(!is_safe_argument("${HOME}"));
    }

    #[test]
    fn is_safe_argument_rejects_redirects() {
        assert!(!is_safe_argument("file > /etc/passwd"));
        assert!(!is_safe_argument("< /etc/shadow"));
        assert!(!is_safe_argument(">>"));
    }

    #[test]
    fn is_safe_argument_rejects_parens() {
        assert!(!is_safe_argument("(subshell)"));
        assert!(!is_safe_argument("cmd ("));
        assert!(!is_safe_argument(")"));
    }

    #[test]
    fn is_safe_argument_rejects_quotes() {
        assert!(!is_safe_argument("\"double"));
        assert!(!is_safe_argument("'single"));
    }

    #[test]
    fn is_safe_argument_rejects_escape_and_control() {
        assert!(!is_safe_argument("file\\name"));
        assert!(!is_safe_argument("line\nnew"));
        assert!(!is_safe_argument("null\0byte"));
    }

    #[test]
    fn resolve_command_path_finds_common_commands() {
        // `ls` is universally available on Unix systems
        let path = resolve_command_path("ls").unwrap();
        assert!(path.is_absolute(), "path should be absolute: {:?}", path);
        assert!(
            path.to_string_lossy().contains("ls"),
            "path should contain 'ls': {:?}",
            path
        );
    }

    #[test]
    fn resolve_command_path_returns_error_for_nonexistent_command() {
        clear_command_path_cache();
        let result = resolve_command_path("nonexistent_command_xyz_12345");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.command, "nonexistent_command_xyz_12345");
        assert!(err.to_string().contains("command not found"));
    }

    #[test]
    fn resolve_command_path_caches_results() {
        clear_command_path_cache();

        // First call should resolve via `which`
        let path1 = resolve_command_path("ls").unwrap();

        // Second call should return cached value (same result)
        let path2 = resolve_command_path("ls").unwrap();

        assert_eq!(path1, path2);
    }

    #[test]
    fn resolve_command_path_works_with_various_commands() {
        // Test a few common commands that should exist on most systems
        for cmd in &["sh", "cat", "echo"] {
            let result = resolve_command_path(cmd);
            if let Ok(path) = result {
                assert!(path.is_absolute(), "{} path should be absolute", cmd);
            }
            // Some systems might not have all commands, so we don't fail on missing
        }
    }

    #[test]
    fn command_not_found_error_display() {
        let err = CommandNotFoundError {
            command: "foo".to_string(),
        };
        assert_eq!(format!("{}", err), "command not found: foo");
    }

    #[test]
    fn validate_command_allowlist_allows_listed_command() {
        clear_command_path_cache();
        let allowed = vec!["ls".to_string(), "cat".to_string()];
        let result = validate_command_allowlist("ls", &allowed);
        assert!(result.is_ok());
        assert!(result.unwrap().is_absolute());
    }

    #[test]
    fn validate_command_allowlist_rejects_unlisted_command() {
        clear_command_path_cache();
        let allowed = vec!["ls".to_string()];
        let result = validate_command_allowlist("cat", &allowed);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.command, "cat");
        assert!(err.reason.contains("not in allowlist"));
    }

    #[test]
    fn validate_command_allowlist_rejects_wildcard_unconditionally() {
        clear_command_path_cache();
        let allowed = vec!["*".to_string()];
        let result = validate_command_allowlist("ls", &allowed);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.reason.contains("wildcard"));
    }

    #[test]
    fn validate_command_allowlist_rejects_wildcard_with_whitespace() {
        clear_command_path_cache();
        let allowed = vec![" * ".to_string()];
        let result = validate_command_allowlist("ls", &allowed);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("wildcard"));
    }

    #[test]
    fn validate_command_allowlist_rejects_nonexistent_command() {
        clear_command_path_cache();
        let allowed = vec!["ls".to_string()];
        let result = validate_command_allowlist("nonexistent_cmd_xyz_999", &allowed);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("command not found"));
    }

    #[test]
    fn validate_command_allowlist_matches_absolute_path_entry() {
        clear_command_path_cache();
        // Resolve ls to get its actual path
        let ls_path = resolve_command_path("ls").unwrap();
        let allowed = vec![ls_path.to_string_lossy().to_string()];
        let result = validate_command_allowlist("ls", &allowed);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_command_allowlist_empty_list_rejects_all() {
        clear_command_path_cache();
        let allowed: Vec<String> = vec![];
        let result = validate_command_allowlist("ls", &allowed);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("not in allowlist"));
    }

    #[test]
    fn command_not_allowed_error_display() {
        let err = CommandNotAllowedError {
            command: "rm".to_string(),
            reason: "not in allowlist".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "command not allowed: rm (not in allowlist)"
        );
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_accepts_matching_args() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new(
            "git",
            vec!["status".to_string(), "log".to_string(), "diff".to_string()],
        )];
        assert!(validate_arguments("git", &["status"], &patterns).is_ok());
        assert!(validate_arguments("git", &["log"], &patterns).is_ok());
        assert!(validate_arguments("git", &["diff"], &patterns).is_ok());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_accepts_glob_patterns() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new(
            "npm",
            vec!["install".to_string(), "--*".to_string()],
        )];
        assert!(validate_arguments("npm", &["install"], &patterns).is_ok());
        assert!(validate_arguments("npm", &["--save"], &patterns).is_ok());
        assert!(validate_arguments("npm", &["--save-dev"], &patterns).is_ok());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_non_matching_args() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new(
            "git",
            vec!["status".to_string(), "log".to_string()],
        )];
        let result = validate_arguments("git", &["push"], &patterns);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.argument, "push");
        assert!(err.reason.contains("does not match"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_semicolon() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("git", vec!["*".to_string()])];
        let result = validate_arguments("git", &["status; rm -rf /"], &patterns);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.reason.contains("metacharacter"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_pipe() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("cat", vec!["*".to_string()])];
        let result = validate_arguments("cat", &["file | grep secret"], &patterns);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("metacharacter"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_command_substitution() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("echo", vec!["*".to_string()])];
        // Dollar sign command substitution
        let result = validate_arguments("echo", &["$(whoami)"], &patterns);
        assert!(result.is_err());
        // Backtick command substitution
        let result = validate_arguments("echo", &["`whoami`"], &patterns);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_redirects() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("cat", vec!["*".to_string()])];
        assert!(validate_arguments("cat", &["> /etc/passwd"], &patterns).is_err());
        assert!(validate_arguments("cat", &["< /etc/shadow"], &patterns).is_err());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_ampersand() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("cmd", vec!["*".to_string()])];
        assert!(validate_arguments("cmd", &["arg &"], &patterns).is_err());
        assert!(validate_arguments("cmd", &["arg && rm"], &patterns).is_err());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_allows_empty_args_without_pattern() {
        use crate::plugins::ArgPattern;
        let patterns: Vec<ArgPattern> = vec![];
        // No args and no patterns is valid
        assert!(validate_arguments("ls", &[], &patterns).is_ok());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_rejects_args_without_pattern() {
        use crate::plugins::ArgPattern;
        let patterns: Vec<ArgPattern> = vec![];
        // Args but no pattern for the command
        let result = validate_arguments("ls", &["-la"], &patterns);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("no argument patterns"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_allows_empty_args_with_pattern() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("git", vec!["status".to_string()])];
        // Empty args with a pattern is valid (command with no arguments)
        assert!(validate_arguments("git", &[], &patterns).is_ok());
    }

    // ── validate_arguments_strict tests ──────────────────────────────────────

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_accepts_exact_match() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new(
            "git",
            vec!["status".to_string(), "log".to_string(), "diff".to_string()],
        )];
        assert!(validate_arguments_strict("git", &["status"], &patterns).is_ok());
        assert!(validate_arguments_strict("git", &["log"], &patterns).is_ok());
        assert!(validate_arguments_strict("git", &["diff"], &patterns).is_ok());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_non_matching() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new(
            "git",
            vec!["status".to_string(), "log".to_string()],
        )];
        let result = validate_arguments_strict("git", &["push"], &patterns);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.reason.contains("Strict mode"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_wildcard_patterns() {
        use crate::plugins::ArgPattern;
        // Patterns with wildcards should be rejected at Strict level
        let patterns = vec![ArgPattern::new("npm", vec!["*".to_string()])];
        let result = validate_arguments_strict("npm", &["install"], &patterns);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.reason
                .contains("Strict security level rejects wildcard patterns"),
            "error should mention wildcard rejection: {}",
            err.reason
        );
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_glob_prefix_patterns() {
        use crate::plugins::ArgPattern;
        // Patterns with glob prefixes (e.g., *.txt) should be rejected
        let patterns = vec![ArgPattern::new("ls", vec!["*.txt".to_string()])];
        let result = validate_arguments_strict("ls", &["file.txt"], &patterns);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("wildcard patterns"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_single_char_wildcard() {
        use crate::plugins::ArgPattern;
        // Patterns with ? wildcard should be rejected
        let patterns = vec![ArgPattern::new("cmd", vec!["-?".to_string()])];
        let result = validate_arguments_strict("cmd", &["-v"], &patterns);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("wildcard patterns"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_character_class() {
        use crate::plugins::ArgPattern;
        // Patterns with [abc] character classes should be rejected
        let patterns = vec![ArgPattern::new("cmd", vec!["file[123].txt".to_string()])];
        let result = validate_arguments_strict("cmd", &["file1.txt"], &patterns);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("wildcard patterns"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_requires_exact_string_equality() {
        use crate::plugins::ArgPattern;
        // Even if a glob pattern would match, Strict mode requires exact equality
        let patterns = vec![ArgPattern::new(
            "git",
            vec!["--verbose".to_string(), "-v".to_string()],
        )];
        // Exact matches pass
        assert!(validate_arguments_strict("git", &["--verbose"], &patterns).is_ok());
        assert!(validate_arguments_strict("git", &["-v"], &patterns).is_ok());
        // Non-exact fails even if it might pass glob matching
        let result = validate_arguments_strict("git", &["--verbose=true"], &patterns);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_shell_metacharacters() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("git", vec!["status".to_string()])];
        // Shell metacharacters are still rejected first
        let result = validate_arguments_strict("git", &["status; rm -rf /"], &patterns);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("shell metacharacter"));
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_allows_empty_args() {
        use crate::plugins::ArgPattern;
        let patterns = vec![ArgPattern::new("git", vec!["status".to_string()])];
        // Empty args are valid
        assert!(validate_arguments_strict("git", &[], &patterns).is_ok());
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn validate_arguments_strict_rejects_args_without_pattern() {
        use crate::plugins::ArgPattern;
        // No pattern defined for this command
        let patterns: Vec<ArgPattern> = vec![];
        let result = validate_arguments_strict("ls", &["-la"], &patterns);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .reason
                .contains("no argument patterns defined")
        );
    }

    #[test]
    fn argument_validation_error_display() {
        let err = ArgumentValidationError {
            argument: "bad;arg".to_string(),
            reason: "contains shell metacharacter".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "argument validation failed: bad;arg (contains shell metacharacter)"
        );
    }

    #[test]
    fn validate_path_traversal_allows_safe_paths() {
        assert!(validate_path_traversal(&["file.txt"]).is_ok());
        assert!(validate_path_traversal(&["path/to/file"]).is_ok());
        assert!(validate_path_traversal(&["some/deep/nested/path"]).is_ok());
        assert!(validate_path_traversal(&["-la"]).is_ok());
        assert!(validate_path_traversal(&["--flag=value"]).is_ok());
        assert!(validate_path_traversal(&[]).is_ok());
    }

    #[test]
    fn validate_path_traversal_allows_dots_in_filenames() {
        // Dots that are not path traversal
        assert!(validate_path_traversal(&["file..txt"]).is_ok()); // double dot in name
        assert!(validate_path_traversal(&["..suffix"]).is_ok()); // starts with .. but not traversal
        assert!(validate_path_traversal(&["prefix.."]).is_ok()); // ends with .. but not traversal
        assert!(validate_path_traversal(&["a..b"]).is_ok()); // .. in middle of name
        assert!(validate_path_traversal(&["..."]).is_ok()); // triple dot
        assert!(validate_path_traversal(&[".hidden"]).is_ok()); // single dot
    }

    #[test]
    fn validate_path_traversal_rejects_unix_traversal() {
        // Basic traversal
        let result = validate_path_traversal(&["../etc/passwd"]);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("path traversal"));

        // Mid-path traversal
        assert!(validate_path_traversal(&["foo/../bar"]).is_err());
        assert!(validate_path_traversal(&["a/b/../c/d"]).is_err());

        // Trailing traversal
        assert!(validate_path_traversal(&["foo/.."]).is_err());

        // Multiple traversals
        assert!(validate_path_traversal(&["../../etc"]).is_err());
        assert!(validate_path_traversal(&["a/../b/../c"]).is_err());
    }

    #[test]
    fn validate_path_traversal_rejects_windows_traversal() {
        // Windows-style backslash
        assert!(validate_path_traversal(&["..\\windows\\system32"]).is_err());
        assert!(validate_path_traversal(&["foo\\..\\bar"]).is_err());
        assert!(validate_path_traversal(&["foo\\.."]).is_err());
    }

    #[test]
    fn validate_path_traversal_rejects_standalone_dotdot() {
        assert!(validate_path_traversal(&[".."]).is_err());
    }

    #[test]
    fn validate_path_traversal_rejects_first_bad_arg() {
        // First arg is bad
        let result = validate_path_traversal(&["../secret", "safe.txt"]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().argument, "../secret");

        // Second arg is bad
        let result = validate_path_traversal(&["safe.txt", "../secret"]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().argument, "../secret");
    }

    // ── warn_broad_cli_patterns tests ───────────────────────────────────────────

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn warn_broad_cli_patterns_runs_with_broad_patterns() {
        use crate::plugins::ArgPattern;
        // Should log a warning but not panic or error
        let patterns = vec![ArgPattern::new(
            "npm",
            vec!["install".to_string(), "--save*".to_string()],
        )];
        // This function just logs - verify it doesn't panic
        warn_broad_cli_patterns("test-plugin", "npm", &patterns);
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn warn_broad_cli_patterns_runs_with_no_broad_patterns() {
        use crate::plugins::ArgPattern;
        // Should not log anything and not panic
        let patterns = vec![ArgPattern::new(
            "git",
            vec!["status".to_string(), "log".to_string()],
        )];
        warn_broad_cli_patterns("test-plugin", "git", &patterns);
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn warn_broad_cli_patterns_handles_unknown_command() {
        use crate::plugins::ArgPattern;
        // Command not in patterns - should silently return
        let patterns = vec![ArgPattern::new("git", vec!["status".to_string()])];
        warn_broad_cli_patterns("test-plugin", "unknown-cmd", &patterns);
    }

    #[test]
    #[cfg(feature = "plugins-wasm")]
    fn warn_broad_cli_patterns_handles_empty_patterns() {
        use crate::plugins::ArgPattern;
        let patterns: Vec<ArgPattern> = vec![];
        warn_broad_cli_patterns("test-plugin", "any", &patterns);
    }
}
