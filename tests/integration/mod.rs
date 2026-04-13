mod agent;
mod agent_robustness;
mod backup_cron_scheduling;
mod channel_matrix;
mod channel_routing;
mod email_attachments;
mod hooks;
mod memory_comparison;
mod memory_loop_continuity;
mod memory_restart;
#[cfg(feature = "plugins-wasm")]
mod plugin_allowed_channels_filter;
#[cfg(feature = "plugins-wasm")]
mod plugin_allowed_tools_filter;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_auth_required;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_disable_plugin;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_enable_plugin;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_feature_gate;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_get_plugin;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_install_auth_required;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_install_endpoint;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_install_error_descriptive;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_install_uses_host_logic;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_list_plugins;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_reload_plugins;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_remove_plugin;
#[cfg(feature = "plugins-wasm")]
mod plugin_api_remove_uses_host_logic;
#[cfg(feature = "plugins-wasm")]
mod plugin_audit_combined;
#[cfg(feature = "plugins-wasm")]
mod plugin_audit_fields;
#[cfg(feature = "plugins-wasm")]
mod plugin_audit_log;
#[cfg(feature = "plugins-wasm")]
mod plugin_audit_output_format;
#[cfg(feature = "plugins-wasm")]
mod plugin_audit_parse;
#[cfg(feature = "plugins-wasm")]
mod plugin_build_functions;
#[cfg(feature = "plugins-wasm")]
mod plugin_call_depth_limit;
#[cfg(feature = "plugins-wasm")]
mod plugin_capabilities_parse;
#[cfg(feature = "plugins-wasm")]
mod plugin_capability_gated_registration;
#[cfg(feature = "plugins-wasm")]
mod plugin_channel_messaging;
#[cfg(feature = "plugins-wasm")]
mod plugin_channel_rate_limit;
#[cfg(feature = "plugins-wasm")]
mod plugin_channel_send_listen;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_allowlist_enforced;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_arg_pattern_validated;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_audit_logged;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_command_args_logged;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_concurrent_limited;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_default_allows_with_warnings;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_defaults_applied;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_exec_implemented;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_exit_code_in_response;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_exit_duration_logged;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_function_accepts_params;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_metachar_blocked;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_no_shell;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_output_captured;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_output_truncated;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_paranoid_denied;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_path_traversal_rejected;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_rate_limit_clear_error;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_rate_limited;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_relaxed_requires_allowlist;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_resource_limits;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_response_indicators;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_security_level_checked;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_strict_exact_patterns;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_timeout_sigkill;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_which_resolution;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_wildcard_rejected;
#[cfg(feature = "plugins-wasm")]
mod plugin_cli_working_dir_restricted;
#[cfg(feature = "plugins-wasm")]
mod plugin_config_edit_nonsensitive;
#[cfg(feature = "plugins-wasm")]
mod plugin_config_persist_toml;
#[cfg(feature = "plugins-wasm")]
mod plugin_config_roundtrip;
#[cfg(feature = "plugins-wasm")]
mod plugin_config_status_display;
#[cfg(feature = "plugins-wasm")]
mod plugin_context_agent_config;
#[cfg(feature = "plugins-wasm")]
mod plugin_context_integration;
#[cfg(feature = "plugins-wasm")]
mod plugin_context_readonly;
#[cfg(feature = "plugins-wasm")]
mod plugin_context_session;
#[cfg(feature = "plugins-wasm")]
mod plugin_context_user_identity;
#[cfg(feature = "plugins-wasm")]
mod plugin_default_mode;
#[cfg(feature = "plugins-wasm")]
mod plugin_delegation_security_limits;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_api_includes_plugins;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_backwards_compat;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_capability_conflict;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_category_severity;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_checks_all;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_diagnostics;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_failure_entries;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_health_section;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_hint_exact_command;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_hint_healthy;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_hint_on_issues;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_invalid_manifest;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_missing_config;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_missing_wasm;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_summary_counts;
#[cfg(feature = "plugins-wasm")]
mod plugin_doctor_summary_only;
#[cfg(feature = "plugins-wasm")]
mod plugin_echo_roundtrip;
#[cfg(feature = "plugins-wasm")]
mod plugin_empty_allowed_paths;
#[cfg(feature = "plugins-wasm")]
mod plugin_encrypted_config;
#[cfg(feature = "plugins-wasm")]
mod plugin_env_sanitization;
#[cfg(feature = "plugins-wasm")]
mod plugin_filesystem_access_display;
#[cfg(feature = "plugins-wasm")]
mod plugin_forbidden_mount;
#[cfg(feature = "plugins-wasm")]
mod plugin_fs_roundtrip;
#[cfg(feature = "plugins-wasm")]
mod plugin_get_channels;
#[cfg(feature = "plugins-wasm")]
mod plugin_hash_mismatch;
#[cfg(feature = "plugins-wasm")]
mod plugin_hash_verification_cycle;
#[cfg(feature = "plugins-wasm")]
mod plugin_host_function_abi;
#[cfg(feature = "plugins-wasm")]
mod plugin_host_function_registry;
#[cfg(feature = "plugins-wasm")]
mod plugin_http_allowed_host;
#[cfg(feature = "plugins-wasm")]
mod plugin_http_audit_log;
#[cfg(feature = "plugins-wasm")]
mod plugin_http_auth_header;
#[cfg(feature = "plugins-wasm")]
mod plugin_http_blocked;
#[cfg(feature = "plugins-wasm")]
mod plugin_http_empty_hosts;
#[cfg(feature = "plugins-wasm")]
mod plugin_http_get;
#[cfg(feature = "plugins-wasm")]
mod plugin_input_redact;
#[cfg(feature = "plugins-wasm")]
mod plugin_install_hash;
#[cfg(feature = "plugins-wasm")]
mod plugin_load_verify_hash;
#[cfg(feature = "plugins-wasm")]
mod plugin_manifest_defaults;
#[cfg(feature = "plugins-wasm")]
mod plugin_manifest_timeout;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_author_tag;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_capability_denial;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_forget;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_host_roundtrip;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_no_capability;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_recall;
#[cfg(feature = "plugins-wasm")]
mod plugin_memory_store;
#[cfg(feature = "plugins-wasm")]
mod plugin_messaging_no_capability;
#[cfg(feature = "plugins-wasm")]
mod plugin_messaging_security;
#[cfg(feature = "plugins-wasm")]
mod plugin_messaging_send;
#[cfg(feature = "plugins-wasm")]
mod plugin_missing_config;
#[cfg(feature = "plugins-wasm")]
mod plugin_multi_tool_register;
#[cfg(feature = "plugins-wasm")]
mod plugin_network_access_display;
#[cfg(feature = "plugins-wasm")]
mod plugin_no_capability_no_imports;
#[cfg(feature = "plugins-wasm")]
mod plugin_no_secret_leak;
#[cfg(feature = "plugins-wasm")]
mod plugin_output_format_spec;
#[cfg(feature = "plugins-wasm")]
mod plugin_paranoid_allowlist;
#[cfg(feature = "plugins-wasm")]
mod plugin_paranoid_context_denied;
#[cfg(feature = "plugins-wasm")]
mod plugin_path_traversal;
#[cfg(feature = "plugins-wasm")]
mod plugin_rate_limit;
#[cfg(feature = "plugins-wasm")]
mod plugin_rate_limit_exceeded;
#[cfg(feature = "plugins-wasm")]
mod plugin_rate_limit_test;
#[cfg(feature = "plugins-wasm")]
mod plugin_read_file;
#[cfg(feature = "plugins-wasm")]
mod plugin_relaxed_mode;
#[cfg(feature = "plugins-wasm")]
mod plugin_reload_behavior;
#[cfg(feature = "plugins-wasm")]
mod plugin_reload_command;
#[cfg(feature = "plugins-wasm")]
mod plugin_reload_new_discovered;
#[cfg(feature = "plugins-wasm")]
mod plugin_reload_removed_unloaded;
#[cfg(feature = "plugins-wasm")]
mod plugin_reload_rescan;
#[cfg(feature = "plugins-wasm")]
mod plugin_risk_level_ceiling;
#[cfg(feature = "plugins-wasm")]
mod plugin_risk_level_display;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_cli_response_struct;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_context_module;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_crate_structure;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_example_e2e;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_memory_module;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_messaging_module;
#[cfg(feature = "plugins-wasm")]
mod plugin_sdk_tools_module;
#[cfg(feature = "plugins-wasm")]
mod plugin_security_audit_display;
#[cfg(feature = "plugins-wasm")]
mod plugin_security_level_config;
#[cfg(feature = "plugins-wasm")]
mod plugin_security_level_enforcement;
#[cfg(feature = "plugins-wasm")]
mod plugin_sensitive_config_masked;
#[cfg(feature = "plugins-wasm")]
mod plugin_sensitive_log_redact;
#[cfg(feature = "plugins-wasm")]
mod plugin_strict_mode;
#[cfg(feature = "plugins-wasm")]
mod plugin_structure;
#[cfg(feature = "plugins-wasm")]
mod plugin_success_failure_status;
#[cfg(feature = "plugins-wasm")]
mod plugin_timeout;
#[cfg(feature = "plugins-wasm")]
mod plugin_timeout_clean_error;
#[cfg(feature = "plugins-wasm")]
mod plugin_timeout_enforcement;
#[cfg(feature = "plugins-wasm")]
mod plugin_tool_call_dispatch;
#[cfg(feature = "plugins-wasm")]
mod plugin_tool_delegation;
#[cfg(feature = "plugins-wasm")]
mod plugin_wildcard_delegation_security;
mod report_template_tool_test;
mod telegram_attachment_fallback;
mod telegram_finalize_draft;
