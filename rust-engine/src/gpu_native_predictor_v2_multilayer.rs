//! Qualification-only 45-layer P1X command. One ordinary request, no warmup,
//! no sidecar enablement. Running this command requires separate runtime authority.
use crate::gpu_native_predictor_v2_observation::{
    ensure_output_absent, parse_request, write_report,
};
use crate::gpu_native_real_benchmark as benchmark;
use crate::predictor_v2::p1x;
use serde::Serialize;
use std::path::PathBuf;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "mer.predictor-v2-p1x-runtime-observation.v1";
const MODE: &str = "qualification-only-multilayer-observation";
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| "P1X count overflow".into())
}
#[derive(Clone, Debug, clap::Args, Serialize)]
pub(crate) struct CommandArgs {
    #[arg(long)]
    pub(crate) config: PathBuf,
    #[arg(long)]
    request_json: PathBuf,
    #[arg(long)]
    expected_adapter_name: String,
    #[arg(long)]
    report_out: PathBuf,
}
#[derive(Serialize)]
struct Report {
    schema: &'static str,
    mode: &'static str,
    provenance: benchmark::BenchmarkProvenance,
    config_path: String,
    config_sha256: String,
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    model_load: Option<crate::greedy_parity::ModelLoadEvidence>,
    adapter: Option<crate::backend::GpuDeviceIdentity>,
    runtime_contract: Option<benchmark::RuntimeContractEvidence>,
    production_configuration: benchmark::ProductionConfiguration,
    request: benchmark::RequestEvidence,
    generated_token_ids: Vec<u32>,
    generated_token_ids_sha256: String,
    completed_position_count: usize,
    raw_p1x_report: Option<p1x::Report>,
    runtime_counters: Option<benchmark::RequestSnapshots>,
    // Structural fact: P1X implements no movement operation. Counter snapshots
    // describe ordinary work and do not establish a no-overhead performance claim.
    predictor_v2_movement_operations: u64,
    runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
    errors: Vec<String>,
    observation_complete: bool,
}
async fn execute_request(
    runtime: &crate::BenchRealRuntime,
    prompt: &[u32],
    report: &mut Report,
) -> Result<()> {
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .ok_or("missing authoritative token loop")?;
    let planned = add(
        prompt.len(),
        report
            .request
            .requested_output_tokens
            .checked_sub(1)
            .ok_or("zero output tokens")?,
    )?;
    if planned > token_loop.max_seq_len() {
        return Err("request exceeds runtime context limit".into());
    }
    let mut request = token_loop.create_request_state()?;
    request.enable_p1x(token_loop, prompt.len(), planned)?;
    let snapshots = benchmark::RequestSnapshotStart::capture(runtime)?;
    let execution: Result<()> = async {
        for position in 0..planned {
            let (token, sample) = if position < prompt.len() {
                (
                    prompt[position],
                    position.checked_add(1) == Some(prompt.len()),
                )
            } else {
                (
                    *report
                        .generated_token_ids
                        .last()
                        .ok_or("missing previous generated token")?,
                    true,
                )
            };
            let sampled = token_loop
                .step_token(&runtime.engine, &mut request, token, position, sample)
                .await?;
            report.completed_position_count = add(report.completed_position_count, 1)?;
            match (sample, sampled) {
                (true, Some(id)) => report.generated_token_ids.push(id),
                (false, None) => {}
                _ => return Err("ordinary step returned an unexpected sampling result".into()),
            }
        }
        Ok(())
    }
    .await;
    match request.finish_p1x() {
        Ok(raw) => report.raw_p1x_report = Some(raw),
        Err(e) => report.errors.push(e.to_string()),
    }
    report.generated_token_ids_sha256 =
        crate::greedy_parity::token_ids_sha256(&report.generated_token_ids);
    match snapshots.finish(runtime) {
        Ok(counters) => report.runtime_counters = Some(counters),
        Err(e) => report.errors.push(e.to_string()),
    }
    // Preserve the ordinary execution error before checking accounting evidence.
    execution?;
    let counters = report
        .runtime_counters
        .as_ref()
        .ok_or("runtime counters unavailable")?;
    benchmark::validate_request_postconditions(
        prompt.len(),
        report.request.requested_output_tokens,
        report.generated_token_ids.len(),
        counters.token_loop_delta,
        counters.recovery_delta,
        counters.routed_execution_delta,
    )?;
    let c = counters.token_loop_delta;
    if c.queue_submissions != c.token_attempts
        || c.boundary_maps != c.token_attempts
        || c.boundary_readbacks != c.token_attempts
    {
        return Err("ordinary attempt/submission/boundary accounting mismatch".into());
    }
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<()> {
    ensure_output_absent(&args.report_out)?;
    if args.expected_adapter_name.trim().is_empty() {
        return Err("exact expected adapter name is required".into());
    }
    let (prompt, output_tokens) = parse_request(&std::fs::read(&args.request_json)?)?;
    let build = crate::qualification::BuildProvenance::embedded();
    benchmark::validate_preflight_provenance(&build)?;
    let config_bytes = std::fs::read(&args.config)?;
    let config_sha256 = crate::greedy_parity::sha256_hex(&config_bytes);
    let cfg: crate::config::Config = toml::from_str(std::str::from_utf8(&config_bytes)?)?;
    cfg.validate()?;
    benchmark::validate_source_config(&cfg)?;
    let (artifacts, errors) = crate::qualification_artifacts(&args.config, &cfg);
    benchmark::validate_artifacts(&artifacts, &errors)?;
    if artifacts.config.as_ref().map(|v| v.sha256.as_str()) != Some(config_sha256.as_str()) {
        return Err("config changed between parsed snapshot and artifact hashing".into());
    }
    let metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|e| format!("expert metadata unavailable: {e}"))?;
    benchmark::validate_expert_metadata(&metadata)?;
    let mode = crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark;
    let spec = crate::resolve_real_cli_spec_from_config(cfg, mode)?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err("strict Qwen3-MoE Q4_0 model identity required".into());
    }
    let resolved_hash = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(&spec.cfg, mode)?;
    let prompt_ids = tokenizer.encode(&prompt)?;
    let positions = add(
        prompt_ids.len(),
        output_tokens.checked_sub(1).ok_or("zero output tokens")?,
    )?;
    if prompt_ids.is_empty()
        || positions > p1x::MAX_POSITIONS
        || positions > spec.cfg.real_transformer.gpu_native_max_seq_len
    {
        return Err("request exceeds configured context limit".into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let configuration = benchmark::ProductionConfiguration::from_config(&spec.cfg, &metadata);
    let mut report = Report {
        schema: SCHEMA,
        mode: MODE,
        provenance: benchmark::BenchmarkProvenance {
            build,
            executable_canonical_path: std::fs::canonicalize(executable)?.display().to_string(),
            executable_sha256,
            resolved_config_sha256: resolved_hash.clone(),
            artifacts,
            expert_metadata: metadata,
        },
        config_path: std::fs::canonicalize(&args.config)?.display().to_string(),
        config_sha256,
        model_identity,
        model_load: None,
        adapter: None,
        runtime_contract: None,
        production_configuration: configuration,
        request: benchmark::RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(prompt.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: output_tokens,
            greedy: true,
        },
        generated_token_ids: Vec::new(),
        generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&[]),
        completed_position_count: 0,
        raw_p1x_report: None,
        runtime_counters: None,
        predictor_v2_movement_operations: 0,
        runtime_shutdown: None,
        errors: Vec::new(),
        observation_complete: false,
    };
    // Exactly one production bootstrap. All paths after successful construction
    // reach normal isolated shutdown, including validation and stepping failures.
    match crate::build_isolated_greedy_runtime(&spec, mode, tokenizer).await {
        Err(e) => report.errors.push(e.to_string()),
        Ok(runtime) => {
            let execution: Result<()> = async {
                let observed_hash = crate::resolved_real_runtime_identity_sha256(
                    &runtime.cfg,
                    runtime.model.config.architecture,
                    runtime.model.config.first_k_dense_replace,
                    &runtime.model.config.advanced,
                )?;
                if observed_hash != resolved_hash {
                    return Err("runtime configuration identity drift".into());
                }
                let model_load = crate::greedy_parity_model_load(&runtime);
                let input = benchmark::RuntimeContractInput {
                    real_transformer_enabled: runtime.cfg.real_transformer.enabled,
                    real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
                    compute_offload: runtime.cfg.real_transformer.compute_offload,
                    legacy_execution_plan: runtime.engine.execution_context().plan().into(),
                    token_loop_geometry: runtime
                        .gpu_native_token_loop
                        .as_ref()
                        .map(|t| t.model_geometry()),
                    authoritative_device: runtime.engine.gpu_device_identity(),
                    model_load: model_load.clone(),
                    routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
                };
                report.model_load = Some(model_load);
                report.adapter = input.authoritative_device.clone();
                let (contract, _) =
                    benchmark::validate_runtime_contract(&input, &args.expected_adapter_name)?;
                report.runtime_contract = Some(contract);
                let t = runtime
                    .gpu_native_token_loop
                    .as_ref()
                    .ok_or("missing token loop")?;
                if t.snapshot() != Default::default()
                    || t.recovery_snapshot() != Default::default()
                    || runtime.engine.routed_expert_execution_snapshot() != Default::default()
                {
                    return Err("isolated runtime has nonzero initial execution counters".into());
                }
                execute_request(&runtime, &prompt_ids, &mut report).await
            }
            .await;
            if let Err(e) = execution {
                report.errors.push(e.to_string());
            }
            match runtime.shutdown_isolated().await {
                Ok(evidence) => {
                    if !evidence.controlled_shutdown_requested
                        || !evidence.all_runtime_resources_released
                    {
                        report
                            .errors
                            .push("isolated runtime shutdown incomplete".into());
                    }
                    report.runtime_shutdown = Some(evidence);
                }
                Err(e) => report.errors.push(e.to_string()),
            }
        }
    }
    report.observation_complete = report.errors.is_empty()
        && report.completed_position_count == positions
        && report.raw_p1x_report.as_ref().is_some_and(|r| {
            r.error.is_none() && r.completed_positions == positions && r.layers.len() == 45
        })
        && report
            .runtime_shutdown
            .is_some_and(|s| s.controlled_shutdown_requested && s.all_runtime_resources_released);
    if !report.observation_complete {
        report.errors.push("P1X observation incomplete".into());
    }
    write_report(&args.report_out, &report)?;
    if !report.errors.is_empty() {
        return Err(report.errors.join("; ").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    fn between<'a>(s: &'a str, start: &str, end: &str) -> &'a str {
        s.split_once(start).unwrap().1.split_once(end).unwrap().0
    }
    #[test]
    fn p1x_source_capabilities_exclude_movement_io_and_gpu_operations() {
        let driver = include_str!("gpu_native_predictor_v2_multilayer.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let core = include_str!("predictor_v2_p1x.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let token = include_str!("gpu_native_token_loop.rs")
            .split("#[cfg(test)]\npub(crate) mod tests")
            .next()
            .unwrap();
        let manager = include_str!("gpu_native_residency.rs");
        let backend = include_str!("backend/gpu_native.rs");
        let f = between(
            token,
            "    fn observe_p1x_completed(",
            "    fn observe_p1x_deadline(",
        );
        let d = between(
            token,
            "    fn observe_p1x_deadline(",
            "    fn observe_p1e_completed(",
        );
        let helper = between(manager, "fn try_p1x_record(", "fn p1e_resident(");
        let api = between(
            manager,
            "    pub(crate) fn try_observe_p1x_physical(",
            "    pub(crate) fn executor(",
        );
        let arena = between(
            backend,
            "    pub(crate) fn try_contains_exact_residency(",
            "    pub(crate) const fn vram_plan(",
        );
        for code in [core, f, d, helper, api, arena] {
            for forbidden in [
                "queue.submit",
                "device.poll",
                "write_buffer",
                "map_async",
                "read_expert",
                "fetch_with_retry",
                "HostBackedLease",
                "try_lease_host_backed",
                "enable_p1j",
                "try_prepare",
                "writer.spawn",
                "p1e::PhysicalSnapshot",
            ] {
                assert!(!code.contains(forbidden), "{forbidden}");
            }
        }
        for code in [d, helper, api, arena] {
            for forbidden in [
                ".lock(",
                ".await",
                ".join(",
                ".wait(",
                ".sleep(",
                ".get_mut(",
                ".put(",
                "fetch_add",
                "Vec::",
            ] {
                assert!(
                    !code.contains(forbidden),
                    "blocking/mutating/allocating D: {forbidden}"
                );
            }
        }
        assert!(helper.contains("state.try_lock()"));
        assert!(helper.contains("residents.peek(&global)"));
        assert!(arena.contains("self.state.try_lock()?"));
        assert!(api.contains("try_contains_exact_residency"));
        for forbidden in [
            "enable_predictor_v2_p1j",
            "enable_predictor_v2_p1e",
            "try_lease",
            "fetch_with_retry",
            "queue.submit",
            "device.poll",
            "map_async",
        ] {
            assert!(!driver.contains(forbidden), "driver {forbidden}");
        }
        assert_eq!(driver.matches("build_isolated_greedy_runtime(").count(), 1);
        assert!(driver.contains(".enable_p1x("));
        let create = between(
            token,
            "pub fn create_request_state(",
            "/// Calculate the required context capacity",
        );
        assert!(create.contains("p1j: None"));
        assert!(create.contains("p1x: None"));
        let enable = between(
            token,
            "pub(crate) fn enable_p1x(",
            "pub(crate) fn finish_p1x(",
        );
        assert!(!enable.contains("enable_p1j"));
        assert!(!enable.contains("enable_predictor_v2_p1e"));
        let launch = between(
            token,
            "    fn launch_p1j_at_freeze(",
            "    fn finish_p1j_target(",
        );
        assert!(launch.contains("let Some(movement) = request.p1j.as_mut() else"));
        let seam = token
            .find("self.observe_p1x_completed(engine, request, position, &report.selected_ids)")
            .unwrap();
        let commit = token.find("request.committed_position += 1;").unwrap();
        assert!(commit < seam);
        let d_seam = token
            .find("self.observe_p1x_deadline(request, position, layer_idx)")
            .unwrap();
        assert!(token[d_seam..].find("// 1. Attention Pre-Norm").unwrap() < 1500);
    }
    #[test]
    fn p1x_cli_is_explicit_and_preserves_portable_command_contract() {
        let main = include_str!("main.rs");
        assert!(main.contains("observe-gpu-native-predictor-v2-p1x"));
        assert!(main.contains("gpu_native_predictor_v2_multilayer::run_command(args)"));
        let driver = include_str!("gpu_native_predictor_v2_multilayer.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(driver.contains("validate_source_config"));
        assert!(driver.contains("validate_runtime_contract"));
        assert!(driver.contains("queue_submissions != c.token_attempts"));
        assert!(driver.contains("boundary_maps != c.token_attempts"));
        assert!(driver.contains("boundary_readbacks != c.token_attempts"));
    }
}
