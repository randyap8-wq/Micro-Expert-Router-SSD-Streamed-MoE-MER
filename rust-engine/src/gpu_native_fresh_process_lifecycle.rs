//! HMA-1H: one unchanged mapped-baseline arm per fresh OS process.
//! Parent and auditor only handle CPU evidence. Production execution is inherited.
use super::*;
use std::io::Read;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-fresh-process-lifecycle.v1";
pub(crate) const MODE: &str = "qualify-gpu-native-fresh-process-lifecycle";
const WORKER: &str = "hma1h-worker-internal";
const PREFIX: &str = "HMA1H_";
const CLEANUP: &str = "One fresh child and one mapped-baseline runtime per arm; complete child exit before next launch; inherited exact HMA-1F-B cleanup; no cooldown or device probes.";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct ProcessIdentity {
    pid: u32,
    ppid: u32,
    boot_id: String,
    start_ticks: u64,
}
impl ProcessIdentity {
    fn parse(stat: &str, boot: &str) -> Result<Self> {
        // comm is parenthesized and may itself contain spaces or parentheses.
        let (pid, _) = stat.split_once(" (").ok_or("malformed proc stat pid")?;
        let (_, tail) = stat.rsplit_once(") ").ok_or("malformed proc stat comm")?;
        let fields: Vec<_> = tail.split_whitespace().collect();
        require(fields.len() >= 20, "truncated proc stat")?;
        let id = Self {
            pid: pid.parse()?,
            ppid: fields[1].parse()?,
            boot_id: boot.trim_end_matches('\n').into(),
            start_ticks: fields[19].parse()?,
        };
        id.validate()?;
        Ok(id)
    }
    fn validate(&self) -> Result<()> {
        require(
            self.pid > 0 && self.ppid > 0 && self.start_ticks > 0,
            "invalid process identity",
        )?;
        require(
            self.boot_id.len() == 36
                && self.boot_id.bytes().enumerate().all(|(i, b)| {
                    if [8, 13, 18, 23].contains(&i) {
                        b == b'-'
                    } else {
                        b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
                    }
                }),
            "invalid Linux boot ID",
        )
    }
    fn capture(pid: u32) -> Result<Self> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
        let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
        let id = Self::parse(&stat, &boot)?;
        require(id.pid == pid, "proc PID mismatch")?;
        Ok(id)
    }
    fn release(&self, index: usize) -> String {
        format!(
            "HMA1H_GO index={index} identity={}\n",
            serde_json::to_string(self).unwrap()
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildPayload {
    schema: String,
    mode: String,
    index: usize,
    name: String,
    process: ProcessIdentity,
    frozen_workload: Value,
    arm: Option<ArmData>,
    failure: Option<FailedArm>,
    worker_error: Option<String>,
}

pub(crate) async fn run_worker(args: CommandArgs, index: usize) -> Result<()> {
    crate::gpu_native_mapped_pin::require_platform()?;
    require(index < ORDER.len(), "one arm index in 0..3 required")?;
    let process = ProcessIdentity::capture(std::process::id())?;
    let mut payload = ChildPayload {
        schema: SCHEMA.into(),
        mode: MODE.into(),
        index,
        name: ORDER[index].into(),
        process,
        frozen_workload: serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
        arm: None,
        failure: None,
        worker_error: None,
    };
    let result: Result<()> = async {
        let mut release = String::new();
        std::io::stdin().read_to_string(&mut release)?;
        require(
            release == payload.process.release(index),
            "parent/child process handshake mismatch",
        )?;
        println!(
            "HMA1H_WORKER_IDENTITY {}",
            serde_json::to_string(&payload.process)?
        );
        std::io::stdout().flush()?;
        require(
            args.expected_adapter_name == "NVIDIA L4",
            "HMA-1H requires NVIDIA L4",
        )?;
        let prepared = prepare(&args)?;
        // Keep the inherited logical-arm markers truthful inside the parent segment.
        let link = TranscriptLink::arm(index);
        println!("{}", link.begin_marker);
        let attempt = super::execute_arm(&prepared, &args, index).await;
        println!("{}", link.end_marker);
        match attempt? {
            Attempt::Complete(arm) => {
                payload.arm = Some(arm);
                let retained = payload.arm.as_ref().unwrap();
                validate_benchmark_header(&retained.production["benchmark"])?;
                validate_arm(retained)?;
                Ok(())
            }
            Attempt::Failed(failure) => {
                let error = format!("{}: {}", failure.primary.code, failure.primary.detail);
                payload.failure = Some(failure);
                Err(error.into())
            }
        }
    }
    .await;
    if let Err(error) = &result {
        payload.worker_error = Some(error.to_string());
    }
    let bytes = serde_json::to_vec(&payload)?;
    write_new(&args.report_out, &bytes)?;
    println!("HMA1H_WORKER_PAYLOAD index={index} sha256={}", sha(&bytes));
    std::io::stdout().flush()?;
    result
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ChildExit {
    unix_wait_status: i32,
    code: Option<i32>,
    signal: Option<i32>,
    core_dumped: bool,
}
impl ChildExit {
    #[cfg(unix)]
    fn capture(status: std::process::ExitStatus) -> Self {
        use std::os::unix::process::ExitStatusExt;
        Self {
            unix_wait_status: status.into_raw(),
            code: status.code(),
            signal: status.signal(),
            core_dumped: status.core_dumped(),
        }
    }
    fn expected(code: i32) -> Self {
        Self {
            unix_wait_status: code << 8,
            code: Some(code),
            signal: None,
            core_dumped: false,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildBinding {
    index: usize,
    name: String,
    launch_ns: u64,
    exit_ns: Option<u64>,
    spawned_pid: Option<u32>,
    process: Option<ProcessIdentity>,
    exit: Option<ChildExit>,
    payload_hex: Option<String>,
    payload_sha256: Option<String>,
    evidence_error: Option<String>,
}
impl ChildBinding {
    fn begin(&self) -> String {
        format!(
            "HMA1H_CHILD_BEGIN index={} name={} launch_ns={}",
            self.index, self.name, self.launch_ns
        )
    }
    fn identity(&self) -> Result<String> {
        Ok(format!(
            "HMA1H_PARENT_IDENTITY index={} pid={} process={}",
            self.index,
            self.spawned_pid.ok_or("missing spawned PID")?,
            serde_json::to_string(&self.process)?
        ))
    }
    fn end(&self) -> Result<String> {
        Ok(format!(
            "HMA1H_CHILD_END index={} name={} exit_ns={} status={} payload_sha256={} error={}",
            self.index,
            self.name,
            serde_json::to_string(&self.exit_ns)?,
            serde_json::to_string(&self.exit)?,
            serde_json::to_string(&self.payload_sha256)?,
            serde_json::to_string(&self.evidence_error)?
        ))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    mode: String,
    primary: String,
    cleanup_contract: String,
    parent: ProcessIdentity,
    executable_sha256: String,
    children: Vec<ChildBinding>,
    transcript_sha256: String,
    transcript_bytes: usize,
    parent_error: Option<String>,
}
fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn decode_hex(s: &str) -> Result<Vec<u8>> {
    require(
        s.len() % 2 == 0
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid exact payload hex",
    )?;
    s.as_bytes()
        .chunks_exact(2)
        .map(|c| Ok(u8::from_str_radix(std::str::from_utf8(c)?, 16)?))
        .collect()
}
fn worker_arguments(
    config: &Path,
    draft: &Path,
    index: usize,
    raw: &[std::ffi::OsString],
) -> Result<Vec<std::ffi::OsString>> {
    require(index < ORDER.len(), "no fifth child")?;
    // Reuse historical forwarding, with only the public/hidden command identities replaced.
    let inherited: Vec<_> = raw
        .iter()
        .map(|a| {
            if a == MODE {
                super::MODE.into()
            } else {
                a.clone()
            }
        })
        .collect();
    let mut args = super::worker_arguments(config, draft, &inherited)?;
    let slot = args
        .iter_mut()
        .find(|a| *a == "hma1g-worker-internal")
        .ok_or("missing worker command")?;
    *slot = WORKER.into();
    args.extend(["--arm-index".into(), index.to_string().into()]);
    Ok(args)
}
fn elapsed_ns(start: Instant) -> Result<u64> {
    Ok(start.elapsed().as_nanos().try_into()?)
}
fn emit(file: &mut std::fs::File, line: &str) -> Result<()> {
    writeln!(file, "{line}")?;
    file.flush()?;
    Ok(())
}

pub(crate) fn launch(
    config: &Path,
    report_out: &Path,
    transcript_out: &Path,
    raw: &[std::ffi::OsString],
) -> Result<()> {
    crate::gpu_native_mapped_pin::require_platform()?;
    #[cfg(not(unix))]
    {
        let _ = (config, report_out, transcript_out, raw);
        Err("Linux required".into())
    }
    #[cfg(unix)]
    {
        launch_unix(config, report_out, transcript_out, raw)
    }
}
#[cfg(unix)]
fn launch_unix(
    config: &Path,
    report_out: &Path,
    transcript_out: &Path,
    raw: &[std::ffi::OsString],
) -> Result<()> {
    let drafts: Vec<_> = (0..ORDER.len())
        .map(|i| {
            let mut name = report_out.as_os_str().to_owned();
            name.push(format!(".baseline-{i}.payload.json"));
            PathBuf::from(name)
        })
        .collect();
    let paths: Vec<_> = [report_out, transcript_out]
        .into_iter()
        .chain(drafts.iter().map(PathBuf::as_path))
        .map(canonical_output)
        .collect::<Result<_>>()?;
    require(
        paths.iter().collect::<BTreeSet<_>>().len() == 6,
        "six distinct fresh output paths required",
    )?;
    let executable = std::env::current_exe()?;
    let mut envelope = Envelope {
        schema: SCHEMA.into(),
        mode: MODE.into(),
        primary: PRIMARY.into(),
        cleanup_contract: CLEANUP.into(),
        parent: ProcessIdentity::capture(std::process::id())?,
        executable_sha256: sha(&std::fs::read(&executable)?),
        children: Vec::new(),
        transcript_sha256: String::new(),
        transcript_bytes: 0,
        parent_error: None,
    };
    // Validate forwarding before any child work. Reserve the report before starting.
    let arguments: Vec<_> = drafts
        .iter()
        .enumerate()
        .map(|(i, p)| worker_arguments(config, p, i, raw))
        .collect::<Result<_>>()?;
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_out)?;
    let mut transcript = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(transcript_out)?;
    emit(&mut transcript, &run_begin(&envelope)?)?;
    let clock = Instant::now();
    for index in 0..ORDER.len() {
        let mut binding = ChildBinding {
            index,
            name: ORDER[index].into(),
            launch_ns: elapsed_ns(clock)?,
            exit_ns: None,
            spawned_pid: None,
            process: None,
            exit: None,
            payload_hex: None,
            payload_sha256: None,
            evidence_error: None,
        };
        emit(&mut transcript, &binding.begin())?;
        let attempted = (|| -> Result<()> {
            // Both descriptors share one sequential open-file description. No pipe drains or threads.
            let mut child = std::process::Command::new(&executable)
                .args(&arguments[index])
                .stdin(std::process::Stdio::piped())
                .stdout(transcript.try_clone()?)
                .stderr(transcript.try_clone()?)
                .spawn()?;
            binding.spawned_pid = Some(child.id());
            // The worker cannot enter prepare/execute until this authoritative /proc handshake.
            let handshake = (|| -> Result<()> {
                let id = ProcessIdentity::capture(child.id())?;
                require(
                    id.ppid == envelope.parent.pid && id.boot_id == envelope.parent.boot_id,
                    "child parent/boot mismatch",
                )?;
                binding.process = Some(id.clone());
                emit(&mut transcript, &binding.identity()?)?;
                let mut input = child.stdin.take().ok_or("child stdin unavailable")?;
                input.write_all(id.release(index).as_bytes())?;
                Ok(())
            })();
            drop(child.stdin.take()); // EOF also releases a rejected handshake without executing work.
                                      // Always wait, including handshake errors; never leave an unwaited child and continue.
            let status = child.wait()?;
            binding.exit_ns = Some(elapsed_ns(clock)?);
            binding.exit = Some(ChildExit::capture(status));
            handshake?;
            Ok(())
        })();
        if let Err(error) = attempted {
            binding.evidence_error = Some(error.to_string());
        }
        // One exact byte snapshot, including malformed payloads, is retained for every available file.
        match std::fs::read(&drafts[index]) {
            Ok(bytes) => {
                binding.payload_sha256 = Some(sha(&bytes));
                binding.payload_hex = Some(encode_hex(&bytes));
            }
            Err(error) => {
                binding
                    .evidence_error
                    .get_or_insert_with(|| error.to_string());
            }
        }
        emit(&mut transcript, &binding.end()?)?;
        envelope.children.push(binding);
        // CPU-only prefix checks stop immediately after a failure or invalid completed child.
        match collect_children(&envelope) {
            Ok((arms, failure)) => {
                if failure.is_some() {
                    break;
                }
                if let Err(error) = validate_completed(&arms) {
                    envelope.parent_error = Some(error.to_string());
                    break;
                }
            }
            Err(error) => {
                envelope.parent_error = Some(error.to_string());
                break;
            }
        }
    }
    emit(&mut transcript, &run_end(&envelope))?;
    transcript.sync_all()?;
    drop(transcript);
    let bytes = std::fs::read(transcript_out)?;
    envelope.transcript_sha256 = sha(&bytes);
    envelope.transcript_bytes = bytes.len();
    let raw = serde_json::to_vec_pretty(&envelope)?;
    report.write_all(&raw)?;
    report.sync_all()?;
    let audit = audit_bytes(&raw, &bytes);
    eprintln!(
        "HMA1H_RESULT={} raw_report_sha256={} transcript_sha256={}",
        audit.classification,
        sha(&raw),
        envelope.transcript_sha256
    );
    require(
        audit.authoritative,
        "HMA-1H NON_AUTHORITATIVE; see bound report/transcript",
    )
}

fn run_begin(e: &Envelope) -> Result<String> {
    Ok(format!(
        "HMA1H_QUALIFIER_BEGIN parent={} executable_sha256={}",
        serde_json::to_string(&e.parent)?,
        e.executable_sha256
    ))
}
fn run_end(e: &Envelope) -> String {
    format!("HMA1H_QUALIFIER_END attempted={}", e.children.len())
}
fn collect_children(e: &Envelope) -> Result<(Vec<ArmData>, Option<FailedArm>)> {
    e.parent.validate()?;
    require(
        !e.children.is_empty() && e.children.len() <= 4,
        "one to four attempted children required",
    )?;
    let mut identities = BTreeSet::new();
    let mut previous_exit = None;
    let mut arms = Vec::new();
    let mut failure = None;
    for (index, c) in e.children.iter().enumerate() {
        require(
            c.index == index && c.name == ORDER[index] && failure.is_none(),
            "child order/failed prefix mismatch",
        )?;
        require(c.evidence_error.is_none(), "parent child evidence error")?;
        let id = c
            .process
            .as_ref()
            .ok_or("missing attempted child identity")?;
        id.validate()?;
        require(
            c.spawned_pid == Some(id.pid)
                && id.pid != e.parent.pid
                && id.ppid == e.parent.pid
                && id.boot_id == e.parent.boot_id
                && id.start_ticks >= e.parent.start_ticks
                && identities.insert((id.boot_id.clone(), id.pid, id.start_ticks)),
            "process identity/PPID reuse or mismatch",
        )?;
        let exit_ns = c.exit_ns.ok_or("child not completely waited")?;
        require(
            c.launch_ns < exit_ns && previous_exit.is_none_or(|n| n < c.launch_ns),
            "exit must precede next launch",
        )?;
        previous_exit = Some(exit_ns);
        let bytes = decode_hex(c.payload_hex.as_deref().ok_or("missing child payload")?)?;
        require(
            c.payload_sha256.as_deref() == Some(sha(&bytes).as_str()),
            "exact child payload hash mismatch",
        )?;
        let p: ChildPayload = serde_json::from_slice(&bytes)?;
        require(
            serde_json::to_value(&p)? == serde_json::from_slice::<Value>(&bytes)?,
            "missing/extra child fields",
        )?;
        require(
            p.schema == SCHEMA
                && p.mode == MODE
                && p.index == index
                && p.name == ORDER[index]
                && &p.process == id
                && p.frozen_workload == serde_json::to_value(frozen_workload("NVIDIA L4".into()))?,
            "child identity/workload contract",
        )?;
        match (p.arm, p.failure) {
            (Some(arm), None) => {
                require(
                    p.worker_error.is_none()
                        && c.exit == Some(ChildExit::expected(0))
                        && arm.position == index,
                    "completed child exit/index mismatch",
                )?;
                require(
                    arm.production["benchmark"]["provenance"]["executable_sha256"]
                        == e.executable_sha256,
                    "parent/child executable mismatch",
                )?;
                arms.push(arm);
            }
            (None, Some(f)) => {
                require(
                    c.exit == Some(ChildExit::expected(1))
                        && p.worker_error
                            .as_ref()
                            .is_some_and(|s| !s.trim().is_empty())
                        && f.position == Some(index)
                        && f.name.as_deref() == Some(ORDER[index]),
                    "failed child status/identity mismatch",
                )?;
                if let Some(s) = &f.startup {
                    require(
                        s.benchmark["provenance"]["executable_sha256"] == e.executable_sha256,
                        "failed executable mismatch",
                    )?;
                }
                failure = Some(f);
            }
            _ => return Err("missing/duplicated arm evidence".into()),
        }
    }
    Ok((arms, failure))
}

// HMA-1G work equations verbatim except process-local context IDs may restart.
// Each ID is still validated by validate_arm; OS identity uniqueness is above.
fn validate_completed(arms: &[ArmData]) -> Result<()> {
    require(
        !arms.is_empty() && arms.len() <= 4,
        "one to four completed arms required",
    )?;
    for (i, a) in arms.iter().enumerate() {
        require(
            a.position == i && a.mode == Mode::MappedBaseline,
            "baseline-only arm order mismatch",
        )?;
        validate_benchmark_header(&a.production["benchmark"])?;
        validate_arm(a)?;
        if i == 0 {
            continue;
        }
        let c = &arms[0].production;
        let t = &a.production;
        contracts_equal(&c["benchmark"], &t["benchmark"])?;
        require(
            c["warmup_ram_cache_state_sha256"] == t["warmup_ram_cache_state_sha256"],
            "warmup cache/reset drift",
        )?;
        for (key, measured) in [("warmup_results", false), ("per_run_results", true)] {
            let cr = if measured {
                &c["benchmark"][key]
            } else {
                &c[key]
            };
            let tr = if measured {
                &t["benchmark"][key]
            } else {
                &t[key]
            };
            for (x, y) in cr.as_array().unwrap().iter().zip(tr.as_array().unwrap()) {
                for field in [
                    "run_index",
                    "generated_tokens",
                    "generated_token_ids_sha256",
                    "generated_text_sha256",
                ] {
                    require(x[field] == y[field], "token/text work drift")?;
                }
                if measured {
                    require(
                        x["generated_token_ids"] == y["generated_token_ids"],
                        "generated token stream drift",
                    )?;
                }
            }
        }
        for prefix in ["warmup_", ""] {
            let mk = if prefix.is_empty() {
                "mechanism"
            } else {
                "warmup_mechanism"
            };
            let cs: Snapshot = decode(&c[mk])?;
            let ts: Snapshot = decode(&t[mk])?;
            let cu: UploadSnapshot = decode(&c[format!("{prefix}upload")])?;
            let tu: UploadSnapshot = decode(&t[format!("{prefix}upload")])?;
            let cp: GpuNativeProductionPhysicalInstallSnapshot =
                decode(&c[format!("{prefix}production_physical_install")])?;
            let tp = decode(&t[format!("{prefix}production_physical_install")])?;
            let cds: ProductionDemandSourceSnapshot = decode(&c[format!("{prefix}production")])?;
            let tds = decode(&t[format!("{prefix}production")])?;
            let cw: ArmWorkEvidence = decode(&c[format!("{prefix}work")])?;
            let tw: ArmWorkEvidence = decode(&t[format!("{prefix}work")])?;
            let g = pair_mechanism_gate(&cs, &ts, &cp, &tp, &cu, &tu, &cds, &tds);
            require(
                g.source_and_route_streams_exact
                    && g.physical_install_reservation_victim_and_publication_exact
                    && g.source_scheduler_exact
                    && work_pair_exact(&cw, &tw)
                    && cw.gpu_expert_io.expert_weight_upload_bytes
                        == tw.gpu_expert_io.expert_weight_upload_bytes
                    && cw.gpu_native_residency.vram_hits == tw.gpu_native_residency.vram_hits
                    && cw.gpu_native_residency.vram_misses == tw.gpu_native_residency.vram_misses
                    && cs.primary_pool_capacity == ts.primary_pool_capacity
                    && cu.logical_admission_ids_sha256 == tu.logical_admission_ids_sha256
                    && cu.logical_generation_ids_sha256 == tu.logical_generation_ids_sha256
                    && cu.metrics.logical_admissions == tu.metrics.logical_admissions
                    && cu.metrics.logical_generation_observations
                        == tu.metrics.logical_generation_observations
                    && cu.metrics.logical_materialization_operations
                        == tu.metrics.logical_materialization_operations
                    && cu.metrics.logical_materialization_bytes
                        == tu.metrics.logical_materialization_bytes
                    && cu.metrics.shared_payload_reuse == tu.metrics.shared_payload_reuse
                    && cu.source_upload_fd_proof == tu.source_upload_fd_proof,
                "production route/source/cache/work drift",
            )?;
        }
        require(
            arms[0].records.len() == a.records.len(),
            "source stream length drift",
        )?;
        for (x, y) in arms[0].records.iter().zip(&a.records) {
            require(
                (x.measured, x.request_index, x.source_index, &x.ids, x.bytes)
                    == (y.measured, y.request_index, y.source_index, &y.ids, y.bytes),
                "source IDs/order/width/request drift",
            )?;
        }
    }
    Ok(())
}

fn validate_transcript(e: &Envelope, bytes: &[u8]) -> Result<Vec<TranscriptDiagnostics>> {
    require(
        sha(bytes) == e.transcript_sha256 && bytes.len() == e.transcript_bytes,
        "exact transcript identity mismatch",
    )?;
    require(
        pattern_counts(bytes).values().all(|&n| n == 0),
        "retry/breaker contamination",
    )?;
    let text = std::str::from_utf8(bytes)?;
    let mut expected = vec![run_begin(e)?];
    let mut diagnostics = Vec::new();
    let mut previous = expected[0].len() + 1;
    require(
        text.starts_with(&format!("{}\n", expected[0])),
        "parent begin must precede every child byte",
    )?;
    for c in &e.children {
        let id = c.process.as_ref().ok_or("missing process identity")?;
        let hash = c.payload_sha256.as_deref().ok_or("missing payload hash")?;
        let begin = c.begin();
        let end = c.end()?;
        expected.extend([
            begin.clone(),
            c.identity()?,
            format!("HMA1H_WORKER_IDENTITY {}", serde_json::to_string(id)?),
            format!("HMA1H_WORKER_PAYLOAD index={} sha256={hash}", c.index),
            end.clone(),
        ]);
        let left = marker_offset(text, &begin)?;
        let right = marker_offset(text, &end)?;
        require(
            left == previous && left < right,
            "parent segments must be contiguous and ordered",
        )?;
        let link = TranscriptLink::arm(c.index);
        let arm_left = marker_offset(text, &link.begin_marker)?;
        let arm_right = marker_offset(text, &link.end_marker)?;
        require(
            left < arm_left && arm_left < arm_right && arm_right < right,
            "single inherited arm must lie inside its child segment",
        )?;
        let exact = &bytes[left..right + end.len()];
        diagnostics.push(TranscriptDiagnostics {
            source: "bound-parent-child-transcript",
            begin_byte: left,
            end_byte: right + end.len(),
            sha256: sha(exact),
            exact_utf8: String::from_utf8(exact.to_vec())?,
        });
        previous = right + end.len() + 1;
    }
    let end = run_end(e);
    require(
        marker_offset(text, &end)? == previous && text.ends_with(&format!("{end}\n")),
        "parent completion must follow final exit",
    )?;
    expected.push(end);
    let markers: Vec<_> = text.lines().filter(|l| l.starts_with(PREFIX)).collect();
    require(
        markers == expected.iter().map(String::as_str).collect::<Vec<_>>(),
        "missing/duplicate/out-of-order parent or child markers",
    )?;
    require(
        text.matches("HMA1G_ARM_BEGIN").count() == e.children.len()
            && text.matches("HMA1G_ARM_END").count() == e.children.len(),
        "exactly one inherited arm per child",
    )?;
    Ok(diagnostics)
}
fn classify(analysis: &Analysis) -> &'static str {
    match super::classify(analysis) {
        "RUNTIME_LARGE_DRIFT" => "FRESH_PROCESS_LARGE_DRIFT",
        "RUNTIME_MATERIAL_DRIFT" => "FRESH_PROCESS_MATERIAL_DRIFT",
        "RUNTIME_STABLE" => "FRESH_PROCESS_STABLE",
        _ => "AMBIGUOUS",
    }
}
#[derive(Debug, Serialize)]
struct Audit {
    schema: &'static str,
    raw_report_sha256: String,
    transcript_sha256: String,
    authoritative: bool,
    classification: &'static str,
    attempted_child_count: usize,
    completed_arm_count: usize,
    child_processes: Vec<ProcessIdentity>,
    lifecycle_failure: Option<FailedArm>,
    transcript_diagnostics: Vec<TranscriptDiagnostics>,
    analysis: Option<Analysis>,
    failure: Option<String>,
}
fn audit_bytes(raw: &[u8], transcript: &[u8]) -> Audit {
    let mut audit = Audit {
        schema: "mer.gpu-native-fresh-process-lifecycle.audit.v1",
        raw_report_sha256: sha(raw),
        transcript_sha256: sha(transcript),
        authoritative: false,
        classification: "NON_AUTHORITATIVE",
        attempted_child_count: 0,
        completed_arm_count: 0,
        child_processes: Vec::new(),
        lifecycle_failure: None,
        transcript_diagnostics: Vec::new(),
        analysis: None,
        failure: None,
    };
    let result = (|| -> Result<()> {
        let e: Envelope = serde_json::from_slice(raw)?;
        require(
            serde_json::to_value(&e)? == serde_json::from_slice::<Value>(raw)?,
            "missing/extra envelope fields",
        )?;
        require(
            e.schema == SCHEMA
                && e.mode == MODE
                && e.primary == PRIMARY
                && e.cleanup_contract == CLEANUP
                && e.parent_error.is_none()
                && crate::gpu_native_real_benchmark::is_hex(&e.executable_sha256, 64),
            "frozen parent contract/evidence failure",
        )?;
        audit.attempted_child_count = e.children.len();
        audit.child_processes = e
            .children
            .iter()
            .filter_map(|c| c.process.clone())
            .collect();
        let (arms, failure) = collect_children(&e)?;
        audit.completed_arm_count = arms.len();
        audit.lifecycle_failure = failure.clone();
        audit.transcript_diagnostics = validate_transcript(&e, transcript)?;
        validate_completed(&arms)?;
        if let Some(failure) = failure {
            // This is a view of actual retained evidence, never a fabricated arm/report.
            let mut inherited = empty_payload()?;
            inherited.arms = arms;
            inherited.failure = Some(failure);
            super::validate_lifecycle(&inherited, &audit.transcript_diagnostics)?;
            audit.classification = "FRESH_PROCESS_LIFECYCLE_FAILURE";
        } else {
            require(
                arms.len() == 4 && e.children.len() == 4,
                "all four fresh children required for performance",
            )?;
            let analysis = analyze(&arms)?;
            audit.classification = classify(&analysis);
            audit.analysis = Some(analysis);
        }
        audit.authoritative = true;
        Ok(())
    })();
    if let Err(error) = result {
        audit.failure = Some(error.to_string());
    }
    audit
}
pub(crate) fn audit_command(raw: &Path, transcript: &Path, report_out: &Path) -> Result<()> {
    require(
        raw != transcript && raw != report_out && transcript != report_out,
        "artifact paths must differ",
    )?;
    canonical_output(report_out)?;
    let bytes = std::fs::read(raw)?;
    let transcript = std::fs::read(transcript)?;
    let audit = audit_bytes(&bytes, &transcript);
    write_new(report_out, &serde_json::to_vec_pretty(&audit)?)?;
    require(
        audit.authoritative,
        "HMA-1H NON_AUTHORITATIVE; see audit report",
    )
}

#[cfg(test)]
#[path = "gpu_native_fresh_process_lifecycle_tests.rs"]
mod tests;
