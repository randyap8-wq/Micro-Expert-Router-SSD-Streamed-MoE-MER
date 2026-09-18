use super::*;

fn multiply(v: &mut Value, factor: u64) {
    for (key, value) in v.as_object_mut().unwrap() {
        if [
            "ring_capacity",
            "high_water",
            "install_set_width_min",
            "install_set_width_max",
            "rayon_num_threads",
            "max_in_flight_physical_staging",
            "primary_pool_capacity",
            "shadow_pool_capacity",
        ]
        .contains(&key.as_str())
        {
            continue;
        }
        if let Some(n) = value.as_u64() {
            *value = json!(n * factor);
        }
    }
}
fn inherited_fixture(walls: [u64; 4]) -> Payload {
    let inherited = super::super::super::hma1fb_tests::fixture();
    let mut p = empty_payload().unwrap();
    for (position, wall) in walls.into_iter().enumerate() {
        let mut arm = inherited.arms[position].clone();
        arm.mode = Mode::MappedBaseline;
        arm.records.clear();
        for measured in [false, true] {
            for request in 0..if measured { 3 } else { 1 } {
                for width in 1..=8 {
                    let mut r = super::super::super::hma1fb_tests::record(
                        Mode::MappedBaseline,
                        measured,
                        request,
                        width,
                        wall,
                    );
                    r.source_index = width - 1;
                    arm.records.push(r);
                }
            }
        }
        let b = &mut arm.production["benchmark"];
        use crate::gpu_native_real_benchmark as benchmark;
        for (field, value) in [
            ("schema", benchmark::SCHEMA),
            ("mode", benchmark::MODE),
            ("optimization", benchmark::OPTIMIZATION),
            (
                "immediate_comparison_commit",
                benchmark::IMMEDIATE_COMPARISON_COMMIT,
            ),
            (
                "pr1b_a_experiment_commit",
                benchmark::PR1B_A_EXPERIMENT_COMMIT,
            ),
            (
                "pr1b_a_experiment_report_sha256",
                benchmark::PR1B_A_EXPERIMENT_REPORT_SHA256,
            ),
            (
                "original_baseline_commit",
                benchmark::ORIGINAL_BASELINE_COMMIT,
            ),
        ] {
            b[field] = json!(value);
        }
        b["correctness_qualification_pending"] = json!(true);
        b["qualification_pass"] = json!(false);
        b["aggregate"] = Value::Null;
        for r in b["per_run_results"].as_array_mut().unwrap() {
            for key in [
                "engine_storage_before",
                "engine_storage_after",
                "engine_storage_delta",
            ] {
                let c = &mut r["counters"][key];
                c["nvme_read_operations"] = json!(c["nvme_read_operations"].as_u64().unwrap() * 8);
                c["nvme_bytes_read"] = json!(c["nvme_bytes_read"].as_u64().unwrap() * 18);
            }
        }
        for measured in [false, true] {
            let prefix = if measured { "" } else { "warmup_" };
            let mk = if measured {
                "mechanism"
            } else {
                "warmup_mechanism"
            };
            let count = if measured { 3 } else { 1 };
            for key in [
                mk.to_string(),
                format!("{prefix}upload"),
                format!("{prefix}production_physical_install"),
            ] {
                multiply(&mut arm.production[key], 18);
            }
            let s: Snapshot = serde_json::from_value(arm.production[mk].clone()).unwrap();
            arm.production[format!("{prefix}source")] =
                serde_json::to_value(concurrency_common_snapshot(&s)).unwrap();
            let u = &mut arm.production[format!("{prefix}upload")];
            multiply(&mut u["source_upload_fd_proof"], 18);
            let mut h = Sha256::new();
            for r in arm.records.iter().filter(|r| r.measured == measured) {
                for id in &r.ids {
                    h.update(id.to_le_bytes());
                }
            }
            u["ordered_nvme_ids_sha256"] = json!(format!("{:x}", h.finalize()));
            let source = &mut arm.production[format!("{prefix}production")];
            source["production_source_sets"] = json!(18 * count);
            for key in [
                "production_batch_eligible_sets",
                "production_batch_attempts",
                "production_batch_successes",
            ] {
                source[key] = json!(7 * count);
            }
            source["production_batch_experts"] = json!(35 * count);
            source["production_batch_width_min"] = json!(2);
            source["production_batch_width_max"] = json!(8);
            source["production_batch_width_mean"] = json!(5.0);
            let work = &mut arm.production[format!("{prefix}work")];
            multiply(&mut work["gpu_native_residency"], 18);
            work["engine_storage"]["nvme_read_operations"] = json!(8 * count);
            work["engine_storage"]["nvme_bytes_read"] = json!(36 * FULL as u64 * count);
        }
        p.arms.push(arm);
    }
    p.complete = true;
    p
}
fn inherited_lifecycle(index: usize, constructed: bool) -> Payload {
    let mut p = inherited_fixture([1000; 4]);
    let failed = p.arms[index].clone();
    let mut b = failed.production["benchmark"].clone();
    b["benchmark_complete"] = json!(false);
    b["warmup_runs_completed"] = json!(0);
    b["per_run_results"] = json!([]);
    for field in ["hardware", "model_load", "runtime_contract", "aggregate"] {
        b[field] = Value::Null;
    }
    if !constructed {
        b["runtime_constructions"] = json!([]);
        b["runtime_shutdowns"] = json!([]);
    }
    let primary = FailureIdentity {
        stage: "startup".into(),
        code: if constructed {
            "wrong-adapter"
        } else {
            "runtime-construction-failed"
        }
        .into(),
        detail: if constructed {
            "selected adapter NVIDIA L4/PCIe/SSE2 did not equal NVIDIA L4"
        } else {
            "adapter found but request_device failed: Vulkan initialization failed"
        }
        .into(),
    };
    let shutdown = if constructed {
        Shutdown::Succeeded
    } else {
        Shutdown::NotAttempted
    };
    p.failure = Some(FailedArm {
        position: Some(index),
        name: Some(ORDER[index].into()),
        primary: primary.clone(),
        startup: Some(StartupEvidence {
            primary,
            benchmark: b,
            construction_completed: constructed,
            qualification_enable_completed: constructed,
            runtime_validation_completed: false,
            shutdown: shutdown.clone(),
        }),
        incomplete_arm: None,
        shutdown: Some(shutdown),
        before: Some(failed.before),
        after: Some(failed.after),
        records: Vec::new(),
        transcript: Some(TranscriptLink::arm(index)),
    });
    p.arms.truncate(index);
    p.complete = false;
    p
}
fn identity(pid: u32, start: u64) -> ProcessIdentity {
    ProcessIdentity {
        pid,
        ppid: 100,
        boot_id: "01234567-89ab-cdef-0123-456789abcdef".into(),
        start_ticks: start,
    }
}
fn envelope(p: Payload) -> Envelope {
    let executable = p
        .arms
        .first()
        .map(|a| {
            a.production["benchmark"]["provenance"]["executable_sha256"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .unwrap_or_else(|| {
            p.failure
                .as_ref()
                .unwrap()
                .startup
                .as_ref()
                .unwrap()
                .benchmark["provenance"]["executable_sha256"]
                .as_str()
                .unwrap()
                .to_string()
        });
    let mut e = Envelope {
        schema: SCHEMA.into(),
        mode: MODE.into(),
        primary: PRIMARY.into(),
        cleanup_contract: CLEANUP.into(),
        parent: ProcessIdentity {
            ppid: 1,
            ..identity(100, 10)
        },
        executable_sha256: executable,
        children: Vec::new(),
        transcript_sha256: String::new(),
        transcript_bytes: 0,
        parent_error: None,
    };
    let count = p.arms.len() + usize::from(p.failure.is_some());
    for index in 0..count {
        let process = identity(200 + index as u32, 20 + index as u64);
        let mut arm = p.arms.get(index).cloned();
        if let Some(a) = &mut arm {
            // The context counter restarts in each process. It is not an OS identity.
            a.production["benchmark"]["runtime_contract"]["legacy_execution_plan"]["context_id"] =
                json!("1");
        }
        let failure = if arm.is_none() {
            p.failure.clone()
        } else {
            None
        };
        let payload = ChildPayload {
            schema: SCHEMA.into(),
            mode: MODE.into(),
            index,
            name: ORDER[index].into(),
            process: process.clone(),
            frozen_workload: serde_json::to_value(frozen_workload("NVIDIA L4".into())).unwrap(),
            worker_error: failure
                .as_ref()
                .map(|f| format!("{}: {}", f.primary.code, f.primary.detail)),
            arm,
            failure,
        };
        let bytes = serde_json::to_vec(&payload).unwrap();
        e.children.push(ChildBinding {
            index,
            name: ORDER[index].into(),
            launch_ns: 100 + index as u64 * 100,
            exit_ns: Some(150 + index as u64 * 100),
            spawned_pid: Some(process.pid),
            process: Some(process),
            exit: Some(ChildExit::expected(if payload.arm.is_some() {
                0
            } else {
                1
            })),
            payload_hex: Some(encode_hex(&bytes)),
            payload_sha256: Some(sha(&bytes)),
            evidence_error: None,
        });
    }
    e
}
fn fixture(walls: [u64; 4]) -> Envelope {
    envelope(inherited_fixture(walls))
}
fn mutate_payload(e: &mut Envelope, index: usize, f: impl FnOnce(&mut ChildPayload)) {
    let c = &mut e.children[index];
    let bytes = decode_hex(c.payload_hex.as_ref().unwrap()).unwrap();
    let mut p: ChildPayload = serde_json::from_slice(&bytes).unwrap();
    f(&mut p);
    let bytes = serde_json::to_vec(&p).unwrap();
    c.payload_sha256 = Some(sha(&bytes));
    c.payload_hex = Some(encode_hex(&bytes));
}
fn bind_transcript(mut e: Envelope, text: String) -> (Vec<u8>, Vec<u8>) {
    e.transcript_sha256 = sha(text.as_bytes());
    e.transcript_bytes = text.len();
    (serde_json::to_vec(&e).unwrap(), text.into_bytes())
}
fn bound(e: &Envelope) -> (Vec<u8>, Vec<u8>) {
    let mut t = format!("{}\n", run_begin(e).unwrap());
    for c in &e.children {
        t.push_str(&format!("{}\n", c.begin()));
        if let Ok(id) = c.identity() {
            t.push_str(&format!("{id}\n"));
        }
        if let Some(id) = &c.process {
            t.push_str(&format!(
                "HMA1H_WORKER_IDENTITY {}\n",
                serde_json::to_string(id).unwrap()
            ));
        }
        let link = TranscriptLink::arm(c.index.min(3));
        t.push_str(&format!("{}\nwgpu adapter visible name=NVIDIA L4 backend=vulkan\nwgpu request_device failed adapter=NVIDIA L4 backend=vulkan\nselected wgpu compute plane\n{}\n",link.begin_marker,link.end_marker));
        if let Some(hash) = &c.payload_sha256 {
            t.push_str(&format!(
                "HMA1H_WORKER_PAYLOAD index={} sha256={hash}\n",
                c.index
            ));
        }
        t.push_str(&format!("{}\n", c.end().unwrap()));
    }
    t.push_str(&format!("{}\n", run_end(e)));
    bind_transcript(e.clone(), t)
}
fn audit(e: &Envelope) -> Audit {
    let (raw, t) = bound(e);
    audit_bytes(&raw, &t)
}

#[test]
fn hma1h_four_fresh_process_classifications_use_unchanged_integer_analysis() {
    for (walls, expected) in [
        ([1000; 4], "FRESH_PROCESS_STABLE"),
        ([1000, 1100, 1100, 1100], "FRESH_PROCESS_LARGE_DRIFT"),
        ([1000, 900, 900, 900], "FRESH_PROCESS_LARGE_DRIFT"),
        ([1000, 1030, 1030, 1030], "FRESH_PROCESS_MATERIAL_DRIFT"),
        ([1000, 970, 970, 970], "FRESH_PROCESS_MATERIAL_DRIFT"),
        ([1000, 1020, 1040, 1060], "AMBIGUOUS"),
    ] {
        let e = fixture(walls);
        let a = audit(&e);
        assert!(a.authoritative, "{expected}: {:?}", a.failure);
        assert_eq!(a.classification, expected);
        let inherited = analyze(&inherited_fixture(walls).arms).unwrap();
        assert_eq!(
            serde_json::to_value(a.analysis.unwrap()).unwrap(),
            serde_json::to_value(inherited).unwrap()
        );
        assert_eq!(a.child_processes.len(), 4);
    }
}
#[test]
fn hma1h_recognized_lifecycle_failure_at_every_later_child() {
    for index in 1..4 {
        for constructed in [false, true] {
            let p = inherited_lifecycle(index, constructed);
            let failure = serde_json::to_value(&p.failure).unwrap();
            let e = envelope(p);
            let a = audit(&e);
            assert!(a.authoritative, "{index}/{constructed}: {:?}", a.failure);
            assert_eq!(a.classification, "FRESH_PROCESS_LIFECYCLE_FAILURE");
            assert_eq!(a.completed_arm_count, index);
            assert_eq!(a.attempted_child_count, index + 1);
            assert!(a.analysis.is_none());
            assert_eq!(serde_json::to_value(a.lifecycle_failure).unwrap(), failure);
            assert_eq!(a.child_processes.len(), index + 1);
            assert_eq!(a.transcript_diagnostics.len(), index + 1);
        }
    }
}
#[test]
fn hma1h_b0_failure_never_authoritative() {
    for constructed in [false, true] {
        let a = audit(&envelope(inherited_lifecycle(0, constructed)));
        assert!(!a.authoritative);
        assert_eq!(a.classification, "NON_AUTHORITATIVE");
        assert!(a.analysis.is_none());
    }
}
#[test]
fn hma1h_parent_process_and_binding_mutations_reject() {
    let cases: &[(&str, fn(&mut Envelope))] = &[
        ("duplicate process", |e| {
            e.children[1].process = e.children[0].process.clone();
            e.children[1].spawned_pid = e.children[0].spawned_pid;
        }),
        ("wrong PPID", |e| {
            e.children[1].process.as_mut().unwrap().ppid = 101
        }),
        ("parent reused", |e| {
            e.children[1].process = Some(e.parent.clone())
        }),
        ("different boot", |e| {
            e.children[1].process.as_mut().unwrap().boot_id =
                "11234567-89ab-cdef-0123-456789abcdef".into()
        }),
        ("zero start", |e| {
            e.children[1].process.as_mut().unwrap().start_ticks = 0
        }),
        ("child predates parent", |e| {
            e.children[1].process.as_mut().unwrap().start_ticks = 1
        }),
        ("spawn PID", |e| e.children[1].spawned_pid = Some(999)),
        ("missing identity", |e| e.children[1].process = None),
        ("missing exit", |e| e.children[1].exit = None),
        ("missing wait", |e| e.children[1].exit_ns = None),
        ("overlap", |e| {
            e.children[1].launch_ns = e.children[0].exit_ns.unwrap()
        }),
        ("exit before launch", |e| e.children[1].exit_ns = Some(1)),
        ("exit raw", |e| {
            e.children[1].exit.as_mut().unwrap().unix_wait_status = 256
        }),
        ("exit code", |e| {
            e.children[1].exit.as_mut().unwrap().code = Some(1)
        }),
        ("signal", |e| {
            e.children[1].exit.as_mut().unwrap().signal = Some(9)
        }),
        ("core dump", |e| {
            e.children[1].exit.as_mut().unwrap().core_dumped = true
        }),
        ("missing payload", |e| e.children[1].payload_hex = None),
        ("malformed bytes", |e| {
            e.children[1].payload_hex = Some("ff00".into());
            e.children[1].payload_sha256 = Some(sha(&[255, 0]));
        }),
        ("payload hash", |e| {
            e.children[1].payload_sha256 = Some("0".repeat(64))
        }),
        ("missing arm", |e| {
            e.children.remove(1);
        }),
        ("reordered arms", |e| e.children.swap(1, 2)),
        ("fifth child", |e| e.children.push(e.children[3].clone())),
        ("incomplete prefix", |e| {
            e.children.pop();
        }),
        ("parent error", |e| e.parent_error = Some("failed".into())),
        ("evidence error", |e| {
            e.children[1].evidence_error = Some("failed".into())
        }),
        ("executable drift", |e| e.executable_sha256 = "0".repeat(64)),
        ("schema drift", |e| e.schema = "changed".into()),
    ];
    for (name, f) in cases {
        let mut e = fixture([1000; 4]);
        f(&mut e);
        let a = audit(&e);
        assert!(!a.authoritative, "{name}");
        assert!(a.analysis.is_none(), "{name}");
    }
}
#[test]
fn hma1h_payload_authority_mutations_reject_after_rehash() {
    let cases: &[(&str, fn(&mut ChildPayload))] = &[
        ("process substitution", |p| p.process.pid = 999),
        ("wrong position", |p| p.index = 0),
        ("frozen workload", |p| {
            p.frozen_workload["output_tokens"] = json!(127)
        }),
        ("missing arm", |p| p.arm = None),
        ("worker error", |p| p.worker_error = Some("failed".into())),
        ("pinned", |p| {
            p.arm.as_mut().unwrap().mode = Mode::MappedPinned
        }),
        ("registration", |p| {
            p.arm.as_mut().unwrap().records[0].pin.registration.attempts = 1
        }),
        ("unregister", |p| {
            p.arm.as_mut().unwrap().records[0].pin.unregister.attempts = 1
        }),
        ("SQE", |p| {
            p.arm.as_mut().unwrap().records[0].pin.sqes_submitted = 1
        }),
        ("startup count", |p| {
            let b = &mut p.arm.as_mut().unwrap().production["benchmark"];
            b["runtime_constructions"] = json!([]);
        }),
        ("shutdown count", |p| {
            p.arm.as_mut().unwrap().production["benchmark"]["runtime_shutdowns"] = json!([])
        }),
        ("software fallback", |p| {
            p.arm.as_mut().unwrap().production["benchmark"]["hardware"]["software_adapter"] =
                json!(true)
        }),
        ("GL fallback", |p| {
            p.arm.as_mut().unwrap().production["benchmark"]["hardware"]["wgpu_backend"] =
                json!("gl")
        }),
        ("source ids", |p| {
            p.arm.as_mut().unwrap().records[0].ids[0] += 1
        }),
        ("missing record", |p| {
            p.arm.as_mut().unwrap().records.pop();
        }),
        ("duplicate record", |p| {
            let a = p.arm.as_mut().unwrap();
            a.records.push(a.records[0].clone());
        }),
        ("provenance", |p| {
            p.arm.as_mut().unwrap().production["benchmark"]["provenance"]["build"]["git_sha"] =
                json!("f".repeat(40))
        }),
        ("dirty", |p| {
            p.arm.as_mut().unwrap().production["benchmark"]["provenance"]["build"]["dirty"] =
                json!(true)
        }),
        ("noncanonical context", |p| {
            p.arm.as_mut().unwrap().production["benchmark"]["runtime_contract"]
                ["legacy_execution_plan"]["context_id"] = json!("01")
        }),
    ];
    for (name, f) in cases {
        let mut e = fixture([1000; 4]);
        mutate_payload(&mut e, 1, f);
        let a = audit(&e);
        assert!(!a.authoritative, "{name}");
        assert!(a.analysis.is_none());
    }
}
#[test]
fn hma1h_unrecognized_or_unclean_failures_never_promote() {
    let cases: &[(&str, fn(&mut ChildPayload))] = &[
        ("unknown code", |p| {
            p.failure.as_mut().unwrap().primary.code = "anything".into()
        }),
        ("model failure", |p| {
            let f = p.failure.as_mut().unwrap();
            f.primary.detail = "model cannot load".into();
            f.startup.as_mut().unwrap().primary = f.primary.clone();
        }),
        ("post-work", |p| {
            p.failure.as_mut().unwrap().incomplete_arm = Some(json!({}))
        }),
        ("missing startup", |p| {
            p.failure.as_mut().unwrap().startup = None
        }),
        ("missing after", |p| {
            p.failure.as_mut().unwrap().after = None
        }),
        ("shutdown failed", |p| {
            p.failure.as_mut().unwrap().shutdown = Some(Shutdown::Failed {
                failure: FailureIdentity {
                    stage: "shutdown".into(),
                    code: "failed".into(),
                    detail: "failure".into(),
                },
            })
        }),
        ("measured startup", |p| {
            p.failure
                .as_mut()
                .unwrap()
                .startup
                .as_mut()
                .unwrap()
                .benchmark["per_run_results"] = json!([{}])
        }),
        ("startup config", |p| {
            p.failure
                .as_mut()
                .unwrap()
                .startup
                .as_mut()
                .unwrap()
                .benchmark["production_configuration"] = json!({})
        }),
        ("work started", |p| {
            p.failure
                .as_mut()
                .unwrap()
                .records
                .push(super::super::super::hma1fb_tests::record(
                    Mode::MappedBaseline,
                    false,
                    0,
                    2,
                    1000,
                ))
        }),
    ];
    for (name, f) in cases {
        let mut e = envelope(inherited_lifecycle(2, false));
        mutate_payload(&mut e, 2, f);
        let a = audit(&e);
        assert!(!a.authoritative, "{name}");
        assert!(a.analysis.is_none());
    }
    let mut e = envelope(inherited_lifecycle(2, false));
    e.children[2].exit = Some(ChildExit {
        unix_wait_status: 9,
        code: None,
        signal: Some(9),
        core_dumped: false,
    });
    assert!(!audit(&e).authoritative);
}
#[test]
fn hma1h_transcript_rebinding_cannot_hide_order_or_marker_corruption() {
    let e = fixture([1000; 4]);
    let (_, t) = bound(&e);
    let text = String::from_utf8(t).unwrap();
    let first = e.children[0].begin();
    let end = e.children[0].end().unwrap();
    for (name, mutated) in [
        ("missing begin", text.replace(&format!("{first}\n"), "")),
        (
            "duplicate begin",
            text.replace(&first, &format!("{first}\n{first}")),
        ),
        ("end before begin", text.replacen(&first, &end, 1)),
        (
            "wrong identity",
            text.replacen("HMA1H_WORKER_IDENTITY", "HMA1H_BAD_IDENTITY", 1),
        ),
        (
            "fifth marker",
            text.replace(
                "HMA1H_QUALIFIER_END",
                "HMA1H_CHILD_BEGIN index=4\nHMA1H_QUALIFIER_END",
            ),
        ),
        (
            "retry",
            text.replace(
                "selected wgpu compute plane",
                "transient I/O error; retrying",
            ),
        ),
        (
            "extra arm",
            text.replace(
                &first,
                &format!("{first}\nHMA1G_ARM_BEGIN index=4 name=baseline-4"),
            ),
        ),
        ("unbound prefix", format!("extra\n{text}")),
        ("unbound suffix", format!("{text}extra\n")),
        ("missing exit marker", text.replace(&format!("{end}\n"), "")),
    ] {
        let (raw, t) = bind_transcript(e.clone(), mutated);
        assert!(!audit_bytes(&raw, &t).authoritative, "{name}");
    }
    let (raw, mut t) = bound(&e);
    t.push(b'!');
    assert!(!audit_bytes(&raw, &t).authoritative);
}
#[test]
fn hma1h_missing_optional_wire_fields_are_not_synthesized() {
    let (raw, t) = bound(&fixture([1000; 4]));
    let base: Value = serde_json::from_slice(&raw).unwrap();
    for key in ["parent_error", "transcript_sha256", "parent", "children"] {
        let mut v = base.clone();
        v.as_object_mut().unwrap().remove(key);
        assert!(!audit_bytes(&serde_json::to_vec(&v).unwrap(), &t).authoritative);
    }
    let mut e = fixture([1000; 4]);
    let c = &mut e.children[0];
    let mut p: Value =
        serde_json::from_slice(&decode_hex(c.payload_hex.as_ref().unwrap()).unwrap()).unwrap();
    p.as_object_mut().unwrap().remove("failure");
    let bytes = serde_json::to_vec(&p).unwrap();
    c.payload_hex = Some(encode_hex(&bytes));
    c.payload_sha256 = Some(sha(&bytes));
    assert!(!audit(&e).authoritative);
}
#[test]
fn hma1h_proc_stat_parser_uses_field_22_after_parenthesized_comm() {
    let boot = "01234567-89ab-cdef-0123-456789abcdef\n";
    let mut fields = vec!["0"; 20];
    fields[0] = "S";
    fields[1] = "100";
    fields[19] = "123456";
    let stat = format!("200 (worker with ) parentheses) {}", fields.join(" "));
    assert_eq!(
        ProcessIdentity::parse(&stat, boot).unwrap(),
        identity(200, 123456)
    );
    for s in ["200 x", "200 (x) S 1", "0 (x) S 1"] {
        assert!(ProcessIdentity::parse(s, boot).is_err());
    }
    assert!(ProcessIdentity::parse(&stat, "bad-boot").is_err());
}
#[test]
fn hma1h_forwarding_preserves_process_options_and_exact_one_arm() {
    let raw: Vec<std::ffi::OsString> = [
        "mer",
        "--rayon-threads",
        "8",
        MODE,
        "--report-out=old",
        "--transcript-out",
        "old-log",
        "--config=old-config",
        "--log",
        "debug",
        "--worker-threads",
        "2",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    for index in 0..4 {
        let a =
            worker_arguments(Path::new("frozen.toml"), Path::new("new.json"), index, &raw).unwrap();
        assert_eq!(a.iter().filter(|s| *s == WORKER).count(), 1);
        assert_eq!(a.iter().filter(|s| *s == "--arm-index").count(), 1);
        assert_eq!(
            a.last().unwrap(),
            &std::ffi::OsString::from(index.to_string())
        );
        for value in [
            "--rayon-threads",
            "8",
            "--worker-threads",
            "2",
            "info",
            "frozen.toml",
            "new.json",
        ] {
            assert!(a.contains(&value.into()));
        }
        assert!(!a.contains(&"debug".into()));
        assert!(!a.contains(&"old-log".into()));
    }
    assert!(worker_arguments(Path::new("c"), Path::new("p"), 4, &raw).is_err());
    let mut bad = raw.clone();
    bad.push("--autotune".into());
    assert!(worker_arguments(Path::new("c"), Path::new("p"), 0, &bad).is_err());
}
#[test]
fn hma1h_cli_worker_is_hidden_and_index_bounded() {
    use clap::{CommandFactory, Parser};
    let c = crate::Cli::command();
    assert!(c
        .get_subcommands()
        .find(|c| c.get_name() == WORKER)
        .unwrap()
        .is_hide_set());
    for index in ["0", "3"] {
        assert!(crate::Cli::try_parse_from([
            "mer",
            WORKER,
            "--arm-index",
            index,
            "--config",
            "c",
            "--report-out",
            "p"
        ])
        .is_ok());
    }
    for index in ["4", "-1", "256"] {
        assert!(crate::Cli::try_parse_from([
            "mer",
            WORKER,
            "--arm-index",
            index,
            "--config",
            "c",
            "--report-out",
            "p"
        ])
        .is_err());
    }
}
#[test]
fn hma1h_exact_thresholds_and_request_width_gates() {
    for sign in [-1, 1] {
        let mut a = analyze(
            &inherited_fixture(if sign == 1 {
                [1000, 1100, 1100, 1100]
            } else {
                [1000, 900, 900, 900]
            })
            .arms,
        )
        .unwrap();
        let neutral = Comparison::new(
            Endpoints {
                critical: 1000,
                max_wall: 1000,
            },
            Endpoints {
                critical: 1000,
                max_wall: 1000,
            },
        );
        a.adjacent[0].measured_requests[0].comparison = neutral.clone();
        for w in &mut a.adjacent[0].widths[..2] {
            w.comparison = neutral.clone();
        }
        assert_eq!(classify(&a), "FRESH_PROCESS_LARGE_DRIFT");
        a.adjacent[0].measured_requests[1].comparison = neutral.clone();
        assert_eq!(classify(&a), "AMBIGUOUS");
        a.adjacent[0].measured_requests[1].comparison = a.adjacent[0].pooled_k_gt_one.clone();
        a.adjacent[0].widths[2].comparison = neutral;
        assert_eq!(classify(&a), "AMBIGUOUS");
    }
    assert!(!Delta::new(100, 103).stable());
    assert!(!Delta::new(100, 97).stable());
    assert!(Delta::new(101, 104).stable());
    assert!(Delta::new(u64::MAX, 0).reaches(100));
    assert!(!Delta::new(0, 1).stable());
}
#[test]
fn hma1h_coherent_identity_reuse_rejects_but_pid_recycling_is_distinct() {
    let mut e = fixture([1000; 4]);
    let id = e.children[0].process.clone().unwrap();
    e.children[1].process = Some(id.clone());
    e.children[1].spawned_pid = Some(id.pid);
    mutate_payload(&mut e, 1, |p| p.process = id.clone());
    assert!(!audit(&e).authoritative);
    let recycled = ProcessIdentity {
        start_ticks: id.start_ticks + 1,
        ..id
    };
    e.children[1].process = Some(recycled.clone());
    mutate_payload(&mut e, 1, |p| p.process = recycled);
    assert!(audit(&e).authoritative);
}
#[test]
fn hma1h_cpu_auditor_fresh_outputs_and_exact_payload_snapshot() {
    let mut e = fixture([1000; 4]);
    let c = &mut e.children[0];
    let bytes = decode_hex(c.payload_hex.as_ref().unwrap()).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let pretty = serde_json::to_vec_pretty(&value).unwrap();
    assert_ne!(sha(&pretty), sha(&bytes));
    c.payload_hex = Some(encode_hex(&pretty));
    c.payload_sha256 = Some(sha(&pretty));
    let (raw, transcript) = bound(&e);
    let dir = std::env::temp_dir().join(format!(
        "hma1h-cpu-audit-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let raw_path = dir.join("raw.json");
    let transcript_path = dir.join("transcript.log");
    let audit_path = dir.join("audit.json");
    write_new(&raw_path, &raw).unwrap();
    write_new(&transcript_path, &transcript).unwrap();
    audit_command(&raw_path, &transcript_path, &audit_path).unwrap();
    let saved: Value = serde_json::from_slice(&std::fs::read(&audit_path).unwrap()).unwrap();
    assert_eq!(saved["classification"], "FRESH_PROCESS_STABLE");
    assert_eq!(saved["raw_report_sha256"], sha(&raw));
    assert!(audit_command(&raw_path, &transcript_path, &audit_path).is_err());
    assert!(write_new(&raw_path, b"overwrite").is_err());
    #[cfg(unix)]
    {
        let dangling = dir.join("dangling");
        std::os::unix::fs::symlink(dir.join("missing"), &dangling).unwrap();
        assert!(canonical_output(&dangling).is_err());
    }
    assert_eq!(std::fs::read(&raw_path).unwrap(), raw);
    std::fs::remove_dir_all(&dir).unwrap();
}
