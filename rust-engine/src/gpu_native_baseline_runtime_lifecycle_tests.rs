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
fn fixture(walls: [u64; 4]) -> Payload {
    let inherited = super::super::hma1fb_tests::fixture();
    let mut p = empty_payload().unwrap();
    for (position, wall) in walls.into_iter().enumerate() {
        let mut arm = inherited.arms[position].clone();
        arm.mode = Mode::MappedBaseline;
        arm.records.clear();
        for measured in [false, true] {
            for request in 0..if measured { 3 } else { 1 } {
                for width in 1..=8 {
                    let mut r = super::super::hma1fb_tests::record(
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
fn bound(p: &Payload) -> (Vec<u8>, Vec<u8>) {
    let payload = serde_json::to_string(p).unwrap();
    let mut transcript = format!("{BEGIN}\n");
    let attempted = if p.complete {
        4
    } else {
        p.failure
            .as_ref()
            .and_then(|f| f.position)
            .map_or(0, |i| i + 1)
    };
    for index in 0..attempted {
        let link = TranscriptLink::arm(index);
        transcript.push_str(&format!(
            "{}\nwgpu adapter visible name=NVIDIA L4 backend=vulkan\n",
            link.begin_marker
        ));
        if !p.complete && Some(index) == p.failure.as_ref().and_then(|f| f.position) {
            transcript.push_str("wgpu request_device failed adapter=NVIDIA L4 backend=vulkan\n");
        }
        transcript.push_str(&format!(
            "selected wgpu compute plane\n{}\n",
            link.end_marker
        ));
    }
    transcript.push_str(&format!("{END}{}\n", sha(payload.as_bytes())));
    if !p.complete {
        transcript.push_str("Error: worker failed after preserving payload\n");
    }
    let e = Envelope {
        schema: SCHEMA.into(),
        payload_sha256: sha(payload.as_bytes()),
        payload_json_utf8: payload,
        transcript_sha256: sha(transcript.as_bytes()),
        transcript_bytes: transcript.len(),
        worker_exit_success: p.complete,
    };
    (serde_json::to_vec(&e).unwrap(), transcript.into_bytes())
}
fn audit(p: &Payload) -> Audit {
    let (raw, transcript) = bound(p);
    audit_bytes(&raw, &transcript)
}
fn lifecycle(index: usize, constructed: bool) -> Payload {
    let mut p = fixture([1000; 4]);
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
#[test]
fn hma1g_four_arm_classifications_and_real_width_evidence() {
    for (walls, expected) in [
        ([1000; 4], "RUNTIME_STABLE"),
        ([1000, 1100, 1100, 1100], "RUNTIME_LARGE_DRIFT"),
        ([1000, 900, 900, 900], "RUNTIME_LARGE_DRIFT"),
        ([1000, 1030, 1030, 1030], "RUNTIME_MATERIAL_DRIFT"),
        ([1000, 970, 970, 970], "RUNTIME_MATERIAL_DRIFT"),
        ([1000, 1020, 1040, 1060], "AMBIGUOUS"),
    ] {
        let p = fixture(walls);
        let a = audit(&p);
        assert!(a.authoritative, "{expected}: {:?}", a.failure);
        assert_eq!(a.classification, expected);
        let analysis = a.analysis.unwrap();
        assert_eq!(analysis.adjacent.len(), 3);
        for pair in analysis.adjacent {
            assert_eq!(pair.measured_requests.len(), 3);
            assert_eq!(pair.widths.len(), 7);
            assert!(pair
                .widths
                .iter()
                .all(|w| w.comparison.read_critical_span_ns.left > 0));
        }
    }
}
#[test]
fn hma1g_later_startup_failure_retains_exact_prefix_and_no_missing_timing() {
    for index in 1..4 {
        for constructed in [false, true] {
            let p = lifecycle(index, constructed);
            let arms = serde_json::to_value(&p.arms).unwrap();
            let a = audit(&p);
            assert!(a.authoritative, "{:?}", a.failure);
            assert_eq!(a.classification, "RUNTIME_LIFECYCLE_FAILURE");
            assert_eq!(a.completed_arm_count, index);
            assert!(a.analysis.is_none());
            assert_eq!(serde_json::to_value(&p.arms).unwrap(), arms);
            let f = a.lifecycle_failure.unwrap();
            assert_eq!(f.position, Some(index));
            assert!(f.incomplete_arm.is_none());
            assert_eq!(f.startup.unwrap().benchmark["hardware"], Value::Null);
            assert_eq!(a.transcript_diagnostics.len(), index + 1);
        }
    }
}
#[test]
fn hma1g_b0_failure_is_not_authoritative() {
    for constructed in [false, true] {
        assert!(!audit(&lifecycle(0, constructed)).authoritative);
    }
}
#[test]
fn hma1g_exact_integer_boundaries_and_sign_consistency() {
    for sign in [-1, 1] {
        let mut a = analyze(
            &fixture(if sign == 1 {
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
        for width in &mut a.adjacent[0].widths[..2] {
            width.comparison = neutral.clone();
        }
        assert_eq!(classify(&a), "RUNTIME_LARGE_DRIFT");
        a.adjacent[0].measured_requests[1].comparison = neutral.clone();
        assert_eq!(classify(&a), "AMBIGUOUS");
        a.adjacent[0].measured_requests[1].comparison = a.adjacent[0].pooled_k_gt_one.clone();
        a.adjacent[0].widths[2].comparison = neutral;
        assert_eq!(classify(&a), "AMBIGUOUS");
    }
    assert!(!Delta::new(100, 103).stable());
    assert!(!Delta::new(100, 97).stable());
    assert!(Delta::new(101, 104).stable());
    assert!(!Delta::new(101, 104).reaches(3));
    assert!(Delta::new(u64::MAX, 0).reaches(100));
    assert!(!Delta::new(0, 1).stable());
}
#[test]
fn hma1g_authority_mutations_precede_timing() {
    let mutations: &[(&str, fn(&mut Payload))] = &[
        ("pinned arm", |p| p.arms[1].mode = Mode::MappedPinned),
        ("registration", |p| {
            p.arms[1].records[0].pin.registration.attempts = 1
        }),
        ("unregister", |p| {
            p.arms[1].records[0].pin.unregister.attempts = 1
        }),
        ("SQE", |p| p.arms[1].records[0].pin.sqes_submitted = 1),
        ("active VmPin", |p| {
            p.arms[1].records[0]
                .pin
                .process
                .active
                .as_mut()
                .unwrap()
                .vmpin_bytes += 1
        }),
        ("after VmPin", |p| {
            p.arms[1].records[0]
                .pin
                .process
                .after
                .as_mut()
                .unwrap()
                .vmpin_bytes += 1
        }),
        ("VmLck", |p| {
            p.arms[1].records[0]
                .pin
                .process
                .active
                .as_mut()
                .unwrap()
                .vmlck_bytes += 1
        }),
        ("ring cleanup", |p| {
            p.arms[1].records[0].pin.ring_dropped = false
        }),
        ("source IDs", |p| p.arms[1].records[2].ids.reverse()),
        ("source bytes", |p| p.arms[1].records[2].bytes += 1),
        ("source order", |p| p.arms[1].records[2].source_index += 1),
        ("request index", |p| p.arms[1].records[2].request_index = 1),
        ("warmup identity", |p| p.arms[1].records[0].measured = true),
        ("context reused", |p| {
            p.arms[1].production["benchmark"]["runtime_contract"]["legacy_execution_plan"]
                ["context_id"] = json!("1")
        }),
        ("GL completed", |p| {
            p.arms[1].production["benchmark"]["hardware"]["wgpu_backend"] = json!("gl")
        }),
        ("wrong adapter", |p| {
            p.arms[1].production["benchmark"]["hardware"]["name"] = json!("NVIDIA L4/PCIe/SSE2")
        }),
        ("non isolated", |p| {
            p.arms[1].production["isolated_runtime"] = json!(false)
        }),
        ("missing arm", |p| {
            p.arms.pop();
        }),
        ("fifth arm", |p| p.arms.push(p.arms[0].clone())),
        ("schema", |p| p.schema = "wrong".into()),
        ("mode", |p| p.mode = "wrong".into()),
        ("tokens", |p| {
            p.arms[1].production["benchmark"]["request"]["requested_output_tokens"] = json!(127)
        }),
        ("config", |p| {
            p.arms[1].production["benchmark"]["provenance"]["artifacts"]["config"]["sha256"] =
                json!("0".repeat(64))
        }),
        ("build", |p| {
            p.arms[1].production["benchmark"]["provenance"]["build"]["dirty"] = json!(true)
        }),
        ("model", |p| {
            p.arms[1].production["benchmark"]["model_identity"]["num_layers"] = json!(47)
        }),
        ("route", |p| {
            p.arms[1].production["mechanism"]["selected_route_ids_sha256"] = json!("0".repeat(64))
        }),
        ("shutdown", |p| {
            p.arms[1].production["benchmark"]["runtime_shutdowns"][0]["evidence"]
                ["all_runtime_resources_released"] = json!(false)
        }),
    ];
    for (name, mutate) in mutations {
        let mut p = fixture([1000, 1200, 1200, 1200]);
        mutate(&mut p);
        let a = audit(&p);
        assert!(!a.authoritative, "{name}");
        assert_eq!(a.classification, "NON_AUTHORITATIVE", "{name}");
        assert!(a.analysis.is_none());
    }
}
#[test]
fn hma1g_lifecycle_authority_mutations() {
    for mutation in 0..12 {
        let mut p = lifecycle(3, true);
        let f = p.failure.as_mut().unwrap();
        match mutation {
            0 => f.position = Some(2),
            1 => f.name = Some("baseline-2".into()),
            2 => f.startup.as_mut().unwrap().runtime_validation_completed = true,
            3 => f.startup.as_mut().unwrap().qualification_enable_completed = false,
            4 => f.startup.as_mut().unwrap().benchmark["runtime_shutdowns"] = json!([]),
            5 => {
                f.startup.as_mut().unwrap().benchmark["hardware"] =
                    p.arms[0].production["benchmark"]["hardware"].clone()
            }
            6 => {
                f.primary.code = "wrong-model-geometry".into();
                f.startup.as_mut().unwrap().primary = f.primary.clone();
            }
            7 => f.records.push(p.arms[0].records[0].clone()),
            8 => f.after.as_mut().unwrap().vmpin_bytes += 1,
            9 => {
                f.startup.as_mut().unwrap().benchmark["provenance"]["build"]["dirty"] = json!(true)
            }
            10 => f.transcript.as_mut().unwrap().source = "structured-report".into(),
            _ => f.startup = None,
        }
        let a = audit(&p);
        assert!(!a.authoritative, "mutation {mutation}");
        assert!(a.analysis.is_none());
    }
    let mut p = lifecycle(2, false);
    let f = p.failure.as_mut().unwrap();
    f.primary.detail = "model file missing".into();
    f.startup.as_mut().unwrap().primary = f.primary.clone();
    assert!(!audit(&p).authoritative);
}
#[test]
fn hma1g_bound_envelope_mutations() {
    for mutation in 0..9 {
        let (raw, mut transcript) = bound(&fixture([1000; 4]));
        let mut e: Envelope = serde_json::from_slice(&raw).unwrap();
        match mutation {
            0 => e.payload_sha256 = "0".repeat(64),
            1 => e.transcript_sha256 = "0".repeat(64),
            2 => e.transcript_bytes += 1,
            3 => e.schema = "wrong".into(),
            4 => e.worker_exit_success = false,
            5 => e.payload_json_utf8.push(' '),
            6 => {
                transcript.extend_from_slice(b"HMA1G_ARM_BEGIN index=4 name=baseline-4\n");
                e.transcript_bytes = transcript.len();
                e.transcript_sha256 = sha(&transcript);
            }
            7 => {
                transcript.extend_from_slice(b"transient I/O error; retrying\n");
                e.transcript_bytes = transcript.len();
                e.transcript_sha256 = sha(&transcript);
            }
            _ => {
                e.payload_json_utf8 = "{}".into();
                e.payload_sha256 = sha(e.payload_json_utf8.as_bytes());
            }
        }
        assert!(
            !audit_bytes(&serde_json::to_vec(&e).unwrap(), &transcript).authoritative,
            "mutation {mutation}"
        );
    }
}
#[test]
fn hma1g_seam_construct_enable_validate_and_success_projection() {
    fn failure(code: &str) -> BenchmarkFailure {
        BenchmarkFailure::new("startup", code, "original detail")
    }
    for constructed in [false, true] {
        for enabled in [false, true] {
            if enabled && !constructed {
                continue;
            }
            for shutdown_failed in [false, true] {
                let primary = failure(if !constructed {
                    "runtime-construction-failed"
                } else if !enabled {
                    "qualification-arm-enable-failed"
                } else {
                    "wrong-adapter"
                });
                let shutdown = if !constructed {
                    ObservedShutdown::NotAttempted
                } else if shutdown_failed {
                    ObservedShutdown::Failed {
                        failure: failure("runtime-shutdown-failed"),
                    }
                } else {
                    ObservedShutdown::Succeeded
                };
                let partial = json!({"runtime_constructions":if constructed {vec![1]} else {vec![]},"hardware":null});
                let rich = ObservedStartupFailure {
                    primary: primary.clone(),
                    benchmark: partial.clone(),
                    construction_completed: constructed,
                    qualification_enable_completed: enabled,
                    runtime_validation_completed: false,
                    shutdown,
                };
                assert_eq!(rich.benchmark, partial);
                assert_eq!(
                    serde_json::to_value(&rich.primary).unwrap(),
                    serde_json::to_value(&primary).unwrap()
                );
                let outcome: ObservedArmOutcome<u32, Value> =
                    ObservedArmOutcome::StartupFailed(Box::new(rich));
                let old = outcome.historical().unwrap_err();
                if enabled && shutdown_failed {
                    assert_eq!(old.stage, "postcondition");
                    assert_eq!(old.code, "runtime-validation-and-shutdown-failed");
                    assert_eq!(
                        old.detail,
                        format!("{primary}; {}", failure("runtime-shutdown-failed"))
                    );
                } else {
                    assert_eq!(
                        serde_json::to_value(old).unwrap(),
                        serde_json::to_value(primary).unwrap()
                    );
                }
            }
        }
    }
    let value = Box::new(vec![1, 2, 3]);
    let pointer = value.as_ptr();
    let outcome: ObservedArmOutcome<_, Value> = ObservedArmOutcome::Run {
        run: value,
        primary_failure: None,
        shutdown: ObservedShutdown::Succeeded,
    };
    let projected = outcome.historical().unwrap();
    assert_eq!(projected.as_ptr(), pointer);
    assert_eq!(*projected, vec![1, 2, 3]);
}
#[test]
#[cfg(not(all(target_os = "linux", feature = "io_uring")))]
fn hma1g_platform_gate_precedes_paths_or_runtime() {
    assert!(launch(
        Path::new("/missing/config"),
        Path::new("/missing/report"),
        Path::new("/missing/transcript"),
        &[]
    )
    .unwrap_err()
    .to_string()
    .contains("Linux + io_uring"));
}

#[test]
fn hma1g_width_one_is_descriptive_and_opposed_endpoints_are_ambiguous() {
    let mut p = fixture([1000; 4]);
    for arm in &mut p.arms[1..] {
        for r in arm.records.iter_mut().filter(|r| r.width == 1) {
            *r = super::super::hma1fb_tests::record(
                Mode::MappedBaseline,
                r.measured,
                r.request_index,
                1,
                100000,
            );
        }
    }
    let a = audit(&p);
    assert!(a.authoritative, "{:?}", a.failure);
    assert_eq!(a.classification, "RUNTIME_STABLE");
    let mut a = analyze(&fixture([1000, 1200, 1200, 1200]).arms).unwrap();
    a.adjacent[0].pooled_k_gt_one.batch_max_read_wall_ns = Delta::new(1000, 800);
    assert_eq!(classify(&a), "AMBIGUOUS");
}
#[test]
fn hma1g_malformed_reports_and_failed_shutdown_never_qualify() {
    for mutation in 0..5 {
        let mut p = lifecycle(3, true);
        let f = p.failure.as_mut().unwrap();
        let s = f.startup.as_mut().unwrap();
        match mutation {
            0 => {
                s.benchmark.as_object_mut().unwrap().remove("hardware");
            }
            1 => {
                s.benchmark["mode"] = json!("wrong");
            }
            2 => {
                let failure = FailureIdentity {
                    stage: "postcondition".into(),
                    code: "runtime-shutdown-failed".into(),
                    detail: "shutdown detail".into(),
                };
                s.shutdown = Shutdown::Failed { failure };
                f.shutdown = Some(s.shutdown.clone());
            }
            3 => {
                s.benchmark["per_run_results"] = json!([{"run_index":0}]);
            }
            _ => {
                s.benchmark["extra"] = json!(true);
            }
        }
        assert!(!audit(&p).authoritative, "mutation {mutation}");
    }
}
#[test]
fn hma1g_offline_command_create_new_and_exact_payload_spelling() {
    let directory = std::env::temp_dir().join(format!(
        "hma1g-offline-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let (raw, transcript) = bound(&lifecycle(3, true));
    let input = directory.join("raw.json");
    let log = directory.join("transcript.log");
    let output = directory.join("audit.json");
    std::fs::write(&input, &raw).unwrap();
    std::fs::write(&log, &transcript).unwrap();
    audit_command(&input, &log, &output).unwrap();
    let report: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["classification"], "RUNTIME_LIFECYCLE_FAILURE");
    assert_eq!(report["analysis"], Value::Null);
    assert!(audit_command(&input, &log, &output).is_err());
    assert_eq!(std::fs::read(&input).unwrap(), raw);
    assert_eq!(std::fs::read(&log).unwrap(), transcript);
    let mut e: Envelope = serde_json::from_slice(&raw).unwrap();
    e.payload_json_utf8 = format!("\n{}\n", e.payload_json_utf8);
    let previous = e.payload_sha256.clone();
    e.payload_sha256 = sha(e.payload_json_utf8.as_bytes());
    let transcript = String::from_utf8(transcript)
        .unwrap()
        .replace(&previous, &e.payload_sha256)
        .into_bytes();
    e.transcript_sha256 = sha(&transcript);
    e.transcript_bytes = transcript.len();
    assert!(audit_bytes(&serde_json::to_vec(&e).unwrap(), &transcript).authoritative);
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn hma1g_seam_run_projection_preserves_incomplete_result_and_primary_shutdown_separately() {
    let primary = BenchmarkFailure::new("measured", "inference-failed", "first");
    let shutdown = BenchmarkFailure::new("postcondition", "runtime-shutdown-failed", "second");
    let run = json!({"complete":false,"failure":{"stage":"postcondition","code":"execution-and-shutdown-failed","detail":format!("{primary}; {shutdown}")}});
    let rich: ObservedArmOutcome<_, Value> = ObservedArmOutcome::Run {
        run: run.clone(),
        primary_failure: Some(primary.clone()),
        shutdown: ObservedShutdown::Failed {
            failure: shutdown.clone(),
        },
    };
    if let ObservedArmOutcome::Run {
        primary_failure: Some(f),
        shutdown: ObservedShutdown::Failed { failure: s },
        ..
    } = &rich
    {
        assert_eq!(
            serde_json::to_value(f).unwrap(),
            serde_json::to_value(primary).unwrap()
        );
        assert_eq!(
            serde_json::to_value(s).unwrap(),
            serde_json::to_value(shutdown).unwrap()
        );
    } else {
        panic!("lost independent failure evidence");
    }
    assert_eq!(rich.historical().unwrap(), run);
}

#[test]
fn hma1g_malformed_index_and_missing_optional_fields_reject_without_panicking() {
    let (raw, transcript) = bound(&lifecycle(3, true));
    for mutation in 0..3 {
        let mut e: Envelope = serde_json::from_slice(&raw).unwrap();
        let previous = e.payload_sha256.clone();
        let mut payload: Value = serde_json::from_str(&e.payload_json_utf8).unwrap();
        match mutation {
            0 => payload["failure"]["position"] = json!(usize::MAX),
            1 => {
                payload["failure"]
                    .as_object_mut()
                    .unwrap()
                    .remove("incomplete_arm");
            }
            _ => {
                payload["failure"]["startup"]["benchmark"]
                    .as_object_mut()
                    .unwrap()
                    .remove("model_load");
            }
        }
        e.payload_json_utf8 = serde_json::to_string(&payload).unwrap();
        e.payload_sha256 = sha(e.payload_json_utf8.as_bytes());
        let transcript = String::from_utf8(transcript.clone())
            .unwrap()
            .replace(&previous, &e.payload_sha256)
            .into_bytes();
        e.transcript_sha256 = sha(&transcript);
        e.transcript_bytes = transcript.len();
        let a = audit_bytes(&serde_json::to_vec(&e).unwrap(), &transcript);
        assert!(!a.authoritative, "mutation {mutation}");
        assert!(a.analysis.is_none());
    }
}
