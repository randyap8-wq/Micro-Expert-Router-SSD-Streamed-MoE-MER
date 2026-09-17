#!/usr/bin/env python3
"""Offline HMA-1G branch proof and in-memory mutations; never runs hardware."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = '7ddf34d55dcd6fcd2ce4091a52e80fd83ee03702'
TREE = 'de079a1c96ceaffc5890b44e5dafd7e8a1fd3f26'
P = 'rust-engine/src/'
SHARED = P + 'gpu_native_physical_install_staging.rs'
PIN = P + 'gpu_native_source_mapped_pin_production.rs'
NEW = P + 'gpu_native_baseline_runtime_lifecycle.rs'
Q4 = P + 'gpu_native_q4_route_parallel.rs'
EXISTING = {SHARED, PIN, Q4, P + 'main.rs'}
ADDED = {NEW, P + 'gpu_native_baseline_runtime_lifecycle_tests.rs', 'scripts/hma1g-lifecycle-source-proof.py'}

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])
def sha(data):
    return hashlib.sha256(data).hexdigest()
def compact(value):
    return re.sub(r'\s+', '', value)
spec = importlib.util.spec_from_file_location('frozen_extractor', ROOT / 'scripts/hma1f-source-proof.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
def function(data, name):
    return legacy.function(data, name).decode()


def prove(old, current, child=None):
    checks = []
    def check(name, ok):
        checks.append({'check': name, 'pass': bool(ok)})
    def body(path, name):
        return function(current[path], name)
    check('exact frozen base tree', git('rev-parse', BASE + '^{tree}').decode().strip() == TREE)
    check('exact branch', git('branch', '--show-current').decode().strip() == 'diag/hma1g-baseline-runtime-lifecycle')
    if child:
        check('single child of frozen base', git('rev-list', '--parents', '-n', '1', child).decode().split() == [child, BASE])
    else:
        check('uncommitted work starts from exact base', git('rev-parse', 'HEAD').decode().strip() == BASE)
    changed = {p for p in current if old.get(p) != current[p]}
    check('exact seven-file boundary', changed == EXISTING | ADDED)
    protected = {p: sha(data) for p, data in old.items() if p not in EXISTING}
    check('every other historical file byte-identical', all(current.get(p) == old[p] for p in protected))
    retry = legacy.function(current[P + 'io_provider.rs'], 'read_at_with_retries')
    check('protected retry exact bytes/hash', len(retry) == 6909 and sha(retry) == '9f3241f79c6ebf02d4fa796b3e2bd3e2b2d45fc9740f2532872e5e3330b258c0')
    for name in ['backend/mod.rs','gpu_native_real_benchmark.rs','engine.rs','io_provider.rs','io_uring_storage.rs','gpu_native_source_upload.rs','gpu_native_mapped_pin.rs','gpu_native_mapped_fixed_buffer.rs','gpu_native_mapped_lock.rs']:
        check(name + ' byte-identical', current[P + name] == old[P + name])
    pinned = current[PIN].decode()
    restored = pinned.replace('#[path = "gpu_native_baseline_runtime_lifecycle.rs"]\npub(crate) mod baseline_lifecycle;\n\n', '', 1)
    restored = restored.replace('    pub(super) fn fixture() -> Payload {', '    fn fixture() -> Payload {', 1)
    restored = restored.replace('    pub(super) fn record(\n        mode: Mode,\n        measured: bool,\n        request_index: usize,\n        width: usize,\n        wall: u64,\n    ) -> Record {', '    fn record(mode: Mode, measured: bool, request_index: usize, width: usize, wall: u64) -> Record {', 1)
    check('HMA-1F-B only module declaration and two test-fixture visibility changes', restored.encode() == old[PIN])
    check('HMA-1F-B calls only historical runner', 'run_physical_install_arm_observed_inner(' in body(PIN, 'run_worker') and 'run_physical_install_arm_observed_outcome' not in pinned)
    check('Q4 only exact shared-file SHA substitution', current[Q4] == old[Q4].replace(sha(old[SHARED]).encode(), sha(current[SHARED]).encode(), 1))
    check('Q4 production bytes unchanged', current[Q4].split(b'#[cfg(test)]')[0] == old[Q4].split(b'#[cfg(test)]')[0])
    shared = current[SHARED].decode()
    base_shared = old[SHARED].decode()
    check('shared file prefix outside seam unchanged', shared.split('/// HMA-1G can retain startup evidence;')[0] == base_shared.split('async fn run_physical_install_arm_observed_inner(')[0])
    check('shared file suffix outside seam unchanged', shared.split('\nasync fn run_arm(',1)[1] == base_shared.split('\nasync fn run_arm(',1)[1])
    old_run = function(old[SHARED], 'run_physical_install_arm_observed_inner')
    signature = old_run.split('    use crate::engine::', 1)[0]
    expected_wrapper = signature + '''    run_physical_install_arm_observed_outcome(
        prepared, args, run, diagnostic_mode, mapped_observer, pin_observer,
    ).await.historical()
}'''
    check('exact historical wrapper API and projection', compact(body(SHARED,'run_physical_install_arm_observed_inner')) == compact(expected_wrapper))
    # Enumerate every allowed production-runner edit. The complete residual body
    # must equal the frozen implementation, not just a list of plausible tokens.
    expected = compact(old_run)
    def replace(before, after):
        nonlocal expected
        a,b=compact(before),compact(after)
        if expected.count(a) != 1:
            raise ValueError('proof replacement not unique: ' + before)
        expected=expected.replace(a,b,1)
    replace('run_physical_install_arm_observed_inner(', 'run_physical_install_arm_observed_outcome(')
    replace('-> Result<PhysicalInstallArmRun, BenchmarkFailure>', '-> ObservedArmOutcome')
    replace('let runtime = crate::gpu_native_real_benchmark::construct_runtime(', 'let runtime = match crate::gpu_native_real_benchmark::construct_runtime(')
    replace('''&mut benchmark, ).await?;''','''&mut benchmark, ).await {
        Ok(runtime) => runtime,
        Err(primary) => { return ObservedArmOutcome::StartupFailed(Box::new(ObservedStartupFailure {
            primary, benchmark, construction_completed: false,
            qualification_enable_completed: false, runtime_validation_completed: false,
            shutdown: ObservedShutdown::NotAttempted,
        })) }
    };''')
    replace('let _ = crate::gpu_native_real_benchmark::shutdown_runtime(', 'let shutdown = crate::gpu_native_real_benchmark::shutdown_runtime(')
    replace('return Err(failure);','''return ObservedArmOutcome::StartupFailed(Box::new(ObservedStartupFailure {
        primary: failure, benchmark, construction_completed: true,
        qualification_enable_completed: false, runtime_validation_completed: false,
        shutdown: ObservedShutdown::from_result(shutdown),
    }));''')
    replace('''return match shutdown {
        Ok(()) => Err(validation_error),
        Err(shutdown_error) => Err(BenchmarkFailure::new(
            "postcondition", "runtime-validation-and-shutdown-failed",
            format!("{validation_error}; {shutdown_error}"),
        )),
    };''','''return ObservedArmOutcome::StartupFailed(Box::new(ObservedStartupFailure {
        primary: validation_error, benchmark, construction_completed: true,
        qualification_enable_completed: true, runtime_validation_completed: false,
        shutdown: ObservedShutdown::from_result(shutdown),
    }));''')
    replace('''let shutdown = crate::gpu_native_real_benchmark::shutdown_runtime(runtime, arm_name, None, &mut benchmark).await;
    if let Err(error) = shutdown {''','''let primary_failure = execution_failure.clone();
    let shutdown = crate::gpu_native_real_benchmark::shutdown_runtime(runtime, arm_name, None, &mut benchmark).await;
    let observed_shutdown = ObservedShutdown::from_result(shutdown.clone());
    if let Err(error) = shutdown {''')
    replace('Ok(PhysicalInstallArmRun {', 'ObservedArmOutcome::Run { primary_failure, shutdown: observed_shutdown, run: PhysicalInstallArmRun {')
    replace('warmup_concurrency, concurrency, }) }', 'warmup_concurrency, concurrency, }, } }')
    rich = body(SHARED, 'run_physical_install_arm_observed_outcome')
    check('entire runner equals frozen runner plus enumerated evidence-only edits', compact(rich) == expected)
    for token in ['construct_runtime(', 'shutdown_runtime(', 'validate_and_record_runtime(', 'Instant::now(', 'begin_request(', 'execute_request(']:
        check('no additional runner call ' + token, rich.count(token) == old_run.count(token))
    qualifier = current[NEW].decode()
    execute = body(NEW,'execute_arm')
    check('worker observer baseline constant', 'Observer::new(Mode::MappedBaseline)' in execute and 'Mode::MappedPinned' not in qualifier)
    check('one rich call from HMA-1G', qualifier.count('run_physical_install_arm_observed_outcome(') == 1)
    consumers = [p for p,data in current.items() if p.startswith(P) and p.endswith('.rs') and b'run_physical_install_arm_observed_outcome(' in data]
    check('rich runner consumers limited to historical projection and HMA-1G', sorted(consumers) == sorted([SHARED,NEW]))
    check('four fixed arm names, no fifth', 'const ORDER: [&str; 4] = ["baseline-0", "baseline-1", "baseline-2", "baseline-3"];' in qualifier)
    worker=body(NEW,'run_worker')
    check('sequential worker awaits each arm', 'for index in 0..ORDER.len()' in worker and 'execute_arm(&prepared, &args, index).await' in worker)
    check('per-arm authority checked before next runtime', 'validate_completed(&payload.arms)?;' in worker)
    check('worker always serializes completed prefix before returning result', worker.index('serde_json::to_vec(&payload)') < worker.index('println!("{END}') and worker.rstrip().endswith('result\n}'))
    work_expected = function(old[PIN], 'validate_work').replace('fn validate_work(', 'fn validate_completed(').replace('arms.len() == 4, "exactly four arms required"', '!arms.is_empty() && arms.len() <= 4, "one to four completed arms required",').replace('a.mode == ORDER[i]', 'a.mode == Mode::MappedBaseline').replace('"four-arm order mismatch"', '"baseline-only arm order mismatch"')
    work_expected = work_expected.replace('        validate_arm(a)?;', '        validate_benchmark_header(&a.production["benchmark"])?;\n        validate_arm(a)?;', 1)
    check('entire production work authority reused with only baseline prefix order/count changes', compact(body(NEW,'validate_completed')) == compact(work_expected))
    args_expected = function(old[PIN], 'worker_arguments').replace('HMA-1F-B','HMA-1G').replace('hma1fb-worker-internal','hma1g-worker-internal')
    check('worker process argument preservation unchanged', compact(body(NEW,'worker_arguments')) == compact(args_expected))
    for token in ['register_buffers','unregister_buffers','register_files','mlock','munlock','mlockall','mlock2','request_device','enumerate_adapters','read_at_with_retries','read_experts_batch_into_aligned_slices','construct_runtime','shutdown_runtime','submission','submit','submit_and_wait','sched_setaffinity','set_num_threads']:
        check('new module has no ' + token + ' call', not re.search(r'\b'+token+r'\s*\(', qualifier))
    check('no new io_uring implementation', 'io_uring::' not in qualifier)
    for name in ['audit_command','audit_bytes','validate_completed','validate_lifecycle','analyze','classify','validate_transcript']:
        code=body(NEW,name)
        check('CPU-only ' + name, not re.search(r'\b(?:prepare|execute_arm|construct_runtime|run_worker|launch|run_physical_install_arm_observed_outcome)\s*\(',code))
    audit=body(NEW,'audit_bytes')
    check('binding and work checks precede timing', audit.index('validate_transcript(') < audit.index('validate_completed(') < audit.index('analyze('))
    check('no performance on lifecycle branch', 'analyze(' not in audit.split('validate_lifecycle(',1)[1])
    launcher=body(NEW,'launch')
    check('platform before filesystem/worker', launcher.index('require_platform()?') < launcher.index('canonical_output(') < launcher.index('std::process::Command'))
    check('one transcript read and one payload read', launcher.count('std::fs::read(')==2)
    check('exact payload preserved', 'String::from_utf8(payload_bytes)?' in launcher and 'payload_sha256: sha(&payload_bytes)' in launcher)
    check('shared stdout stderr transcript description', '.stdout(transcript.try_clone()?)' in launcher and '.stderr(transcript)' in launcher)
    check('all report writes use frozen create-new helper', 'write_new(' in launcher and '.create_new(true)' in body(PIN,'write_new'))
    check('integer thresholds', 'self.signed_delta.abs() * 100' in qualifier and 'i128::from(self.left) * percent' in qualifier and 'i128::from(self.left) * 3' in qualifier)
    main=current[P+'main.rs'].decode(); before=old[P+'main.rs'].decode()
    restored=main
    for start,end in [
        ('    /// HMA-1G: four isolated baseline runtimes;', '    /// Standalone full-file O_DIRECT'),
        ('        Cmd::AuditGpuNativeBaselineRuntimeLifecycle {', '        Cmd::AuditGpuNativeSourceMappedPinProduction {'),
        ('        Cmd::Hma1gWorkerInternal { config, report_out } =>', '        Cmd::Hma1fbWorkerInternal { config, report_out } =>'),
    ]:
        if start not in restored or end not in restored:
            check('main expected HMA-1G seam ' + start,False);continue
        a=restored.index(start);b=restored.index(end,a);restored=restored[:a]+restored[b:]
    check('main only HMA-1G command/dispatch additions',restored==before)
    dispatch=main.split('fn main() ->',1)[1]
    check('audit/launcher/platform gate before logging and runtime', all(dispatch.index(s)<dispatch.index('init_logging(') for s in ['baseline_lifecycle::audit_command','baseline_lifecycle::launch','Cmd::Hma1gWorkerInternal { .. }']))
    return {'base':BASE,'base_tree':TREE,'child':child or 'WORKTREE','checks':checks,'pass':all(c['pass'] for c in checks), 'protected_file_count':len(protected),'retry':{'bytes':len(retry),'sha256':sha(retry)},'q4_pin':{'old':sha(old[SHARED]),'new':sha(current[SHARED])}}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--child');ap.add_argument('--out',required=True,type=Path);ap.add_argument('--mutations',action='store_true');args=ap.parse_args()
    old={p:git('show',f'{BASE}:{p}') for p in git('ls-tree','-r','--name-only',BASE).decode().splitlines()}
    if args.child:
        current={p:git('show',f'{args.child}:{p}') for p in git('ls-tree','-r','--name-only',args.child).decode().splitlines()}
    else:
        paths=set(git('ls-files').decode().splitlines())|set(git('ls-files','--others','--exclude-standard').decode().splitlines())
        current={p:(ROOT/p).read_bytes() for p in paths if (ROOT/p).is_file()}
    report=prove(old,current,args.child)
    mutations=[]
    if args.mutations:
        cases=[
            ('retry bytes',P+'io_provider.rs',b'read_at_with_retries',b'read_at_with_retries_changed'),
            ('upload timing',P+'gpu_native_source_upload.rs',b'let started = Instant::now();',b'let started = Instant::now(); extra_clock();'),
            ('backend enumeration',P+'backend/mod.rs',b'enumerate_adapters(',b'enumerate_adapters_changed('),
            ('extra construction',SHARED,b'let primary_failure = execution_failure.clone();',b'construct_runtime(); let primary_failure = execution_failure.clone();'),
            ('old API removed',SHARED,b'.historical()',b'.bypass_historical()'),
            ('pinned observer',NEW,b'Observer::new(Mode::MappedBaseline)',b'Observer::new(Mode::MappedPinned)'),
            ('registration',NEW,b'let before = ProcessMemory::capture()?;',b'register_buffers(); let before = ProcessMemory::capture()?;'),
            ('unregister',NEW,b'let before = ProcessMemory::capture()?;',b'unregister_buffers(); let before = ProcessMemory::capture()?;'),
            ('SQE',NEW,b'let before = ProcessMemory::capture()?;',b'ring.submit(); let before = ProcessMemory::capture()?;'),
            ('mlock',NEW,b'let before = ProcessMemory::capture()?;',b'mlock(); let before = ProcessMemory::capture()?;'),
            ('auditor hardware',NEW,b'let bytes = std::fs::read(raw)?;',b'construct_runtime(); let bytes = std::fs::read(raw)?;'),
            ('HMA-1F-B semantics',PIN,b'PIN_NO_MATERIAL_EFFECT',b'ALTERED_CLASSIFICATION'),
        ]
        for name,path,before,after in cases:
            altered=dict(current)
            if before not in altered[path]:raise ValueError('mutation absent '+name)
            altered[path]=altered[path].replace(before,after,1)
            try:
                result=prove(old,altered,args.child);rejected=not result['pass']
            except (ValueError,KeyError):rejected=True
            mutations.append({'mutation':name,'rejected':rejected})
        report['mutations']=mutations
        report['pass']=report['pass'] and all(m['rejected'] for m in mutations)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':report['pass'],'checks':len(report['checks']),'failed':[c['check'] for c in report['checks'] if not c['pass']],'mutations':len(mutations),'mutation_failures':[m['mutation'] for m in mutations if not m['rejected']],'retry':report['retry'],'q4_pin':report['q4_pin']},indent=2))
    return 0 if report['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
