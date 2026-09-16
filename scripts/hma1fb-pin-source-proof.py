#!/usr/bin/env python3
"""Offline HMA-1F-B source, ownership, timing, and zero-SQE proof; no hardware."""
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
BASE = '29d0ce6d2085a9e0dff4680b91e4aff93f8ace8b'
TREE = '69ff4937d7fe7c4541c07c505b0c5da7b5f4ab69'
ACCEPTED = '3069ceb9d46e072304180c18e2c29951491f31fb'
ACCEPTED_TREE = '01bf945d6427b44f752b5117485e4b38edef1df6'
PREFIX = 'rust-engine/src/'
EXISTING = {PREFIX+p for p in ['main.rs', 'engine.rs', 'gpu_native_source_upload.rs',
    'gpu_native_physical_install_staging.rs', 'gpu_native_source_to_upload_copy_elision_production.rs',
    'gpu_native_q4_route_parallel.rs']}
NEW = {PREFIX+'gpu_native_mapped_pin.rs', PREFIX+'gpu_native_source_mapped_pin_production.rs',
       'scripts/hma1fb-pin-source-proof.py'}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


spec = importlib.util.spec_from_file_location('frozen_hma1f_proof', ROOT/'scripts/hma1f-source-proof.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
def function(data, name):
    try:
        return legacy.function(data, name)
    except ValueError:
        # The protected extractor predates unsafe qualification-only methods.
        return legacy.function(data.replace(b'unsafe fn ', b'fn '), name)


def compact(s):
    # Used only for formatting-insensitive comparisons of known protected seams.
    return re.sub(r'\s+', '', s)


def prove(child=None):
    checks = []
    def check(name, value):
        checks.append({'check': name, 'pass': bool(value)})
    def read(path):
        return git('show', f'{child}:{path}') if child else (ROOT/path).read_bytes()
    def old(path):
        return git('show', f'{BASE}:{path}')
    def src(name):
        return read(PREFIX+name).decode()
    def body(name, fn):
        return function(read(PREFIX+name), fn).decode()
    check('exact base tree', git('rev-parse', BASE+'^{tree}').decode().strip() == TREE)
    check('accepted implementation remains exact child of frozen base', git('rev-list', '--parents', '-n', '1', ACCEPTED).decode().split() == [ACCEPTED, BASE])
    check('accepted implementation tree unchanged', git('rev-parse', ACCEPTED+'^{tree}').decode().strip() == ACCEPTED_TREE)
    if child:
        check('exact single repair child of accepted implementation', git('rev-list', '--parents', '-n', '1', child).decode().split() == [child, ACCEPTED])
    else:
        check('worktree HEAD remains accepted implementation before repair commit', git('rev-parse', 'HEAD').decode().strip() == ACCEPTED)
    repair_changed = set(git('diff', '--name-only', ACCEPTED, *([child] if child else [])).decode().splitlines())
    if not child:
        repair_changed.update(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    check('repair changes only source upload and this proof', repair_changed == {PREFIX+'gpu_native_source_upload.rs', 'scripts/hma1fb-pin-source-proof.py'})
    changed = set(git('diff', '--name-only', BASE, *([child] if child else [])).decode().splitlines())
    if not child:
        changed.update(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    check('exact nine-file amended boundary', changed == EXISTING | NEW)
    protected = []
    for path in git('ls-tree', '-r', '--name-only', BASE).decode().splitlines():
        if path not in EXISTING:
            a,b=old(path),read(path)
            protected.append({'path': path, 'base_sha256':sha(a), 'child_sha256':sha(b), 'equal':a==b})
    check('every other existing file byte-identical including prior proofs', all(p['equal'] for p in protected))
    for name in ['io_provider.rs','io_uring_storage.rs','gpu_native_mapped_lock.rs','gpu_native_mapped_vma.rs','gpu_native_mapped_fixed_buffer.rs','gpu_native_source_mapped_lock_production.rs']:
        check(name+' byte-identical', old(PREFIX+name)==read(PREFIX+name))
    io=read(PREFIX+'io_provider.rs')
    retry=function(io,'read_at_with_retries')
    retry_evidence={'bytes':len(retry),'sha256':sha(retry)}
    check('retry exact 6909 bytes and SHA',len(retry)==6909 and sha(retry)=='9f3241f79c6ebf02d4fa796b3e2bd3e2b2d45fc9740f2532872e5e3330b258c0')
    helper=function(io,'read_experts_batch_into_aligned_slices')
    check('whole helper and existing worker timing unchanged',helper==function(old(PREFIX+'io_provider.rs'),'read_experts_batch_into_aligned_slices'))
    for token,count in [(b'tokio::task::block_in_place',1),(b'std::thread::scope',1),(b'scope.spawn',1),(b'.join()',1),(b'std::time::Instant::now()',4)]:
        check('unchanged worker topology '+token.decode(),helper.count(token)==count)
    upload=src('gpu_native_source_upload.rs')
    source=body('gpu_native_source_upload.rs','read_source')
    pin=src('gpu_native_mapped_pin.rs').split('#[cfg(test)]')[0]
    qualifier=src('gpu_native_source_mapped_pin_production.rs').split('#[cfg(test)]')[0]
    for token in ['opcode::','squeue::','READ_FIXED','WRITE_FIXED','ReadFixed','WriteFixed','register_files','IoUringStorage','BufferPool']:
        check('new path excludes '+token,token not in pin+qualifier)
    for method in ['submission','submission_shared','submit','submit_and_wait','split']:
        check('no queue/I/O method '+method,not re.search(r'\.\s*'+method+r'\s*\(',pin+qualifier))
    check('no queue push',not re.search(r'\b(?:ring|sq|queue)\s*\.\s*push\s*\(',pin+qualifier))
    for method in ['mlock','munlock','mlockall','mlock2']:
        check('no '+method+' calls',not re.search(r'\b'+method+r'\s*\(',pin+qualifier))
    check('new path does not use old intervention guard','LockGuard' not in pin+qualifier)
    check('closed zero-SQE constant','sqes_submitted: 0' in pin and 'self.sqes_submitted != 0' in pin)
    check('exactly one identical ring API both modes',pin.count('io_uring::IoUring::new(2)')==1)
    check('exactly one system registration',pin.count('ring.submitter().register_buffers(&iovecs)')==1)
    check('exactly one system unregister',pin.count('ring.submitter().unregister_buffers()')==1)
    check('only registration submitter accesses',pin.count('ring.submitter()')==2)
    check('K exact iovecs from ranges',all(x in compact(pin) for x in ['letiovecs:Vec<libc::iovec>=ranges.iter().map(', 'iov_base:pointer.cast(),','iov_len:FULL,']))
    acquire=body('gpu_native_mapped_pin.rs','acquire')
    finish=body('gpu_native_mapped_pin.rs','finish')
    ordered=['validate_ranges(', 'R::create()', 'process.pre = Some(ring.status()?)', 'if guard.evidence.mode == Mode::MappedPinned', 'ring.register(', 'process.active = Some(ring.status()?)', 'validate_active()']
    positions=[acquire.find(s) for s in ordered]
    check('ranges then equal ring then pre/register/active/exact gate',min(positions)>=0 and positions==sorted(positions))
    check('baseline skips registration', 'if guard.evidence.mode == Mode::MappedPinned' in acquire and acquire.count('ring.register(')==1)
    check('baseline skips unregister', 'if self.registered' in finish and finish.count('ring.unregister()')==1)
    check('registered flag set only after successful registration',compact(acquire).find('ring.register(')<compact(acquire).find('guard.registered=true'))
    check('setup failures explicitly cleanup','let cleanup = guard.finish();' in acquire)
    positions=[finish.find(s) for s in ['ring.unregister()', 'ring.status()', 'drop(self.ring.take())', 'self.evidence.ring_dropped = true']]
    check('explicit unregister then after snapshot then ring drop',min(positions)>=0 and positions==sorted(positions))
    check('unregister failure cannot bypass ring drop', 'unregister.and(cleanup)' in finish and '.record(ring.unregister())?' not in finish)
    check('unwind fallback explicit cleanup','if !self.finished' in pin and 'let _ = self.finish();' in pin)
    check('one unchanged source helper call',source.count('.read_experts_batch_into_aligned_slices(')==1)
    pin_stop='let stopped = pin_observer.map(|_| Instant::now());'
    source_compact=compact(source)
    timed=source.partition('let started = Instant::now();')[2].partition(pin_stop)[0]
    expected='let result = storage.read_experts_batch_into_aligned_slices(ids, &mut destinations, raw.as_mut()).await;'
    check('only existing helper between start and stop',compact(timed)==compact(expected))
    check('no unconditional HMA-1F-B stop timestamp',not re.search(r'let\s+stopped\s*=\s*Instant\s*::\s*now\s*\(\s*\)\s*;', source))
    check('pin stop conditional only on pin observer',source_compact.count(compact(pin_stop))==1)
    check('conditional pin stop immediately after unchanged helper',compact(expected+pin_stop) in source_compact)
    ordered=['let mut leases','let mut views','let offsets','let mut destinations','PinGuard::<','let started = Instant::now();',pin_stop,'g.finish()','drop(pin_guard)','drop(destinations)','self.materialize_source_payload','drop(views)','lease.unmap()']
    positions=[{'step':s,'offset':source.find(s)} for s in ordered]
    check('mapped lifetime and timing boundaries',all(x['offset']>=0 for x in positions) and [x['offset'] for x in positions]==sorted(x['offset'] for x in positions))
    check('unregister immediately after stop before accounting',source_compact.partition(compact(pin_stop))[2].startswith('letpin_cleanup=pin_guard.as_mut().map(|g|g.finish()).transpose();'))
    accounting='''if let Some(stopped) = stopped {
        self.add(|m| &mut m.fused_source_us, stopped.duration_since(started).as_micros() as u64,);
    } else {
        self.add(|m| &mut m.fused_source_us, elapsed(started));
    }
    let lock_stopped = observer.map(|_| Instant::now());'''
    check('pin accounting uses captured helper interval and disabled path uses historical elapsed',compact(accounting) in source_compact)
    post_helper=source_compact.partition(compact(expected))[2].partition('letunlock=')[0]
    cleanup='let pin_cleanup = pin_guard.as_mut().map(|g| g.finish()).transpose(); drop(pin_guard);'
    check('post-helper sequence contains only conditional stop cleanup accounting and historical lock stop',post_helper==compact(pin_stop+cleanup+accounting))
    before_source=function(old(PREFIX+'gpu_native_source_upload.rs'),'read_source').decode()
    historical='self.add(|m| &mut m.fused_source_us, elapsed(started)); let stopped = observer.map(|_| Instant::now());'
    check('disabled accounting then lock stop equals frozen historical sequence',compact(historical) in compact(before_source) and compact('else { '+historical.replace('let stopped = observer', '} let lock_stopped = observer')) in source_compact)
    pin_record=source.partition('if let Some(o) = pin_observer {')[2].partition('let evidence = pin_evidence.unwrap();')[0]
    expected_record='''let stopped = stopped.expect("pin observer always captures the helper stop timestamp");
        let caller_ns = stopped.duration_since(started).as_nanos() as u64;
        let timing = raw.as_ref().unwrap().reconstruct(ids.len(), started, stopped);'''
    check('pin record caller and RawBatch use the same necessarily present conditional stop',compact(pin_record)==compact(expected_record))
    post_stop=source.partition(pin_stop)[2].partition('drop(destinations);')[0]
    check('no later synthesized pin timestamp',compact(post_stop).count('Instant::now()')==1 and 'letlock_stopped=observer.map(|_|Instant::now());' in compact(post_stop))
    check('read errors handled after explicit cleanup',source.find('g.finish()')<source.find('result.as_ref().err()'))
    check('shared original RawBatch timing','crate::gpu_native_mapped_lock::RawBatch::default' in source)
    check('pin observer absent in all constructors',upload.count('mapped_pin_observer: std::sync::OnceLock::new()')==2)
    check('lock observer absent in all constructors',upload.count('mapped_lock_observer: std::sync::OnceLock::new()')==2)
    for kind,other in [('pin','lock'),('lock','pin')]:
        install=body('gpu_native_source_upload.rs',f'enable_mapped_{kind}_observer')
        check(kind+' observer installation serialized','qualification_observer_install.lock()' in install)
        check(kind+' excludes '+other, f'mapped_{other}_observer.get().is_some()' in install)
        check(kind+' requires idle production mapped treatment', all(s in install for s in ['!self.production_owned','self.arm != Arm::Treatment','self.active_leases() != 0','!self.pending.lock().is_empty()']))
        check(kind+' rejects duplicate',f'mapped_{kind}_observer.set(observer)' in compact(install))
    for name in ['new_production','acquire','materialize_source_payload','take_lease','reset']:
        check('unchanged upload '+name,function(old(PREFIX+'gpu_native_source_upload.rs'),name)==function(read(PREFIX+'gpu_native_source_upload.rs'),name))
    before_source=function(old(PREFIX+'gpu_native_source_upload.rs'),'read_source').decode()
    check('entire materialization/unmap/copy continuation byte-identical',before_source.split('        drop(destinations);',1)[1]==source.split('        drop(destinations);',1)[1])
    check('checked K*FULL arithmetic','checked_mul(FULL as u64)' in pin)
    check('checked VmPin addition','checked_add(pin_bytes(self.ranges.len())?)' in pin)
    check('exact active pin and lock equality',all(s in pin for s in ['active.vmpin_bytes != expected','active.vmlck_bytes != pre.vmlck_bytes']))
    check('baseline uses pre, not absolute zero','else {\n            pre.vmpin_bytes\n        }' in pin)
    check('strict status parser','rows.len() != 1' in pin and 'checked_mul(1024)' in pin)
    check('platform feature gate','cfg!(all(target_os = "linux", feature = "io_uring"))' in pin)
    main=src('main.rs').split('fn main() ->',1)[1]
    check('public launcher and worker gate before logging',all(main.index(s)<main.index('init_logging(') for s in ['mapped_pin::launch','Cmd::Hma1fbWorkerInternal { .. } => crate::gpu_native_mapped_pin::require_platform()?']))
    check('launcher rejects before outputs/model/runtime',body('gpu_native_source_mapped_pin_production.rs','launch').index('require_platform()?')<body('gpu_native_source_mapped_pin_production.rs','launch').index('OpenOptions'))
    check('worker rejects before preparation',body('gpu_native_source_mapped_pin_production.rs','run_worker').index('require_platform()?')<body('gpu_native_source_mapped_pin_production.rs','run_worker').index('prepare(&args)'))
    check('exact schema','mer.gpu-native-source-mapped-pin-production.v1' in qualifier)
    check('exact four-arm order','Mode::MappedBaseline,Mode::MappedPinned,Mode::MappedPinned,Mode::MappedBaseline,' in compact(pin))
    check('authority before analysis/classification',qualifier.index('validate_work(&payload.arms)?;')<qualifier.index('audit.classification = classify(&analysis);'))
    # Reuse all existing production authority equations without changing their meaning.
    old_qual=old(PREFIX+'gpu_native_source_mapped_lock_production.rs')
    for name in ['context_id','contracts_equal','mapped_ring_exact','validate_request_counters','validate_work']:
        check('unchanged production authority '+name,function(old_qual,name)==function(read(PREFIX+'gpu_native_source_mapped_pin_production.rs'),name))
    runner=src('gpu_native_physical_install_staging.rs')
    check('existing runner callers pass no pin observer','run_physical_install_arm_observed_inner(prepared,args,run,diagnostic_mode,mapped_observer,None,).await' in compact(runner))
    old_runner=function(old(PREFIX+'gpu_native_physical_install_staging.rs'),'run_physical_install_arm_observed').decode()
    new_runner=body('gpu_native_physical_install_staging.rs','run_physical_install_arm_observed_inner')
    new_body=new_runner[new_runner.index('    use crate::engine::'):]
    added='''    let enable_result = enable_result.and_then(|()| match &pin_observer {
        Some(observer) => runtime.engine.enable_mapped_pin_observer(observer.clone()),
        None => Ok(()),
    });
'''
    new_body=new_body.replace(added,'')
    for measured in ['false','true']:
        new_body=new_body.replace(f'            if let Some(observer) = &pin_observer {{ observer.begin_request({measured}, index); }}\n','')
    check('runner body only pin handoff and request boundaries',new_body==old_runner[old_runner.index('    use crate::engine::'):])
    module=PREFIX+'gpu_native_source_to_upload_copy_elision_production.rs'
    expected=old(module).replace(b'pub(crate) mod mapped_lock;',b'pub(crate) mod mapped_lock;\n\n#[path = "gpu_native_source_mapped_pin_production.rs"]\npub(crate) mod mapped_pin;',1)
    check('source qualifier parent only module declaration',read(module)==expected)
    engine=read(PREFIX+'engine.rs');new_method=function(engine,'enable_mapped_pin_observer')
    old_method=function(old(PREFIX+'engine.rs'),'enable_mapped_lock_observer')
    expected_method=old_method.replace(b'enable_mapped_lock_observer',b'enable_mapped_pin_observer').replace(b'gpu_native_mapped_lock::Observer',b'gpu_native_mapped_pin::Observer').replace(b'HMA-1F observer',b'HMA-1F-B observer')
    check('engine pin handoff analogous to existing handoff',compact(new_method.decode()).replace(",)", ")")==compact(expected_method.decode()).replace(",)", ")"))
    comment=b'    /// HMA-1F-B only: called after enabling source/upload qualification on an idle isolated runtime.\n'
    check('engine otherwise byte-identical',engine.replace(comment+new_method+b'\n\n',b'',1)==old(PREFIX+'engine.rs'))
    q4=PREFIX+'gpu_native_q4_route_parallel.rs';expected=old(q4)
    for path in [PREFIX+'engine.rs',PREFIX+'gpu_native_physical_install_staging.rs']:
        expected=expected.replace(sha(old(path)).encode(),sha(read(path)).encode(),1)
    check('PR2-C exactly two allowed SHA literal replacements',read(q4)==expected)
    return {'base':BASE,'base_tree':TREE,'child':child or 'WORKTREE','checks':checks,'protected_files':protected,
            'retry':retry_evidence,'source_lifetime_positions':positions,'pass':all(c['pass'] for c in checks)}


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--child')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); report=prove(args.child)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':report['pass'],'checks':len(report['checks']),'failed':[c['check'] for c in report['checks'] if not c['pass']],'retry':report['retry']},indent=2))
    raise SystemExit(0 if report['pass'] else 1)
