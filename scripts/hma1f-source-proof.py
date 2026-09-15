#!/usr/bin/env python3
"""Offline exact-base source proof. No GPU, storage runtime, model or benchmark."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
BASE='1d41abdcc841fea740a59a0c4dab482730f388ca'
ROOT=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def sha(b):return hashlib.sha256(b).hexdigest()
def function(source,name):
    text=source.decode();m=re.search(r'(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn '+re.escape(name)+r'\b',text)
    if not m:raise ValueError('missing function '+name)
    opening=text.index('{',m.end());i=opening;depth=0
    while i<len(text):
        if text.startswith('//',i):
            end=text.find('\n',i);i=len(text) if end<0 else end;continue
        if text.startswith('/*',i):
            j=i+2;n=1
            while n:
                if text.startswith('/*',j):n+=1;j+=2
                elif text.startswith('*/',j):n-=1;j+=2
                else:j+=1
            i=j;continue
        raw=re.match(r'r(#+)?"',text[i:])
        if raw:
            delim='"'+(raw.group(1) or '');i=text.index(delim,i+raw.end())+len(delim);continue
        if text[i]=='"':
            i+=1
            while text[i]!='"':i+=2 if text[i]=='\\' else 1
            i+=1;continue
        char=re.match(r"'(?:\\.|[^'\n])'",text[i:])
        if char:i+=char.end();continue
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[m.start():i+1].encode()
        i+=1
    raise ValueError('unterminated '+name)
def permitted_helper(base):
    s=base.decode()
    s=s.replace('        destinations: &mut [&mut [u8]],\n','        destinations: &mut [&mut [u8]],\n        observation: Option<&mut crate::gpu_native_mapped_lock::RawBatch>,\n')
    s=s.replace('        if self.is_packed()','        if observation.as_ref().is_some_and(|_| ids.len() > crate::gpu_native_mapped_lock::WIDTH)\n            || self.is_packed()',1)
    s=s.replace('        tokio::task::block_in_place(|| -> io::Result<usize> {','''        // Exclusive slots are prepared before donation; None allocates no storage.
        let mut timing_slots = observation.into_iter().flat_map(|o| o.reads.iter_mut())
            .map(Some).chain(std::iter::repeat_with(|| None));
        tokio::task::block_in_place(|| -> io::Result<usize> {''')
    s=s.replace('                return self.read_at_with_retries(&files[0], id_vec[0], 0, destinations[0]);','''                let timing = timing_slots.next().flatten();
                let start = timing.as_ref().map(|_| std::time::Instant::now());
                let result = self.read_at_with_retries(&files[0], id_vec[0], 0, destinations[0]);
                let end = timing.as_ref().map(|_| std::time::Instant::now());
                if let Some(slot) = timing {
                    *slot = crate::gpu_native_mapped_lock::RawRead { start, end, success: result.is_ok() };
                }
                return result;''')
    s=s.replace('.map(|((file, dst), &id)| {','.zip(timing_slots)\n                    .map(|(((file, dst), &id), timing)| {')
    s=s.replace('scope.spawn(move || self.read_at_with_retries(file, id, 0, dst))','''scope.spawn(move || {
                            let start = timing.as_ref().map(|_| std::time::Instant::now());
                            let result = self.read_at_with_retries(file, id, 0, dst);
                            let end = timing.as_ref().map(|_| std::time::Instant::now());
                            if let Some(slot) = timing {
                                *slot = crate::gpu_native_mapped_lock::RawRead { start, end, success: result.is_ok() };
                            }
                            result
                        })''')
    return s.encode()
def prove(child=None):
    file='rust-engine/src/io_provider.rs';old=git('show',f'{BASE}:{file}');new=git('show',f'{child}:{file}') if child else (ROOT/file).read_bytes()
    checks=[];slices=[]
    def check(name,passed):checks.append({'check':name,'pass':bool(passed)})
    for name in ['read_at_with_retries','is_transient_io_error','note_read_success','note_read_failure','try_admit_probe','is_drive_unavailable','is_expert_unavailable','fd_for','prove_source_upload_fd','prove_source_upload_fd_with','read_expert','read_experts_batch']:
        a=function(old,name);b=function(new,name)
        slices.append({'function':name,'base_sha256':sha(a),'child_sha256':sha(b),'base_bytes':len(a),'child_bytes':len(b),'equal':a==b})
        check(name+' byte-identical',a==b)
    for name in ['STORAGE_RETRY_ATTEMPTS','STORAGE_RETRY_BACKOFF','STORAGE_RETRY_MAX_BACKOFF','STORAGE_BREAKER_THRESHOLD','STORAGE_BREAKER_PROBE_INTERVAL']:
        pat=rb'(?m)^pub const '+name.encode()+rb'\b[^;]*;';a=re.search(pat,old);b=re.search(pat,new);check(name+' unchanged',a and b and a[0]==b[0])
    helper=function(new,'read_experts_batch_into_aligned_slices');expected=permitted_helper(function(old,'read_experts_batch_into_aligned_slices'))
    check('entire mapped helper equals base plus only enumerated observation exception',helper==expected)
    # All production bytes outside the helper must differ only by the fd snapshot Deserialize derive.
    old_production=old.split(b'#[cfg(test)]\nmod tests')[0]
    new_production=new.split(b'#[cfg(test)]\nmod tests')[0]
    old_production=old_production.replace(function(old,'read_experts_batch_into_aligned_slices'),b'HELPER')
    new_production=new_production.replace(helper,b'HELPER')
    new_production=new_production.replace(b'#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]',b'#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]',1)
    check('all other io_provider production bytes unchanged except snapshot Deserialize',old_production==new_production)
    h=helper.decode();worker=h.split('scope.spawn(move || {',1)[1].split('.collect();',1)[0]
    for token,n in [('tokio::task::block_in_place',1),('std::thread::scope',1),('scope.spawn',1),('.join()',1),('std::time::Instant::now()',4)]:check(token+' count '+str(n),h.count(token)==n)
    for token in ['Mutex','RwLock','channel','Vec','HashMap','tracing','mlock','munlock','/proc','Atomic','attempt_starts']:
        check('worker excludes '+token,token not in worker)
    check('no HMA-1E RawRead dependency in io_provider',b'gpu_native_source_order_straggler_production' not in new_production)
    def read(path):return git('show',f'{child}:{path}') if child else (ROOT/path).read_bytes()
    upload=read('rust-engine/src/gpu_native_source_upload.rs')
    check('new_production constructor source-identical',function(git('show',f'{BASE}:rust-engine/src/gpu_native_source_upload.rs'),'new_production')==function(upload,'new_production'))
    check('observer default is empty OnceLock',b'mapped_lock_observer: std::sync::OnceLock::new()' in upload)
    for name in ['acquire','materialize_source_payload','take_lease']:
        check('upload '+name+' source-identical',function(git('show',f'{BASE}:rust-engine/src/gpu_native_source_upload.rs'),name)==function(upload,name))
    q4_path='rust-engine/src/gpu_native_q4_route_parallel.rs'
    q4_base=git('show',f'{BASE}:{q4_path}');q4_child=read(q4_path);q4_expected=q4_base
    for path in ['rust-engine/src/engine.rs','rust-engine/src/gpu_native_physical_install_staging.rs']:
        q4_expected=q4_expected.replace(sha(git('show',f'{BASE}:{path}')).encode(),sha(read(path)).encode())
    check('q4 route-parallel source differs only in two reviewed whole-file test hash pins',q4_child==q4_expected)
    check('q4 route-parallel production code byte-identical',q4_base.split(b'#[cfg(test)]')[0]==q4_child.split(b'#[cfg(test)]')[0])
    locks=read('rust-engine/src/gpu_native_mapped_lock.rs').split(b'#[cfg(test)]\nmod hma1f_tests')[0]
    source=function(upload,'read_source')
    positions=[source.find(x) for x in [b'let mut views',b'let mut destinations',b'LockGuard::acquire',b'let started = Instant::now()',b'.read_experts_batch_into_aligned_slices',b'let stopped',b'g.release()',b'drop(guard)',b'drop(destinations)',b'self.materialize_source_payload',b'drop(views)',b'lease.unmap()']]
    check('lock/source-timer/unlock/materialize/view-drop/unmap order',all(p>=0 for p in positions) and positions==sorted(positions))
    for forbidden in ['mlockall','mlock2','madvise','mbind','move_pages','set_mempolicy','sched_setaffinity','io_uring','register_buffers']:
        check('intervention excludes '+forbidden,not re.search(rb'\b'+forbidden.encode()+rb'\s*\(',locks+source+helper))
    diff=git('diff','--no-ext-diff','--unified=3',BASE,*([child] if child else []),'--',file).decode()
    hunks=[]
    for chunk in re.split(r'(?m)(?=^@@ )',diff)[1:]:
        production='read_experts_batch_into_aligned_slices' in chunk or 'timing_slots' in chunk or 'scope.spawn' in chunk or 'observation:' in chunk
        reason='Optional preallocated wrapper timing only; original fd resolution/proof, retry calls, spawn/join topology, error order and byte accumulation preserved.' if production else 'Deserialize for offline fd-proof reconstruction, or portable test/call-site plumbing; no runtime retry/breaker behavior change.'
        hunks.append({'hunk':chunk,'explanation':reason})
    return {'base':BASE,'child':child or 'WORKTREE','checks':checks,'function_slices':slices,'io_provider_hunks':hunks,'pass':all(c['pass'] for c in checks)}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--child');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();report=prove(args.child);args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'pass':report['pass'],'checks':len(report['checks']),'failed':[c['check'] for c in report['checks'] if not c['pass']],'read_at_with_retries':report['function_slices'][0]},indent=2));raise SystemExit(0 if report['pass'] else 1)
