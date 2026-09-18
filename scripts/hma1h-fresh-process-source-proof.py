#!/usr/bin/env python3
"""Offline HMA-1H exact-scope, protected-source and mutation proof. No hardware."""
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
BASE = 'f77dc64cdcaa79cf24773daa45a6ccb141b1b822'
TREE = '3d55a81f4a5a574d7bd3540b87c0eecc50eb0f3b'
BRANCH = 'diag/hma1h-fresh-process-lifecycle'
P = 'rust-engine/src/'
G = P + 'gpu_native_baseline_runtime_lifecycle.rs'
H = P + 'gpu_native_fresh_process_lifecycle.rs'
MAIN = P + 'main.rs'
TEST = P + 'gpu_native_fresh_process_lifecycle_tests.rs'
SELF = 'scripts/hma1h-fresh-process-source-proof.py'
DECL = '#[path = "gpu_native_fresh_process_lifecycle.rs"]\npub(crate) mod fresh_process_lifecycle;\n\n'
ALLOWED = {G, H, MAIN, TEST, SELF}
# Reviewed implementation pin complements structural checks and rejects unlisted edits.
IMPLEMENTATION_SHA256 = 'a05adeded4afdc91973cce3742da2d38a72ac569ed25677191397819faac1fdc'
def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])
def sha(data):
    return hashlib.sha256(data).hexdigest()
def compact(text):
    return re.sub(r'\s+', '', text)

def parse_child_sha(value):
    if re.fullmatch(r'[0-9a-f]{40}', value) is None:
        raise ValueError('child must be canonical lowercase 40-hex')
    return value

def validate_provenance(child, head, base_tree, branch, parents, status):
    """Pure mode/lineage validation; status is Git porcelain bytes, including untracked."""
    checks=[]
    def check(name, ok):
        checks.append({'check':name,'pass':bool(ok)})
    if child is None:
        check('exact frozen base HEAD',head==BASE)
    else:
        try:
            parse_child_sha(child)
            canonical=True
        except ValueError:
            canonical=False
        check('canonical lowercase 40-hex child',canonical)
        check('exact requested child HEAD',canonical and head==child)
        check('child has exactly one parent, the frozen base',canonical and parents==[BASE])
        check('committed working tree completely clean',status==b'')
    check('exact frozen base tree',base_tree==TREE)
    check('exact local HMA-1H branch',branch==BRANCH)
    passed=all(c['pass'] for c in checks)
    return {
        'mode':'precommit-working-tree' if child is None else 'committed-child',
        'base_sha':BASE,'base_tree':base_tree,'observed_head_sha':head,
        'requested_child_sha':child,'validated_child_sha':child if child is not None and passed else None,
        'child_parent_sha':parents[0] if child is not None and parents and len(parents)==1 else None,
        'branch':branch,'working_tree_clean':status==b'',
        'implementation_sha256':None,'retry_sha256':None,
        'checks':checks,'mutations':[],'pass':passed,
    }

def provenance(child):
    parents=None
    if child is not None:
        try:
            parse_child_sha(child)
            # Read all parents, so a merge with BASE as first parent cannot pass.
            lineage=git('rev-list','--parents','-n','1',child,'--').decode().split()
            if lineage and lineage[0]==child:
                parents=lineage[1:]
        except (ValueError,subprocess.CalledProcessError):
            pass
    return validate_provenance(
        child,git('rev-parse','HEAD').decode().strip(),
        git('rev-parse',BASE+'^{tree}').decode().strip(),
        git('branch','--show-current').decode().strip(),parents,
        git('status','--porcelain=v1','--untracked-files=all','-z'))

def committed_files(revision):
    paths=git('ls-tree','-r','--name-only','-z',revision).decode().split('\0')
    return {p:git('show',f'{revision}:{p}') for p in paths if p}

def working_files():
    paths=set(git('ls-files','-z').decode().split('\0'))|set(git('ls-files','--others','--exclude-standard','-z').decode().split('\0'))
    return {p:(ROOT/p).read_bytes() for p in paths if p and (ROOT/p).is_file()}

def self_test():
    """Deterministic helper coverage: no Git commands, commits, or source writes."""
    child='0123456789abcdef0123456789abcdef01234567'
    valid=dict(child=child,head=child,base_tree=TREE,branch=BRANCH,parents=[BASE],status=b'')
    tests=[]
    def check(name, ok):
        tests.append({'test':name,'pass':bool(ok)})
    check('canonical valid 40-hex parsing',parse_child_sha(child)==child)
    for name,value in [
        ('non-hex','g'*40),('short',child[:-1]),('long',child+'0'),
        ('uppercase',child.upper()),('leading whitespace',' '+child),
        ('trailing newline',child+'\n'),('empty',''),
    ]:
        try:
            parse_child_sha(value)
            rejected=False
        except ValueError:
            rejected=True
        check(name+' child rejected',rejected and not validate_provenance(**{**valid,'child':value})['pass'])
    accepted=validate_provenance(**valid)
    check('clean direct child accepted',accepted['pass'] and accepted['mode']=='committed-child'
          and accepted['validated_child_sha']==child and accepted['child_parent_sha']==BASE)
    for name,changes,failed_check in [
        ('wrong child',{'head':'f'*40},'exact requested child HEAD'),
        ('wrong parent',{'parents':['e'*40]},'child has exactly one parent, the frozen base'),
        ('merge child',{'parents':[BASE,'e'*40]},'child has exactly one parent, the frozen base'),
        ('root child',{'parents':[]},'child has exactly one parent, the frozen base'),
        ('unresolved child',{'parents':None},'child has exactly one parent, the frozen base'),
        ('dirty tracked state',{'status':b' M tracked\0'},'committed working tree completely clean'),
        ('staged state',{'status':b'M  tracked\0'},'committed working tree completely clean'),
        ('untracked state',{'status':b'?? extra\0'},'committed working tree completely clean'),
        ('wrong branch',{'branch':'other'},'exact local HMA-1H branch'),
        ('detached HEAD',{'branch':''},'exact local HMA-1H branch'),
        ('wrong base tree',{'base_tree':'d'*40},'exact frozen base tree'),
    ]:
        result=validate_provenance(**{**valid,**changes})
        check(name+' rejected',not result['pass'] and result['validated_child_sha'] is None
              and any(c['check']==failed_check and not c['pass'] for c in result['checks']))
    precommit={**valid,'child':None,'head':BASE,'parents':None,'status':b' M tracked\0?? new\0'}
    result=validate_provenance(**precommit)
    check('precommit working tree accepted',result['pass'] and result['mode']=='precommit-working-tree'
          and not result['working_tree_clean'] and result['requested_child_sha'] is None
          and result['validated_child_sha'] is None and result['child_parent_sha'] is None)
    check('precommit wrong HEAD rejected',not validate_provenance(**{**precommit,'head':child})['pass'])
    check('report fields present',set(accepted)=={
        'mode','base_sha','base_tree','observed_head_sha','requested_child_sha','validated_child_sha',
        'child_parent_sha','branch','working_tree_clean','implementation_sha256','retry_sha256','checks','mutations','pass'})
    return {'pass':all(t['pass'] for t in tests),'tests':tests}

spec = importlib.util.spec_from_file_location('hma1f_extractor', ROOT/'scripts/hma1f-source-proof.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
def function(data, name):
    return legacy.function(data, name).decode()
def without_main_wiring(text):
    for start, end in [
        ('    /// HMA-1H: four fresh child processes;', '    /// HMA-1G: four isolated baseline runtimes;'),
        ('        Cmd::AuditGpuNativeFreshProcessLifecycle {', '        Cmd::AuditGpuNativeBaselineRuntimeLifecycle { raw_report'),
        ('        Cmd::Hma1hWorkerInternal {\n', '        Cmd::Hma1gWorkerInternal { config, report_out } => {'),
    ]:
        a = text.index(start); b = text.index(end, a)
        text = text[:a] + text[b:]
    return text

def prove(old, current, context):
    checks=list(context['checks'])
    def check(name, ok):
        checks.append({'check':name,'pass':bool(ok)})
    h=current[H].decode(); g=current[G].decode(); main=current[MAIN].decode()
    def body(name):return function(current[H],name)
    changed={p for p in set(old)|set(current) if old.get(p)!=current.get(p)}
    check('exact five-file scope',changed==ALLOWED)
    check('all other historical files byte-identical',all(current.get(p)==data for p,data in old.items() if p not in {G,MAIN}))
    check('HMA-1G only authorized child declaration',g.replace(DECL,'',1).encode()==old[G] and g.count(DECL)==1)
    check('main only three CLI/dispatch insertions',without_main_wiring(main).encode()==old[MAIN])
    check('reviewed complete HMA-1H implementation pinned',sha(current[H])==IMPLEMENTATION_SHA256)
    retry=legacy.function(current[P+'io_provider.rs'],'read_at_with_retries')
    check('protected retry identity',len(retry)==6909 and sha(retry)=='9f3241f79c6ebf02d4fa796b3e2bd3e2b2d45fc9740f2532872e5e3330b258c0')
    for p in ['gpu_native_physical_install_staging.rs','gpu_native_source_to_upload_copy_elision_production.rs','gpu_native_source_mapped_pin_production.rs','gpu_native_mapped_pin.rs','gpu_native_source_upload.rs','backend/mod.rs','io_provider.rs','io_uring_storage.rs','engine.rs']:
        check('protected '+p,current[P+p]==old[P+p])
    check('exact inherited four arm names','const ORDER: [&str; 4] = ["baseline-0", "baseline-1", "baseline-2", "baseline-3"];' in g)
    check('single inherited wrapper call',h.count('super::execute_arm(')==1)
    worker=body('run_worker');launch=body('launch_unix');collect=body('collect_children');audit=body('audit_bytes');transcript=body('validate_transcript')
    check('worker index bounded before preparation',worker.index('index < ORDER.len()')<worker.index('prepare(&args)'))
    check('worker has no arm loop','for ' not in worker and 'while ' not in worker)
    check('worker handshake before preparation',worker.index('read_to_string(')<worker.index('payload.process.release(index)')<worker.index('prepare(&args)')<worker.index('super::execute_arm('))
    check('worker retains arm before validating',worker.index('payload.arm = Some(arm)')<worker.index('validate_arm(retained)'))
    check('worker writes payload before hash marker',worker.index('write_new(')<worker.index('HMA1H_WORKER_PAYLOAD'))
    check('one spawn site',h.count('.spawn()?')==1)
    check('four sequential children','for index in 0..ORDER.len()' in launch and 'child.wait()?' in launch)
    check('release captured identity only after parent marker',launch.index('ProcessIdentity::capture(child.id())')<launch.index('binding.identity()')<launch.index('input.write_all(')<launch.index('child.wait()?'))
    check('wait then exit then payload then prefix check',launch.index('child.wait()?')<launch.index('binding.exit = Some(')<launch.index('std::fs::read(&drafts[index])')<launch.index('collect_children(&envelope)'))
    check('shared sequential stdout/stderr descriptor','.stdout(transcript.try_clone()?)' in launch and '.stderr(transcript.try_clone()?)' in launch)
    check('fresh outputs reserved before launch',launch.count('.create_new(true)')==2 and launch.index('.create_new(true)')<launch.index('.spawn()?'))
    check('exact six output paths', 'len() == 6' in launch and 'canonical_output' in launch)
    check('payload exact bytes including invalid UTF8','encode_hex(&bytes)' in launch and 'sha(&bytes)' in launch)
    check('transcript exactly one final read',launch.count('std::fs::read(transcript_out)')==1)
    check('Linux stat and boot sources','/proc/{pid}/stat' in h and '/proc/sys/kernel/random/boot_id' in h)
    check('stat field22 after full comm','rsplit_once(") ")' in h and 'start_ticks: fields[19].parse()?' in h)
    for token in ['id.ppid == e.parent.pid','id.boot_id == e.parent.boot_id','id.pid != e.parent.pid','c.spawned_pid == Some(id.pid)','identities.insert((id.boot_id.clone(), id.pid, id.start_ticks))','id.start_ticks >= e.parent.start_ticks','c.launch_ns < exit_ns','n < c.launch_ns','&p.process == id','c.exit == Some(ChildExit::expected(0))','c.exit == Some(ChildExit::expected(1))','e.children.len() <= 4','failure.is_none()']:
        check('process/status authority '+token,token in collect)
    check('exact wait status captured',all(x in h for x in ['status.into_raw()','status.code()','status.signal()','status.core_dumped()']))
    check('strict payload wire shape','serde_json::to_value(&p)? == serde_json::from_slice::<Value>(&bytes)?' in collect)
    check('parent executable tied to children','== e.executable_sha256' in collect)
    for token in ['sha(bytes) == e.transcript_sha256','bytes.len() == e.transcript_bytes','pattern_counts(bytes)','left == previous','arm_left < arm_right','arm_right < right','markers == expected.iter()','text.ends_with','text.matches("HMA1G_ARM_BEGIN").count() == e.children.len()']:
        check('transcript authority '+token,token in transcript)
    expected=function(old[G],'validate_completed').replace('    let mut ids = BTreeSet::new();\n','').replace('''        require(
            ids.insert(context_id(&a.production["benchmark"])?),
            "runtime context reused across arms",
        )?;
''','')
    check('exact inherited work checks except process-local context uniqueness',compact(body('validate_completed'))==compact(expected))
    check('unchanged baseline wrapper uses baseline observer','Observer::new(Mode::MappedBaseline)' in function(old[G],'execute_arm'))
    check('binding then transcript then work before timing',audit.index('collect_children(')<audit.index('validate_transcript(')<audit.index('validate_completed(')<audit.index('analyze('))
    check('lifecycle retains exact inherited recognition','super::validate_lifecycle(&inherited, &audit.transcript_diagnostics)?' in audit and 'inherited.arms = arms;' in audit and 'inherited.failure = Some(failure);' in audit)
    check('four complete children before performance','arms.len() == 4 && e.children.len() == 4' in audit)
    expected_class='''fn classify(analysis: &Analysis) -> &'static str {
        match super::classify(analysis) {
            "RUNTIME_LARGE_DRIFT" => "FRESH_PROCESS_LARGE_DRIFT",
            "RUNTIME_MATERIAL_DRIFT" => "FRESH_PROCESS_MATERIAL_DRIFT",
            "RUNTIME_STABLE" => "FRESH_PROCESS_STABLE",
            _ => "AMBIGUOUS",
        }
    }'''
    check('classification only renames inherited exact thresholds',compact(body('classify'))==compact(expected_class))
    check('no new timing equations','fn analyze(' not in h and 'fn totals(' not in h and 'fn validate_lifecycle(' not in h)
    for forbidden in ['register_buffers','unregister_buffers','mlock','munlock','submit','submit_and_wait','enumerate_adapters','request_device','construct_runtime','shutdown_runtime','run_physical_install_arm_observed_outcome','read_at_with_retries','read_experts_batch_into_aligned_slices','sleep','set_num_threads','sched_setaffinity']:
        check('no new '+forbidden+' call',not re.search(r'\b'+forbidden+r'\s*\(',h))
    check('no hardware helper commands',not any(s in h for s in ['nvidia-smi','vulkaninfo','io_uring::']))
    for role in ['launch','launch_unix','audit_command','audit_bytes','collect_children','validate_transcript','validate_completed','classify']:
        code=body(role)
        check('CPU-only '+role,not re.search(r'\b(?:prepare|execute_arm|run_worker|construct_runtime|run_physical_install_arm_observed_outcome)\s*\(',code))
    cmd=body('audit_command')
    check('auditor exact two read-once snapshots',cmd.count('std::fs::read(')==2)
    dispatch=main.split('fn main() ->',1)[1]
    check('parent and auditor dispatch before init_logging',dispatch.index('Cmd::AuditGpuNativeFreshProcessLifecycle')<dispatch.index('init_logging(') and dispatch.index('Cmd::QualifyGpuNativeFreshProcessLifecycle')<dispatch.index('init_logging('))
    check('CLI bounds arm index','value_parser = clap::value_parser!(u8).range(0..=3)' in main and '#[command(name = "hma1h-worker-internal", hide = true)]' in main)
    return {**context,'pass':all(c['pass'] for c in checks),'checks':checks,'protected_historical_files':len(old)-2,'implementation_sha256':sha(current[H]),'retry_sha256':sha(retry)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--mutations',action='store_true')
    parser.add_argument('--child',help='canonical lowercase 40-hex committed HMA-1H child; omit for working-tree mode')
    parser.add_argument('--self-test',action='store_true',help='run deterministic offline provenance helper tests')
    args=parser.parse_args()
    if args.self_test:
        if args.child is not None or args.mutations:
            parser.error('--self-test cannot be combined with --child or --mutations')
        report=self_test()
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'pass':report['pass'],'tests':len(report['tests']),
            'failed':[t['test'] for t in report['tests'] if not t['pass']]},indent=2))
        return 0 if report['pass'] else 1
    context=provenance(args.child)
    report=context
    if context['pass']:
        old=committed_files(BASE)
        current=committed_files(args.child) if args.child is not None else working_files()
        if args.child is not None:
            # Include mode-only tree changes in the committed scope gate too.
            changed=set(git('diff','--no-renames','--name-only','-z',BASE,args.child,'--').decode().split('\0'))-{''}
            context['checks'].append({'check':'exact committed five-file tree scope','pass':changed==ALLOWED})
        report=prove(old,current,context)
    mutations=[]
    if args.mutations and context['pass']:
        cases=[
            ('fifth arm',G,b'const ORDER: [&str; 4]',b'const ORDER: [&str; 5]'),
            ('HMA-1G threshold',G,b'p.drift(10)',b'p.drift(11)'),
            ('HMA-1G behavior',G,b'Mode::MappedBaseline',b'Mode::MappedPinned'),
            ('one-arm worker loop',H,b'let attempt = super::execute_arm',b'for i in 0..4 {} let attempt = super::execute_arm'),
            ('remove child wait',H,b'child.wait()?',b'pretend_exit()?'),
            ('sleep between children',H,b'for index in 0..ORDER.len() {',b'for index in 0..ORDER.len() { sleep();'),
            ('parent GPU',H,b'let executable = std::env::current_exe()?;',b'enumerate_adapters(); let executable = std::env::current_exe()?;'),
            ('auditor GPU',H,b'let bytes = std::fs::read(raw)?;',b'construct_runtime(); let bytes = std::fs::read(raw)?;'),
            ('stat field',H,b'fields[19]',b'fields[18]'),
            ('PPID',H,b'id.ppid == e.parent.pid',b'id.ppid != e.parent.pid'),
            ('boot identity',H,b'id.boot_id == e.parent.boot_id',b'true'),
            ('process uniqueness',H,b'identities.insert((id.boot_id.clone(), id.pid, id.start_ticks))',b'true'),
            ('exit overlap',H,b'n < c.launch_ns',b'n <= c.launch_ns'),
            ('signal status',H,b'c.exit == Some(ChildExit::expected(0))',b'true'),
            ('payload hash',H,b'Some(sha(&bytes).as_str())',b'Some("ignored")'),
            ('payload exactness',H,b'encode_hex(&bytes)',b'encode_hex(b"synthetic")'),
            ('transcript chronology',H,b'left == previous',b'left >= previous'),
            ('transcript hash',H,b'sha(bytes) == e.transcript_sha256',b'true'),
            ('fresh output',H,b'.create_new(true)',b'.create(true)'),
            ('missing B0 validation',H,b'validate_completed(&arms)?;',b'// omitted'),
            ('lifecycle bypass',H,b'super::validate_lifecycle(&inherited, &audit.transcript_diagnostics)?;',b'// omitted'),
            ('classification rename',H,b'"FRESH_PROCESS_LARGE_DRIFT"',b'"FRESH_PROCESS_STABLE"'),
            ('registration',H,b'let prepared = prepare(&args)?;',b'register_buffers(); let prepared = prepare(&args)?;'),
            ('unregister',H,b'let prepared = prepare(&args)?;',b'unregister_buffers(); let prepared = prepare(&args)?;'),
            ('SQE',H,b'let prepared = prepare(&args)?;',b'submit(); let prepared = prepare(&args)?;'),
            ('mlock',H,b'let prepared = prepare(&args)?;',b'mlock(); let prepared = prepare(&args)?;'),
            ('retry helper',P+'io_provider.rs',b'read_at_with_retries',b'changed_read_at_with_retries'),
            ('protected runner',P+'gpu_native_physical_install_staging.rs',b'async fn run_physical_install_arm_observed_outcome',b'async fn altered_runner'),
            ('hidden CLI bound',MAIN,b'.range(0..=3)',b'.range(0..=4)'),
        ]
        for name,path,before,after in cases:
            if before not in current[path]:raise ValueError('mutation anchor absent: '+name)
            altered=dict(current);altered[path]=altered[path].replace(before,after,1)
            try: rejected=not prove(old,altered,context)['pass']
            except (ValueError,KeyError,IndexError): rejected=True
            mutations.append({'mutation':name,'rejected':rejected})
    report['mutations']=mutations;report['pass']=report['pass'] and all(m['rejected'] for m in mutations)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'mode':report['mode'],'pass':report['pass'],'checks':len(report['checks']),'failed':[c['check'] for c in report['checks'] if not c['pass']],
        'mutations':len(mutations),'mutation_failures':[m['mutation'] for m in mutations if not m['rejected']]},indent=2))
    return 0 if report['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
