#!/usr/bin/env python3
"""Independent read-only review of S2-R3 physically collected authority receipts.

This is not an execution approval. No Cargo, Rust, worker, one-use marker, or
authorization.json write. Failure is terminal; no automatic retries.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import stat
import sys
import urllib.request

ROOT=Path('/home/ubuntu/mer-tools/mer2-s2-r3-authority.a872b5b9cb5e81')
STAGED=Path('/home/ubuntu/mer-tools/mer2-s2-r3-inputs.jOhiQVIb')
RUN_PARENT=Path('/mnt/localssd/mer2-s2-r3-runs')
API='https://api.github.com/repos/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER'
REQ_SHA='0ebd0032e0228ce39dfabbc159c48a3fdb284aa856355174a7ff91e15ebe6457'
DRAFT_SHA='4f3c67faeaf13b303425d75583932a6b90d76666d6761e3e9f61279a1fee555e'
NOTION_SHA='b88aca8a02af6226d7369c4249414442a7d1dbd2520cdb200c7eca9c75649b6d'
RUNNER_SHA='2593a3054e490378f052d115611a885c66fc0a298cb3a389140963f43137b8a2'
SPEC_SHA='7e32a9354b78ec5da3771521c574b9492c5eed1a428be6f49dbd9c5fc10ec74b'
R2_SHA='a0695dcd8198d720002fd37e20033c61b25a61344ad4988f89da6d73fa17cc89'
PRS={'354':('6150f4bb047b18679ddd48e468b819d7f39ad2ac','1b72de2fdb72fd10c8125d766a56b8876eb081af'),
     '361':('79310a5d3aab9bae6f327fb268a40e4b4610fcaf','6150f4bb047b18679ddd48e468b819d7f39ad2ac'),
     '368':('144f93b94bba2f4a2d111cc28def350e56fb3f13','79310a5d3aab9bae6f327fb268a40e4b4610fcaf')}
ENDPOINTS={'issue-369':'/issues/369','issue-369-comments':'/issues/369/comments?per_page=100',
           'issue-367':'/issues/367','issue-367-comments':'/issues/367/comments?per_page=100',
           'pr-354':'/pulls/354','pr-361':'/pulls/361','pr-368':'/pulls/368',
           'pr-368-reviews':'/pulls/368/reviews?per_page=100',
           'pr-368-review-comments':'/pulls/368/comments?per_page=100',
           'explicit-execution-authorization':'/issues/371'}
LABELS=set(ENDPOINTS)|{'pr-368-discussion','notion-correction'}

class STOP(Exception): pass
def require(test,why):
    if not test: raise STOP(why)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def instant(s):
    x=dt.datetime.fromisoformat(s.replace('Z','+00:00'))
    require(x.tzinfo is not None,'BAD_TIMESTAMP_TIMEZONE')
    return x
def live(endpoint):
    url=API+endpoint
    req=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json',
        'User-Agent':'MER-S2-R3-independent-receipt-review/1.0',
        'X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(req,timeout=20) as resp:
        require(resp.status==200 and resp.geturl()==url,'GITHUB_UNEXPECTED_REDIRECT')
        require('rel="next"' not in resp.headers.get('Link',''),'GITHUB_PAGINATION')
        raw=resp.read(4_000_001)
    require(len(raw)<4_000_001,'GITHUB_RESPONSE_TOO_LARGE')
    return json.loads(raw)
def run():
    require(len(sys.argv)==1,'NO_ARGUMENTS')
    require(os.getuid()==pwd.getpwnam('ubuntu').pw_uid and os.geteuid()!=0,'UBUNTU_OPERATOR_ONLY')
    require(platform.system()=='Linux' and platform.machine()=='x86_64','HOST_NOT_LINUX_X86_64')
    require(ROOT.is_dir() and not ROOT.is_symlink(),'AUTHORITY_ROOT')
    request_file=ROOT/'review-request.json';draft_file=ROOT/'authorization-DRAFT-NOT-VALID.json'
    require(sha(request_file)==REQ_SHA and sha(draft_file)==DRAFT_SHA,'FREEZE_REQUEST_SHA256')
    q=json.loads(request_file.read_text());d=json.loads(draft_file.read_text())
    require(q['execution_authorized'] is False and d['execution_authorized'] is False,'DRAFT_MUST_NOT_AUTHORIZE')
    require(q['status']=='COLLECTED_PENDING_EXTERNAL_REVIEW_NO_EXECUTION'
            and d['review_status']=='REVIEW_REQUIRED_NOT_APPROVED','REVIEW_STATUS')
    require(q['gate']==d['gate']=='MER2-G3-P6A-A3-S2-R3','GATE_ID')
    for k,val in [('runner_sha256',RUNNER_SHA),('spec_sha256',SPEC_SHA),('zip_sha256',R2_SHA)]:
        require(q[k]==d[k]==val,'FREEZE_'+k)
    require(q['source_sha256']==d['source_sha256'] and len(q['source_sha256'])==8,'SOURCE_8')
    require(q['receipts']==d['authority_files'] and set(q['receipts'])==LABELS,'RECEIPT_COVERAGE')
    require(d['review_result']=='S2_R2_INDEPENDENT_CORRECTION_REVIEW=PASS'
            and d['issue_369_is_not_corrected_execution_authority'] is True,'OLD_AUTHORITY_BOUNDARY')
    require(q['authorization_reference']==d['authorization_reference']
            =='https://github.com/randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER/issues/371','ISSUE_371_REFERENCE')
    require(q['tests']==q['builds']==q['markers']==0,'NONZERO_EXECUTION_CLAIM')
    t=dt.datetime.now(dt.timezone.utc)
    chk=instant(q['fresh_authority_checked_utc'])
    notion_time=instant(q['notion_snapshot_as_of_utc'])
    expiry=instant(q['expires_utc'])
    require(q['expires_utc']==d['expires_utc']
            and q['fresh_authority_checked_utc']==d['fresh_authority_checked_utc'],'FRESHNESS_FIELDS')
    require(0<=(t-chk).total_seconds()<=1800 and t<expiry,'AUTHORITY_EXPIRED')
    require(expiry<=notion_time+dt.timedelta(minutes=30) and (expiry-t).total_seconds()>60,'NOTION_EXPIRED')
    host={'node':platform.node(),'release':platform.release(),'machine':platform.machine()}
    require(q['host']==d['host']==host,'HOST_DRIFT')
    run_root=Path(q['run_root'])
    require(q['run_root']==d['run_root'] and run_root.is_absolute()
            and run_root.parent==RUN_PARENT and not run_root.exists(),'RUN_ROOT_USED_OR_DRIFT')
    require(RUN_PARENT.is_dir() and RUN_PARENT.stat().st_uid==os.getuid()
            and stat.S_IMODE(RUN_PARENT.stat().st_mode)==0o700,'RUN_ROOT_PARENT_IDENTITY')
    require(q['tools']==d['tools'] and set(q['tools'])=={'cargo','rustc','rustfmt','cargo-fmt'},'TOOL_INVENTORY')
    for name,tool in q['tools'].items():
        p=Path(tool['path'])
        require(p.name==name and p.is_file() and p.resolve()==p and os.access(p,os.X_OK),'TOOL_PATH_'+name)
        with p.open('rb') as f: require(f.read(4)==b'\x7fELF','TOOL_NOT_NATIVE_ELF_'+name)
        require(sha(p)==tool['sha256'],'TOOL_SHA_'+name)
    inputs={'MER2-G3-P6A-A3-S2-R3-R1-linux-runner.py':RUNNER_SHA,
            'MER2-G3-P6A-A3-S2-R3-R1-execution-spec.json':SPEC_SHA,
            'MER2-G3-P6A-A3-S2-R2-Evidence.zip':R2_SHA}
    for name,expected in inputs.items():
        require(sha(STAGED/name)==expected,'STAGED_INPUT_SHA_'+name)
    spec=json.loads((STAGED/'MER2-G3-P6A-A3-S2-R3-R1-execution-spec.json').read_text())
    require(spec['source_sha256']==q['source_sha256']
            and spec['marker']=='S2_R3_LINUX_CPU_TEST_ONCE','SPEC_SOURCE_AND_MARKER')
    require([(x['stage'],x.get('passed')) for x in spec['commands']]==[
        ('format',None),('build',None),('success_boundaries',12),
        ('socket_lifecycle',18),('os01',1),('cpu',85)],'FROZEN_COMMANDS')
    require(spec['commands'][4]['filtered']==84 and spec['acceptance']['named_results']==116
            and spec['acceptance']['worker_reports']==45,'FROZEN_TEST_COUNTS')
    receipts={}
    for label,record in q['receipts'].items():
        p=Path(record['path'])
        require(p.parent==ROOT/'receipts' and p.is_file() and not p.is_symlink(),'RECEIPT_FILE_'+label)
        require(sha(p)==record['sha256'],'RECEIPT_HASH_'+label)
        receipts[label]=p.read_bytes()
    require(hashlib.sha256(receipts['notion-correction']).hexdigest()==NOTION_SHA,'NOTION_SOURCE_SHA')
    nt=receipts['notion-correction'].decode()
    require('S2_R2_INDEPENDENT_CORRECTION_REVIEW=PASS' in nt
        and R2_SHA in nt and 'issues/371' in nt and 'No Linux test result yet' in nt,'NOTION_SOURCE_CONTENT')
    obj={k:json.loads(v) for k,v in receipts.items() if k!='notion-correction'}
    require(obj['issue-369']['number']==369 and obj['issue-367']['number']==367
        and obj['explicit-execution-authorization']['number']==371,'ISSUE_NUMBER')
    require('seven source SHA256' in (obj['issue-369']['body'] or ''),'ISSUE_369_OLD_SEVEN_SOURCE')
    issuebody=obj['explicit-execution-authorization']['body'] or ''
    require(all(v in issuebody for v in ['FOUNDER_CPU_ONLY_ONE_USE_AUTHORIZED=YES',
        'EXECUTION_RECEIPT=NOT_YET_SEALED',R2_SHA,RUNNER_SHA,SPEC_SHA]),'ISSUE_371_AUTHORITY')
    require(any('R3_CPU_FAKE_WORKER_85_OF_85=PASS' in (x.get('body') or '')
        for x in obj['issue-367-comments']),'HISTORICAL_R3_PASS_COMMENT')
    for num,(head,base) in PRS.items():
        pr=obj['pr-'+num]
        require(pr['number']==int(num) and pr['head']['sha']==head and pr['base']['sha']==base
            and pr['state']=='open' and pr['draft'] is True
            and pr['merged'] is False and pr['merged_at'] is None,'PR_STATE_'+num)
    discussion=obj['pr-368-discussion']
    require(discussion['reviews']==obj['pr-368-reviews']
        and discussion['inline_review_comments']==obj['pr-368-review-comments'],'DISCUSSION_COVERAGE')
    for label,endpoint in ENDPOINTS.items():
        require(live(endpoint)==obj[label],'GITHUB_SOURCE_CHANGED_'+label)
    require(live('/issues/368/comments?per_page=100')==discussion['issue_comments'],
        'GITHUB_DISCUSSION_CHANGED')
    output={'status':'INDEPENDENT_STATIC_AND_FRESH_GITHUB_READS_PASS_FOR_APPROVAL',
        'execution_authorized':False,'review_request_sha256':REQ_SHA,
        'draft_sha256':DRAFT_SHA,'notion_source_sha256':NOTION_SHA,
        'receipt_sha256':{k:v['sha256'] for k,v in sorted(q['receipts'].items())},
        'github_live_sources_matched':11,'host':host,'run_root':q['run_root'],
        'fresh_authority_checked_utc':q['fresh_authority_checked_utc'],
        'expires_utc':q['expires_utc'],'seconds_remaining_at_review':int((expiry-t).total_seconds()),
        'mer_builds':0,'mer_tests':0,'markers':0,'next':'EXTERNAL_INDEPENDENT_APPROVAL_STILL_REQUIRED'}
    raw=json.dumps(output,sort_keys=True,separators=(',',':')).encode()
    print('S2_R3_INDEPENDENT_READ_ONLY_RECEIPT_REVIEW=PASS_FOR_APPROVAL')
    print('REVIEW_REPORT_SHA256='+hashlib.sha256(raw).hexdigest())
    print('REVIEW_REPORT_JSON='+raw.decode())
    print('EXECUTION_AUTHORIZED=FALSE MER_BUILDS=0 MER_TESTS=0 EXECUTION_MARKERS=0')
if __name__=='__main__':
    try: run()
    except Exception as exc:
        print('STOP_S2_R3_INDEPENDENT_RECEIPT_REVIEW: '+type(exc).__name__+': '+str(exc),file=sys.stderr)
        sys.exit(1)
