#!/usr/bin/env python3
"""Independent read-only S2-R3 receipt reviewer; NO execution authority issued.

Requires the frozen on-host 12 receipts and previously verified R2 package.
Independently GETs GitHub APIs to compare the actual receipt content; a private
Notion snapshot is matched against a SHA256 independently recorded through the
connected Notion source. Does not run Cargo, MER, or change any source/marker.
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
import urllib.error
import urllib.request

ROOT=Path('/home/ubuntu/mer-tools/mer2-s2-r3-authority.faad8e55d1de1e')
RUN_PARENT=Path('/mnt/localssd/mer2-s2-r3-runs')
STAGED=Path('/home/ubuntu/mer-tools/mer2-s2-r3-inputs.jOhiQVIb')
REPO='randyap8-wq/Micro-Expert-Router-SSD-Streamed-MoE-MER'
API='https://api.github.com/repos/'+REPO
EXPECTED_REQUEST='96c39ebccab9ec7d26a1819c50ee0e2d52b1db6fd427eb1b4b5831c8ba87b963'
EXPECTED_DRAFT='0220f22455e48e16ca8ce7d8eb341fe4bcfd6788f586cf0d8cb567a15c620231'
EXPECTED_NOTION='d7cb3af4a8cd37caac16f65e077ac342f845d545c11232d0d0b7a013f7ac7f58'
RUNNER_SHA='2593a3054e490378f052d115611a885c66fc0a298cb3a389140963f43137b8a2'
SPEC_SHA='7e32a9354b78ec5da3771521c574b9492c5eed1a428be6f49dbd9c5fc10ec74b'
R2_SHA='a0695dcd8198d720002fd37e20033c61b25a61344ad4988f89da6d73fa17cc89'
EXPECTED_PRS={
  '354':('6150f4bb047b18679ddd48e468b819d7f39ad2ac','1b72de2fdb72fd10c8125d766a56b8876eb081af'),
  '361':('79310a5d3aab9bae6f327fb268a40e4b4610fcaf','6150f4bb047b18679ddd48e468b819d7f39ad2ac'),
  '368':('144f93b94bba2f4a2d111cc28def350e56fb3f13','79310a5d3aab9bae6f327fb268a40e4b4610fcaf')}
EXPECTED_LABELS={'issue-369','issue-369-comments','issue-367','issue-367-comments',
    'pr-354','pr-361','pr-368','pr-368-discussion','pr-368-reviews',
    'pr-368-review-comments','notion-correction','explicit-execution-authorization'}

class STOP(Exception):pass


def require_pr_identity(pr, number):
    expected_head, expected_base = EXPECTED_PRS[number]
    require(pr.get('number') == int(number)
            and pr.get('head', {}).get('sha') == expected_head
            and pr.get('base', {}).get('sha') == expected_base
            and pr.get('state') == 'open'
            and pr.get('draft') is True
            and pr.get('merged') is False
            and pr.get('merged_at') is None,
            'FROZEN_PR_IDENTITY_DRIFT_'+number)


def pr_core_same(stored, live):
    # Exclude ONLY the four mutable nested repository metadata fields seen in
    # the frozen R1 STOP forensics (not HEAD/base source SHA or PR state).
    # Preserve original raw receipt bytes, hashes and all other comparisons.
    import copy
    before=copy.deepcopy(stored)
    after=copy.deepcopy(live)
    for record in (before,after):
        for side in ('head','base'):
            repo=record[side]['repo']
            require(isinstance(repo,dict),'PR_REPO_METADATA_MISSING_'+side)
            repo.pop('pushed_at',None)
            repo.pop('size',None)
    return before==after

def require(ok,what):
    if not ok:raise STOP(what)
def hashfile(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return dt.datetime.now(dt.timezone.utc)
def date(s):
    t=dt.datetime.fromisoformat(s.replace('Z','+00:00'))
    require(t.tzinfo is not None,'TIMEZONE_MISSING')
    return t

def live_get(endpoint):
    url=API+endpoint
    req=urllib.request.Request(url,headers={
        'Accept':'application/vnd.github+json',
        'User-Agent':'MER-S2-R3-independent-review/1.0',
        'X-GitHub-Api-Version':'2022-11-28'})
    try:
        with urllib.request.urlopen(req,timeout=20) as response:
            require(response.status==200 and response.geturl()==url,'REDIRECT_OR_STATUS_'+endpoint)
            require('rel="next"' not in response.headers.get('Link',''),'PAGINATION_'+endpoint)
            raw=response.read(4_000_001)
    except (urllib.error.URLError,TimeoutError) as err:
        raise STOP('GITHUB_INDEPENDENT_READ_'+endpoint+'_'+type(err).__name__) from None
    require(len(raw)<4_000_001,'GITHUB_RESPONSE_TOO_LARGE_'+endpoint)
    return json.loads(raw)

def main():
    require(len(sys.argv)==1,'NO_ARGUMENTS')
    require(os.getuid()==pwd.getpwnam('ubuntu').pw_uid and os.geteuid()!=0,'EXPECTED_UBUNTU')
    require(platform.system()=='Linux' and platform.machine()=='x86_64','HOST_OS_ARCH')
    require(ROOT.is_dir() and not ROOT.is_symlink(),'COLLECTION_ROOT_MISSING')
    reqfile=ROOT/'review-request.json'; draftfile=ROOT/'authorization-DRAFT-NOT-VALID.json'
    require(hashfile(reqfile)==EXPECTED_REQUEST,'REVIEW_REQUEST_SHA256')
    require(hashfile(draftfile)==EXPECTED_DRAFT,'DRAFT_SHA256')
    req=json.loads(reqfile.read_text());draft=json.loads(draftfile.read_text())
    require(req['execution_authorized'] is False and draft['execution_authorized'] is False,'FALSE_REQUIRED')
    require(req['status']=='COLLECTED_PENDING_EXTERNAL_REVIEW_NO_EXECUTION'
        and draft['review_status']=='REVIEW_REQUIRED_NOT_APPROVED','REVIEW_STATUS')
    require(req['gate']==draft['gate']=='MER2-G3-P6A-A3-S2-R3','GATE_MISMATCH')
    require(req['runner_sha256']==draft['runner_sha256']==RUNNER_SHA,'RUNNER_SHA')
    require(req['spec_sha256']==draft['spec_sha256']==SPEC_SHA,'SPEC_SHA')
    require(req['zip_sha256']==draft['zip_sha256']==R2_SHA,'R2_SHA')
    require(req['source_sha256']==draft['source_sha256'],'R2_SOURCES_DIFFER')
    require(len(req['source_sha256'])==8,'SOURCE_COUNT')
    require(req['receipts']==draft['authority_files'],'RECEIPTS_DIFFER')
    require(set(req['receipts'])==EXPECTED_LABELS,'RECEIPT_LABELS')
    require(req['authorization_reference']==draft['authorization_reference']
        =='https://github.com/'+REPO+'/issues/371','AUTHORITY_REFERENCE')
    require(draft['review_result']=='S2_R2_INDEPENDENT_CORRECTION_REVIEW=PASS','CORRECTION_REVIEW')
    require(draft['issue_369_is_not_corrected_execution_authority'] is True,'OLD_369_NOT_AUTH')
    require(draft['permitted_existing_linux_cpu_environment'] is True,'HOST_PERMISSION')
    require(req['tests']==req['builds']==req['markers']==0,'NO_EXECUTION_CLAIM')
    t=now()
    chk=date(req['fresh_authority_checked_utc']); exp=date(req['expires_utc'])
    notion_asof=date(req['notion_snapshot_as_of_utc'])
    require(req['fresh_authority_checked_utc']==draft['fresh_authority_checked_utc']
        and req['expires_utc']==draft['expires_utc'],'TIME_FIELD_MISMATCH')
    require(0<=(t-chk).total_seconds()<=1800 and t<exp,'FRESHNESS_EXPIRED')
    require(exp<=notion_asof+dt.timedelta(minutes=30),'NOTION_FRESHNESS_WINDOW')
    require((t-notion_asof).total_seconds()>=0,'NOTION_FUTURE')
    require((exp-t).total_seconds()>60,'INSUFFICIENT_APPROVAL_WINDOW')
    require(req['host']==draft['host']=={'node':platform.node(),'release':platform.release(),'machine':platform.machine()},'HOST_DRIFT')
    runroot=Path(req['run_root'])
    require(req['run_root']==draft['run_root'] and runroot.parent==RUN_PARENT
        and runroot.is_absolute() and not runroot.exists(),'FROZEN_UNUSED_RUN_ROOT')
    require(RUN_PARENT.is_dir() and RUN_PARENT.stat().st_uid==os.getuid()
        and stat.S_IMODE(RUN_PARENT.stat().st_mode)==0o700,'RUN_PARENT')
    require(req['tools']==draft['tools'] and set(req['tools'])=={'cargo','rustc','rustfmt','cargo-fmt'},'TOOLS_SET')
    for name,record in req['tools'].items():
        p=Path(record['path'])
        require(p.is_file() and p.name==name and p.resolve()==p and os.access(p,os.X_OK), 'TOOL_PATH_'+name)
        with p.open('rb') as f:require(f.read(4)==b'\x7fELF','NATIVE_TOOL_'+name)
        require(hashfile(p)==record['sha256'],'TOOL_HASH_'+name)
    specpath=STAGED/'MER2-G3-P6A-A3-S2-R3-R1-execution-spec.json'
    runnerpath=STAGED/'MER2-G3-P6A-A3-S2-R3-R1-linux-runner.py'
    r2path=STAGED/'MER2-G3-P6A-A3-S2-R2-Evidence.zip'
    require(hashfile(specpath)==SPEC_SHA and hashfile(runnerpath)==RUNNER_SHA and hashfile(r2path)==R2_SHA,'STAGED_INPUT_SHA')
    spec=json.loads(specpath.read_text())
    require(spec['source_sha256']==req['source_sha256'] and spec['marker']=='S2_R3_LINUX_CPU_TEST_ONCE','SPEC_SOURCE_MARKER')
    require([(x['stage'],x.get('passed')) for x in spec['commands']]==[
        ('format',None),('build',None),('success_boundaries',12),('socket_lifecycle',18),('os01',1),('cpu',85)],'FROZEN_COMMANDS')
    require(spec['commands'][4]['filtered']==84 and spec['acceptance']['named_results']==116
        and spec['acceptance']['worker_reports']==45,'FROZEN_COUNTS')
    receipts={}
    for label,entry in req['receipts'].items():
        p=Path(entry['path'])
        require(p.is_file() and p.parent==ROOT/'receipts' and not p.is_symlink(),'RECEIPT_FILE_'+label)
        require(hashfile(p)==entry['sha256'],'RECEIPT_HASH_'+label)
        receipts[label]=p.read_text()
    require(hashlib.sha256(receipts['notion-correction'].encode()).hexdigest()==EXPECTED_NOTION,'NOTION_SNAPSHOT_INDEPENDENT_HASH')
    notion=receipts['notion-correction']
    require('S2_R2_INDEPENDENT_CORRECTION_REVIEW=PASS' in notion and 'issues/371' in notion
        and R2_SHA in notion and 'No Linux test result yet' in notion,'NOTION_CONTENT')
    github={k:json.loads(v) for k,v in receipts.items() if k!='notion-correction'}
    require(github['issue-367']['number']==367 and github['issue-369']['number']==369
        and github['explicit-execution-authorization']['number']==371,'ISSUE_NUMBERS')
    require(all(github[x]['state']=='open' for x in ['issue-367','issue-369','explicit-execution-authorization']),'ISSUE_STATES')
    require('seven source SHA256' in github['issue-369'].get('body',''),'OLD_ISSUE_369_SCOPE')
    body=github['explicit-execution-authorization'].get('body') or ''
    require('FOUNDER_CPU_ONLY_ONE_USE_AUTHORIZED=YES' in body
        and R2_SHA in body and RUNNER_SHA in body and SPEC_SHA in body
        and 'EXECUTION_RECEIPT=NOT_YET_SEALED' in body
        and 'S2_R3_LINUX_CPU_TEST_ONCE' in body,'GITHUB_FOUNDER_AUTHORITY')
    require(isinstance(github['issue-369-comments'],list)
        and isinstance(github['issue-367-comments'],list)
        and any('R3_CPU_FAKE_WORKER_85_OF_85=PASS' in (item.get('body') or '')
                for item in github['issue-367-comments']),'HISTORICAL_COMMENTS')
    for pr,(head,base) in EXPECTED_PRS.items():
        item=github['pr-'+pr]
        require(item['number']==int(pr) and item['head']['sha']==head and item['base']['sha']==base
            and item['state']=='open' and item['draft'] is True
            and item['merged'] is False and item.get('merged_at') is None,'PR_'+pr+'_MISMATCH')
    discussion=github['pr-368-discussion']
    require(set(discussion)=={'issue_comments','reviews','inline_review_comments'}
        and discussion['reviews']==github['pr-368-reviews']
        and discussion['inline_review_comments']==github['pr-368-review-comments'],'DISCUSSION_MATCH')
    # Independent HTTP reads, no local collector code imported or run.
    endpoints={
        'issue-369':'/issues/369','issue-369-comments':'/issues/369/comments?per_page=100',
        'issue-367':'/issues/367','issue-367-comments':'/issues/367/comments?per_page=100',
        'pr-354':'/pulls/354','pr-361':'/pulls/361','pr-368':'/pulls/368',
        'pr-368-reviews':'/pulls/368/reviews?per_page=100',
        'pr-368-review-comments':'/pulls/368/comments?per_page=100',
        'explicit-execution-authorization':'/issues/371'}
    remote_matches=[]
    for label,endpoint in endpoints.items():
        live=live_get(endpoint)
        if label in ('pr-354','pr-361','pr-368'):
            num=label.split('-')[1]
            require_pr_identity(live, num)
            require_pr_identity(github[label], num)
            require(pr_core_same(github[label],live), 'LIVE_PR_SUBSTANTIVE_DRIFT_'+label)
        else:
            require(live==github[label],'LIVE_GITHUB_CHANGED_'+label)
        remote_matches.append(label)
    discussion_live=live_get('/issues/368/comments?per_page=100')
    require(discussion_live==discussion['issue_comments'],'LIVE_DISCUSSION_CHANGED')
    report={
        'status':'S2_R3_RECEIPTS_INDEPENDENT_STATIC_AND_LIVE_GITHUB_REVIEW_PASS_FOR_HUMAN_APPROVAL',
        'execution_authorized':False,'review_request_sha256':EXPECTED_REQUEST,
        'draft_sha256':EXPECTED_DRAFT,'runner_sha256':RUNNER_SHA,'spec_sha256':SPEC_SHA,
        'r2_zip_sha256':R2_SHA,'notion_source_sha256':EXPECTED_NOTION,
        'receipt_count':len(receipts),'receipts_sha256':{k:v['sha256'] for k,v in sorted(req['receipts'].items())},
        'live_github_matches':len(remote_matches)+1,
        'github_prs':{k:{'head':v[0],'base':v[1],'state':'open','draft':True,'merged':False} for k,v in EXPECTED_PRS.items()},
        'host':req['host'],'run_root':req['run_root'],
        'fresh_authority_checked_utc':req['fresh_authority_checked_utc'],
        'notion_snapshot_as_of_utc':req['notion_snapshot_as_of_utc'],
        'expires_utc':req['expires_utc'],
        'remaining_seconds_at_review':int((exp-t).total_seconds()),
        'fmt_runs':0,'build_runs':0,'test_runs':0,'marker_created':False,
        'next':'INDEPENDENT_REVIEWER_APPROVAL_REQUIRED_BEFORE_EXECUTION_RECEIPT'
    }
    output=json.dumps(report,sort_keys=True,separators=(',',':')).encode()
    print('S2_R3_INDEPENDENT_READ_ONLY_REVIEW=PASS_FOR_HUMAN_APPROVAL')
    print('REVIEW_REPORT_SHA256='+hashlib.sha256(output).hexdigest())
    print('REVIEW_REPORT_JSON='+output.decode())
    print('EXECUTION_AUTHORIZED=FALSE MER_BUILDS=0 MER_TESTS=0 EXECUTION_MARKERS=0')

if __name__=='__main__':
    try:main()
    except (STOP,OSError,ValueError,KeyError,TypeError,PermissionError) as err:
        print('STOP_S2_R3_INDEPENDENT_RECEIPT_REVIEW: '+str(err),file=sys.stderr)
        sys.exit(1)
