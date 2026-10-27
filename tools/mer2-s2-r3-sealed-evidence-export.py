#!/usr/bin/env python3
"""Read-only export of MER2 S2-R3 *already sealed* Linux CPU evidence.

Never imports/runs the frozen MER runner, Cargo, Rust, tests, or workers.
Never writes to the sealed run root or consumes a one-use marker.
Exports sensitive receipts/logs to a *private local ZIP* for off-host audit.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import stat
import sys
import tempfile
import zipfile

ROOT = Path('/mnt/localssd/mer2-s2-r3-runs/s2-r3-cpu-fcd4f508c69bcef8')
EXPECTED_DISPOSITION_SHA = '389a047a54937ba1de4b1f157e53688175852df43400d6ea927da28bcbbd3046'
EXPECTED_PAYLOADS = 81
EXPECTED_RESULT = 'S2_R3_LINUX_CPU_CONFORMANCE_PASS_FOR_INDEPENDENT_REVIEW'
EXPECTED_TESTS = {'success_boundaries': (12, 12), 'socket_lifecycle': (18, 0),
                  'os01': (1, 1), 'cpu': (85, 32)}
EXPECTED_MARKER = 'S2_R3_LINUX_CPU_TEST_ONCE'
FILE_NAME = 'MER2-S2-R3-Linux-CPU-Sealed-Evidence.zip'


class ExportStop(Exception):
    pass


def require(cond, reason):
    if not cond:
        raise ExportStop(reason)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def inspect_evidence(root):
    require(root.is_absolute() and root.resolve() == root and root.is_dir()
            and not root.is_symlink(), 'FROZEN_RUN_ROOT_MISSING_OR_SYMLINK')
    evidence = root / 'evidence'
    require(evidence.is_dir() and not evidence.is_symlink(), 'SEALED_EVIDENCE_DIR_MISSING')
    disposition_file = evidence / 'disposition.json'
    manifest_file = evidence / 'SHA256SUMS'
    require(disposition_file.is_file() and manifest_file.is_file(), 'NO_FINAL_SEALED_MANIFEST')
    require(not disposition_file.is_symlink() and not manifest_file.is_symlink(), 'UNSAFE_SYMLINK')
    require(sha(disposition_file) == EXPECTED_DISPOSITION_SHA, 'FROZEN_DISPOSITION_SHA256')
    disposition = read_json(disposition_file)
    manifest_bytes = manifest_file.read_bytes()
    require(disposition.get('result') == EXPECTED_RESULT and disposition.get('first_error') is None,
            'FINAL_DISPOSITION_NOT_PASS')
    require(disposition.get('manifest_sha256') == digest_bytes(manifest_bytes),
            'FROZEN_MANIFEST_SHA256')
    require(disposition.get('manifest_excludes') ==
            ['SHA256SUMS', 'disposition.json', '.disposition.pending'],
            'MANIFEST_EXCLUSIONS')
    manifest = {}
    for line in manifest_bytes.decode('utf-8').splitlines():
        require('  ' in line, 'BAD_MANIFEST_FORMAT')
        checksum, name = line.split('  ', 1)
        pp = PurePosixPath(name)
        require(len(checksum) == 64 and all(c in '0123456789abcdef' for c in checksum),
                'BAD_MANIFEST_SHA')
        require(bool(name) and not pp.is_absolute() and '..' not in pp.parts
                and str(pp) == name and name not in manifest, 'BAD_MANIFEST_NAME')
        manifest[name] = checksum
    require(len(manifest) == EXPECTED_PAYLOADS, 'MANIFEST_COUNT')
    actual = {p.relative_to(evidence).as_posix() for p in evidence.rglob('*') if p.is_file()}
    require(actual == set(manifest) | {'SHA256SUMS', 'disposition.json'}, 'COMPLETE_COVERAGE')
    for name, expected in manifest.items():
        path = evidence / name
        require(path.is_file() and not path.is_symlink(), 'UNSAFE_PAYLOAD_' + name)
        require(sha(path) == expected, 'PAYLOAD_HASH_' + name)
    require((evidence / EXPECTED_MARKER).is_file(), 'ONE_USE_MARKER_NOT_PRESENT')
    marker = read_json(evidence / EXPECTED_MARKER)
    require(marker.get('name') == EXPECTED_MARKER and marker.get('retry_forbidden') is True,
            'ONE_USE_MARKER_CONTRACT')
    accounting = read_json(evidence / 'execution-accounting.json')
    for key,expected in {'format_invocations':1, 'test_build_invocations':1,
                         'test_invocations':4, 'test_cases_observed':116,
                         'worker_reports_observed':45, 'marker_created':1,
                         'test_retries':0, 'source_edits':0,
                         'old_marker_operations':0}.items():
        require(accounting.get(key) == expected, 'ACCOUNTING_' + key)
    require(len(accounting.get('invocations', [])) == 9, 'INVOCATION_COUNT')
    expected_stages = ['version-rustc','version-cargo','version-rustfmt','format','build',
                       'success_boundaries','socket_lifecycle','os01','cpu']
    for i, stage in enumerate(expected_stages):
        a = accounting['invocations'][i]
        require(a.get('stage') == stage and a.get('spawned') is True
                and a.get('returncode') == 0 and a.get('launcher_reaped') is True
                and a.get('watchdog_fired') is False and a.get('retries') == 0,
                'INVOCATION_' + stage)
        require(not a.get('residual_group_observation'), 'RESIDUAL_GROUP_' + stage)
    counts = {}
    for stage,(num_tests,num_workers) in EXPECTED_TESTS.items():
        cases = read_json(evidence / (stage + '-tests.json'))
        children = read_json(evidence / (stage + '-children.json'))
        require(len(cases) == num_tests and len(children) == num_workers,
                'TEST_WORKER_COUNTS_' + stage)
        require(len({c['test'] for c in cases}) == num_tests
                and all(c.get('statuses') == ['ok'] for c in cases),
                'TEST_STATUS_' + stage)
        require(all(c.get('reaped') is True and c.get('process_exit_observed') is True
                and c.get('exit_status') not in (None, 'None') for c in children),
                'WORKER_REAP_' + stage)
        counts[stage] = {'tests':len(cases),'workers':len(children)}
    return evidence, manifest, disposition, counts


def export(evidence, manifest, disposition, counts, private_parent):
    require(private_parent.is_dir() and not private_parent.is_symlink(), 'PRIVATE_EXPORT_PARENT')
    os.umask(0o077)
    location = Path(tempfile.mkdtemp(prefix='mer2-s2-r3-sealed-export.', dir=str(private_parent)))
    archive = location / FILE_NAME
    audit = {'export_type':'MER2_S2_R3_FROZEN_CPU_LINUX_EVIDENCE',
             'scope':'PRIVATE_READ_ONLY_EXPORT_NOT_FINAL_INDEPENDENT_ACCEPTANCE',
             'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'run_root':str(ROOT), 'disposition_sha256':EXPECTED_DISPOSITION_SHA,
             'manifest_sha256':disposition['manifest_sha256'], 'payload_count':len(manifest),
             'total_test_executions':sum(c['tests'] for c in counts.values()),
             'total_worker_reports':sum(c['workers'] for c in counts.values()),
             'stage_counts':counts,'one_use_marker_consumed':True,'candidate_runs_from_export':0}
    member_names = ['evidence/' + n for n in sorted(manifest)] + [
        'evidence/SHA256SUMS','evidence/disposition.json','export-verification.json']
    try:
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as z:
            for name in sorted(manifest) + ['SHA256SUMS', 'disposition.json']:
                path = evidence / name
                expected = manifest[name] if name in manifest else sha(path)
                h = hashlib.sha256()
                zi = zipfile.ZipInfo('evidence/' + name)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (stat.S_IFREG | 0o600) << 16
                with path.open('rb') as f, z.open(zi, 'w', force_zip64=True) as output:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        h.update(chunk)
                require(h.hexdigest() == expected, 'SOURCE_CHANGED_DURING_ARCHIVE_' + name)
            z.writestr('export-verification.json', json.dumps(audit, indent=2, sort_keys=True) + '\n')
        require(sha(evidence / 'disposition.json') == EXPECTED_DISPOSITION_SHA,
                'DISPOSITION_CHANGED_AFTER_EXPORT')
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and set(names) == set(member_names),
                    'ZIP_COVERAGE')
            require(z.testzip() is None, 'ZIP_CRC')
            for name in sorted(manifest):
                with z.open('evidence/' + name) as f:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1024*1024), b''):
                        h.update(chunk)
                require(h.hexdigest() == manifest[name], 'ZIP_PAYLOAD_HASH_' + name)
            require(digest_bytes(z.read('evidence/disposition.json')) == EXPECTED_DISPOSITION_SHA
                    and digest_bytes(z.read('evidence/SHA256SUMS')) == disposition['manifest_sha256'],
                    'ZIP_FINAL_SEAL')
        archive.chmod(0o600)
        print('S2_R3_PRIVATE_EVIDENCE_EXPORT=PASS_ZIP_CRC_AND_ALL_HASHES')
        print('SOURCE_DISPOSITION_SHA256=' + EXPECTED_DISPOSITION_SHA)
        print('MANIFEST_ENTRIES=' + str(len(manifest)))
        print('TEST_EXECUTIONS=116 WORKER_REPORTS=45')
        print('EXPORT_ZIP=' + str(archive))
        print('EXPORT_ZIP_SHA256=' + sha(archive))
        print('EXPORT_ZIP_BYTES=' + str(archive.stat().st_size))
        print('ORIGINAL_RUN_ROOT_MODIFICATIONS=0 MER_BUILDS=0 MER_TESTS=0')
        print('EXECUTION_MARKER_REUSED=NO')
        print('NEXT=PRIVATE_TRANSFER_FOR_INDEPENDENT_EVIDENCE_AUDIT')
    except BaseException:
        print('STOP_S2_R3_EXPORT: partial private archive, DO NOT PUBLISH', file=sys.stderr)
        raise


def main():
    require(len(sys.argv)==1, 'NO_ARGS')
    require(os.geteuid()!=0 and os.getuid()==pwd.getpwnam('ubuntu').pw_uid,
            'EXPECTED_UBUNTU_NOT_ROOT')
    evidence, manifest, disposition, counts = inspect_evidence(ROOT)
    export(evidence, manifest, disposition, counts, Path.home() / 'mer-tools')


if __name__=='__main__':
    try:
        main()
    except (ExportStop, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print('STOP_S2_R3_PRIVATE_EVIDENCE_EXPORT: '+type(exc).__name__+': '+str(exc), file=sys.stderr)
        sys.exit(1)
