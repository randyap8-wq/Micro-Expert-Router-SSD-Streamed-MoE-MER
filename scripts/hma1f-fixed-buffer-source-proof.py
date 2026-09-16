#!/usr/bin/env python3
"""Offline source/lifetime proof for the capability-only fixed-buffer probe."""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = '5ee2941d8b0c82fc4b412d444f739123f59bf9fd'
TREE = '22cf2448fe0372a948b2a53dfd3c7838f18c7615'
MODULE = 'rust-engine/src/gpu_native_mapped_fixed_buffer.rs'
MAIN = 'rust-engine/src/main.rs'
SELF = 'scripts/hma1f-fixed-buffer-source-proof.py'
ALLOWED = {MODULE, MAIN, SELF}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prove(child=None):
    checks = []

    def check(name, value):
        checks.append({'check': name, 'pass': bool(value)})

    def read(path):
        return git('show', f'{child}:{path}') if child else (ROOT / path).read_bytes()

    check('exact base tree', git('rev-parse', BASE + '^{tree}').decode().strip() == TREE)
    base_paths = git('ls-tree', '-r', '--name-only', BASE).decode().splitlines()
    # Every tracked base file other than the explicitly bounded CLI additions
    # is protected, not just the named production/authoritative subset.
    unchanged = []
    for path in base_paths:
        if path == MAIN:
            continue
        old = git('show', f'{BASE}:{path}')
        new = read(path)
        unchanged.append({'path': path, 'base_sha256': sha(old), 'child_sha256': sha(new), 'equal': old == new})
    check('every existing file except main is byte-identical', all(row['equal'] for row in unchanged))
    changed = set(git('diff', '--name-only', BASE, *([child] if child else [])).decode().splitlines())
    if not child:
        changed.update(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    check('only the three authorized implementation files changed', changed == ALLOWED)
    old_main = git('show', f'{BASE}:{MAIN}').decode()
    expected_main = old_main.replace('mod gpu_native_mapped_vma;',
        'mod gpu_native_mapped_fixed_buffer;\nmod gpu_native_mapped_vma;', 1)
    expected_main = expected_main.replace('    /// Read-only smaps characterization', '''    /// Capability only: register one mapped L4/Vulkan range; no I/O. Requires Linux + io_uring.
    ProbeGpuNativeMappedDestinationFixedBuffer {
        #[arg(long)]
        report_out: PathBuf,
    },
    /// Read-only smaps characterization''', 1)
    expected_main = expected_main.replace('        Cmd::ProbeGpuNativeMappedDestinationVma { report_out } =>', '''        Cmd::ProbeGpuNativeMappedDestinationFixedBuffer { report_out } =>
            return crate::gpu_native_mapped_fixed_buffer::probe_command(report_out),
        Cmd::ProbeGpuNativeMappedDestinationVma { report_out } =>''', 1)
    expected_main = expected_main.replace('        | Cmd::ProbeGpuNativeMappedDestinationVma { .. }',
        '        | Cmd::ProbeGpuNativeMappedDestinationFixedBuffer { .. }\n        | Cmd::ProbeGpuNativeMappedDestinationVma { .. }', 1)
    check('main changes are exactly module, CLI, early dispatch and unreachable arm', read(MAIN).decode() == expected_main)

    spec = importlib.util.spec_from_file_location('frozen_hma1f_proof', ROOT / 'scripts/hma1f-source-proof.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    protected = []
    for ref in [BASE, child]:
        data = git('show', f'{ref}:rust-engine/src/io_provider.rs') if ref else read('rust-engine/src/io_provider.rs')
        body = legacy.function(data, 'read_at_with_retries')
        protected.append({'ref': ref or 'WORKTREE', 'bytes': len(body), 'sha256': sha(body)})
        check(f'protected retry function exact at {ref or "WORKTREE"}', len(body) == 6909 and sha(body) == '9f3241f79c6ebf02d4fa796b3e2bd3e2b2d45fc9740f2532872e5e3330b258c0')

    source = read(MODULE).decode().split('#[cfg(test)]')[0]
    for forbidden in ['BufferPool', 'IoUringStorage', 'NvmeStorage', 'RealModel', 'Config::', 'Runtime::',
                      'read_expert', 'File::open', 'O_DIRECT', 'pread', 'readv', 'writev',
                      'mlock', 'munlock', 'madvise', 'opcode::', 'squeue::', 'register_files',
                      'READ_FIXED', 'WRITE_FIXED', 'ReadFixed', 'WriteFixed', 'thread::spawn',
                      'create_command_encoder', 'copy_buffer']:
        check('excludes ' + forbidden, forbidden not in source)
    for method in ['submit', 'submit_and_wait', 'submission', 'submission_shared', 'split', 'push', 'sync']:
        # Match an actual call, not the authorized submitter() spelling.
        pattern = r'\.\s*' + method + r'\s*\('
        # Vec::push collects evidence/errors; the ring has no queue accessor.
        if method == 'push':
            pattern = r'\bring\s*\.\s*push\s*\('
        check('excludes queue/submission call ' + method, not re.search(pattern, source))
    for token in ['libc::iovec {', 'device.create_buffer(', 'get_mapped_range_mut()',
                  'io_uring::IoUring::new(2)', 'ring.submitter().register_buffers(&[iov])',
                  'ring.submitter().unregister_buffers()']:
        check('exactly one ' + token, source.count(token) == 1)
    check('only two submitter accesses', source.count('ring.submitter()') == 2)
    check('registration implementation has exact feature/platform gate',
          '#[cfg(all(target_os = "linux", feature = "io_uring"))]\nfn capture_hardware' in source)
    check('unsupported build has explicit pre-hardware stub',
          '#[cfg(not(all(target_os = "linux", feature = "io_uring")))]\nfn capture_hardware' in source)
    check('schema exact', 'mer.gpu-native-mapped-destination-fixed-buffer.v1' in source)
    check('one exact iovec uses live pointer and FULL',
          'iov_base: pointer.cast(),' in source and 'iov_len: FULL,' in source)
    ordered = ['let mut view =', 'aligned_offset(', '.get_mut(offset..end)', 'io_uring::IoUring::new(2)',
               'registration_lifecycle(', 'register_buffers(&[iov])', 'unregister_buffers()',
               'drop(ring)', 'drop(view)', 'buffer.unmap()', 'device.poll(wgpu::Maintain::Wait)', 'device.pop_error_scope()']
    offset = source.find('#[cfg(all(target_os = "linux", feature = "io_uring"))]')
    positions = []
    for token in ordered:
        pos = source.find(token, offset)
        positions.append({'step': token, 'offset': pos})
        if pos < 0:
            break
        offset = pos + len(token)
    check('view and ring lifetime and WGPU cleanup order', len(positions) == len(ordered) and all(p['offset'] >= 0 for p in positions))
    lifecycle = legacy.function(source.encode(), 'registration_lifecycle').decode()
    steps = ['capture(report, Phase::Baseline)', 'report.register =', 'if report.register.succeeded()',
             'capture(report, Phase::Registered)', 'report.unregister =', 'capture(report, Phase::AfterUnregister)',
             '} else {', 'capture(report, Phase::PostFailure)']
    positions_lifecycle = [lifecycle.find(x) for x in steps]
    check('baseline/register/registered/unregister/after or failure sequence',
          all(n >= 0 for n in positions_lifecycle) and positions_lifecycle == sorted(positions_lifecycle))
    check('negative registration branch has no unregister',
          'unregister' not in lifecycle.split('} else {', 1)[1])
    check('proc reads are ordered and single per observation',
          source.count('read("/proc/self/smaps")') == 1 and source.count('read("/proc/self/status")') == 1
          and source.index('read("/proc/self/smaps")') < source.index('read("/proc/self/status")'))
    check('only filesystem readers are proc/kernel capture',
          re.findall(r'std::fs::read(?:_to_string)?\([^\n]+', source) == [
              'std::fs::read_to_string("/proc/sys/kernel/osrelease")?);',
              'std::fs::read(path)),'])
    check('only filesystem open is create-new evidence',
          source.count('.open(') == 1 and 'create_new(true).open(path)' in source)
    main = read(MAIN).decode().split('fn main()', 1)[1]
    check('dispatch before logging/config/runtime',
          main.index('return crate::gpu_native_mapped_fixed_buffer::probe_command') < main.index('init_logging('))
    if child:
        check('child is exactly one commit from base', git('rev-list', '--parents', '-n', '1', child).decode().split() == [child, BASE])
    return {'base': BASE, 'base_tree': TREE, 'child': child or 'WORKTREE', 'checks': checks,
            'unchanged_files': unchanged, 'protected_function': protected, 'lifetime_source_positions': positions,
            'pass': all(c['pass'] for c in checks)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = prove(args.child)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pass': report['pass'], 'checks': len(report['checks']),
                      'unchanged_files': len(report['unchanged_files']),
                      'failed': [c['check'] for c in report['checks'] if not c['pass']],
                      'protected_function': report['protected_function']}, indent=2))
    raise SystemExit(0 if report['pass'] else 1)
