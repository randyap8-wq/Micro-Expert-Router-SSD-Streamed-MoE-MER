//! Shared `/proc/cpuinfo` helpers.
//!
//! Both [`super`] (the kernel dispatcher) and [`super::amx`] need to
//! probe `/proc/cpuinfo` for CPU feature flags. The helpers used to be
//! duplicated in both modules; keeping a single implementation here
//! ensures that any future fix (e.g. supporting a non-Linux
//! `/proc/cpuinfo` alternative) only has to be applied in one place.
//!
//! The module is `pub(super)` (via `pub(crate)` on the items, scoped
//! through `kernels::`), so it stays an internal implementation detail
//! of the `kernels` module tree.

/// Returns `true` when `/proc/cpuinfo` lists `flag` in either the
/// `flags:` (x86) or `Features:` (arm64) line of any CPU. Returns
/// `false` if `/proc/cpuinfo` cannot be read (e.g. non-Linux hosts or
/// sandboxed environments where the file is filtered).
pub(crate) fn cpuinfo_has_flag(flag: &str) -> bool {
    let Some(s) = read_proc_cpuinfo() else {
        return false;
    };
    s.lines()
        .filter(|l| l.starts_with("flags") || l.starts_with("Features"))
        .any(|l| l.split_whitespace().any(|tok| tok == flag))
}

/// Reads the entire `/proc/cpuinfo` file into a `String`, returning
/// `None` on any I/O error (including the file not existing on
/// non-Linux platforms).
pub(crate) fn read_proc_cpuinfo() -> Option<String> {
    use std::io::Read;
    let mut s = String::new();
    let mut f = std::fs::File::open("/proc/cpuinfo").ok()?;
    f.read_to_string(&mut s).ok()?;
    Some(s)
}
