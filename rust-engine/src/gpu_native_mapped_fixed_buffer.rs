//! HMA-1F fixed-buffer registration capability only. No source I/O.
//! Proc parsing is locally frozen from the authoritative VMA module so that
//! module and the production backend remain byte-identical.
//! Every memory quantity reported under its Linux field name is in bytes.
// Parsing and lifecycle code is retained for portable tests even when the
// hardware implementation is not compiled into this binary.
#![cfg_attr(not(all(target_os = "linux", feature = "io_uring")), allow(dead_code))]
#[cfg(all(target_os = "linux", feature = "io_uring"))]
use crate::gpu_native_source_upload::aligned_offset;
use crate::gpu_native_source_upload::{ALIGN, FULL, UPLOAD_BYTES};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "mer.gpu-native-mapped-destination-fixed-buffer.v1";
const REQUIRED_BYTES: &[&str] = &[
    "Size",
    "KernelPageSize",
    "MMUPageSize",
    "Rss",
    "Pss",
    "Anonymous",
    "Locked",
];
const OPTIONAL_BYTES: &[&str] = &[
    "Private_Clean",
    "Private_Dirty",
    "Shared_Clean",
    "Shared_Dirty",
    "Referenced",
    "AnonHugePages",
    "Swap",
    "SwapPss",
];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Serialize)]
struct Hardware {
    name: String,
    backend: String,
    device_type: String,
    vendor: u32,
    device: u32,
    driver: String,
    driver_info: String,
}

#[derive(Debug, Serialize)]
struct MappedRange {
    pointer: usize,
    length: usize,
    pointer_mod_4096: usize,
    aligned_offset: usize,
    upload_buffer_bytes: usize,
}
impl MappedRange {
    fn new(pointer: usize, length: usize, offset: usize) -> Result<Self> {
        let range = Self {
            pointer,
            length,
            pointer_mod_4096: pointer % ALIGN,
            aligned_offset: offset,
            upload_buffer_bytes: UPLOAD_BYTES,
        };
        range.end()?;
        Ok(range)
    }

    fn end(&self) -> Result<u64> {
        if self.pointer == 0
            || self.pointer % ALIGN != 0
            || self.length != FULL
            || self.aligned_offset >= ALIGN
            || self.aligned_offset % 4 != 0
            || self
                .aligned_offset
                .checked_add(self.length)
                .is_none_or(|n| n > UPLOAD_BYTES)
        {
            return Err("invalid exact aligned mapped range".into());
        }
        Ok(u64::try_from(
            self.pointer
                .checked_add(self.length)
                .ok_or("mapped range overflow")?,
        )?)
    }
}

// Numeric fields retain their original Linux keys. Bytes and scalars remain
// distinct internally so a missing/wrong unit cannot silently pass validation.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Field {
    Bytes(u64),
    Scalar(u64),
    Flags(Vec<String>),
}

#[derive(Debug, Serialize)]
struct Vma {
    start: u64,
    end: u64,
    length_bytes: u64,
    permissions: String,
    file_offset: u64,
    dev: String,
    inode: u64,
    pathname: Option<String>,
    #[serde(flatten)]
    fields: BTreeMap<String, Field>,
    raw_header_line: String,
    raw_entry: String,
    raw_entry_sha256: String,
}

#[derive(Debug, Serialize)]
struct MatchingVma {
    #[serde(flatten)]
    entry: Vma,
    pointer_offset_within_vma: u64,
    range_fully_contained: bool,
    vm_io: bool,
    vm_pfnmap: bool,
    vm_mixedmap: bool,
    vm_dontexpand: bool,
    vm_locked: bool,
    accountable: bool,
    shared: bool,
}

fn decimal(text: &str) -> Result<u64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("invalid unsigned decimal: {text:?}").into());
    }
    Ok(text.parse()?)
}
fn hexadecimal(text: &str) -> Result<u64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("invalid hexadecimal: {text:?}").into());
    }
    Ok(u64::from_str_radix(text, 16)?)
}
fn kb(text: &str) -> Result<u64> {
    let parts: Vec<_> = text.split_ascii_whitespace().collect();
    if parts.len() != 2 || parts[1] != "kB" {
        return Err(format!("expected unsigned kB field: {text:?}").into());
    }
    decimal(parts[0])?
        .checked_mul(1024)
        .ok_or_else(|| "kB conversion overflow".into())
}
fn token<'a>(rest: &mut &'a str) -> Result<&'a str> {
    *rest = rest.trim_start_matches([' ', '\t']);
    let n = rest.find([' ', '\t']).unwrap_or(rest.len());
    if n == 0 {
        return Err("missing VMA header token".into());
    }
    let result = &rest[..n];
    *rest = &rest[n..];
    Ok(result)
}
fn parse_header(header: &str) -> Result<Vma> {
    let mut rest = header;
    let addresses = token(&mut rest)?;
    let (start, end) = addresses
        .split_once('-')
        .ok_or("missing VMA address separator")?;
    let (start, end) = (hexadecimal(start)?, hexadecimal(end)?);
    if start >= end {
        return Err("empty or reversed VMA range".into());
    }
    let permissions = token(&mut rest)?;
    let p = permissions.as_bytes();
    if p.len() != 4
        || !matches!(p[0], b'r' | b'-')
        || !matches!(p[1], b'w' | b'-')
        || !matches!(p[2], b'x' | b'-')
        || !matches!(p[3], b'p' | b's')
    {
        return Err("invalid VMA permissions".into());
    }
    let file_offset = hexadecimal(token(&mut rest)?)?;
    let dev = token(&mut rest)?;
    let (major, minor) = dev.split_once(':').ok_or("invalid VMA device")?;
    hexadecimal(major)?;
    hexadecimal(minor)?;
    let inode = decimal(token(&mut rest)?)?;
    // Only the delimiter is trimmed: spaces within (or at the end of) a path
    // are retained, and the original complete header is always preserved.
    let pathname = rest.trim_start_matches([' ', '\t']);
    Ok(Vma {
        start,
        end,
        length_bytes: end - start,
        permissions: permissions.into(),
        file_offset,
        dev: dev.into(),
        inode,
        pathname: (!pathname.is_empty()).then(|| pathname.to_owned()),
        fields: BTreeMap::new(),
        raw_header_line: header.into(),
        raw_entry: String::new(),
        raw_entry_sha256: String::new(),
    })
}
fn parse_field(line: &str) -> Result<(String, Field)> {
    let (key, value) = line
        .split_once(':')
        .ok_or("missing smaps field separator")?;
    if !key.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
        || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err("invalid smaps field name".into());
    }
    let parts: Vec<_> = value.split_ascii_whitespace().collect();
    let field = if key == "VmFlags" {
        if parts
            .iter()
            .any(|p| *p != "??" && (p.len() != 2 || !p.bytes().all(|b| b.is_ascii_lowercase())))
        {
            return Err("invalid VmFlags token".into());
        }
        // Linux emits "??" for unknown bits. Preserve those and all unknown
        // future mnemonics in original order, without assigning them meaning.
        Field::Flags(parts.iter().map(|p| (*p).to_owned()).collect())
    } else if parts.len() == 2 {
        Field::Bytes(kb(value)?)
    } else if parts.len() == 1 {
        Field::Scalar(decimal(parts[0])?)
    } else {
        return Err(format!("invalid smaps field {key}").into());
    };
    Ok((key.into(), field))
}
fn finish_vma(mut vma: Vma, raw: &str) -> Result<Vma> {
    for key in REQUIRED_BYTES {
        if !matches!(vma.fields.get(*key), Some(Field::Bytes(_))) {
            return Err(format!("missing or invalid required kB field {key}").into());
        }
    }
    for key in OPTIONAL_BYTES {
        if vma
            .fields
            .get(*key)
            .is_some_and(|v| !matches!(v, Field::Bytes(_)))
        {
            return Err(format!("invalid optional kB field {key}").into());
        }
    }
    if !matches!(vma.fields.get("THPeligible"), Some(Field::Scalar(0 | 1))) {
        return Err("missing or invalid THPeligible".into());
    }
    if !matches!(vma.fields.get("VmFlags"), Some(Field::Flags(_))) {
        return Err("missing VmFlags".into());
    }
    if vma
        .fields
        .get("ProtectionKey")
        .is_some_and(|v| !matches!(v, Field::Scalar(_)))
    {
        return Err("invalid ProtectionKey".into());
    }
    vma.raw_entry = raw.into();
    vma.raw_entry_sha256 = sha256(raw.as_bytes());
    Ok(vma)
}

/// Parse every entry from a completed, immutable snapshot, including entries
/// after the match. Raw slices retain original spacing and line terminators.
fn parse_smaps(snapshot: &str) -> Result<Vec<Vma>> {
    if snapshot.is_empty() || !snapshot.ends_with('\n') {
        return Err("empty or unterminated smaps snapshot".into());
    }
    let mut entries = Vec::new();
    let mut current = None;
    let mut entry_start = 0;
    let mut offset = 0;
    for raw_line in snapshot.split_inclusive('\n') {
        let line = raw_line
            .strip_suffix('\n')
            .ok_or("unterminated smaps line")?;
        // Header authority is address geometry, never pathname. A field starts
        // with a colon-terminated key; all other lines must parse as headers.
        let first = line
            .split_ascii_whitespace()
            .next()
            .ok_or("blank smaps line")?;
        if first.ends_with(':') {
            let vma: &mut Vma = current.as_mut().ok_or("smaps field before header")?;
            let (key, field) = parse_field(line)?;
            if vma.fields.insert(key.clone(), field).is_some() {
                return Err(format!("duplicate smaps field {key}").into());
            }
        } else {
            if let Some(vma) = current.take() {
                entries.push(finish_vma(vma, &snapshot[entry_start..offset])?);
            }
            current = Some(parse_header(line)?);
            entry_start = offset;
        }
        offset += raw_line.len();
    }
    entries.push(finish_vma(
        current.ok_or("no smaps entries")?,
        &snapshot[entry_start..],
    )?);
    Ok(entries)
}
fn containing_vma(entries: Vec<Vma>, range: &MappedRange) -> Result<MatchingVma> {
    let end = range.end()?;
    let pointer = u64::try_from(range.pointer)?;
    let mut matches = entries
        .into_iter()
        .filter(|v| v.start <= pointer && pointer < v.end && end <= v.end);
    let entry = matches
        .next()
        .ok_or("zero VMAs fully contain the mapped range")?;
    if matches.next().is_some() {
        return Err("more than one VMA fully contains the mapped range".into());
    }
    let Some(Field::Flags(flags)) = entry.fields.get("VmFlags") else {
        return Err("missing VmFlags".into());
    };
    let has = |flag: &str| flags.iter().any(|f| f == flag);
    Ok(MatchingVma {
        pointer_offset_within_vma: pointer - entry.start,
        range_fully_contained: true,
        vm_io: has("io"),
        vm_pfnmap: has("pf"),
        vm_mixedmap: has("mm"),
        vm_dontexpand: has("de"),
        vm_locked: has("lo"),
        accountable: has("ac"),
        shared: has("sh"),
        entry,
    })
}
fn parse_status(snapshot: &str) -> Result<BTreeMap<String, u64>> {
    let mut fields = BTreeMap::new();
    for line in snapshot.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if matches!(key, "VmLck" | "VmPin" | "VmRSS")
            && fields.insert(key.into(), kb(value)?).is_some()
        {
            return Err(format!("duplicate status field {key}").into());
        }
    }
    if fields.len() != 3 {
        return Err("missing VmLck, VmPin or VmRSS".into());
    }
    Ok(fields)
}

#[derive(Debug, Default, Serialize)]
struct Operation {
    attempted: bool,
    success: bool,
    raw_os_error: Option<i32>,
    error_kind: Option<String>,
    error_message: Option<String>,
}
impl Operation {
    fn from_result(result: std::io::Result<()>) -> Self {
        match result {
            Ok(()) => Self {
                attempted: true,
                success: true,
                ..Self::default()
            },
            Err(e) => Self {
                attempted: true,
                success: false,
                raw_os_error: e.raw_os_error(),
                error_kind: Some(format!("{:?}", e.kind())),
                error_message: Some(e.to_string()),
            },
        }
    }
    fn succeeded(&self) -> bool {
        self.attempted
            && self.success
            && self.raw_os_error.is_none()
            && self.error_kind.is_none()
            && self.error_message.is_none()
    }
    fn kernel_error(&self) -> bool {
        self.attempted
            && !self.success
            && self.raw_os_error.is_some()
            && self.error_kind.is_some()
            && self.error_message.is_some()
    }
    fn failed(&self) -> bool {
        self.attempted && !self.success && self.error_kind.is_some() && self.error_message.is_some()
    }
    fn not_attempted(&self) -> bool {
        !self.attempted
            && !self.success
            && self.raw_os_error.is_none()
            && self.error_kind.is_none()
            && self.error_message.is_none()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Baseline,
    Registered,
    AfterUnregister,
    PostFailure,
}
impl Phase {
    const ALL: [Self; 4] = [
        Self::Baseline,
        Self::Registered,
        Self::AfterUnregister,
        Self::PostFailure,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Registered => "registered",
            Self::AfterUnregister => "after-unregister",
            Self::PostFailure => "post-failure",
        }
    }
}
#[derive(Debug, Serialize)]
struct Snapshot {
    path: Option<PathBuf>,
    byte_length: usize,
    sha256: String,
    #[serde(skip)]
    bytes: Vec<u8>,
}
impl Snapshot {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            path: None,
            byte_length: bytes.len(),
            sha256: sha256(&bytes),
            bytes,
        }
    }
}
#[derive(Debug, Default, Serialize)]
struct Observation {
    smaps_snapshot: Option<Snapshot>,
    status_snapshot: Option<Snapshot>,
    smaps_vma_count: Option<usize>,
    vma: Option<MatchingVma>,
    process_status: Option<BTreeMap<String, u64>>,
    failures: Vec<String>,
}
impl Observation {
    fn valid(&self) -> bool {
        self.failures.is_empty()
            && self.smaps_snapshot.is_some()
            && self.status_snapshot.is_some()
            && self.smaps_vma_count.is_some()
            && self.vma.is_some()
            && self.process_status.is_some()
    }
}
fn observe(
    report: &mut Report,
    phase: Phase,
    mut read: impl FnMut(&str) -> std::io::Result<Vec<u8>>,
) -> bool {
    let mut observation = Observation::default();
    // Both reads complete before parsing; no reread can change the bound bytes.
    let smaps = read("/proc/self/smaps");
    let status = read("/proc/self/status");
    for (result, slot, name) in [
        (smaps, &mut observation.smaps_snapshot, "smaps"),
        (status, &mut observation.status_snapshot, "status"),
    ] {
        match result {
            Ok(bytes) => *slot = Some(Snapshot::new(bytes)),
            Err(e) => observation.failures.push(format!("{name} capture: {e}")),
        }
    }
    let parsed = (|| -> Result<()> {
        let raw = &observation
            .smaps_snapshot
            .as_ref()
            .ok_or("smaps unavailable")?
            .bytes;
        let entries = parse_smaps(std::str::from_utf8(raw)?)?;
        observation.smaps_vma_count = Some(entries.len());
        observation.vma = Some(containing_vma(
            entries,
            report
                .mapped_range
                .as_ref()
                .ok_or("mapped range unavailable")?,
        )?);
        Ok(())
    })();
    if let Err(e) = parsed {
        observation.failures.push(e.to_string());
    }
    let parsed = (|| -> Result<_> {
        let raw = &observation
            .status_snapshot
            .as_ref()
            .ok_or("status unavailable")?
            .bytes;
        parse_status(std::str::from_utf8(raw)?)
    })();
    match parsed {
        Ok(fields) => observation.process_status = Some(fields),
        Err(e) => observation.failures.push(e.to_string()),
    }
    let valid = observation.valid();
    for e in &observation.failures {
        report.failures.push(format!("{}: {e}", phase.name()));
    }
    if report.observations.insert(phase, observation).is_some() {
        report.failures.push("duplicate observation phase".into());
        return false;
    }
    valid
}

#[derive(Debug, Default, Serialize)]
struct Cleanup {
    // IoUring's Drop has no fallible close API. This witnesses completion of
    // that destructor, not an invented close return code.
    ring_drop_completed: bool,
    mapped_view_dropped: bool,
    buffer_unmapped: bool,
    device_polled: bool,
    validation_scope_popped: bool,
    validation_error: Option<String>,
}
impl Cleanup {
    fn clean(&self, ring_created: bool) -> bool {
        self.ring_drop_completed == ring_created
            && self.mapped_view_dropped
            && self.buffer_unmapped
            && self.device_polled
            && self.validation_scope_popped
            && self.validation_error.is_none()
    }
}
#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Classification {
    FixedBufferRegisterable,
    FixedBufferNotRegisterable,
    FixedBufferUnregisterFailed,
    RingUnavailable,
    NonAuthoritative,
}
#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    classification: Classification,
    memory_field_unit: &'static str,
    required_platform: &'static str,
    hardware: Option<Hardware>,
    mapped_range: Option<MappedRange>,
    kernel_osrelease: Option<String>,
    ring_queue_depth: u32,
    ring_parameters: Option<String>,
    ring_creation: Operation,
    register: Operation,
    unregister: Operation,
    observations: BTreeMap<Phase, Observation>,
    sqes_submitted: u64,
    cleanup: Cleanup,
    artifacts_valid: bool,
    failures: Vec<String>,
}
impl Report {
    fn new() -> Self {
        Self {
            schema: SCHEMA,
            classification: Classification::NonAuthoritative,
            memory_field_unit: "bytes",
            required_platform: "Linux + io_uring",
            hardware: None,
            mapped_range: None,
            kernel_osrelease: None,
            ring_queue_depth: 2,
            ring_parameters: None,
            ring_creation: Operation::default(),
            register: Operation::default(),
            unregister: Operation::default(),
            observations: BTreeMap::new(),
            sqes_submitted: 0,
            cleanup: Cleanup::default(),
            artifacts_valid: false,
            failures: Vec::new(),
        }
    }
    fn classify(&self) -> Classification {
        use Classification::*;
        let hardware_valid = self.hardware.as_ref().is_some_and(|h| {
            h.name == "NVIDIA L4"
                && h.backend == "Vulkan"
                && h.device_type == "DiscreteGpu"
                && h.vendor == 0x10de
        });
        if !self.failures.is_empty()
            || !self.artifacts_valid
            || !hardware_valid
            || self.mapped_range.as_ref().is_none_or(|r| r.end().is_err())
            || self
                .kernel_osrelease
                .as_ref()
                .is_none_or(|s| s.trim().is_empty())
            || self.ring_queue_depth != 2
            || self.sqes_submitted != 0
            || !self.cleanup.clean(self.ring_creation.succeeded())
        {
            return NonAuthoritative;
        }
        if self.ring_creation.failed()
            && self.register.not_attempted()
            && self.unregister.not_attempted()
            && self.observations.is_empty()
        {
            return RingUnavailable;
        }
        let has = |p| self.observations.get(&p).is_some_and(Observation::valid);
        if !self.ring_creation.succeeded()
            || self.ring_parameters.is_none()
            || !has(Phase::Baseline)
        {
            return NonAuthoritative;
        }
        if self.register.succeeded()
            && has(Phase::Registered)
            && has(Phase::AfterUnregister)
            && self.observations.len() == 3
        {
            if self.unregister.succeeded() {
                return FixedBufferRegisterable;
            }
            if self.unregister.failed() {
                return FixedBufferUnregisterFailed;
            }
        }
        if self.register.kernel_error()
            && self.unregister.not_attempted()
            && has(Phase::PostFailure)
            && self.observations.len() == 2
        {
            return FixedBufferNotRegisterable;
        }
        NonAuthoritative
    }
}

// The same branching is exercised by portable fakes. Captures never propagate
// an early return after successful registration: explicit unregister still runs
// once even when registered-state evidence fails. No retry is permitted.
fn registration_lifecycle(
    report: &mut Report,
    mut capture: impl FnMut(&mut Report, Phase) -> bool,
    register: impl FnOnce() -> std::io::Result<()>,
    unregister: impl FnOnce() -> std::io::Result<()>,
) {
    if !capture(report, Phase::Baseline) {
        return;
    }
    report.register = Operation::from_result(register());
    if report.register.succeeded() {
        capture(report, Phase::Registered);
        report.unregister = Operation::from_result(unregister());
        // The label means after the explicit unregister *attempt*. Its actual
        // result is always retained; failure cannot imply successful release.
        capture(report, Phase::AfterUnregister);
    } else {
        capture(report, Phase::PostFailure);
    }
}

#[cfg(not(all(target_os = "linux", feature = "io_uring")))]
fn capture_hardware(_report: &mut Report) -> Result<()> {
    Err("fixed-buffer probe requires Linux + io_uring; hardware was not constructed".into())
}

#[cfg(all(target_os = "linux", feature = "io_uring"))]
fn capture_hardware(report: &mut Report) -> Result<()> {
    report.kernel_osrelease = Some(std::fs::read_to_string("/proc/sys/kernel/osrelease")?);
    if report.kernel_osrelease.as_ref().unwrap().trim().is_empty() {
        return Err("empty kernel osrelease".into());
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .into_iter()
        .find(|a| {
            let i = a.get_info();
            i.name == "NVIDIA L4"
                && i.backend == wgpu::Backend::Vulkan
                && i.device_type == wgpu::DeviceType::DiscreteGpu
                && i.vendor == 0x10de
        })
        .ok_or("exact NVIDIA L4 Vulkan adapter unavailable")?;
    let info = adapter.get_info();
    report.hardware = Some(Hardware {
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        vendor: info.vendor,
        device: info.device,
        driver: info.driver,
        driver_info: info.driver_info,
    });
    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("HMA-1F fixed-buffer capability"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))?;
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HMA-1F single fixed-buffer capability upload buffer"),
        size: UPLOAD_BYTES as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // After buffer creation all fallible returns remain inside this closure.
    let observed = (|| -> Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Write, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv_timeout(std::time::Duration::from_secs(30))??;
        let mut view = buffer.slice(..).get_mapped_range_mut();
        let observed = (|| -> Result<()> {
            let offset = aligned_offset(view.as_ptr() as usize, view.len())?;
            let end = offset
                .checked_add(FULL)
                .ok_or("mapped view offset overflow")?;
            let pointer = view
                .get_mut(offset..end)
                .ok_or("mapped view too short")?
                .as_mut_ptr();
            report.mapped_range = Some(MappedRange::new(pointer as usize, FULL, offset)?);
            // The ring exists before baseline and remains live for every capture.
            let ring = match io_uring::IoUring::new(2) {
                Ok(ring) => {
                    report.ring_creation = Operation::from_result(Ok(()));
                    ring
                }
                Err(e) => {
                    report.ring_creation = Operation::from_result(Err(e));
                    return Ok(());
                }
            };
            report.ring_parameters = Some(format!("{:?}", ring.params()));
            registration_lifecycle(
                report,
                |r, phase| observe(r, phase, |path| std::fs::read(path)),
                || {
                    let iov = libc::iovec {
                        iov_base: pointer.cast(),
                        iov_len: FULL,
                    };
                    // SAFETY: this is the checked mutable WGPU slice. The single
                    // view stays alive through explicit unregister and ring drop,
                    // including errors, and no requests can reference the buffer.
                    unsafe { ring.submitter().register_buffers(&[iov]) }
                },
                || ring.submitter().unregister_buffers(),
            );
            drop(ring);
            report.cleanup.ring_drop_completed = true;
            Ok(())
        })();
        drop(view);
        report.cleanup.mapped_view_dropped = true;
        observed
    })();
    buffer.unmap();
    report.cleanup.buffer_unmapped = true;
    device.poll(wgpu::Maintain::Wait);
    report.cleanup.device_polled = true;
    report.cleanup.validation_error =
        pollster::block_on(device.pop_error_scope()).map(|e| e.to_string());
    report.cleanup.validation_scope_popped = true;
    if let Some(e) = &report.cleanup.validation_error {
        report.failures.push(format!("WGPU validation: {e}"));
    }
    observed
}

fn write_synced(file: &mut File, bytes: &[u8]) -> Result<()> {
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn new_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}
fn evidence_path(report_out: &Path, phase: Phase, kind: &str) -> PathBuf {
    let mut path = report_out.as_os_str().to_owned();
    path.push(format!(".{}.{}", phase.name(), kind));
    PathBuf::from(path)
}
struct EvidenceFiles {
    files: BTreeMap<Phase, [(PathBuf, File); 2]>,
}
impl EvidenceFiles {
    fn reserve(report_out: &Path) -> Result<Self> {
        let mut files = BTreeMap::new();
        // Reserve all potential siblings before hardware. Unused phases retain
        // empty reserved files and are absent from observations in the report.
        for phase in Phase::ALL {
            let smaps = evidence_path(report_out, phase, "smaps");
            let status = evidence_path(report_out, phase, "status");
            let smaps_file = new_file(&smaps)?;
            let status_file = new_file(&status)?;
            files.insert(phase, [(smaps, smaps_file), (status, status_file)]);
        }
        Ok(Self { files })
    }
    fn persist(&mut self, report: &mut Report) {
        let mut valid = true;
        for (phase, observation) in &mut report.observations {
            let files = self.files.get_mut(phase).expect("all phases reserved");
            for (snapshot, (path, file)) in [
                &mut observation.smaps_snapshot,
                &mut observation.status_snapshot,
            ]
            .into_iter()
            .zip(files.iter_mut())
            {
                if let Some(snapshot) = snapshot {
                    match write_synced(file, &snapshot.bytes) {
                        Ok(()) => snapshot.path = Some(path.clone()),
                        Err(e) => {
                            valid = false;
                            report
                                .failures
                                .push(format!("evidence {}: {e}", path.display()));
                        }
                    }
                }
            }
        }
        report.artifacts_valid = valid;
    }
}
fn run_with_capture(
    report_out: &Path,
    capture: impl FnOnce(&mut Report) -> Result<()>,
) -> Result<()> {
    let report_out = std::path::absolute(report_out)?;
    // Existing files and symlinks are never followed/overwritten. If the report
    // itself cannot be created, return an error without any hardware access.
    let mut report_file = new_file(&report_out)?;
    let mut report = Report::new();
    match EvidenceFiles::reserve(&report_out) {
        Err(e) => report.failures.push(format!("create-new evidence: {e}")),
        Ok(mut evidence) => {
            if let Err(e) = capture(&mut report) {
                report.failures.push(e.to_string());
            }
            evidence.persist(&mut report);
        }
    }
    report.classification = report.classify();
    write_synced(&mut report_file, &serde_json::to_vec_pretty(&report)?)?;
    match report.classification {
        Classification::FixedBufferRegisterable
        | Classification::FixedBufferNotRegisterable
        | Classification::RingUnavailable => Ok(()),
        Classification::FixedBufferUnregisterFailed => {
            Err("explicit buffer unregister failed; later performance work blocked".into())
        }
        Classification::NonAuthoritative => {
            Err(format!("NON_AUTHORITATIVE: {}", report.failures.join("; ")).into())
        }
    }
}
pub(crate) fn probe_command(report_out: &Path) -> Result<()> {
    run_with_capture(report_out, capture_hardware)
}

#[cfg(test)]
mod tests {
    use super::*;
    const ANON: &str = "10000000-10400000 rw-p 00000000 00:00 0\nSize:               4096 kB\nKernelPageSize:         4 kB\nMMUPageSize:            4 kB\nRss:                  12 kB\nPss:                   6 kB\nAnonymous:             8 kB\nLocked:                0 kB\nTHPeligible:           0\nVmFlags: rd wr mr mw me ac\n";
    fn range() -> MappedRange {
        MappedRange::new(0x10001000, FULL, 0).unwrap()
    }
    fn matching(snapshot: &str) -> Result<MatchingVma> {
        containing_vma(parse_smaps(snapshot)?, &range())
    }
    fn value(snapshot: &str) -> serde_json::Value {
        serde_json::to_value(matching(snapshot).unwrap()).unwrap()
    }

    #[test]
    fn anonymous_header_and_checked_byte_fields() {
        let v = value(ANON);
        assert_eq!(v["start"], 0x10000000u64);
        assert_eq!(v["end"], 0x10400000u64);
        assert_eq!(v["length_bytes"], 4 * 1024 * 1024);
        assert_eq!(v["pointer_offset_within_vma"], 4096);
        assert_eq!(v["range_fully_contained"], true);
        assert_eq!(v["permissions"], "rw-p");
        assert_eq!(v["file_offset"], 0);
        assert_eq!(v["dev"], "00:00");
        assert_eq!(v["inode"], 0);
        assert!(v["pathname"].is_null());
        for (key, bytes) in [
            ("Size", 4194304),
            ("KernelPageSize", 4096),
            ("MMUPageSize", 4096),
            ("Rss", 12288),
            ("Pss", 6144),
            ("Anonymous", 8192),
            ("Locked", 0),
        ] {
            assert_eq!(v[key], bytes, "{key}");
        }
        assert_eq!(v["THPeligible"], 0);
        assert_eq!(v["accountable"], true);
        assert_eq!(v["vm_io"], false);
        assert_eq!(v["vm_locked"], false);
    }
    #[test]
    fn file_backed_path_spaces_and_hexadecimal_metadata() {
        let s = ANON.replace(
            "rw-p 00000000 00:00 0",
            "rw-s 00aBc000 fF:0a 123456 /dev/nvidia mapped file (deleted)",
        );
        let v = value(&s);
        assert_eq!(v["permissions"], "rw-s");
        assert_eq!(v["file_offset"], 0xabc000);
        assert_eq!(v["dev"], "fF:0a");
        assert_eq!(v["inode"], 123456);
        assert_eq!(v["pathname"], "/dev/nvidia mapped file (deleted)");
        let executable = s.replace("rw-s", "rwxp");
        assert_eq!(value(&executable)["permissions"], "rwxp");
    }
    #[test]
    fn preserves_exact_raw_entry_header_and_flag_order() {
        let s = ANON.replace("rd wr mr mw me ac", "rd io pf mm de lo ac sh zz ??");
        let v = value(&s);
        assert_eq!(v["raw_entry"], s);
        assert_eq!(v["raw_header_line"], s.lines().next().unwrap());
        assert_eq!(v["raw_entry_sha256"], sha256(s.as_bytes()));
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            v["VmFlags"],
            serde_json::json!(["rd", "io", "pf", "mm", "de", "lo", "ac", "sh", "zz", "??"])
        );
        for key in [
            "vm_io",
            "vm_pfnmap",
            "vm_mixedmap",
            "vm_dontexpand",
            "vm_locked",
            "accountable",
            "shared",
        ] {
            assert_eq!(v[key], true, "{key}");
        }
    }
    #[test]
    fn optional_fields_preserved_in_bytes_and_protection_key_scalar() {
        let mut s = ANON.to_string();
        for key in OPTIONAL_BYTES {
            s.push_str(&format!("{key}: 2 kB\n"));
        }
        s.push_str("ProtectionKey: 3\nFutureBytes: 5 kB\nFutureScalar: 7\n");
        let v = value(&s);
        for key in OPTIONAL_BYTES {
            assert_eq!(v[*key], 2048);
        }
        assert_eq!(v["ProtectionKey"], 3);
        assert_eq!(v["FutureBytes"], 5120);
        assert_eq!(v["FutureScalar"], 7);
        assert!(value(ANON).get("ProtectionKey").is_none());
    }
    #[test]
    fn parses_every_entry_and_selects_by_pointer_not_path() {
        let other = ANON.replace("10000000-10400000", "20000000-20400000");
        let s = format!("{other}{ANON}{other}");
        assert_eq!(parse_smaps(&s).unwrap().len(), 3);
        assert_eq!(matching(&s).unwrap().entry.raw_entry, ANON);
        assert!(matching(&format!("{ANON}malformed later entry\n")).is_err());
        assert!(matching(&format!("{ANON}20000000-20400000 rw-p 0 00:00 0\n")).is_err());
    }
    #[test]
    fn exact_containment_boundaries() {
        for pointer in [0x10000000, 0x10400000 - FULL] {
            let r = MappedRange::new(pointer, FULL, 0).unwrap();
            assert!(containing_vma(parse_smaps(ANON).unwrap(), &r).is_ok());
        }
        for pointer in [0x10000000 - ALIGN, 0x10400000 - FULL + ALIGN, 0x10400000] {
            let r = MappedRange::new(pointer, FULL, 0).unwrap();
            assert!(containing_vma(parse_smaps(ANON).unwrap(), &r).is_err());
        }
    }
    #[test]
    fn rejects_zero_multiple_and_split_vma_containment() {
        let other = ANON.replace("10000000-10400000", "20000000-20400000");
        assert!(matching(&other)
            .unwrap_err()
            .to_string()
            .contains("zero VMAs"));
        assert!(matching(&format!("{ANON}{ANON}"))
            .unwrap_err()
            .to_string()
            .contains("more than one"));
        let first = ANON.replace("10400000", "10200000");
        let second = ANON.replace("10000000", "10200000");
        assert!(matching(&format!("{first}{second}")).is_err());
    }
    #[test]
    fn rejects_invalid_mapped_range_and_overflow() {
        for (p, n, offset) in [
            (0, FULL, 0),
            (0x10000001, FULL, 0),
            (0x10000000, FULL - 1, 0),
            (usize::MAX - (ALIGN - 1), FULL, 0),
            (0x10000000, FULL, ALIGN),
            (0x10000000, FULL, 1),
        ] {
            assert!(MappedRange::new(p, n, offset).is_err());
        }
        assert_eq!(FULL, 2_658_304);
        assert_eq!(ALIGN, 4096);
        assert_eq!(UPLOAD_BYTES, FULL + ALIGN);
    }
    #[test]
    fn rejects_malformed_headers() {
        for (from, to) in [
            ("10000000", "xyz"),
            ("10000000", "10400000"),
            ("10000000", "fffffffffffffffff"),
            ("10400000", "00001000"),
            ("rw-p", "rw-q"),
            ("rw-p", "rwxpp"),
            ("00000000 00:00", "+1 00:00"),
            ("00:00 0", "0g:00 0"),
            ("00:00 0", "00:00 -1"),
        ] {
            assert!(parse_smaps(&ANON.replacen(from, to, 1)).is_err(), "{to}");
        }
    }
    #[test]
    fn rejects_missing_duplicate_wrong_unit_and_overflow_fields() {
        for key in REQUIRED_BYTES
            .iter()
            .copied()
            .chain(["THPeligible", "VmFlags"])
        {
            let s: String = ANON
                .split_inclusive('\n')
                .filter(|l| !l.starts_with(&format!("{key}:")))
                .collect();
            assert!(parse_smaps(&s).is_err(), "missing {key}");
        }
        for tail in [
            "Rss: 1 kB\n",
            "VmFlags: rd\n",
            "Private_Clean: 1\n",
            "ProtectionKey: 1 kB\n",
        ] {
            assert!(parse_smaps(&format!("{ANON}{tail}")).is_err());
        }
        for replacement in [
            "0",
            "-1 kB",
            "+1 kB",
            "1 MB",
            "1 kB extra",
            "1.5 kB",
            "18446744073709551615 kB",
            "18446744073709551616 kB",
        ] {
            assert!(
                parse_smaps(&ANON.replace("0 kB", replacement)).is_err(),
                "{replacement}"
            );
        }
        for eligible in ["2", "1 kB", "-1", "18446744073709551616"] {
            assert!(parse_smaps(&ANON.replace(
                "THPeligible:           0",
                &format!("THPeligible: {eligible}")
            ))
            .is_err());
        }
        assert_eq!(kb("18014398509481983 kB").unwrap(), 18446744073709550592);
        assert!(kb("18014398509481984 kB").is_err());
    }
    #[test]
    fn rejects_incomplete_or_malformed_snapshots() {
        for s in ["", "\n", "Size: 4 kB\n", ANON.trim_end()] {
            assert!(parse_smaps(s).is_err());
        }
        assert!(parse_smaps(&ANON.replace("rd wr", "rd WR")).is_err());
        assert!(parse_smaps(&ANON.replace("rd wr", "rd toolong")).is_err());
        assert!(parse_smaps(&format!("{ANON}\n")).is_err());
        assert!(parse_smaps(&format!("{ANON}start: 4\n")).is_err());
    }
    #[test]
    fn status_is_descriptive_and_strict() {
        let s = "Name:\ttest\nVmLck:\t0 kB\nVmPin: 7 kB\nVmRSS: 9 kB\n";
        let v = parse_status(s).unwrap();
        assert_eq!(v["VmLck"], 0);
        assert_eq!(v["VmPin"], 7168);
        assert_eq!(v["VmRSS"], 9216);
        for bad in [
            s.replace("VmPin", "Other"),
            s.replace("7 kB", "7"),
            format!("{s}VmLck: 0 kB\n"),
            s.replace("7 kB", "18446744073709551615 kB"),
        ] {
            assert!(parse_status(&bad).is_err());
        }
    }
    const STATUS: &str = "VmLck: 0 kB\nVmPin: 0 kB\nVmRSS: 8192 kB\n";
    fn ready_report() -> Report {
        let mut r = Report::new();
        r.hardware = Some(Hardware {
            name: "NVIDIA L4".into(),
            backend: "Vulkan".into(),
            device_type: "DiscreteGpu".into(),
            vendor: 0x10de,
            device: 0,
            driver: "fixture".into(),
            driver_info: "fixture".into(),
        });
        r.mapped_range = Some(range());
        r.kernel_osrelease = Some("fixture-only\n".into());
        r.ring_creation = Operation::from_result(Ok(()));
        r.ring_parameters = Some("fixture-only".into());
        r.cleanup = Cleanup {
            ring_drop_completed: true,
            mapped_view_dropped: true,
            buffer_unmapped: true,
            device_polled: true,
            validation_scope_popped: true,
            validation_error: None,
        };
        r.artifacts_valid = true;
        r
    }
    fn fixture_capture(r: &mut Report, p: Phase) -> bool {
        observe(r, p, |path| {
            Ok(if path.ends_with("smaps") {
                ANON
            } else {
                STATUS
            }
            .as_bytes()
            .to_vec())
        })
    }
    fn fixture_result(register: std::io::Result<()>, unregister: std::io::Result<()>) -> Report {
        let mut r = ready_report();
        registration_lifecycle(&mut r, fixture_capture, || register, || unregister);
        r
    }
    #[test]
    fn classifier_registerable_without_any_accounting_delta() {
        let r = fixture_result(Ok(()), Ok(()));
        assert_eq!(r.classify(), Classification::FixedBufferRegisterable);
        assert_eq!(r.observations.len(), 3);
        for o in r.observations.values() {
            assert_eq!(o.process_status.as_ref().unwrap()["VmPin"], 0);
            assert_eq!(o.process_status.as_ref().unwrap()["VmLck"], 0);
        }
    }
    #[test]
    fn classifier_not_registerable_retains_exact_kernel_error() {
        for errno in [libc::EFAULT, libc::EINVAL, libc::EPERM, libc::ENOMEM] {
            let mut r = ready_report();
            registration_lifecycle(
                &mut r,
                fixture_capture,
                || Err(std::io::Error::from_raw_os_error(errno)),
                || panic!("negative registration must never unregister"),
            );
            assert_eq!(r.classify(), Classification::FixedBufferNotRegisterable);
            assert_eq!(r.register.raw_os_error, Some(errno));
            assert!(r.unregister.not_attempted());
            assert_eq!(r.observations.len(), 2);
            assert!(!r.observations.contains_key(&Phase::AfterUnregister));
        }
    }
    #[test]
    fn classifier_unregister_failed_blocks_performance() {
        let r = fixture_result(Ok(()), Err(std::io::Error::from_raw_os_error(libc::EINVAL)));
        assert_eq!(r.classify(), Classification::FixedBufferUnregisterFailed);
        assert!(r.unregister.attempted);
        assert!(r.observations.contains_key(&Phase::AfterUnregister));
    }
    #[test]
    fn classifier_ring_unavailable_is_not_a_mapping_classification() {
        let mut r = ready_report();
        r.ring_creation =
            Operation::from_result(Err(std::io::Error::from_raw_os_error(libc::EPERM)));
        r.ring_parameters = None;
        r.cleanup.ring_drop_completed = false;
        assert_eq!(r.classify(), Classification::RingUnavailable);
        assert!(r.register.not_attempted() && r.unregister.not_attempted());
        r.cleanup.validation_error = Some("failure".into());
        assert_eq!(r.classify(), Classification::NonAuthoritative);
    }
    #[test]
    fn errno_serialization_preserves_actual_error_without_invented_names() {
        let e = std::io::Error::from_raw_os_error(libc::EFAULT);
        let expected = serde_json::json!({
            "attempted": true, "success": false, "raw_os_error": e.raw_os_error(),
            "error_kind": format!("{:?}", e.kind()), "error_message": e.to_string(),
        });
        let a = Operation::from_result(Err(e));
        let b = Operation::from_result(Err(std::io::Error::from_raw_os_error(libc::EFAULT)));
        assert_eq!(serde_json::to_value(&a).unwrap(), expected);
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        let r = fixture_result(Err(std::io::Error::other("not a kernel error")), Ok(()));
        assert_eq!(r.classify(), Classification::NonAuthoritative);
    }
    #[test]
    fn all_cleanup_and_provenance_failures_are_non_authoritative() {
        for mutation in 0..15 {
            let mut r = fixture_result(Ok(()), Ok(()));
            match mutation {
                0 => r.cleanup.ring_drop_completed = false,
                1 => r.cleanup.mapped_view_dropped = false,
                2 => r.cleanup.buffer_unmapped = false,
                3 => r.cleanup.device_polled = false,
                4 => r.cleanup.validation_scope_popped = false,
                5 => r.cleanup.validation_error = Some("validation".into()),
                6 => r.artifacts_valid = false,
                7 => r.hardware.as_mut().unwrap().vendor = 0,
                8 => r.mapped_range.as_mut().unwrap().length -= 1,
                9 => r.sqes_submitted = 1,
                10 => r.failures.push("evidence write failure".into()),
                11 => {
                    r.observations.remove(&Phase::Registered);
                }
                12 => r.kernel_osrelease = None,
                13 => r.ring_queue_depth = 4,
                _ => r.unregister = Operation::default(),
            }
            assert_eq!(r.classify(), Classification::NonAuthoritative, "{mutation}");
        }
    }
    #[test]
    fn lifecycle_calls_exactly_once_in_order_and_always_unregisters_after_success() {
        use std::cell::RefCell;
        for fail_phase in [
            None,
            Some(Phase::Baseline),
            Some(Phase::Registered),
            Some(Phase::AfterUnregister),
        ] {
            let events = RefCell::new(Vec::new());
            let mut r = ready_report();
            registration_lifecycle(
                &mut r,
                |r, p| {
                    events.borrow_mut().push(p.name());
                    if Some(p) == fail_phase {
                        r.failures.push("synthetic capture failure".into());
                        false
                    } else {
                        fixture_capture(r, p)
                    }
                },
                || {
                    events.borrow_mut().push("register");
                    Ok(())
                },
                || {
                    events.borrow_mut().push("unregister");
                    Ok(())
                },
            );
            let expected: &[&str] = if fail_phase == Some(Phase::Baseline) {
                &["baseline"]
            } else {
                &[
                    "baseline",
                    "register",
                    "registered",
                    "unregister",
                    "after-unregister",
                ]
            };
            assert_eq!(&*events.borrow(), expected);
            assert_eq!(
                r.classify(),
                if fail_phase.is_none() {
                    Classification::FixedBufferRegisterable
                } else {
                    Classification::NonAuthoritative
                }
            );
        }
    }
    #[test]
    fn post_failure_capture_failure_stays_non_authoritative_without_unregister() {
        let mut r = ready_report();
        registration_lifecycle(
            &mut r,
            |r, p| {
                if p == Phase::PostFailure {
                    observe(r, p, |_| Err(std::io::Error::other("capture failed")))
                } else {
                    fixture_capture(r, p)
                }
            },
            || Err(std::io::Error::from_raw_os_error(libc::EFAULT)),
            || panic!("unregister forbidden"),
        );
        assert!(r.unregister.not_attempted());
        assert_eq!(r.classify(), Classification::NonAuthoritative);
    }
    #[test]
    fn capture_order_and_partial_bytes_are_preserved_on_parse_failure() {
        let mut r = ready_report();
        let mut calls = Vec::new();
        assert!(!observe(&mut r, Phase::Baseline, |p| {
            calls.push(p.to_string());
            Ok(if p.ends_with("smaps") {
                b"malformed\n".to_vec()
            } else {
                STATUS.as_bytes().to_vec()
            })
        }));
        assert_eq!(calls, ["/proc/self/smaps", "/proc/self/status"]);
        let o = &r.observations[&Phase::Baseline];
        assert_eq!(
            o.smaps_snapshot.as_ref().unwrap().sha256,
            sha256(b"malformed\n")
        );
        assert!(o.process_status.is_some());
        assert!(!o.valid());
    }
    #[test]
    fn required_classification_strings_and_schema_are_exact() {
        assert_eq!(SCHEMA, "mer.gpu-native-mapped-destination-fixed-buffer.v1");
        for (c, s) in [
            (
                Classification::FixedBufferRegisterable,
                "FIXED_BUFFER_REGISTERABLE",
            ),
            (
                Classification::FixedBufferNotRegisterable,
                "FIXED_BUFFER_NOT_REGISTERABLE",
            ),
            (
                Classification::FixedBufferUnregisterFailed,
                "FIXED_BUFFER_UNREGISTER_FAILED",
            ),
            (Classification::RingUnavailable, "RING_UNAVAILABLE"),
            (Classification::NonAuthoritative, "NON_AUTHORITATIVE"),
        ] {
            assert_eq!(serde_json::to_value(c).unwrap(), s);
        }
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("mer-fixed-buffer-test-{}-{n}", std::process::id()));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
        fn report(&self) -> PathBuf {
            self.0.join("report.json")
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn load(p: &Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
    }
    #[test]
    fn baseline_registered_after_snapshots_are_distinct_and_bound_to_exact_bytes() {
        let t = Temp::new();
        run_with_capture(&t.report(), |r| {
            *r = ready_report();
            registration_lifecycle(
                r,
                |r, p| {
                    observe(r, p, |path| {
                        Ok(if path.ends_with("smaps") {
                            ANON.replace(
                                "Rss:                  12",
                                &format!("Rss:                  {}", 12 + p as u64),
                            )
                        } else {
                            STATUS.replace("VmRSS: 8192", &format!("VmRSS: {}", 8192 + p as u64))
                        }
                        .into_bytes())
                    })
                },
                || Ok(()),
                || Ok(()),
            );
            Ok(())
        })
        .unwrap();
        let r = load(&t.report());
        assert_eq!(r["classification"], "FIXED_BUFFER_REGISTERABLE");
        let mut hashes = std::collections::BTreeSet::new();
        for (p, key) in [
            (Phase::Baseline, "baseline"),
            (Phase::Registered, "registered"),
            (Phase::AfterUnregister, "after_unregister"),
        ] {
            let o = &r["observations"][key];
            for (kind, field) in [("smaps", "smaps_snapshot"), ("status", "status_snapshot")] {
                let path = evidence_path(&t.report(), p, kind);
                let raw = std::fs::read(&path).unwrap();
                assert_eq!(o[field]["path"], path.to_str().unwrap());
                assert_eq!(o[field]["byte_length"], raw.len());
                assert_eq!(o[field]["sha256"], sha256(&raw));
                hashes.insert(sha256(&raw));
            }
            let entry = o["vma"]["raw_entry"].as_str().unwrap();
            assert_eq!(o["vma"]["raw_entry_sha256"], sha256(entry.as_bytes()));
            assert_eq!(
                std::fs::read(evidence_path(&t.report(), p, "smaps")).unwrap(),
                entry.as_bytes()
            );
        }
        assert_eq!(hashes.len(), 6);
        assert!(r["observations"].get("post_failure").is_none());
    }
    #[test]
    fn negative_capability_is_a_successful_command_with_only_post_failure_evidence() {
        let t = Temp::new();
        run_with_capture(&t.report(), |r| {
            *r = fixture_result(Err(std::io::Error::from_raw_os_error(libc::EFAULT)), Ok(()));
            Ok(())
        })
        .unwrap();
        let r = load(&t.report());
        assert_eq!(r["classification"], "FIXED_BUFFER_NOT_REGISTERABLE");
        assert_eq!(r["unregister"]["attempted"], false);
        assert_eq!(r["observations"].as_object().unwrap().len(), 2);
        assert!(r["observations"]["post_failure"]["smaps_snapshot"]["path"].is_string());
    }
    #[test]
    fn unregister_failure_command_fails_and_preserves_evidence() {
        let t = Temp::new();
        assert!(run_with_capture(&t.report(), |r| {
            *r = fixture_result(Ok(()), Err(std::io::Error::from_raw_os_error(libc::EINVAL)));
            Ok(())
        })
        .is_err());
        assert_eq!(
            load(&t.report())["classification"],
            "FIXED_BUFFER_UNREGISTER_FAILED"
        );
    }
    #[test]
    fn existing_artifacts_fail_before_hardware_and_are_never_overwritten() {
        for path_index in 0..9 {
            let t = Temp::new();
            let path = if path_index == 0 {
                t.report()
            } else {
                evidence_path(
                    &t.report(),
                    Phase::ALL[(path_index - 1) / 2],
                    if path_index % 2 == 1 {
                        "smaps"
                    } else {
                        "status"
                    },
                )
            };
            std::fs::write(&path, b"preserve").unwrap();
            assert!(run_with_capture(&t.report(), |_| panic!("hardware must not run")).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"preserve");
            if path_index != 0 {
                assert_eq!(load(&t.report())["classification"], "NON_AUTHORITATIVE");
            }
        }
    }
    #[cfg(unix)]
    #[test]
    fn artifact_symlinks_are_rejected_without_following_targets() {
        for is_report in [true, false] {
            let t = Temp::new();
            let target = t.0.join("target");
            std::fs::write(&target, b"preserve").unwrap();
            let path = if is_report {
                t.report()
            } else {
                evidence_path(&t.report(), Phase::Registered, "smaps")
            };
            std::os::unix::fs::symlink(&target, path).unwrap();
            assert!(run_with_capture(&t.report(), |_| panic!("hardware must not run")).is_err());
            assert_eq!(std::fs::read(target).unwrap(), b"preserve");
        }
    }
    #[test]
    fn evidence_write_failure_demotes_an_otherwise_successful_lifecycle() {
        let t = Temp::new();
        let mut files = EvidenceFiles::reserve(&t.report()).unwrap();
        let (path, file) = &mut files.files.get_mut(&Phase::Registered).unwrap()[0];
        *file = File::open(path).unwrap(); // read-only handle injects a deterministic write failure
        let mut r = fixture_result(Ok(()), Ok(()));
        files.persist(&mut r);
        assert!(!r.artifacts_valid);
        assert_eq!(r.classify(), Classification::NonAuthoritative);
    }
    #[test]
    fn capture_error_preserves_partial_raw_evidence_after_cleanup() {
        let t = Temp::new();
        assert!(run_with_capture(&t.report(), |r| {
            *r = fixture_result(Ok(()), Ok(()));
            Err("synthetic cleanup failure".into())
        })
        .is_err());
        let r = load(&t.report());
        assert_eq!(r["classification"], "NON_AUTHORITATIVE");
        assert!(r["observations"]["registered"]["smaps_snapshot"]["path"].is_string());
    }
    #[test]
    fn cli_has_exact_command_requires_report_and_bypasses_startup_config() {
        use clap::Parser;
        let args = [
            "mer",
            "probe-gpu-native-mapped-destination-fixed-buffer",
            "--report-out",
            "report.json",
        ];
        let cli = crate::Cli::try_parse_from(args).unwrap();
        assert!(crate::startup_config_path(&cli.cmd).is_none());
        assert!(matches!(
            cli.cmd,
            crate::Cmd::ProbeGpuNativeMappedDestinationFixedBuffer { .. }
        ));
        assert!(crate::Cli::try_parse_from(&args[..2]).is_err());
        assert!(
            crate::Cli::try_parse_from(args.into_iter().chain(["--config", "model.toml"])).is_err()
        );
    }
    #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
    #[test]
    fn unsupported_platform_or_feature_fails_before_hardware() {
        let t = Temp::new();
        assert!(probe_command(&t.report()).is_err());
        let r = load(&t.report());
        assert_eq!(r["classification"], "NON_AUTHORITATIVE");
        assert!(r["failures"][0]
            .as_str()
            .unwrap()
            .contains("requires Linux + io_uring"));
        for key in [
            "hardware",
            "mapped_range",
            "kernel_osrelease",
            "ring_parameters",
        ] {
            assert!(r[key].is_null());
        }
        for key in ["ring_creation", "register", "unregister"] {
            assert_eq!(r[key]["attempted"], false);
        }
        assert_eq!(r["cleanup"]["mapped_view_dropped"], false);
        assert!(r["observations"].as_object().unwrap().is_empty());
    }
    #[test]
    fn source_proves_one_iovec_zero_sqes_and_live_view_cleanup_order() {
        let s = include_str!("gpu_native_mapped_fixed_buffer.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "BufferPool",
            "IoUringStorage",
            "NvmeStorage",
            "RealModel",
            "Config::",
            "Runtime::",
            "read_expert",
            "File::open",
            "O_DIRECT",
            "pread",
            "readv",
            "writev",
            "mlock",
            "munlock",
            "madvise",
            "opcode::",
            "squeue::",
            ".submission(",
            ".submission_shared(",
            ".split(",
            ".submit(",
            ".submit_and_wait(",
            "ReadFixed",
            "WriteFixed",
            "READ_FIXED",
            "WRITE_FIXED",
            "register_files",
            "thread::spawn",
            "create_command_encoder",
            "copy_buffer",
        ] {
            assert!(!s.contains(forbidden), "forbidden {forbidden}");
        }
        for unique in [
            "libc::iovec {",
            "device.create_buffer(",
            "get_mapped_range_mut()",
            "io_uring::IoUring::new(2)",
            "ring.submitter().register_buffers(&[iov])",
            "ring.submitter().unregister_buffers()",
        ] {
            assert_eq!(s.matches(unique).count(), 1, "{unique}");
        }
        assert_eq!(s.matches("ring.submitter()").count(), 2);
        assert!(s.contains(
            "#[cfg(all(target_os = \"linux\", feature = \"io_uring\"))]\nfn capture_hardware"
        ));
        let mut position = s.find("let mut view =").unwrap();
        for step in [
            "aligned_offset(",
            "view.get_mut(",
            "io_uring::IoUring::new(2)",
            "registration_lifecycle(",
            "register_buffers(&[iov])",
            "unregister_buffers()",
            "drop(ring)",
            "drop(view)",
            "buffer.unmap()",
            "device.poll(wgpu::Maintain::Wait)",
            "device.pop_error_scope()",
        ] {
            // rustfmt may place get_mut on the following line.
            let needle = if step == "view.get_mut(" {
                ".get_mut(offset..end)"
            } else {
                step
            };
            position += s[position..]
                .find(needle)
                .unwrap_or_else(|| panic!("missing ordered {needle}"));
        }
        let main = include_str!("main.rs").split("fn main()").nth(1).unwrap();
        assert!(
            main.find("return crate::gpu_native_mapped_fixed_buffer::probe_command")
                .unwrap()
                < main.find("init_logging(").unwrap()
        );
    }
}
