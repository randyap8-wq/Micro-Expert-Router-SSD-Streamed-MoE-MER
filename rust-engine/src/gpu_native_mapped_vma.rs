//! Read-only HMA-1F VMA characterization. No runtime or source-I/O construction.
//! Every memory quantity reported under its Linux field name is in bytes.
use crate::gpu_native_source_upload::{aligned_offset, ALIGN, FULL, UPLOAD_BYTES};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SCHEMA: &str = "mer.gpu-native-mapped-destination-vma.v1";
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

#[derive(Debug, Serialize)]
struct SnapshotEvidence {
    path: PathBuf,
    sha256: String,
    byte_length: usize,
}
#[derive(Debug, Serialize)]
struct Cleanup {
    mapped_view_dropped: bool,
    buffer_unmapped: bool,
    device_polled: bool,
    validation_scope_popped: bool,
    validation_error: Option<String>,
}
#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    complete: bool,
    characterization_pass: bool,
    failure: Option<String>,
    memory_field_unit: &'static str,
    hardware: Option<Hardware>,
    mapped_range: Option<MappedRange>,
    vma: Option<MatchingVma>,
    smaps_snapshot: Option<SnapshotEvidence>,
    smaps_vma_count: Option<usize>,
    process_status: Option<BTreeMap<String, u64>>,
    cleanup: Option<Cleanup>,
}
impl Report {
    fn new() -> Self {
        Self {
            schema: SCHEMA,
            complete: false,
            characterization_pass: false,
            failure: None,
            memory_field_unit: "bytes",
            hardware: None,
            mapped_range: None,
            vma: None,
            smaps_snapshot: None,
            smaps_vma_count: None,
            process_status: None,
            cleanup: None,
        }
    }
}

// Both reads finish while the one mutable mapped view remains alive. Parsing,
// hashing and evidence writes use these same immutable bytes after cleanup.
#[derive(Default)]
struct Captured {
    smaps: Option<String>,
    status: Option<String>,
}
fn capture_proc(
    captured: &mut Captured,
    mut read: impl FnMut(&str) -> std::io::Result<String>,
) -> Result<()> {
    let smaps = read("/proc/self/smaps");
    let status = read("/proc/self/status");
    let mut errors = Vec::new();
    match smaps {
        Ok(text) => captured.smaps = Some(text),
        Err(e) => errors.push(format!("smaps capture: {e}")),
    }
    match status {
        Ok(text) => captured.status = Some(text),
        Err(e) => errors.push(format!("status capture: {e}")),
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; ").into())
    }
}
fn capture_hardware(report: &mut Report, captured: &mut Captured) -> Result<()> {
    if !cfg!(target_os = "linux") {
        return Err("VMA probe requires Linux NVIDIA L4/Vulkan".into());
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
            label: Some("HMA-1F read-only mapped VMA characterization"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))?;
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HMA-1F single upload VMA buffer"),
        size: UPLOAD_BYTES as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut mapped_view_dropped = false;
    // No fallible return may bypass cleanup after the buffer is constructed.
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
                .as_mut_ptr() as usize;
            report.mapped_range = Some(MappedRange::new(pointer, FULL, offset)?);
            capture_proc(captured, |path| std::fs::read_to_string(path))
        })();
        drop(view);
        mapped_view_dropped = true;
        observed
    })();
    buffer.unmap();
    device.poll(wgpu::Maintain::Wait);
    let validation_error = pollster::block_on(device.pop_error_scope()).map(|e| e.to_string());
    let result = match (observed, &validation_error) {
        (Ok(()), None) => Ok(()),
        (Err(e), None) => Err(e),
        (Ok(()), Some(e)) => Err(format!("WGPU validation: {e}").into()),
        (Err(e), Some(gpu)) => Err(format!("{e}; WGPU validation: {gpu}").into()),
    };
    report.cleanup = Some(Cleanup {
        mapped_view_dropped,
        buffer_unmapped: true,
        device_polled: true,
        validation_scope_popped: true,
        validation_error,
    });
    result
}
fn write_synced(file: &mut File, bytes: &[u8]) -> Result<()> {
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn new_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}
fn evidence_path(report_out: &Path) -> PathBuf {
    let mut path = report_out.as_os_str().to_owned();
    path.push(".smaps");
    PathBuf::from(path)
}
fn record_observation(
    report: &mut Report,
    captured: &Captured,
    evidence: &mut File,
    path: &Path,
) -> Result<()> {
    // Preserve available raw smaps even if hardware cleanup or status failed.
    if let Some(snapshot) = &captured.smaps {
        write_synced(evidence, snapshot.as_bytes())?;
        report.smaps_snapshot = Some(SnapshotEvidence {
            path: path.to_path_buf(),
            sha256: sha256(snapshot.as_bytes()),
            byte_length: snapshot.len(),
        });
    }
    let status_result = captured
        .status
        .as_ref()
        .ok_or("status snapshot unavailable")
        .map_err(Into::into)
        .and_then(|s| parse_status(s));
    let vma_result = (|| -> Result<()> {
        let entries = parse_smaps(
            captured
                .smaps
                .as_ref()
                .ok_or("smaps snapshot unavailable")?,
        )?;
        report.smaps_vma_count = Some(entries.len());
        report.vma = Some(containing_vma(
            entries,
            report
                .mapped_range
                .as_ref()
                .ok_or("mapped range unavailable")?,
        )?);
        Ok(())
    })();
    match status_result {
        Ok(status) => report.process_status = Some(status),
        Err(status) => {
            return Err(match vma_result {
                Ok(()) => status,
                Err(vma) => format!("{status}; {vma}").into(),
            })
        }
    }
    vma_result
}

// Reserve both artifacts before constructing hardware; never overwrite evidence.
// A failed attempt has complete=false, characterization_pass=false and failure.
fn run_with_capture(
    report_out: &Path,
    capture: impl FnOnce(&mut Report, &mut Captured) -> Result<()>,
) -> Result<()> {
    let report_out = std::path::absolute(report_out)?;
    let mut report_file = new_file(&report_out)?;
    let mut report = Report::new();
    let result = (|| -> Result<()> {
        let path = evidence_path(&report_out);
        let mut evidence = new_file(&path)?;
        let mut captured = Captured::default();
        let hardware = capture(&mut report, &mut captured);
        let observation = record_observation(&mut report, &captured, &mut evidence, &path);
        match (hardware, observation) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
            (Err(hardware), Err(observation)) => Err(format!("{hardware}; {observation}").into()),
        }
    })();
    report.complete = result.is_ok();
    report.characterization_pass = result.is_ok();
    report.failure = result.as_ref().err().map(ToString::to_string);
    write_synced(&mut report_file, &serde_json::to_vec_pretty(&report)?)?;
    result
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
    #[test]
    fn proc_capture_reads_each_file_once_even_on_failure() {
        for fail in [false, true] {
            let mut calls = Vec::new();
            let mut captured = Captured::default();
            let result = capture_proc(&mut captured, |path| {
                calls.push(path.to_string());
                if path.ends_with("smaps") && fail {
                    Err(std::io::Error::other("synthetic read failure"))
                } else {
                    Ok(path.to_string())
                }
            });
            assert_eq!(calls, ["/proc/self/smaps", "/proc/self/status"]);
            assert_eq!(result.is_err(), fail);
            assert!(captured.status.is_some());
            assert_eq!(captured.smaps.is_none(), fail);
        }
    }

    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("mer-vma-test-{}-{n}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
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
    fn synthetic(report: &mut Report, captured: &mut Captured) -> Result<()> {
        report.mapped_range = Some(range());
        captured.smaps = Some(ANON.into());
        captured.status = Some("VmLck: 0 kB\nVmPin: 0 kB\nVmRSS: 8 kB\n".into());
        Ok(())
    }
    fn load(path: &Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }
    #[test]
    fn json_binds_exact_sibling_bytes_and_raw_entry() {
        let t = Temp::new();
        run_with_capture(&t.report(), synthetic).unwrap();
        let v = load(&t.report());
        assert_eq!(v["schema"], SCHEMA);
        assert_eq!(v["complete"], true);
        assert_eq!(v["characterization_pass"], true);
        assert!(v["failure"].is_null());
        assert_eq!(v["memory_field_unit"], "bytes");
        assert_eq!(v["smaps_vma_count"], 1);
        let raw = std::fs::read(evidence_path(&t.report())).unwrap();
        assert_eq!(raw, ANON.as_bytes());
        assert_eq!(v["smaps_snapshot"]["byte_length"], raw.len());
        assert_eq!(v["smaps_snapshot"]["sha256"], sha256(&raw));
        assert_eq!(
            v["smaps_snapshot"]["path"],
            evidence_path(&t.report()).to_str().unwrap()
        );
        assert_eq!(v["vma"]["raw_entry"], ANON);
    }
    #[test]
    fn existing_artifacts_fail_before_capture_and_never_overwrite() {
        for existing_smaps in [false, true] {
            let t = Temp::new();
            let path = if existing_smaps {
                evidence_path(&t.report())
            } else {
                t.report()
            };
            std::fs::write(&path, b"preserve").unwrap();
            let result = run_with_capture(&t.report(), |_, _| panic!("capture must not run"));
            assert!(result.is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"preserve");
            if existing_smaps {
                let v = load(&t.report());
                assert_eq!(v["complete"], false);
                assert_eq!(v["characterization_pass"], false);
                assert!(v["failure"].is_string());
            }
        }
    }
    #[cfg(unix)]
    #[test]
    fn symlink_artifacts_are_not_followed() {
        for link_smaps in [false, true] {
            let t = Temp::new();
            let target = t.0.join("target");
            std::fs::write(&target, b"preserve").unwrap();
            let path = if link_smaps {
                evidence_path(&t.report())
            } else {
                t.report()
            };
            std::os::unix::fs::symlink(&target, path).unwrap();
            assert!(run_with_capture(&t.report(), |_, _| panic!("capture must not run")).is_err());
            assert_eq!(std::fs::read(target).unwrap(), b"preserve");
        }
    }
    #[test]
    fn capture_and_parse_failures_preserve_raw_evidence_and_fail_closed() {
        for kind in ["validation", "parse", "status"] {
            let t = Temp::new();
            let result = run_with_capture(&t.report(), |r, c| {
                synthetic(r, c)?;
                match kind {
                    "validation" => Err("synthetic WGPU validation failure".into()),
                    "parse" => {
                        c.smaps = Some(format!("{ANON}malformed\n"));
                        Ok(())
                    }
                    _ => {
                        c.status = Some("VmLck: 0 kB\n".into());
                        Ok(())
                    }
                }
            });
            assert!(result.is_err());
            let v = load(&t.report());
            assert_eq!(v["complete"], false);
            assert_eq!(v["characterization_pass"], false);
            assert!(v["failure"].is_string());
            let raw = std::fs::read(evidence_path(&t.report())).unwrap();
            assert_eq!(v["smaps_snapshot"]["sha256"], sha256(&raw));
            if kind != "parse" {
                assert!(v["vma"].is_object());
            }
        }
    }
    #[test]
    fn cli_requires_only_report_and_has_no_startup_config() {
        use clap::Parser;
        let args = [
            "mer",
            "probe-gpu-native-mapped-destination-vma",
            "--report-out",
            "vma.json",
        ];
        let cli = crate::Cli::try_parse_from(args).unwrap();
        assert!(crate::startup_config_path(&cli.cmd).is_none());
        assert!(matches!(
            cli.cmd,
            crate::Cmd::ProbeGpuNativeMappedDestinationVma { .. }
        ));
        assert!(crate::Cli::try_parse_from(&args[..2]).is_err());
        assert!(
            crate::Cli::try_parse_from(args.into_iter().chain(["--config", "model.toml"])).is_err()
        );
    }
    #[test]
    fn diagnostic_source_boundary_and_cleanup_order() {
        let source = include_str!("gpu_native_mapped_vma.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "LockGuard",
            "gpu_native_mapped_lock",
            "mlock",
            "munlock",
            "madvise",
            "mbind",
            "move_pages",
            "set_mempolicy",
            "io_uring",
            "queue.submit",
            "create_command_encoder",
            "copy_buffer",
            "NvmeStorage",
            "RealModel",
            "Config::",
            "Runtime::",
            "read_expert",
        ] {
            assert!(
                !source.contains(forbidden),
                "forbidden dependency {forbidden}"
            );
        }
        assert_eq!(source.matches("device.create_buffer(").count(), 1);
        assert_eq!(source.matches("get_mapped_range_mut()").count(), 1);
        assert_eq!(source.matches("read(\"/proc/self/smaps\")").count(), 1);
        assert_eq!(source.matches("read(\"/proc/self/status\")").count(), 1);
        let mut position = source.find("fn capture_hardware(").unwrap();
        for step in [
            "capture_proc(captured",
            "drop(view)",
            "buffer.unmap()",
            "device.poll(wgpu::Maintain::Wait)",
            "device.pop_error_scope()",
        ] {
            position += source[position..].find(step).unwrap();
        }
        let main = include_str!("main.rs").split("fn main()").nth(1).unwrap();
        assert!(
            main.find("return crate::gpu_native_mapped_vma::probe_command")
                .unwrap()
                < main.find("init_logging(").unwrap()
        );
    }
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_fails_before_adapter_construction() {
        let t = Temp::new();
        assert!(probe_command(&t.report()).is_err());
        let v = load(&t.report());
        assert_eq!(v["complete"], false);
        assert!(v["failure"].as_str().unwrap().contains("requires Linux"));
        for field in [
            "hardware",
            "mapped_range",
            "vma",
            "smaps_snapshot",
            "cleanup",
        ] {
            assert!(v[field].is_null());
        }
    }
}
