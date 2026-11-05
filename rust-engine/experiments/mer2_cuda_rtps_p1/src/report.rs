//! Bounded, line-oriented prospective report. No JSON/dependency required.
//! Source manifest is a byte inventory, not a run permit or binary attestation.
use crate::{expected, reference, BASELINE_COMMIT, FIXTURE_ID, NATIVE_STOP,
    STAGE_LENGTHS, STAGE_NAMES, W3_TREE, WORKSPACE_ID};
use std::io::{self, Write};

pub const SOURCE_MANIFEST: &str = include_str!("../SOURCE-PAYLOAD.sha256");

#[derive(Debug, PartialEq, Eq)]
pub struct TerminalReport {
    pub first_failure: &'static str,
    pub attempted_invocations: u8,
    pub native_initializations: u8,
    pub kernels_submitted: u8,
    pub terminal: bool,
    pub cleanup: &'static str,
    pub context_destruction: &'static str,
}
impl TerminalReport {
    pub(crate) fn native_unbound() -> Self {
        Self { first_failure: NATIVE_STOP, attempted_invocations: 0,
            native_initializations: 0, kernels_submitted: 0, terminal: true,
            cleanup: "NO_CUDA_INITIALIZATION_ATTEMPTED",
            context_destruction: "NOT_APPLICABLE_NO_CONTEXT_CREATED" }
    }
    pub fn write<W: Write>(&self, output: &mut W) -> io::Result<()> {
        identity(output)?;
        writeln!(output, "terminal=STOP\nfirst_failure={}\nsecondary_failures=none", self.first_failure)?;
        writeln!(output, "attempted_invocations={}\nnative_initializations={}\nkernels_submitted={}",
            self.attempted_invocations, self.native_initializations, self.kernels_submitted)?;
        writeln!(output, "cleanup={}\ncontext_destruction={}\nresource_leak_freedom=NOT_ASSERTED",
            self.cleanup, self.context_destruction)?;
        writeln!(output, "gpu_output=ABSENT\ncpu_fallback=false\nrun_authority=NONE")
    }
}

pub fn identity<W: Write>(output: &mut W) -> io::Result<()> {
    writeln!(output, "schema=RTPS_P1_PARTIAL_V1\nworkspace={}\nfixture={}\nbaseline={}\nw3_reference_tree={}",
        WORKSPACE_ID, FIXTURE_ID, BASELINE_COMMIT, W3_TREE)?;
    writeln!(output, "native_backend=UNBOUND\nptx_hashes=UNBOUND\ncompiled_binary_hash=EXTERNAL_REQUIRED")?;
    for line in SOURCE_MANIFEST.lines() { writeln!(output, "source={line}")?; }
    Ok(())
}

/// Print all 5120 stage elements, even on numerical mismatch. This path is
/// explicitly CPU_REFERENCE_ONLY and is never called from native_once().
/// At most 4 MiB of report output is proposed; the host must enforce that cap.
pub fn write_reference<W: Write>(output: &mut W, stages: &reference::Stages) -> io::Result<bool> {
    identity(output)?;
    writeln!(output, "kind=CPU_REFERENCE_ONLY\ngpu_output=ABSENT\nacceptance=EXACT_BITS_V1")?;
    let mut first_mismatch = None;
    for (stage, values) in stages.values.iter().enumerate() {
        reference::finite(values, STAGE_LENGTHS[stage])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}")))?;
        for (index, value) in values.iter().enumerate() {
            let delta = expected::difference(stage, index, *value)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}")))?;
            if !delta.exact && first_mismatch.is_none() { first_mismatch = Some((stage, index)); }
            writeln!(output,
                "stage={} index={} expected=0x{:08x} actual=0x{:08x} signed_delta={:.17e} abs_delta={:.17e} exact={}",
                STAGE_NAMES[stage], index, delta.expected_bits, delta.actual_bits,
                delta.signed_delta, delta.absolute_delta, delta.exact)?;
        }
    }
    match first_mismatch {
        Some((stage, index)) => writeln!(output,
            "terminal=STOP_CPU_ORACLE_BITS\nfirst_failure_stage={}\nfirst_failure_index={}", STAGE_NAMES[stage], index)?,
        None => writeln!(output, "terminal=CPU_REFERENCE_BITS_RECORDED\nfirst_failure=none")?,
    }
    writeln!(output, "native_initializations=0\ncleanup=NO_CUDA_INITIALIZATION_ATTEMPTED\nhardware_result=NONE")?;
    Ok(first_mismatch.is_none())
}
