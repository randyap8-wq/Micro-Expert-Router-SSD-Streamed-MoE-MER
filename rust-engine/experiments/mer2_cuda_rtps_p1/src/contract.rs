//! Pure rejection logic for a prospective native adapter's receipts.
//! This neither creates a CUDA owner nor authenticates caller-supplied events.
//! No successful transition here is an execution permit or hardware evidence.
use crate::{Reject, Result};
use std::{marker::PhantomData, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation { Upload, Gate, Up, Silu, Multiply, Down, Readback }
pub const ORDER: [Operation; 7] = [Operation::Upload, Operation::Gate, Operation::Up,
    Operation::Silu, Operation::Multiply, Operation::Down, Operation::Readback];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerTag { pub context: u64, pub request: u64, pub expert: u32 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cleanup {
    NotAttempted,
    NoInitialization,
    CheckedDrainAndRelease,
    Quarantined,
}

/// Non-Send/non-Sync, non-Clone state. This is a receipt checker, not an owner.
/// At most one begin, seven ordered enqueues/fences, and one finish attempt.
pub struct TraceContract {
    owner: OwnerTag,
    started: bool,
    next: usize,
    pending: Option<Operation>,
    first: Option<Reject>,
    secondary: [Option<Reject>; 8],
    secondary_len: usize,
    secondary_overflow: bool,
    terminal: bool,
    cleanup: Cleanup,
    _single_thread: PhantomData<Rc<()>>,
}

impl TraceContract {
    pub fn new(owner: OwnerTag) -> Result<Self> {
        if owner.context == 0 || owner.request != 1 || owner.expert != 0 {
            return Err(Reject::Owner);
        }
        Ok(Self { owner, started: false, next: 0, pending: None, first: None,
            secondary: [None; 8], secondary_len: 0, secondary_overflow: false,
            terminal: false, cleanup: Cleanup::NotAttempted, _single_thread: PhantomData })
    }
    fn fail(&mut self, reason: Reject) -> Reject {
        if self.first.is_none() { self.first = Some(reason); }
        else if self.secondary_len < self.secondary.len() {
            self.secondary[self.secondary_len] = Some(reason);
            self.secondary_len += 1;
        } else { self.secondary_overflow = true; }
        reason
    }
    fn active(&mut self, owner: OwnerTag) -> Result<()> {
        if self.terminal { return Err(self.fail(Reject::Terminal)); }
        if let Some(first) = self.first { return Err(first); }
        if owner != self.owner { return Err(self.fail(Reject::Owner)); }
        if !self.started { return Err(self.fail(Reject::MissingResource)); }
        Ok(())
    }
    pub fn begin(&mut self, owner: OwnerTag) -> Result<()> {
        if self.terminal { return Err(self.fail(Reject::Terminal)); }
        if let Some(first) = self.first { return Err(first); }
        if owner != self.owner { return Err(self.fail(Reject::Owner)); }
        if self.started { return Err(self.fail(Reject::RepeatedInvocation)); }
        // Latch before any prospective driver lookup/constructor.
        self.started = true;
        Ok(())
    }
    pub fn unavailable(&mut self, owner: OwnerTag, reason: Reject) -> Result<()> {
        self.active(owner)?;
        if !matches!(reason, Reject::MissingDriver | Reject::MissingResource) {
            return Err(self.fail(Reject::StageOrder));
        }
        Err(self.fail(reason))
    }
    pub fn enqueue(&mut self, owner: OwnerTag, operation: Operation) -> Result<()> {
        self.active(owner)?;
        if self.pending.is_some() || ORDER.get(self.next) != Some(&operation) {
            return Err(self.fail(Reject::StageOrder));
        }
        self.pending = Some(operation); // Before the fallible enqueue, never after.
        Ok(())
    }
    pub fn enqueue_failed(&mut self, owner: OwnerTag) -> Result<()> {
        self.active(owner)?;
        if self.pending.is_none() { return Err(self.fail(Reject::StageOrder)); }
        Err(self.fail(Reject::Enqueue))
    }
    pub fn fenced(&mut self, owner: OwnerTag, operation: Operation, checked: bool) -> Result<()> {
        self.active(owner)?;
        if self.pending != Some(operation) { return Err(self.fail(Reject::StageOrder)); }
        if !checked { return Err(self.fail(Reject::Fence)); }
        self.pending = None;
        self.next += 1;
        Ok(())
    }
    pub fn readback_failed(&mut self, owner: OwnerTag) -> Result<()> {
        self.active(owner)?;
        if self.pending != Some(Operation::Readback) { return Err(self.fail(Reject::StageOrder)); }
        Err(self.fail(Reject::Readback))
    }
    /// Must remain one shot even when cleanup fails. `checked` is only a future
    /// receipt input; no caller-supplied Boolean can prove native drain/release.
    pub fn finish(&mut self, checked: bool) -> Result<()> {
        if self.terminal { return Err(self.fail(Reject::CleanupRepeated)); }
        self.terminal = true;
        if !self.started {
            self.cleanup = Cleanup::NoInitialization;
            return self.first.map_or(Ok(()), Err);
        }
        if !checked || self.pending.is_some() {
            self.cleanup = Cleanup::Quarantined;
            return Err(self.fail(Reject::CleanupAmbiguous));
        }
        if self.first.is_none() && self.next != ORDER.len() {
            self.cleanup = Cleanup::Quarantined;
            return Err(self.fail(Reject::StageOrder));
        }
        self.cleanup = Cleanup::CheckedDrainAndRelease;
        self.first.map_or(Ok(()), Err)
    }
    pub fn first_failure(&self) -> Option<Reject> { self.first }
    pub fn secondary_failures(&self) -> &[Option<Reject>] { &self.secondary[..self.secondary_len] }
    pub fn secondary_overflow(&self) -> bool { self.secondary_overflow }
    pub fn cleanup(&self) -> Cleanup { self.cleanup }
    pub fn pending(&self) -> Option<Operation> { self.pending }
    pub fn terminal(&self) -> bool { self.terminal }
}
