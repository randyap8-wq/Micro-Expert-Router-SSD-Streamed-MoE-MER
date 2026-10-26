"""A3-S0: pure Python CPU reference contract for an isolated CUDA worker supervisor.

This is a MODEL, not an OS supervisor, GPU adapter, process manager or MER integration.
Every external event is supplied as a value; no process, driver, I/O or network calls.
The model deliberately cannot claim that an in-process CUDA context Drop succeeded.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Optional


class Phase(str, Enum):
    UNSPAWNED = "UNSPAWNED"
    SPAWNING = "SPAWNING"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    IN_FLIGHT = "IN_FLIGHT"
    DRAINING = "DRAINING"
    QUARANTINED = "QUARANTINED"
    TERMINATING = "TERMINATING"
    EXITED = "EXITED_OBSERVED"
    EXIT_UNCONFIRMED = "EXIT_UNCONFIRMED"


class Owner(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    UNRETURNED_POSSIBLE = "UNRETURNED_OWNER_POSSIBLE"
    RETURNED_UNCHECKED = "RETURNED_UNCHECKED"
    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"


class Work(str, Enum):
    NONE = "NONE"
    SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"
    SUBMITTED = "SUBMITTED"
    FENCED = "FENCED"


class Cleanup(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    DRAIN_CHECKED = "DRAIN_CHECKED"
    RELEASE_CHECKED = "RELEASE_CHECKED_SCOPED"
    UNCONFIRMED = "UNCONFIRMED"


class Delivery(str, Enum):
    NONE = "NONE"
    PREPARED = "PREPARED_LOCAL_ONLY"
    SENT_UNACKNOWLEDGED = "SENT_UNACKNOWLEDGED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    AMBIGUOUS = "AMBIGUOUS_DO_NOT_REPLAY"


class ContractStop(Exception):
    """Binding or lifecycle violation. The worker is terminally quarantined."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_SHA = re.compile(r"[a-f0-9]{64}\Z")


def _digest(value: str) -> bool:
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


@dataclass(frozen=True)
class Ticket:
    request_id: str
    generation: int
    nonce: int
    source_sha256: str
    model_sha256: str
    kv_sha256: str


@dataclass
class Ledger:
    """Supervisor-owned request tombstones must survive worker generation changes.

    There is deliberately NO automatic retry/replay operation in this API.
    """

    ever_admitted: set[str] = field(default_factory=set)


@dataclass
class Supervisor:
    """One isolated worker, one admitted request/session, fail-closed CPU model.

    Construct a NEW Supervisor for a NEW generation. Never reset a failed one.
    """

    generation: int
    source_sha256: str
    model_sha256: str
    ledger: Ledger
    phase: Phase = Phase.UNSPAWNED
    pid: Optional[int] = None
    owner: Owner = Owner.NOT_ATTEMPTED
    work: Work = Work.NONE
    cleanup: Cleanup = Cleanup.NOT_ATTEMPTED
    delivery: Delivery = Delivery.NONE
    current: Optional[Ticket] = None
    result_sha256: Optional[str] = None
    first_error: Optional[str] = None
    secondary_errors: list[str] = field(default_factory=list)
    process_exit_observed: bool = False
    post_exit_baseline_observed: bool = False
    cleanup_attempts: int = 0
    unknown_gpu_ownership: bool = False

    def __post_init__(self) -> None:
        if (type(self.generation) is not int or self.generation <= 0
                or not _digest(self.source_sha256) or not _digest(self.model_sha256)
                or not isinstance(self.ledger, Ledger)):
            raise ValueError("invalid initial immutable supervisor identity")

    def _stop(self, code: str) -> None:
        if self.first_error is None:
            self.first_error = code
        else:
            self.secondary_errors.append(code)
        if self.phase not in (Phase.EXITED, Phase.EXIT_UNCONFIRMED):
            self.phase = Phase.QUARANTINED
        if self.owner in (Owner.UNRETURNED_POSSIBLE, Owner.RETURNED_UNCHECKED):
            self.owner = Owner.UNCERTAIN
            self.unknown_gpu_ownership = True
        if self.cleanup != Cleanup.RELEASE_CHECKED:
            self.cleanup = Cleanup.UNCONFIRMED
        if self.current is not None and self.delivery != Delivery.ACKNOWLEDGED:
            self.delivery = Delivery.AMBIGUOUS
        raise ContractStop(code)

    def _require(self, ok: bool, code: str) -> None:
        if not ok:
            self._stop(code)

    def spawn(self, pid: int) -> None:
        self._require(self.phase == Phase.UNSPAWNED and type(pid) is int and pid > 0,
                      "STOP_SPAWN_IDENTITY")
        self.pid = pid
        self.phase = Phase.SPAWNING

    def begin_initialization(self) -> None:
        self._require(self.phase == Phase.SPAWNING, "STOP_INITIALIZATION_ORDER")
        self.phase = Phase.INITIALIZING

    def context_constructor_entered(self) -> None:
        self._require(self.phase == Phase.INITIALIZING and self.owner == Owner.NOT_ATTEMPTED,
                      "STOP_CONSTRUCTOR_ORDER")
        # Set before the fallible API. May include retained but unreturned primary context.
        self.owner = Owner.UNRETURNED_POSSIBLE
        self.unknown_gpu_ownership = True

    def typed_context_returned(self) -> None:
        self._require(self.phase == Phase.INITIALIZING
                      and self.owner == Owner.UNRETURNED_POSSIBLE,
                      "STOP_CONTEXT_OWNER_ORDER")
        self.owner = Owner.RETURNED_UNCHECKED
        # A deferred error may still invalidate the typed ownership guarantee.

    def deferred_context_check_passed(self) -> None:
        self._require(self.phase == Phase.INITIALIZING
                      and self.owner == Owner.RETURNED_UNCHECKED,
                      "STOP_CONTEXT_CHECK_ORDER")
        self.owner = Owner.CONFIRMED
        self.unknown_gpu_ownership = False

    def admission_ready(self) -> None:
        self._require(self.phase == Phase.INITIALIZING and self.owner == Owner.CONFIRMED,
                      "STOP_READY_WITH_UNCERTAIN_OWNER")
        self.phase = Phase.READY

    def admit(self, ticket: Ticket) -> None:
        valid = (isinstance(ticket, Ticket) and self.phase == Phase.READY
                 and self.first_error is None and self.current is None
                 and bool(ticket.request_id) and len(ticket.request_id) <= 128
                 and ticket.generation == self.generation
                 and type(ticket.nonce) is int and ticket.nonce > 0
                 and ticket.source_sha256 == self.source_sha256
                 and ticket.model_sha256 == self.model_sha256
                 and _digest(ticket.kv_sha256)
                 and ticket.request_id not in self.ledger.ever_admitted)
        self._require(valid, "STOP_ADMISSION_REPLAY_OR_IDENTITY")
        self.ledger.ever_admitted.add(ticket.request_id)
        self.current = ticket
        self.phase = Phase.IN_FLIGHT

    def submission_entered(self, ticket: Ticket) -> None:
        self._require(self.phase == Phase.IN_FLIGHT and self.current == ticket
                      and self.work == Work.NONE, "STOP_SUBMISSION_ORDER")
        # Even if enqueue returns Err, GPU work could already have escaped.
        self.work = Work.SUBMISSION_UNCERTAIN

    def submission_confirmed(self, ticket: Ticket) -> None:
        self._require(self.phase == Phase.IN_FLIGHT and self.current == ticket
                      and self.work == Work.SUBMISSION_UNCERTAIN,
                      "STOP_SUBMISSION_CONFIRMATION")
        self.work = Work.SUBMITTED

    def fence_confirmed(self, ticket: Ticket) -> None:
        self._require(self.phase == Phase.IN_FLIGHT and self.current == ticket
                      and self.work == Work.SUBMITTED,
                      "STOP_FENCE_CONFIRMATION")
        self.work = Work.FENCED

    def prepare_output(self, ticket: Ticket, output_sha256: str) -> None:
        self._require(self.phase == Phase.IN_FLIGHT and self.current == ticket
                      and self.work == Work.FENCED and self.delivery == Delivery.NONE
                      and _digest(output_sha256), "STOP_OUTPUT_UNCOMMITTED")
        self.result_sha256 = output_sha256
        self.delivery = Delivery.PREPARED

    def begin_delivery(self, ticket: Ticket, output_sha256: str) -> None:
        self._require(self.current == ticket and self.delivery == Delivery.PREPARED
                      and self.result_sha256 == output_sha256,
                      "STOP_DELIVERY_COMMITMENT")
        self.delivery = Delivery.SENT_UNACKNOWLEDGED

    def acknowledge_delivery(self, ticket: Ticket, output_sha256: str) -> None:
        self._require(self.current == ticket
                      and self.delivery == Delivery.SENT_UNACKNOWLEDGED
                      and self.result_sha256 == output_sha256,
                      "STOP_DELIVERY_ACK")
        self.delivery = Delivery.ACKNOWLEDGED
        # Mode A: retire after one bounded request/session, never pool this owner.
        self.phase = Phase.DRAINING

    def checked_drain(self) -> None:
        self.cleanup_attempts += 1
        self._require(self.phase == Phase.DRAINING and self.owner == Owner.CONFIRMED
                      and not self.unknown_gpu_ownership and self.work == Work.FENCED
                      and self.delivery == Delivery.ACKNOWLEDGED
                      and self.cleanup == Cleanup.NOT_ATTEMPTED
                      and self.cleanup_attempts == 1,
                      "STOP_DRAIN_UNCONFIRMED")
        self.cleanup = Cleanup.DRAIN_CHECKED

    def checked_release(self) -> None:
        self.cleanup_attempts += 1
        self._require(self.phase == Phase.DRAINING and self.cleanup == Cleanup.DRAIN_CHECKED
                      and self.cleanup_attempts == 2 and self.owner == Owner.CONFIRMED,
                      "STOP_RELEASE_UNCONFIRMED")
        self.cleanup = Cleanup.RELEASE_CHECKED
        self.phase = Phase.TERMINATING

    def fault(self, code: str) -> None:
        if not isinstance(code, str) or not code.startswith("STOP_"):
            self._stop("STOP_INVALID_FAULT_LABEL")
        self._stop(code)

    def terminate_quarantined(self) -> None:
        self._require(self.phase == Phase.QUARANTINED, "STOP_TERMINATION_ORDER")
        # Termination is a supervisor signal, not driver cleanup proof.
        self.phase = Phase.TERMINATING

    def observe_exit(self, exited_pid: int, generation: int) -> None:
        self._require(self.phase == Phase.TERMINATING and type(exited_pid) is int
                      and exited_pid == self.pid and generation == self.generation,
                      "STOP_EXIT_IDENTITY")
        self.process_exit_observed = True
        self.phase = Phase.EXITED

    def exit_unconfirmed(self) -> None:
        self._require(self.phase == Phase.TERMINATING, "STOP_EXIT_UNCONFIRMED_ORDER")
        self.phase = Phase.EXIT_UNCONFIRMED

    def observe_post_exit_baseline(self, *, samples: tuple[tuple[int, int], ...],
                                   baseline_mib: int) -> None:
        """Verify eight external post-exit (memory MiB, compute PID count) samples.

        All samples must be process-free; the final four must return to baseline
        within 2 MiB. Earlier memory samples may contain normal settling.
        Nothing here establishes the final in-process context destructor.
        """
        sample_shape = (isinstance(samples, tuple) and len(samples) == 8
                        and all(isinstance(s, tuple) and len(s) == 2
                                and type(s[0]) is int and s[0] >= 0
                                and type(s[1]) is int and s[1] == 0
                                for s in samples))
        final_stable = (sample_shape and type(baseline_mib) is int
                        and baseline_mib >= 0
                        and all(abs(used - baseline_mib) <= 2
                                for used, _ in samples[-4:]))
        self._require(self.phase == Phase.EXITED and self.process_exit_observed
                      and not self.post_exit_baseline_observed and final_stable,
                      "STOP_POST_EXIT_OBSERVATION")
        self.post_exit_baseline_observed = True

    def receipt(self) -> dict:
        return {
            "generation": self.generation,
            "worker_pid": self.pid,
            "phase": self.phase.value,
            "first_error": self.first_error,
            "secondary_errors": list(self.secondary_errors),
            "owner_state": self.owner.value,
            "unknown_gpu_ownership": self.unknown_gpu_ownership,
            "gpu_work_state": self.work.value,
            "cleanup": self.cleanup.value,
            "delivery": self.delivery.value,
            "request_id": self.current.request_id if self.current else None,
            "process_exit_observed": self.process_exit_observed,
            "post_exit_baseline_observed": self.post_exit_baseline_observed,
            "last_cuda_context_drop": "UNOBSERVABLE",
            "leak_free": "NOT_CLAIMED",
            "automatic_retry": "FORBIDDEN",
        }
