"""Deterministic CPU-only A3-S0 contract tests. No subprocess, CUDA, or I/O."""
import unittest

from supervisor_contract import (Cleanup, ContractStop, Delivery, Ledger,
                                 Owner, Phase, Supervisor, Ticket, Work)

SRC = 'a' * 64
MODEL = 'b' * 64
KV = 'c' * 64
OUT = 'd' * 64


def fresh(generation=1, ledger=None, pid=1234):
    worker = Supervisor(generation, SRC, MODEL, ledger if ledger is not None else Ledger())
    worker.spawn(pid)
    worker.begin_initialization()
    worker.context_constructor_entered()
    worker.typed_context_returned()
    worker.deferred_context_check_passed()
    worker.admission_ready()
    return worker


def ticket(generation=1, request='req-1', nonce=10, kv=KV, source=SRC, model=MODEL):
    return Ticket(request, generation, nonce, source, model, kv)


def in_flight():
    w = fresh()
    t = ticket()
    w.admit(t)
    return w, t


def fenced():
    w, t = in_flight()
    w.submission_entered(t)
    w.submission_confirmed(t)
    w.fence_confirmed(t)
    return w, t


def delivered():
    w, t = fenced()
    w.prepare_output(t, OUT)
    w.begin_delivery(t, OUT)
    w.acknowledge_delivery(t, OUT)
    return w, t


class ContractTests(unittest.TestCase):
    def assert_stop(self, code, fn, worker):
        first_before = worker.first_error
        phase_before = worker.phase
        with self.assertRaises(ContractStop) as ctx:
            fn()
        self.assertEqual(ctx.exception.code, code)
        # A subsequent attempted illegal operation cannot replace the first error.
        self.assertEqual(worker.first_error, first_before or code)
        # An already-observed process exit cannot be undone by a later bad sample.
        expected_phase = phase_before if phase_before in (Phase.EXITED, Phase.EXIT_UNCONFIRMED) else Phase.QUARANTINED
        self.assertEqual(worker.phase, expected_phase)
        self.assertNotEqual(worker.receipt()['leak_free'], 'TRUE')

    def test_01_clean_one_session_retirement_and_observations(self):
        w, _ = delivered()
        w.checked_drain()
        w.checked_release()
        self.assertEqual(w.cleanup, Cleanup.RELEASE_CHECKED)
        self.assertEqual(w.phase, Phase.TERMINATING)
        w.observe_exit(1234, 1)
        w.observe_post_exit_baseline(samples=((0, 0),) * 8, baseline_mib=0)
        r = w.receipt()
        self.assertEqual(r['phase'], 'EXITED_OBSERVED')
        self.assertTrue(r['post_exit_baseline_observed'])
        self.assertEqual(r['last_cuda_context_drop'], 'UNOBSERVABLE')
        self.assertEqual(r['leak_free'], 'NOT_CLAIMED')

    def test_02_init_without_owner_cannot_be_ready(self):
        w = Supervisor(1, SRC, MODEL, Ledger())
        w.spawn(123)
        w.begin_initialization()
        self.assert_stop('STOP_READY_WITH_UNCERTAIN_OWNER', w.admission_ready, w)

    def test_03_constructor_unreturned_retain_must_quarantine(self):
        w = Supervisor(1, SRC, MODEL, Ledger())
        w.spawn(123)
        w.begin_initialization()
        w.context_constructor_entered()
        self.assert_stop('STOP_CUDA_CONSTRUCTOR', lambda: w.fault('STOP_CUDA_CONSTRUCTOR'), w)
        self.assertTrue(w.unknown_gpu_ownership)
        self.assertEqual(w.owner, Owner.UNCERTAIN)
        self.assertEqual(w.cleanup, Cleanup.UNCONFIRMED)

    def test_04_returned_unchecked_context_failure_must_quarantine(self):
        w = Supervisor(1, SRC, MODEL, Ledger())
        w.spawn(123)
        w.begin_initialization()
        w.context_constructor_entered()
        w.typed_context_returned()
        self.assert_stop('STOP_DEFERRED', lambda: w.fault('STOP_DEFERRED'), w)
        self.assertEqual(w.owner, Owner.UNCERTAIN)

    def test_05_duplicate_constructor_call_must_quarantine(self):
        w = Supervisor(1, SRC, MODEL, Ledger())
        w.spawn(123); w.begin_initialization(); w.context_constructor_entered()
        self.assert_stop('STOP_CONSTRUCTOR_ORDER', w.context_constructor_entered, w)

    def test_06_deferred_check_without_returned_owner_must_quarantine(self):
        w = Supervisor(1, SRC, MODEL, Ledger())
        w.spawn(123); w.begin_initialization(); w.context_constructor_entered()
        self.assert_stop('STOP_CONTEXT_CHECK_ORDER', w.deferred_context_check_passed, w)

    def test_07_context_owner_return_after_ready_is_forbidden(self):
        w = fresh()
        self.assert_stop('STOP_CONTEXT_OWNER_ORDER', w.typed_context_returned, w)

    def test_08_stale_generation_ticket_quarantined(self):
        w = fresh(generation=2)
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY', lambda: w.admit(ticket()), w)

    def test_09_foreign_source_ticket_quarantined(self):
        w = fresh()
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: w.admit(ticket(source='e' * 64)), w)

    def test_10_foreign_model_ticket_quarantined(self):
        w = fresh()
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: w.admit(ticket(model='e' * 64)), w)

    def test_11_bad_kv_hash_ticket_quarantined(self):
        w = fresh()
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: w.admit(ticket(kv='notsha')), w)

    def test_12_invalid_nonce_quarantined(self):
        w = fresh()
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: w.admit(ticket(nonce=0)), w)

    def test_13_concurrent_admission_forbidden(self):
        w, _ = in_flight()
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: w.admit(ticket(request='req-2')), w)

    def test_14_duplicate_request_id_survives_new_process_generation(self):
        ledger = Ledger()
        first = fresh(generation=1, ledger=ledger)
        first.admit(ticket())
        second = fresh(generation=2, ledger=ledger, pid=456)
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: second.admit(ticket(generation=2, nonce=11)), second)
        self.assertIn('req-1', ledger.ever_admitted)

    def test_15_bad_worker_initial_identity_precluded(self):
        with self.assertRaises(ValueError):
            Supervisor(0, SRC, MODEL, Ledger())
        with self.assertRaises(ValueError):
            Supervisor(1, SRC[:-1], MODEL, Ledger())
        with self.assertRaises(ValueError):
            Supervisor(1, SRC, MODEL, None)

    def test_16_submit_without_active_request_quarantined(self):
        w = fresh()
        self.assert_stop('STOP_SUBMISSION_ORDER', lambda: w.submission_entered(ticket()), w)

    def test_17_submitted_work_without_fence_cannot_return_output(self):
        w, t = in_flight()
        w.submission_entered(t)
        w.submission_confirmed(t)
        self.assert_stop('STOP_OUTPUT_UNCOMMITTED', lambda: w.prepare_output(t, OUT), w)

    def test_18_submission_enqueued_uncertain_fault_requires_quarantine(self):
        w, t = in_flight()
        w.submission_entered(t)
        self.assert_stop('STOP_ENQUEUE_AMBIGUOUS', lambda: w.fault('STOP_ENQUEUE_AMBIGUOUS'), w)
        self.assertEqual(w.work, Work.SUBMISSION_UNCERTAIN)
        self.assertEqual(w.delivery, Delivery.AMBIGUOUS)

    def test_19_foreign_fence_ticket_forbidden(self):
        w, t = in_flight()
        w.submission_entered(t)
        w.submission_confirmed(t)
        self.assert_stop('STOP_FENCE_CONFIRMATION',
                         lambda: w.fence_confirmed(ticket(nonce=t.nonce + 1)), w)

    def test_20_duplicate_fence_forbidden(self):
        w, t = fenced()
        self.assert_stop('STOP_FENCE_CONFIRMATION', lambda: w.fence_confirmed(t), w)

    def test_21_corrupt_output_digest_forbidden(self):
        w, t = fenced()
        self.assert_stop('STOP_OUTPUT_UNCOMMITTED', lambda: w.prepare_output(t, 'wrong'), w)

    def test_22_delivering_wrong_output_digest_forbidden(self):
        w, t = fenced()
        w.prepare_output(t, OUT)
        self.assert_stop('STOP_DELIVERY_COMMITMENT',
                         lambda: w.begin_delivery(t, 'e' * 64), w)

    def test_23_ack_before_delivery_forbidden(self):
        w, t = fenced()
        w.prepare_output(t, OUT)
        self.assert_stop('STOP_DELIVERY_ACK', lambda: w.acknowledge_delivery(t, OUT), w)

    def test_24_sent_unacknowledged_response_must_remain_ambiguous_on_error(self):
        w, t = fenced()
        w.prepare_output(t, OUT)
        w.begin_delivery(t, OUT)
        self.assert_stop('STOP_CLIENT_DISCONNECT', lambda: w.fault('STOP_CLIENT_DISCONNECT'), w)
        self.assertEqual(w.delivery, Delivery.AMBIGUOUS)
        self.assertEqual(w.receipt()['automatic_retry'], 'FORBIDDEN')

    def test_25_clean_ack_only_moves_to_draining(self):
        w, _ = delivered()
        self.assertEqual(w.phase, Phase.DRAINING)
        self.assertFalse(w.process_exit_observed)
        self.assertEqual(w.cleanup, Cleanup.NOT_ATTEMPTED)

    def test_26_first_error_preserved_and_secondary_added(self):
        w, _ = in_flight()
        self.assert_stop('STOP_FIRST', lambda: w.fault('STOP_FIRST'), w)
        with self.assertRaises(ContractStop):
            w.fault('STOP_SECOND')
        self.assertEqual(w.first_error, 'STOP_FIRST')
        self.assertEqual(w.secondary_errors, ['STOP_SECOND'])

    def test_27_no_drain_without_acked_output(self):
        w, _ = fenced()
        self.assert_stop('STOP_DRAIN_UNCONFIRMED', w.checked_drain, w)
        self.assertEqual(w.cleanup_attempts, 1)

    def test_28_no_release_before_drain(self):
        w, _ = delivered()
        self.assert_stop('STOP_RELEASE_UNCONFIRMED', w.checked_release, w)

    def test_29_exactly_one_drain_and_release(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release()
        self.assertEqual(w.cleanup_attempts, 2)
        self.assert_stop('STOP_RELEASE_UNCONFIRMED', w.checked_release, w)

    def test_30_no_second_drain(self):
        w, _ = delivered()
        w.checked_drain()
        self.assert_stop('STOP_DRAIN_UNCONFIRMED', w.checked_drain, w)

    def test_31_quarantined_worker_must_never_readmit(self):
        w, _ = in_flight()
        self.assert_stop('STOP_GPU_UNKNOWN', lambda: w.fault('STOP_GPU_UNKNOWN'), w)
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: w.admit(ticket(request='another')), w)

    def test_32_quarantined_worker_can_be_terminated_and_exit_observed(self):
        w = fresh()
        self.assert_stop('STOP_RESOURCE', lambda: w.fault('STOP_RESOURCE'), w)
        w.terminate_quarantined()
        w.observe_exit(1234, 1)
        self.assertEqual(w.first_error, 'STOP_RESOURCE')
        self.assertEqual(w.phase, Phase.EXITED)
        self.assertNotEqual(w.cleanup, Cleanup.RELEASE_CHECKED)

    def test_33_termination_directly_from_ready_forbidden(self):
        w = fresh()
        self.assert_stop('STOP_TERMINATION_ORDER', w.terminate_quarantined, w)

    def test_34_pid_recycling_must_be_generation_fenced(self):
        w = fresh(generation=3, pid=321)
        self.assert_stop('STOP_EXIT_IDENTITY', lambda: w.observe_exit(321, 2), w)

    def test_35_wrong_pid_on_exit_is_terminal(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release()
        self.assert_stop('STOP_EXIT_IDENTITY', lambda: w.observe_exit(7777, 1), w)

    def test_36_exit_not_observed_is_not_exited(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.exit_unconfirmed()
        self.assertEqual(w.phase, Phase.EXIT_UNCONFIRMED)
        self.assertFalse(w.process_exit_observed)
        self.assertFalse(w.post_exit_baseline_observed)

    def test_37_post_exit_baseline_cannot_precede_exit(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release()
        self.assert_stop('STOP_POST_EXIT_OBSERVATION',
                         lambda: w.observe_post_exit_baseline(samples=((0, 0),) * 8, baseline_mib=0), w)

    def test_38_post_exit_foreign_process_prevents_baseline_claim(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.observe_exit(1234, 1)
        self.assert_stop('STOP_POST_EXIT_OBSERVATION',
                         lambda: w.observe_post_exit_baseline(samples=((0, 1),) * 8, baseline_mib=0), w)

    def test_39_post_exit_memory_outside_tolerance_prevents_baseline_claim(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.observe_exit(1234, 1)
        self.assert_stop('STOP_POST_EXIT_OBSERVATION',
                         lambda: w.observe_post_exit_baseline(samples=((4, 0),) * 8, baseline_mib=0), w)

    def test_40_within_tolerance_baseline_accepted_but_no_drop_proof(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.observe_exit(1234, 1)
        w.observe_post_exit_baseline(samples=((2, 0),) * 8, baseline_mib=0)
        self.assertTrue(w.post_exit_baseline_observed)
        self.assertEqual(w.receipt()['last_cuda_context_drop'], 'UNOBSERVABLE')

    def test_40a_post_exit_requires_eight_samples(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.observe_exit(1234, 1)
        self.assert_stop('STOP_POST_EXIT_OBSERVATION',
                         lambda: w.observe_post_exit_baseline(samples=((0, 0),) * 7, baseline_mib=0), w)

    def test_40b_earlier_memory_settling_does_not_invalidate_final_four(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.observe_exit(1234, 1)
        w.observe_post_exit_baseline(samples=((20, 0),) * 4 + ((0, 0),) * 4,
                                     baseline_mib=0)
        self.assertTrue(w.post_exit_baseline_observed)

    def test_40c_second_baseline_attestation_forbidden(self):
        w, _ = delivered()
        w.checked_drain(); w.checked_release(); w.observe_exit(1234, 1)
        w.observe_post_exit_baseline(samples=((0, 0),) * 8, baseline_mib=0)
        self.assert_stop('STOP_POST_EXIT_OBSERVATION',
                         lambda: w.observe_post_exit_baseline(samples=((0, 0),) * 8, baseline_mib=0), w)

    def test_41_explicit_invalid_fault_label_quarantined(self):
        w = fresh()
        self.assert_stop('STOP_INVALID_FAULT_LABEL', lambda: w.fault('good news'), w)

    def test_42_bool_pid_not_valid_even_if_int_subclass(self):
        w = Supervisor(1, SRC, MODEL, Ledger())
        self.assert_stop('STOP_SPAWN_IDENTITY', lambda: w.spawn(True), w)

    def test_43_no_automatic_replay_after_prepared_output_and_crash(self):
        ledger = Ledger()
        w = fresh(ledger=ledger)
        t = ticket()
        w.admit(t)
        w.submission_entered(t); w.submission_confirmed(t); w.fence_confirmed(t)
        w.prepare_output(t, OUT)
        self.assert_stop('STOP_WORKER_CRASH', lambda: w.fault('STOP_WORKER_CRASH'), w)
        successor = fresh(2, ledger=ledger, pid=1235)
        self.assert_stop('STOP_ADMISSION_REPLAY_OR_IDENTITY',
                         lambda: successor.admit(ticket(generation=2)), successor)

    def test_44_local_receipt_never_claims_global_driver_leak_freedom(self):
        w = fresh()
        self.assertEqual(w.receipt()['leak_free'], 'NOT_CLAIMED')
        self.assertEqual(w.receipt()['last_cuda_context_drop'], 'UNOBSERVABLE')


if __name__ == '__main__':
    unittest.main()
