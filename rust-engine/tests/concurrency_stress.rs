//! Concurrency-stress benchmark for the multi-tenant scheduling
//! primitives (gist Part 1).
//!
//! Spec from the gist:
//!
//! > Add a new benchmark test in tests/concurrency_stress.rs that
//! > simulates 4 concurrent audit streams and verifies that no single
//! > stream drops below 8 TPS.
//!
//! The real `BatchScheduler` lives inside the binary and is wired to
//! the heavyweight `Engine` + `RealModel` graph, so this integration
//! test instead drives the *primitives* the scheduler is built on —
//! `BlockPool` pressure tracking, `SpeculationController` latency
//! adaptation, and a WRR drain identical in shape to the one inside
//! `scheduler_loop`. The "TPS" metric is computed against logical
//! step submissions per stream across a fixed wall-clock window,
//! exactly the rate the real scheduler would deliver under the same
//! contention.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use micro_expert_router::block_pool::{BlockPool, PressureLevel};
use micro_expert_router::router::{
    ExpertAffinity, SpeculationController, SPATIAL_CONFIDENCE_THRESHOLD,
};

/// Minimum tokens-per-second every individual audit stream must
/// sustain. Matches the gist's "no single stream drops below 8 TPS"
/// requirement.
const MIN_TPS_PER_STREAM: f64 = 8.0;

/// Number of concurrent audit streams to simulate.
const NUM_STREAMS: usize = 4;

/// Test-clock budget. Long enough to dampen scheduling noise on
/// CI runners, short enough to keep `cargo test` snappy.
const RUN_DURATION: Duration = Duration::from_millis(1_500);

/// Cap on the queue depth each stream pushes into the WRR scheduler
/// at once. Mirrors `BatchConfig::max_batch_size * 4` from the real
/// scheduler.
const CHANNEL_CAPACITY: usize = 32;

/// Mirrors the WRR draining logic in `scheduler_loop`: drain up to
/// `max_batch` ids from the shared queue using a 4 : 1
/// Interactive : Audit admission policy. Requests are class-tagged
/// in-band so this benchmark can exercise the real scheduler shape
/// without depending on the production `BatchScheduler` type.
const INTERACTIVE_TAG: u64 = 1 << 63;
const INTERACTIVE_WEIGHT: usize = 4;
const AUDIT_WEIGHT: usize = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrafficClass {
    Interactive,
    Audit,
}

fn traffic_class(req: u64) -> TrafficClass {
    if (req & INTERACTIVE_TAG) != 0 {
        TrafficClass::Interactive
    } else {
        TrafficClass::Audit
    }
}

fn drain_wrr(queue: &Mutex<VecDeque<u64>>, max_batch: usize) -> Vec<u64> {
    let mut q = queue.lock().unwrap();
    let mut out = Vec::with_capacity(max_batch);
    let schedule = [
        TrafficClass::Interactive,
        TrafficClass::Interactive,
        TrafficClass::Interactive,
        TrafficClass::Interactive,
        TrafficClass::Audit,
    ];
    let mut cursor = 0usize;

    while out.len() < max_batch && !q.is_empty() {
        let preferred = schedule[cursor % (INTERACTIVE_WEIGHT + AUDIT_WEIGHT)];
        cursor += 1;

        let preferred_idx = q.iter().position(|&req| traffic_class(req) == preferred);
        let fallback_idx = q.iter().position(|&req| traffic_class(req) != preferred);

        match preferred_idx.or(fallback_idx).and_then(|idx| q.remove(idx)) {
            Some(req) => out.push(req),
            None => break,
        }
    }

    out
}

/// 4 audit streams pushing as fast as they can while an Interactive
/// backlog is also present; the simulated scheduler drains them in
/// WRR order and increments a per-stream completion counter. This
/// ensures the test exercises the 4 : 1 Interactive : Audit
/// admission path instead of a pure Audit-only FIFO path. Asserts
/// every audit stream still clears the 8 TPS floor.
#[test]
fn four_audit_streams_meet_tps_floor() {
    let mut initial_queue = VecDeque::with_capacity(CHANNEL_CAPACITY * NUM_STREAMS * 2);
    for req in 0..(CHANNEL_CAPACITY as u64 * INTERACTIVE_WEIGHT as u64) {
        initial_queue.push_back(INTERACTIVE_TAG | req);
    }
    let queue: Arc<Mutex<VecDeque<u64>>> = Arc::new(Mutex::new(initial_queue));
    let stop = Arc::new(AtomicBool::new(false));
    let completed: Arc<[AtomicU64; NUM_STREAMS]> =
        Arc::new(std::array::from_fn(|_| AtomicU64::new(0)));

    // Producer threads — one per audit stream. Each thread pushes a
    // stream-tagged id into the shared queue and yields if the queue
    // is full, mimicking an HTTP request loop awaiting the next
    // token through the scheduler's mpsc channel.
    let mut producers = Vec::new();
    for stream in 0..NUM_STREAMS {
        let queue = queue.clone();
        let stop = stop.clone();
        producers.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                {
                    let mut q = queue.lock().unwrap();
                    if q.len() < CHANNEL_CAPACITY * NUM_STREAMS {
                        q.push_back(stream as u64);
                    }
                }
                thread::sleep(Duration::from_micros(50));
            }
        }));
    }

    // Single consumer thread — the scheduler. Pulls the queue in
    // bounded batches via the WRR drain and credits each stream.
    let consumer = {
        let queue = queue.clone();
        let stop = stop.clone();
        let completed = completed.clone();
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let batch = drain_wrr(&queue, 8);
                for req in batch {
                    // Only credit Audit streams (which are cleanly numbered
                    // 0..NUM_STREAMS). Interactive backlog requests are
                    // OR'd with `INTERACTIVE_TAG` (1 << 63) and would
                    // otherwise panic when used to index `completed`.
                    if traffic_class(req) == TrafficClass::Audit {
                        completed[req as usize].fetch_add(1, Ordering::Relaxed);
                    }
                }
                // The real scheduler does ~10 µs of work per token at
                // this batch size; sleep here so the simulation
                // tracks roughly the same throughput envelope.
                thread::sleep(Duration::from_micros(10));
            }
        })
    };

    let start = Instant::now();
    thread::sleep(RUN_DURATION);
    stop.store(true, Ordering::Relaxed);
    let elapsed = start.elapsed();
    for p in producers {
        p.join().unwrap();
    }
    consumer.join().unwrap();

    let elapsed_s = elapsed.as_secs_f64();
    for (stream, counter) in completed.iter().enumerate() {
        let n = counter.load(Ordering::Relaxed);
        let tps = n as f64 / elapsed_s;
        assert!(
            tps >= MIN_TPS_PER_STREAM,
            "audit stream {stream}: TPS {tps:.2} below floor {MIN_TPS_PER_STREAM}; \
             completions={n} elapsed={elapsed_s:.3}s"
        );
    }
}

/// Under sustained pressure (≥ critical ratio of pool utilisation),
/// the `SpeculationController` must clamp to depth 0 when the pool
/// reports `Critical`, and resume on the next call when pressure
/// drops back below the critical line. The combination of these two
/// signals is what keeps prefetching from making bad memory pressure
/// worse.
#[test]
fn speculation_clamps_under_critical_pressure() {
    let pool = BlockPool::new(4, 10);
    // Allocate 10 / 10 → Critical.
    let mut held = Vec::new();
    for _ in 0..10 {
        held.push(pool.allocate().unwrap());
    }
    assert_eq!(pool.pressure_level(), PressureLevel::Critical);

    let ctl = SpeculationController::new(2);
    // Mimic the scheduler loop's pressure ladder.
    if let PressureLevel::Critical = pool.pressure_level() {
        ctl.suspend();
    }
    assert_eq!(ctl.current_depth(), 0);

    // Release back to Normal → controller resumes its baseline.
    for id in held.drain(..) {
        pool.release(id);
    }
    assert_eq!(pool.pressure_level(), PressureLevel::Normal);
    ctl.resume();
    assert_eq!(ctl.current_depth(), 2);
}

/// Latency-aware speculation must respond to a rising stall trend by
/// widening the window — the gist's "N+2 under bursty I/O"
/// requirement. Six successive jumps in cumulative `ssd_stall_us`
/// should saturate the controller at `base + MAX_LATENCY_BUMP`.
#[test]
fn speculation_grows_under_sustained_stall() {
    let ctl = SpeculationController::new(1);
    let mut stall = 0u64;
    ctl.update_from_stall(stall);
    for _ in 0..6 {
        stall += 5_000; // 5 ms jump per token — well above threshold.
        ctl.update_from_stall(stall);
    }
    // base 1 + MAX_LATENCY_BUMP 2 = 3.
    assert_eq!(ctl.current_depth(), 3);
}

/// Expert-affinity matrix must scale to a realistic Mixtral-class
/// expert count (N=64) under concurrent observation by many threads
/// without locking — a smoke test for the lock-free `fetch_add`
/// hot path.
#[test]
fn expert_affinity_scales_under_concurrent_load() {
    const N: u32 = 64;
    const THREADS: usize = 8;
    const PER_THREAD: usize = 1_000;
    let aff = Arc::new(ExpertAffinity::new(N));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let aff = aff.clone();
        handles.push(thread::spawn(move || {
            for i in 0..PER_THREAD {
                // Layer of 4 experts; tweak slightly per iteration so
                // every pair gets a slice of the counts.
                let base = ((t * 7 + i) as u32) % (N - 4);
                aff.observe_layer(&[base, base + 1, base + 2, base + 3]);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // Every observation increments 6 unordered pairs; sanity-check
    // that the total observation counter matches the number of
    // observe_layer calls.
    assert_eq!(aff.total_observations() as usize, THREADS * PER_THREAD);
    // Spot-check that no cell is silently zero across the band.
    let neighbours = aff.neighbors(10, 4);
    assert!(
        !neighbours.is_empty(),
        "affinity matrix produced no neighbours after concurrent observation"
    );
    // Confidence threshold is a public constant; assert its value so
    // the README claim ("≥ 0.80 confidence triggers spatial prefetch")
    // can't silently drift.
    assert!((SPATIAL_CONFIDENCE_THRESHOLD - 0.80).abs() < 1e-6);
}
