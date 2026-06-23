//! Architecture-agnostic row-parallel execution helper.
//!
//! Every supported family — Mixtral, Qwen3-MoE, DeepSeek-V3 (MLA),
//! GPT-OSS, MiMo, and the dense Mistral / Phi decoders — drives the same
//! handful of dense matrix-vector kernels per token: the attention
//! Q/K/V/O projections, the MoE router gate, the per-expert
//! `gate_up_swiglu` / `down_proj`, and the LM head. These all reduce to
//! "compute `rows` independent output rows, each a dot product over
//! `cols` inputs", so they share one parallelisation primitive here.
//!
//! ## Why not `std::thread::scope` per call
//!
//! The original implementation fanned each call out with
//! `std::thread::scope(|s| { s.spawn(...) })`. That **spawns and joins
//! fresh OS threads on every matmul** — and there are on the order of a
//! few hundred matmuls per token (≈ `layers × (4 attn projections + MoE
//! router + top_k × 2 expert matmuls)` plus the LM head). Thread
//! creation/teardown is tens of microseconds each, so the fixed
//! thread-management cost alone runs into tens of milliseconds per token
//! regardless of how fast the actual SIMD math is.
//!
//! Worse, the engine's headline feature is **continuous batching**: the
//! scheduler runs each in-flight request's `model.step` as a *concurrent*
//! task. With per-call spawning, `N` concurrent requests each fan out to
//! `cores` threads, oversubscribing the box by `N × cores` and thrashing
//! the scheduler exactly when throughput matters most.
//!
//! ## What this does instead
//!
//! [`par_row_chunks`] dispatches disjoint row-chunks onto `rayon`'s
//! process-wide, work-stealing pool, which is created once and shared by
//! every caller. The per-call cost is a fork-join over already-resident
//! workers, and concurrent requests contend for one bounded pool instead
//! of each spawning their own. Output is bit-identical to the serial
//! reference: chunks are disjoint slices of the output and each row's
//! reduction is computed exactly as before.
//!
//! Granularity is bounded from both sides: matmuls below
//! [`MIN_TOTAL_FOR_PARALLEL`] elements run inline on the caller (a tiny
//! MoE router gate or a low-rank MLA projection is not worth a fork-join),
//! and the task count is capped so each task carries at least
//! [`MIN_ELEMS_PER_TASK`] elements of work — preventing a large matmul
//! from being shredded into more tasks than there is work to justify.
//!
//! ## Pool sizing — leave headroom for the async runtime
//!
//! Left to its own devices `rayon` sizes the global pool to *every*
//! logical core. Under continuous batching that is actively harmful: a
//! saturated compute fan-out pins all cores and delays the tokio workers
//! that drive the scheduler (mpsc wakeups), the gRPC server, and io_uring
//! SSD completions, inflating per-token tail latency exactly when
//! throughput matters most. [`init_global_pool`] therefore builds the pool
//! once at startup with [`default_compute_threads`] — logical cores minus
//! a small, bounded reservation — so the engine keeps a couple of cores
//! free for async work by default. An explicit `RAYON_NUM_THREADS` is
//! treated as an operator override and wins.

use tracing::{info, warn};

/// Below this many multiply-accumulates (`rows * cols`) a matmul runs
/// inline on the calling thread. The fork-join handshake costs more than
/// the saved compute for, e.g., a MoE router gate (`num_experts ×
/// d_model`) or DeepSeek's low-rank `q_a_proj`. Chosen so the smallest
/// matmuls that *do* parallelise still carry enough work per task to
/// dwarf the scheduling cost.
pub const MIN_TOTAL_FOR_PARALLEL: usize = 1 << 18; // 262_144

/// Target minimum multiply-accumulates per spawned task. The task count
/// is `min(num_threads, total / MIN_ELEMS_PER_TASK, rows)`, so a matmul
/// only fans out to as many workers as it has work to keep busy.
pub const MIN_ELEMS_PER_TASK: usize = 1 << 16; // 65_536

/// Number of workers in the shared compute pool. `rayon` caches this, so
/// unlike the previous `std::thread::available_parallelism()` call (a
/// `sched_getaffinity` syscall on Linux) it is essentially free to query
/// on the hot path.
#[inline]
pub fn num_threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// Number of logical cores held back from the compute pool for the async
/// runtime, as a function of the host's logical core count.
///
/// The pool would otherwise span *every* core; see the module docs for why
/// that starves tokio under continuous batching. We leave a small, bounded
/// slice free instead:
///
/// | logical cores | reserved | compute |
/// |---------------|----------|---------|
/// | `1..=4`       | 0        | all     |
/// | `5..=8`       | 1        | `n-1`   |
/// | `9..=31`      | 2        | `n-2`   |
/// | `32, 48, 64…` | `n/16`   | `n-n/16`|
///
/// Tiny hosts keep every core — compute is the scarce resource there and a
/// reservation would hurt more than async contention. From nine cores up
/// we hold back two, growing by one per additional sixteen cores so large
/// hosts keep proportionate headroom (e.g. 32 -> 30, 64 -> 60, 128 -> 120).
/// The result is monotonic in `logical` and always at least one.
pub fn default_compute_threads(logical: usize) -> usize {
    let reserved = match logical {
        0..=4 => 0,
        5..=8 => 1,
        _ => (logical / 16).max(2),
    };
    logical.saturating_sub(reserved).max(1)
}

/// A valid, positive `RAYON_NUM_THREADS` is an explicit operator override.
/// Zero or unparseable values are ignored so the smart default applies
/// (rayon itself treats `RAYON_NUM_THREADS=0` as "use the default").
fn env_thread_override() -> Option<usize> {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// Initialise the shared compute pool once, reserving headroom for the
/// async runtime (see [`default_compute_threads`]). Returns the resolved
/// worker count.
///
/// Call exactly once at process start — *after* any NUMA/affinity pinning
/// so the workers inherit the startup affinity mask, and *before* the first
/// [`par_row_chunks`] touches the pool. A valid `RAYON_NUM_THREADS` is left
/// untouched for rayon to apply on first use; otherwise the global pool is
/// built explicitly with the reserved worker count.
pub fn init_global_pool() -> usize {
    if let Some(n) = env_thread_override() {
        info!(
            threads = n,
            source = "RAYON_NUM_THREADS",
            "compute pool: honoring explicit thread-count override"
        );
        return n;
    }

    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let threads = default_compute_threads(logical);

    let build_result = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("mer-compute-{i}"))
        .build_global();

    match build_result {
        Ok(()) => {
            info!(
                logical,
                threads,
                reserved = logical - threads,
                "compute pool: sized with reserved async-runtime headroom"
            );
            threads
        }
        // `build_global` only errors if the global pool was already built
        // (a prior call, or a rayon use before init). Keep what exists.
        Err(e) => {
            let current = rayon::current_num_threads().max(1);
            warn!(
                error = %e,
                current_threads = current,
                "compute pool already initialised; keeping existing configuration"
            );
            current
        }
    }

/// Fill `out` in parallel by computing disjoint row-chunks on the shared
/// `rayon` pool.
///
/// `f(row_start, out_chunk)` must write `out_chunk[i]` with the result for
/// global row `row_start + i`. `cols` is the per-row reduction width and
/// is used only to size the work estimate (`out.len() * cols`); it does
/// not have to correspond to any particular buffer length.
///
/// The closure runs once per chunk, possibly on a worker thread, and is
/// required to be `Sync`. Chunks are non-overlapping `&mut` sub-slices of
/// `out`, so the writes never alias. For small `out` (or a single row, or
/// a single-threaded pool) the closure is invoked once, inline, with the
/// whole slice — no pool interaction at all.
#[inline]
pub fn par_row_chunks<T, F>(out: &mut [T], cols: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let rows = out.len();
    let total = rows.saturating_mul(cols.max(1));
    let nthreads = num_threads();

    // Inline fast path: not enough work, nothing to split, or no pool.
    if rows <= 1 || nthreads <= 1 || total < MIN_TOTAL_FOR_PARALLEL {
        f(0, out);
        return;
    }

    // Fan out to at most `nthreads`, and never to more tasks than there
    // is work to keep each one busy (`MIN_ELEMS_PER_TASK`) or rows to
    // hand out.
    let max_tasks_by_work = (total / MIN_ELEMS_PER_TASK).max(1);
    let ntasks = nthreads.min(max_tasks_by_work).min(rows);
    if ntasks <= 1 {
        f(0, out);
        return;
    }

    let chunk = rows.div_ceil(ntasks);
    let f = &f;
    rayon::scope(|s| {
        for (chunk_idx, out_chunk) in out.chunks_mut(chunk).enumerate() {
            let row_start = chunk_idx * chunk;
            s.spawn(move |_| f(row_start, out_chunk));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference row-major mat-vec used as the parity oracle.
    fn serial_matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| {
                let row = &w[r * cols..(r + 1) * cols];
                row.iter().zip(x).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    fn par_matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows];
        par_row_chunks(&mut y, cols, |row_start, out| {
            for (i, slot) in out.iter_mut().enumerate() {
                let r = row_start + i;
                let row = &w[r * cols..(r + 1) * cols];
                *slot = row.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
        y
    }

    #[test]
    fn matches_serial_across_sizes() {
        // Span the inline path (tiny), the boundary, and the fanned-out
        // path (large) to exercise both branches and the chunk seam.
        for &(rows, cols) in &[(1usize, 1usize), (3, 5), (64, 64), (1024, 512), (4096, 256)] {
            let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 17) as f32) * 0.01 - 0.5).collect();
            let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32) * 0.1 - 0.3).collect();
            let got = par_matvec(&w, &x, rows, cols);
            let want = serial_matvec(&w, &x, rows, cols);
            assert_eq!(got.len(), want.len());
            for (g, e) in got.iter().zip(want.iter()) {
                assert!((g - e).abs() <= 1e-4, "rows={rows} cols={cols}: {g} vs {e}");
            }
        }
    }

    #[test]
    fn every_row_written_exactly_once() {
        // A non-arithmetic check that chunking covers the whole output
        // with no gaps or overlaps: each slot records its own global row
        // index, so any double-write or skipped row would corrupt it.
        let rows = 1000usize;
        let mut out = vec![usize::MAX; rows];
        // Force the parallel path regardless of arithmetic width.
        par_row_chunks(&mut out, MIN_TOTAL_FOR_PARALLEL, |row_start, chunk| {
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = row_start + i;
            }
        });
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i, "row {i} was not written exactly once");
        }
    }

    #[test]
    fn single_row_uses_inline_path() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut out = vec![0i32; 1];
        // `AtomicUsize` keeps the closure `Fn + Sync` (a plain `calls +=
        // 1` capture would make it `FnMut`, which `par_row_chunks`
        // rejects). A single row must take the inline path: one call.
        let calls = AtomicUsize::new(0);
        par_row_chunks(&mut out, 1_000_000, |row_start, chunk| {
            calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(row_start, 0);
            chunk[0] = 42;
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(out[0], 42);
    }

    #[test]
    fn default_threads_reserve_async_headroom() {
        // Tiny hosts keep every core — compute is already scarce.
        assert_eq!(default_compute_threads(1), 1);
        assert_eq!(default_compute_threads(2), 2);
        assert_eq!(default_compute_threads(4), 4);
        // Small hosts hold back a single core.
        assert_eq!(default_compute_threads(5), 4);
        assert_eq!(default_compute_threads(8), 7);
        // From nine cores up we reserve two...
        assert_eq!(default_compute_threads(9), 7);
        assert_eq!(default_compute_threads(16), 14);
        assert_eq!(default_compute_threads(32), 30); // the 32-vCPU reference box
        // ...growing by one per additional sixteen cores.
        assert_eq!(default_compute_threads(48), 45);
        assert_eq!(default_compute_threads(64), 60);
        assert_eq!(default_compute_threads(128), 120);
    }

    #[test]
    fn default_threads_are_positive_and_monotonic() {
        // The pool must never collapse to zero workers, never exceed the
        // logical core count, and never *shrink* as cores are added.
        let mut prev = 0;
        for n in 1..=256 {
            let t = default_compute_threads(n);
            assert!(t >= 1, "n={n}: pool must keep at least one worker");
            assert!(t <= n, "n={n}: cannot use more than {n} logical cores (got {t})");
            assert!(t >= prev, "n={n}: compute threads went backwards {prev}->{t}");
            prev = t;
        }
    }
}
