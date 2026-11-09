//! Per-stage timers and candidate counters for profiling builds.
//!
//! Compiled in only with the `stage-profile` feature; otherwise every call is an
//! inlined no-op, so release builds carry no timing overhead.

/// Detector pass that the following stages and counts belong to.
#[derive(Clone, Copy)]
pub(crate) enum Pass {
    Primary,
    Recovery,
    Fragment,
}

/// Timed stage, or counted event, within a pass.
#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Total,
    Resample,
    TileRanges,
    Threshold,
    Components,
    BlobAccumulate,
    HullFit,
    Plausible,
    Refine,
    Decode,
    GridFit,
    // Counters only.
    Runs,
    Blobs,
    Proposals,
    Skipped,
    PlausibleCalls,
    PlausiblePass,
    RefinePass,
    Decoded,
    Frames,
}

#[cfg(feature = "stage-profile")]
mod imp {
    use super::{Pass, Stage};
    use std::cell::{Cell, RefCell};
    use std::time::Instant;

    const PASSES: [&str; 3] = ["primary", "recovery", "fragment"];
    const STAGES: [&str; 20] = [
        "total",
        "resample",
        "tile_ranges",
        "threshold",
        "components",
        "blob_accumulate",
        "hull_fit",
        "plausible",
        "refine",
        "decode",
        "grid_fit",
        "runs",
        "blobs",
        "proposals",
        "skipped",
        "plausible_calls",
        "plausible_pass",
        "refine_pass",
        "decoded",
        "frames",
    ];
    thread_local! {
        static PASS: Cell<usize> = const { Cell::new(0) };
        static TOTALS: RefCell<[[u64; STAGES.len()]; PASSES.len()]> =
            const { RefCell::new([[0; STAGES.len()]; PASSES.len()]) };
    }

    pub(crate) fn set_pass(pass: Pass) {
        PASS.with(|current| current.set(pass as usize));
    }

    pub(crate) fn add(stage: Stage, amount: u64) {
        let pass = PASS.with(Cell::get);
        TOTALS.with(|totals| totals.borrow_mut()[pass][stage as usize] += amount);
    }

    pub(crate) fn timed<T>(stage: Stage, work: impl FnOnce() -> T) -> T {
        let start = clock();
        let value = work();
        since(stage, start);
        value
    }

    /// Start of a span ended by `since`.
    pub(crate) struct Clock(Instant);

    pub(crate) fn clock() -> Clock {
        Clock(Instant::now())
    }

    pub(crate) fn since(stage: Stage, start: Clock) {
        add(stage, start.0.elapsed().as_nanos() as u64);
    }

    /// The calling thread's `(pass.stage, value)` pairs since its last call, then reset.
    ///
    /// Other threads' totals are neither reported nor reset.
    pub(crate) fn take() -> Vec<(String, u64)> {
        TOTALS.with(|totals| {
            let mut totals = totals.borrow_mut();
            let mut out = Vec::new();
            for (pass, row) in totals.iter().enumerate() {
                for (stage, &value) in row.iter().enumerate() {
                    if value != 0 {
                        out.push((format!("{}.{}", PASSES[pass], STAGES[stage]), value));
                    }
                }
            }
            *totals = [[0; STAGES.len()]; PASSES.len()];
            out
        })
    }
}

#[cfg(feature = "stage-profile")]
pub(crate) use imp::{add, clock, set_pass, since, take, timed};

#[cfg(not(feature = "stage-profile"))]
mod imp {
    use super::{Pass, Stage};

    #[inline(always)]
    pub(crate) fn set_pass(_: Pass) {}

    #[inline(always)]
    pub(crate) fn add(_: Stage, _: u64) {}

    #[inline(always)]
    pub(crate) fn timed<T>(_: Stage, work: impl FnOnce() -> T) -> T {
        work()
    }

    /// Zero-sized stand-in for the profiling span start.
    pub(crate) struct Clock;

    #[inline(always)]
    pub(crate) fn clock() -> Clock {
        Clock
    }

    #[inline(always)]
    pub(crate) fn since(_: Stage, _: Clock) {}
}

#[cfg(not(feature = "stage-profile"))]
pub(crate) use imp::{add, clock, set_pass, since, timed};
