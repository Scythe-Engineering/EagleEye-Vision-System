//! Persistent row workers; small images execute on the calling thread.

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub(crate) struct Workers {
    pool: Option<ThreadPool>,
}

impl Workers {
    /// Create a fixed worker pool without spawning threads for serial detection.
    pub(crate) fn new(threads: usize) -> Result<Self, String> {
        let pool = if threads > 1 {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| format!("cannot create detector workers: {error}"))?,
            )
        } else {
            None
        };
        Ok(Self { pool })
    }

    /// Process disjoint output rows and finish all work before borrowed data expires.
    pub(crate) fn rows<T, F>(&self, output: &mut [T], width: usize, parallel: bool, work: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Send + Sync,
    {
        if let Some(pool) = self.pool.as_ref().filter(|_| parallel) {
            pool.install(|| {
                output
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(row, values)| work(row, values));
            });
        } else {
            for (row, values) in output.chunks_mut(width).enumerate() {
                work(row, values);
            }
        }
    }
}
