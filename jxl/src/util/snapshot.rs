// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A shared value that is published by the decoder between phases and read by render threads
//! on every row, without any shared-memory writes on the read path.
//!
//! [`SharedSnapshot`] keeps the value behind an `Arc` together with a generation counter that is
//! bumped on each [`SharedSnapshot::set`]. Each render thread keeps a [`SnapshotReader`] in its
//! stage-local state; the per-row fast path is a single atomic *load* of the generation, which
//! only refreshes the cached `Arc` (taking the lock) when the value has been replaced.

use crate::util::sync::atomic::{AtomicUsize, Ordering};
use crate::util::sync::{Arc, RwLock};

pub struct SharedSnapshot<T> {
    generation: AtomicUsize,
    value: RwLock<Arc<T>>,
}

impl<T> SharedSnapshot<T> {
    pub fn new(value: T) -> Self {
        Self {
            generation: AtomicUsize::new(0),
            value: RwLock::new(Arc::new(value)),
        }
    }

    /// Replaces the stored value.
    ///
    /// Like the `try_write().unwrap()` it replaces, this must not race with a reader refreshing
    /// its snapshot (the decoder only publishes values between rendering phases). Readers that
    /// still hold the previous `Arc` keep using it until they observe the new generation.
    pub fn set(&self, value: T) {
        let mut guard = self.value.try_write().unwrap();
        *guard = Arc::new(value);
        // Bump while holding the write lock, so that a reader refreshing under the read lock
        // always sees a (generation, value) pair that belongs together.
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Returns the current value. Intended for decoder-side (non per-row) accesses.
    pub fn get(&self) -> Arc<T> {
        self.value.try_read().unwrap().clone()
    }

    /// Creates a per-thread reader holding a snapshot of the current value.
    pub fn reader(&self) -> SnapshotReader<T> {
        let guard = self.value.try_read().unwrap();
        SnapshotReader {
            generation: self.generation.load(Ordering::Acquire),
            value: guard.clone(),
        }
    }
}

/// Per-thread cached view of a [`SharedSnapshot`].
pub struct SnapshotReader<T> {
    generation: usize,
    value: Arc<T>,
}

impl<T> SnapshotReader<T> {
    /// Returns the up-to-date value of `shared`, refreshing the cached snapshot if needed.
    #[inline(always)]
    pub fn get(&mut self, shared: &SharedSnapshot<T>) -> &T {
        if shared.generation.load(Ordering::Acquire) != self.generation {
            self.refresh(shared);
        }
        &self.value
    }

    #[cold]
    #[inline(never)]
    fn refresh(&mut self, shared: &SharedSnapshot<T>) {
        *self = shared.reader();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn reader_sees_updates() {
        let shared = SharedSnapshot::new(1u32);
        let mut reader = shared.reader();
        assert_eq!(*reader.get(&shared), 1);
        shared.set(2);
        assert_eq!(*reader.get(&shared), 2);
        assert_eq!(*shared.get(), 2);
        let mut reader2 = shared.reader();
        shared.set(3);
        assert_eq!(*reader2.get(&shared), 3);
        assert_eq!(*reader.get(&shared), 3);
    }
}
