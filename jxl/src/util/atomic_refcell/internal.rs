// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::AtomicRefCell;

const MUT_BIT: usize = !(usize::MAX >> 1);

/// Atomic counter that performs synchronization of `AtomicRefCell`.
pub(super) struct AtomicRefCounter(AtomicUsize);

impl AtomicRefCounter {
    #[inline]
    pub(super) const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    #[inline]
    fn acquire(&self) -> bool {
        let mut prev_counter = self.0.load(Ordering::Relaxed);
        loop {
            if prev_counter & MUT_BIT != 0 {
                // Mutable borrow exists.
                return false;
            }

            let next_counter = prev_counter + 1;
            if next_counter & MUT_BIT != 0 {
                // Counter overflowed; treat as failure.
                return false;
            }

            // Use compare-exchange to ensure that the counter didn't change since the last time.
            // Acquire ordering synchronizes with Release used in `release` and `release_mut`.
            // Ordering of other accesses doesn't matter, because the borrow happens only when
            // compare-exchange succeeds.
            match self.0.compare_exchange_weak(
                prev_counter,
                next_counter,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(counter) => {
                    // Compare-exchange failed; retry.
                    prev_counter = counter;
                }
            }
        }
    }

    #[inline]
    fn acquire_mut(&self) -> bool {
        // Use compare-exchange to ensure that there's no other reference to the data.
        // Acquire ordering synchronizes with Release used in `release` and `release_mut`.
        self.0
            .compare_exchange(0, MUT_BIT, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    #[inline]
    fn release(&self) {
        // Decrement the reference counter.
        self.0.fetch_sub(1, Ordering::Release);
    }

    #[inline]
    fn release_mut(&self) {
        // Unconditionally set the counter to zero since this is the only reference to the data.
        self.0.store(0, Ordering::Release);
    }
}

/// Indicator that a shared reference to the data is successfully acquired.
struct BorrowToken<'a>(&'a AtomicRefCounter);

impl<'a> BorrowToken<'a> {
    /// Ensures that there's no mutable borrow of the data, and increments the reference counter.
    ///
    /// It is guaranteed that there's no instance of `BorrowTokenMut` that points to the same
    /// counter if this method returned `Some`.
    #[inline]
    fn borrow(counter: &'a AtomicRefCounter) -> Option<Self> {
        let success = counter.acquire();
        success.then(|| Self(counter))
    }
}

impl Clone for BorrowToken<'_> {
    #[inline]
    fn clone(&self) -> Self {
        Self::borrow(self.0).unwrap()
    }
}

impl Drop for BorrowToken<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.release();
    }
}

/// Indicator that a mutable reference to the data is successfully acquired.
struct BorrowTokenMut<'a>(&'a AtomicRefCounter);

impl<'a> BorrowTokenMut<'a> {
    /// Ensures that there's no active borrow of the data, and marks the reference counter as
    /// mutably borrowed.
    ///
    /// It is guaranteed that there's no instance of `BorrowToken` or `BorrowTokenMut` that points
    /// to the same counter if this method returned `Some`.
    #[inline]
    fn borrow_mut(counter: &'a AtomicRefCounter) -> Option<Self> {
        let success = counter.acquire_mut();
        success.then(|| Self(counter))
    }
}

impl Drop for BorrowTokenMut<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.release_mut();
    }
}

// Invariant: `token` and `ptr` are derived from the same `AtomicRefCell`.
pub struct AtomicRef<'a, T: ?Sized> {
    // Token ensures that this is a valid shared reference to `ptr`.
    token: BorrowToken<'a>,
    ptr: NonNull<T>,
}

// SAFETY: `AtomicRefMut` acts like a shared reference (see `deref`).
unsafe impl<'a, T: ?Sized> Send for AtomicRef<'a, T> where for<'r> &'r T: Send {}

// SAFETY: `AtomicRefMut` acts like a shared reference (see `deref`).
unsafe impl<'a, T: ?Sized> Sync for AtomicRef<'a, T> where for<'r> &'r T: Sync {}

impl<'a, T: ?Sized> AtomicRef<'a, T> {
    #[inline]
    pub(super) fn new(cell: &'a AtomicRefCell<T>) -> Option<Self> {
        let token = BorrowToken::borrow(&cell.counter)?;
        // SAFETY: The token and the pointer come from the same `AtomicRefCell`.
        Some(Self {
            token,
            ptr: cell.ptr(),
        })
    }

    #[inline]
    pub fn map<U: ?Sized>(orig: Self, f: impl FnOnce(&T) -> &U) -> AtomicRef<'a, U> {
        AtomicRef {
            ptr: NonNull::from_ref(f(&*orig)),
            token: orig.token,
        }
    }

    #[expect(clippy::should_implement_trait)]
    #[inline]
    pub fn clone(orig: &Self) -> Self {
        // SAFETY: The invariants hold trivially, from the invariants of `orig`.
        Self {
            token: orig.token.clone(),
            ptr: orig.ptr,
        }
    }
}

impl<T: ?Sized> Deref for AtomicRef<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` is synchronized by `token` by the invariant of `AtomicRef`, and the
        // invariant of `token` ensures that we have a valid shared reference to `ptr`.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AtomicRef<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AtomicRef").field(self).finish()
    }
}

// Invariant: `token` and `ptr` are derived from the same `AtomicRefCell`.
pub struct AtomicRefMut<'a, T: ?Sized> {
    // Token ensures that this is the unique reference to `ptr`.
    token: BorrowTokenMut<'a>,
    ptr: NonNull<T>,
    // Marker to make `AtomicRefMut` invariant over `T`.
    _phantom: PhantomData<&'a mut T>,
}

// SAFETY: `AtomicRefMut` acts like a mutable reference (see `deref_mut`).
unsafe impl<'a, T: ?Sized> Send for AtomicRefMut<'a, T> where for<'r> &'r mut T: Send {}

// SAFETY: `AtomicRefMut` acts like a mutable reference (see `deref_mut`).
unsafe impl<'a, T: ?Sized> Sync for AtomicRefMut<'a, T> where for<'r> &'r mut T: Sync {}

impl<'a, T: ?Sized> AtomicRefMut<'a, T> {
    #[inline]
    pub(super) fn new(cell: &'a AtomicRefCell<T>) -> Option<Self> {
        let token = BorrowTokenMut::borrow_mut(&cell.counter)?;
        // SAFETY: The token and the pointer come from the same `AtomicRefCell`.
        Some(Self {
            token,
            ptr: cell.ptr(),
            _phantom: PhantomData,
        })
    }

    #[inline]
    pub fn map<U: ?Sized>(mut orig: Self, f: impl FnOnce(&mut T) -> &mut U) -> AtomicRefMut<'a, U> {
        AtomicRefMut {
            ptr: NonNull::from_ref(f(&mut *orig)),
            token: orig.token,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for AtomicRefMut<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` is synchronized by `_token` by the invariant of `AtomicRefMut`, and the
        // invariant of `token` ensures that we have the unique reference to `ptr`.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for AtomicRefMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `ptr` is synchronized by `_token` by the invariant of `AtomicRefMut`, and the
        // invariant of `token` ensures that we have the unique reference to `ptr`.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AtomicRefMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AtomicRefMut").field(&**self).finish()
    }
}
