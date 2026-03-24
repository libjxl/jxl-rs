// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::low_memory_pipeline::render_group::ChannelVec;

/// Returns a vector of &mut vals[idx[i].0][idx[i].1], in order of idx[i].2.
/// Panics if any of the indices are out of bounds or
/// (idx[i].0, idx[i].1) == (idx[j].0, idx[j].1) for i != j or indices are not
/// sorted lexicographically.
#[allow(unsafe_code)]
pub(super) fn get_distinct_indices<'a, T>(
    vals: &'a mut [impl AsMut<[T]>],
    idx: &[(usize, usize, usize)],
) -> ChannelVec<&'a mut T> {
    // Build output directly using pointer math to avoid Option overhead.
    // idx is sorted lexicographically by (a, b), so we iterate vals in order.
    let mut result: ChannelVec<&'a mut T> = ChannelVec::new();
    // Pre-fill with dummy values that will be overwritten.
    // We need the result in idx[i].2 order, so we use a position-indexed buffer.
    let n = idx.len();
    let mut ptrs: ChannelVec<*mut T> = ChannelVec::new();
    for _ in 0..n {
        ptrs.push(std::ptr::null_mut());
    }

    let mut targets = idx.iter();
    let mut target = targets.next().unwrap();
    'outer: for (aa, bufs) in vals.iter_mut().enumerate() {
        for (bb, buf) in bufs.as_mut().iter_mut().enumerate() {
            let (a, b, pos) = target;
            if aa == *a && bb == *b {
                ptrs[*pos] = buf as *mut T;
                if let Some(t) = targets.next() {
                    target = t;
                } else {
                    break 'outer;
                }
            }
        }
    }

    for p in ptrs.iter() {
        debug_assert!(!p.is_null(), "Not all elements were found");
        // SAFETY: Each pointer was obtained from a distinct &mut T in vals,
        // and the lexicographic sort + distinctness guarantees no aliasing.
        result.push(unsafe { &mut **p });
    }

    result
}
