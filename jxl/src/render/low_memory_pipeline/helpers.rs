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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn get_distinct_indices_arbtest() {
        arbtest::arbtest(|u| {
            let outer_len = (u.arbitrary::<u8>()? % 6) as usize + 1;
            let mut vals = Vec::with_capacity(outer_len);
            let mut coords = Vec::new();

            for a in 0..outer_len {
                let inner_len = (u.arbitrary::<u8>()? % 6) as usize + 1;
                let mut row = Vec::with_capacity(inner_len);
                for b in 0..inner_len {
                    row.push(((a as u32) << 16) | b as u32);
                    coords.push((a, b));
                }
                vals.push(row);
            }

            let selected_len = (u.arbitrary::<u8>()? as usize % coords.len()) + 1;
            let mut selected = (0..coords.len()).collect::<Vec<_>>();
            for i in (1..selected.len()).rev() {
                let j = u.arbitrary::<u16>()? as usize % (i + 1);
                selected.swap(i, j);
            }
            selected.truncate(selected_len);
            selected.sort_unstable();

            let mut pos = (0..selected_len).collect::<Vec<_>>();
            for i in (1..selected_len).rev() {
                let j = u.arbitrary::<u16>()? as usize % (i + 1);
                pos.swap(i, j);
            }

            let mut idx = Vec::with_capacity(selected_len);
            let mut expected_before = vec![0u32; selected_len];
            let mut expected_after = vec![0u32; selected_len];
            for (i, &coord_idx) in selected.iter().enumerate() {
                let (a, b) = coords[coord_idx];
                let p = pos[i];
                idx.push((a, b, p));
                expected_before[p] = vals[a][b];
                expected_after[p] = 0xA5A5_0000 | p as u32;
            }

            let mut refs = get_distinct_indices(&mut vals, &idx);
            assert_eq!(refs.len(), selected_len);
            for (i, r) in refs.iter_mut().enumerate() {
                assert_eq!(**r, expected_before[i]);
                **r = expected_after[i];
            }
            drop(refs);

            for (i, &coord_idx) in selected.iter().enumerate() {
                let (a, b) = coords[coord_idx];
                let p = pos[i];
                assert_eq!(vals[a][b], expected_after[p]);
            }

            Ok(())
        });
    }
}
