// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::util::ChannelVec;

fn collect_distinct_indices<'a, T: 'a>(
    mut vals: &'a mut [impl AsMut<[T]>],
    idx: &[(usize, usize, usize)],
    mut emit: impl FnMut(usize, &'a mut T),
) {
    let mut base_a = 0usize;
    let mut i = 0usize;
    while i < idx.len() {
        let a = idx[i].0;
        let rel_a = a
            .checked_sub(base_a)
            .expect("Indices are not sorted lexicographically");
        let (_, tail_vals) = vals.split_at_mut(rel_a);
        let (head, rest_vals) = tail_vals.split_first_mut().expect("Index out of bounds");
        vals = rest_vals;
        base_a = a + 1;

        let mut row: &'a mut [T] = head.as_mut();
        let mut base_b = 0usize;
        while i < idx.len() && idx[i].0 == a {
            let (_, b, pos) = idx[i];
            let rel_b = b
                .checked_sub(base_b)
                .expect("Indices are not sorted or contain duplicate");
            let (_, tail_row) = row.split_at_mut(rel_b);
            let (elem, rest_row) = tail_row.split_first_mut().expect("Index out of bounds");
            emit(pos, elem);
            row = rest_row;
            base_b = b + 1;
            i += 1;
        }
    }
}

/// Returns a vector of &mut vals[idx[i].0][idx[i].1], in order of idx[i].2.
/// Panics if any of the indices are out of bounds or
/// (idx[i].0, idx[i].1) == (idx[j].0, idx[j].1) for i != j or indices are not
/// sorted lexicographically.
pub(super) fn get_distinct_indices<'a, T>(
    vals: &'a mut [impl AsMut<[T]>],
    idx: &[(usize, usize, usize)],
) -> ChannelVec<&'a mut T> {
    if idx.is_empty() {
        return ChannelVec::new();
    }
    if idx.iter().enumerate().all(|(i, &(_, _, pos))| pos == i) {
        let mut out = ChannelVec::new();
        collect_distinct_indices(vals, idx, |_, item| out.push(item));
        return out;
    }

    let mut answer_buffer: ChannelVec<Option<&'a mut T>> = (0..idx.len()).map(|_| None).collect();
    collect_distinct_indices(vals, idx, |pos, item| {
        assert!(answer_buffer[pos].is_none(), "Duplicate output position");
        answer_buffer[pos] = Some(item);
    });
    answer_buffer
        .iter_mut()
        .map(|x| std::mem::take(x).expect("Not all elements were found"))
        .collect()
}
