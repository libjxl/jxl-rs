// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use crate::render::low_memory_pipeline::row_buffers::RowBuffer;
use crate::util::ChannelVec;

pub struct RowBuffers {
    buffers: Vec<Vec<RowBuffer>>,
    // The input buffer that each channel of each stage should use.
    // This is indexed both by stage index (0 corresponds to input data, 1 to stage[0], etc) and by
    // index *of those channels that are used*.
    //
    // # Safety invariant
    // For each v in stage_input_buffer_index at index idx, we have that:
    // - for all i = 0..v.len(), v[i].0 <= idx, v[i].0 < self.buffers.len()
    //   and v[i].1 < self.buffers[v[i].0].len()
    // - for all i != j in 0..v.len(), v[i] != v[j]
    stage_input_buffer_index: Vec<Vec<(usize, usize)>>,
}

impl RowBuffers {
    pub fn new(
        buffers: Vec<Vec<RowBuffer>>,
        stage_input_buffer_index: Vec<Vec<(usize, usize)>>,
    ) -> Self {
        for (idx, v) in stage_input_buffer_index.iter().enumerate() {
            for i in 0..v.len() {
                for j in i + 1..v.len() {
                    assert_ne!(v[i], v[j]);
                }
                assert!(v[i].0 <= idx);
                assert!(v[i].0 < buffers.len());
                assert!(v[i].1 < buffers[v[i].0].len());
            }
        }
        // Safety note: we just checked the safety invariant.
        Self {
            buffers,
            stage_input_buffer_index,
        }
    }

    pub fn get_input_buffer(&mut self, c: usize) -> &mut RowBuffer {
        &mut self.buffers[0][c]
    }

    pub fn get_num_stage_input_buffers(&self, stage: usize) -> usize {
        self.stage_input_buffer_index[stage].len()
    }

    pub fn get_stage_input_buffer(&mut self, stage: usize, c: usize) -> &mut RowBuffer {
        let (si, ci) = self.stage_input_buffer_index[stage][c];
        &mut self.buffers[si][ci]
    }

    pub fn get_inout_buffers(
        &mut self,
        stage: usize,
    ) -> (ChannelVec<&RowBuffer>, &mut [RowBuffer]) {
        assert!(stage.checked_add(1).unwrap() < self.buffers.len());
        let idx = &self.stage_input_buffer_index[stage];
        let buf_ptr = self.buffers.as_mut_ptr();
        let ret = idx
            .iter()
            .map(|(a, b)| {
                // SAFETY: as per safety invariant, `a` is within bounds of the slice pointed at by `buf_ptr`.
                // Moreover, `b` is within bounds of the slice returned by `as_ptr()`.
                // Finally, the lifetime of the references are tied to `self`, which owns `self.buffers`.
                // Also note that we know that *a <= stage.
                unsafe { &*(*buf_ptr.add(*a)).as_ptr().add(*b) }
            })
            .collect();
        // SAFETY: Above, we only access elements <= stage. `stage + 1` does not overflow (or the
        // checked_add at the start of the function would have failed), so we are accessing
        // a distinct vector element (in bounds due to the assert! above), thus we do not violate
        // aliasing rules. Finally, the reference does not outlive `self` and hence `self.buffers`.
        let outb = unsafe { &mut *buf_ptr.add(stage + 1) };
        (ret, &mut outb[..])
    }

    pub fn get_inplace_buffers(&mut self, stage: usize) -> ChannelVec<&mut RowBuffer> {
        let idx = &self.stage_input_buffer_index[stage];
        let buf_ptr = self.buffers.as_mut_ptr();
        idx.iter()
            .map(|(a, b)| {
                // SAFETY: as per safety invariant, `a` is within bounds of the slice pointed at by `buf_ptr`.
                // Moreover, `b` is within bounds of the slice returned by `as_mut_ptr()`.
                // Moreover, all the `(a, b)` pairs are distinct, so we do not violate aliasing rules.
                // Finally, the lifetime of the references are tied to `self`, which owns `self.buffers`,
                // and `as_mut_ptr` promises to not create tree-borrows-invalidating references to the vec's
                // data slice.
                unsafe { &mut *(*buf_ptr.add(*a)).as_mut_ptr().add(*b) }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::DataTypeTag;

    #[test]
    fn row_buffers_arbtest() {
        arbtest::arbtest(|u| {
            let num_stages = (u.arbitrary::<u8>()? % 8) as usize + 1;
            let buffers_len = num_stages + 1;
            let mut buffers = Vec::with_capacity(buffers_len);
            let mut buffers_lengths = Vec::with_capacity(buffers_len);

            for a in 0..buffers_len {
                let inner_len = (u.arbitrary::<u8>()? % 8) as usize + 1;
                buffers_lengths.push(inner_len);
                let mut inner_buffers = Vec::with_capacity(inner_len);
                for b in 0..inner_len {
                    let mut row = RowBuffer::new(DataTypeTag::U32, 0, 0, 0, 1).unwrap();
                    row.get_row_mut::<u32>(0)[0] = ((a as u32) << 16) | b as u32;
                    inner_buffers.push(row);
                }
                buffers.push(inner_buffers);
            }

            let mut stage_input_buffer_index = Vec::with_capacity(num_stages);
            for s in 0..num_stages {
                let mut options = Vec::new();
                for a in 0..=s {
                    for b in 0..buffers_lengths[a] {
                        options.push((a, b));
                    }
                }

                let selected_len = (u.arbitrary::<u8>()? as usize % options.len()) + 1;
                for i in (1..options.len()).rev() {
                    let j = u.arbitrary::<u16>()? as usize % (i + 1);
                    options.swap(i, j);
                }
                options.truncate(selected_len);
                stage_input_buffer_index.push(options);
            }

            let mut row_buffers = RowBuffers::new(buffers, stage_input_buffer_index.clone());

            // Test get_input_buffer
            for c in 0..buffers_lengths[0] {
                let buf = row_buffers.get_input_buffer(c);
                let val = buf.get_row::<u32>(0)[0];
                assert_eq!(val, c as u32);
            }

            for s in 0..num_stages {
                let num = row_buffers.get_num_stage_input_buffers(s);
                assert_eq!(num, stage_input_buffer_index[s].len());

                // Test get_stage_input_buffer
                for c in 0..num {
                    let buf = row_buffers.get_stage_input_buffer(s, c);
                    let val = buf.get_row::<u32>(0)[0];
                    let (expected_a, expected_b) = stage_input_buffer_index[s][c];
                    assert_eq!(val, ((expected_a as u32) << 16) | expected_b as u32);
                }

                // Test get_inplace_buffers (including mutation and restore)
                {
                    let mut refs = row_buffers.get_inplace_buffers(s);
                    assert_eq!(refs.len(), num);
                    for (c, r) in refs.iter_mut().enumerate() {
                        let val = r.get_row_mut::<u32>(0)[0];
                        let (expected_a, expected_b) = stage_input_buffer_index[s][c];
                        assert_eq!(val, ((expected_a as u32) << 16) | expected_b as u32);
                        r.get_row_mut::<u32>(0)[0] = 0xA5A5_0000 | c as u32;
                    }

                    for (c, r) in refs.iter().enumerate() {
                        let val = r.get_row::<u32>(0)[0];
                        assert_eq!(val, 0xA5A5_0000 | c as u32);
                    }

                    for (c, r) in refs.iter_mut().enumerate() {
                        let (expected_a, expected_b) = stage_input_buffer_index[s][c];
                        r.get_row_mut::<u32>(0)[0] =
                            ((expected_a as u32) << 16) | expected_b as u32;
                    }
                }

                // Test get_inout_buffers
                {
                    let (in_refs, out_slice) = row_buffers.get_inout_buffers(s);
                    assert_eq!(in_refs.len(), num);
                    for (c, r) in in_refs.iter().enumerate() {
                        let val = r.get_row::<u32>(0)[0];
                        let (expected_a, expected_b) = stage_input_buffer_index[s][c];
                        assert_eq!(val, ((expected_a as u32) << 16) | expected_b as u32);
                    }

                    assert_eq!(out_slice.len(), buffers_lengths[s + 1]);
                    for (b, r) in out_slice.iter().enumerate() {
                        let val = r.get_row::<u32>(0)[0];
                        assert_eq!(val, (((s + 1) as u32) << 16) | b as u32);
                    }
                }
            }

            Ok(())
        });
    }
}
