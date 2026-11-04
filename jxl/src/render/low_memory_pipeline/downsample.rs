// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::row_buffers::RowBuffer;
use super::run_stage::ExtraInfo;
use crate::image::{DataTypeTag, box_average};
use crate::render::stages::DownsampleStage;
use crate::util::{ShiftRightCeil, SmallVec};

impl DownsampleStage {
    pub(in crate::render) fn run_lowmem(
        &self,
        info: ExtraInfo,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [RowBuffer],
    ) {
        let shift = self.shift();
        let block_size = 1usize << shift;
        let xpre = if info.start_of_row {
            0
        } else {
            info.out_extra_x
        };
        let xpost = if info.end_of_row { 0 } else { info.out_extra_x };
        let xsize = info.xsize + ((xpre + xpost) << shift);
        debug_assert!((info.group_x0 - (xpre << shift)).is_multiple_of(block_size));
        debug_assert!(xsize.is_multiple_of(block_size) || info.end_of_row);
        debug_assert!(info.current_row.is_multiple_of(block_size));
        let num_input_rows = block_size.min(info.image_height - info.current_row);

        let sample_size = size_of::<f32>();
        let input_start = RowBuffer::x0_byte_offset() - (xpre << shift) * sample_size;
        let input_end = input_start + xsize * sample_size;
        let output_start = RowBuffer::x0_byte_offset() - xpre * sample_size;
        let output_end = output_start + xsize.shrc(shift) * sample_size;

        for (channel, alpha) in self.alpha.iter().enumerate() {
            let source: SmallVec<&[u8], 8> = (0..num_input_rows)
                .map(|y| {
                    &input_buffers[channel].get_row::<u8>(info.current_row + y)
                        [input_start..input_end]
                })
                .collect();
            let alpha_rows = alpha.map(|(channel, _)| {
                (0..num_input_rows)
                    .map(|y| {
                        &input_buffers[channel].get_row::<u8>(info.current_row + y)
                            [input_start..input_end]
                    })
                    .collect::<SmallVec<&[u8], 8>>()
            });
            let dest = &mut output_buffers[channel].get_row_mut::<u8>(info.current_row >> shift)
                [output_start..output_end];
            box_average(
                DataTypeTag::F32,
                &source,
                shift as usize,
                alpha_rows.as_deref(),
                alpha.is_some_and(|(_, clamp)| clamp),
                dest,
            );
        }
    }
}
