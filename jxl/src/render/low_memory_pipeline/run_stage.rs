// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{
    render::{
        RunInPlaceStage,
        internal::{PipelineBuffer, RunInOutStage},
        low_memory_pipeline::helpers::mirror,
    },
    util::{ShiftRightCeil, tracing_wrappers::*},
};

use super::{
    super::{RenderPipelineInOutStage, RenderPipelineInPlaceStage},
    row_buffers::RowBuffer,
};

pub struct ExtraInfo {
    // Number of *input* pixels to process (ignoring additional border pixels).
    pub(super) xsize: usize,
    // Additional border pixels requested in the output on each side, if not first/last xgroup.
    pub(super) out_extra_x: usize,
    pub(super) current_row: usize,
    pub(super) group_origin: (usize, usize),
    pub(super) is_first_xgroup: bool,
    pub(super) is_last_xgroup: bool,
    pub(super) image_height: usize,
}

impl PipelineBuffer for RowBuffer {
    type InPlaceExtraInfo = ExtraInfo;
    type InOutExtraInfo = ExtraInfo;
}

impl<T: RenderPipelineInPlaceStage> RunInPlaceStage<RowBuffer> for T {
    #[instrument(skip_all)]
    fn run_stage_on(
        &self,
        ExtraInfo {
            xsize,
            current_row,
            group_origin,
            out_extra_x,
            image_height: _,
            is_first_xgroup,
            is_last_xgroup,
        }: ExtraInfo,
        buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        let x0 = RowBuffer::x0_offset::<T::Type>();
        let xpre = if is_first_xgroup { 0 } else { out_extra_x };
        let xstart = x0 - xpre;
        let xend = x0 + xsize + if is_last_xgroup { 0 } else { out_extra_x };
        let mut rows: Vec<_> = buffers
            .iter_mut()
            .map(|x| &mut x.get_row_mut::<T::Type>(current_row)[xstart..])
            .collect();

        self.process_row_chunk(
            (group_origin.0 - xpre, current_row),
            xend - xstart,
            &mut rows[..],
            state,
        );
    }
}

impl<T: RenderPipelineInOutStage> RunInOutStage<RowBuffer> for T {
    #[instrument(skip_all)]
    fn run_stage_on(
        &self,
        ExtraInfo {
            xsize,
            current_row,
            group_origin,
            out_extra_x,
            image_height,
            is_first_xgroup,
            is_last_xgroup,
        }: ExtraInfo,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        let ibordery = Self::BORDER.1 as isize;
        let x0 = RowBuffer::x0_offset::<T::InputT>();
        let xpre = if is_first_xgroup {
            0
        } else {
            out_extra_x.shrc(T::SHIFT.0)
        };
        let xstart = x0 - xpre;
        let xend = x0
            + xsize
            + if is_last_xgroup {
                0
            } else {
                out_extra_x.shrc(T::SHIFT.0)
            };
        let input_rows: Vec<_> = input_buffers
            .iter()
            .map(|x| {
                (-ibordery..=ibordery)
                    .map(|iy| {
                        &x.get_row::<T::InputT>(mirror(current_row as isize + iy, image_height))
                            [xstart - Self::BORDER.0 as usize..]
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut output_rows: Vec<_> = output_buffers
            .iter_mut()
            .map(|x| {
                x.get_rows_mut::<T::OutputT>(
                    (current_row << T::SHIFT.1)..((current_row + 1) << T::SHIFT.1),
                    RowBuffer::x0_offset::<T::OutputT>() - (xpre << T::SHIFT.0),
                )
            })
            .collect();

        let input_rows: Vec<_> = input_rows.iter().map(|x| &x[..]).collect();
        let mut output_rows: Vec<_> = output_rows.iter_mut().map(|x| &mut x[..]).collect();

        self.process_row_chunk(
            (group_origin.0 - xpre, current_row),
            xend - xstart,
            &input_rows[..],
            &mut output_rows[..],
            state,
        );
    }
}
