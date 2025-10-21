// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{
    render::{
        RunInPlaceStage,
        internal::{PipelineBuffer, RunInOutStage},
    },
    util::tracing_wrappers::*,
};

use super::{
    super::{RenderPipelineInOutStage, RenderPipelineInPlaceStage},
    row_buffers::RowBuffer,
};

pub struct ExtraInfo {
    pub(super) xsize: usize,
    pub(super) current_row: usize,
    pub(super) group_origin: (usize, usize),
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
        }: ExtraInfo,
        buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        let xoff = RowBuffer::x0_offset::<T::Type>();
        let mut rows: Vec<_> = buffers
            .iter_mut()
            .map(|x| &mut x.get_row_mut::<T::Type>(current_row)[xoff..])
            .collect();

        self.process_row_chunk((group_origin.0, current_row), xsize, &mut rows[..], state);
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
        }: ExtraInfo,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        let xoff_in = RowBuffer::x0_offset::<T::InputT>();
        let input_rows: Vec<_> = input_buffers
            .iter()
            .map(|x| vec![&x.get_row::<T::InputT>(current_row)[xoff_in..]])
            .collect();
        let xoff_out = RowBuffer::x0_offset::<T::OutputT>();
        let mut output_rows: Vec<_> = output_buffers
            .iter_mut()
            .map(|x| x.advance_rows::<T::OutputT>(1 << T::SHIFT.1, xoff_out))
            .collect();

        let input_rows: Vec<_> = input_rows.iter().map(|x| &x[..]).collect();
        let mut output_rows: Vec<_> = output_rows.iter_mut().map(|x| &mut x[..]).collect();

        self.process_row_chunk(
            (group_origin.0, current_row),
            xsize,
            &input_rows[..],
            &mut output_rows[..],
            state,
        );
    }
}
