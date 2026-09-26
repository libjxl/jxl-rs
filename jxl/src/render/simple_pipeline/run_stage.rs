// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use crate::image::{Image, ImageDataType};
use crate::render::internal::PipelineBuffer;
use crate::render::{
    ErasedLocalState, RenderPipelineInOutStage, RenderPipelineInPlaceStage, RowBuffer,
    RunInOutStage, RunInPlaceStage,
};
use crate::util::tracing_wrappers::*;
use crate::util::{SmallVec, StackOnly, mirror, round_up_size_to_cache_line};

impl PipelineBuffer for Image<f64> {
    type InPlaceExtraInfo = usize;
    type InOutExtraInfo = usize;
}

impl<T: RenderPipelineInPlaceStage> RunInPlaceStage<Image<f64>> for T {
    fn run_stage_on(
        &self,
        chunk_size: usize,
        buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut ErasedLocalState>,
    ) {
        debug!("running inplace stage '{self}' in simple pipeline");
        let numc = buffers.len();
        if numc == 0 {
            return;
        }
        let size = buffers[0].size();
        for b in buffers.iter() {
            assert_eq!(size, b.size());
        }
        let mut buffer =
            vec![
                vec![T::Type::default(); round_up_size_to_cache_line::<T::Type>(chunk_size)];
                numc
            ];
        for y in 0..size.1 {
            for x in (0..size.0).step_by(chunk_size) {
                let xsize = size.0.min(x + chunk_size) - x;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    let in_row = buffers[c].row(y);
                    for ix in 0..xsize {
                        buffer[c][ix] = T::Type::from_f64(in_row[x + ix]);
                    }
                }
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [_]).collect();
                self.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut(), false);
                for c in 0..numc {
                    let out_row = buffers[c].row_mut(y);
                    for ix in 0..xsize {
                        out_row[x + ix] = buffer[c][ix].to_f64();
                    }
                }
            }
        }
    }
}

impl<T: RenderPipelineInOutStage> RunInOutStage<Image<f64>> for T {
    #[instrument(skip_all)]
    fn run_stage_on(
        &self,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [Image<f64>],
        mut state: Option<&mut ErasedLocalState>,
    ) {
        assert_ne!(chunk_size, 0);
        debug!("running inout stage '{self}' in simple pipeline");
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        assert_eq!(output_buffers.len(), numc);
        let input_size = input_buffers[0].size();
        let output_size = output_buffers[0].size();
        for c in 1..numc {
            assert_eq!(input_size, input_buffers[c].size());
            assert_eq!(output_size, output_buffers[c].size());
        }
        debug!(
            ?input_size,
            ?output_size,
            SHIFT = ?Self::SHIFT,
            BORDER = ?Self::BORDER,
            numc
        );
        assert_eq!(input_size.0, output_size.0.div_ceil(1 << Self::SHIFT.0));
        assert_eq!(input_size.1, output_size.1.div_ceil(1 << Self::SHIFT.1));
        let mut buffer_in: Vec<RowBuffer> = (0..numc)
            .map(|_| {
                RowBuffer::new(
                    T::InputT::DATA_TYPE_ID,
                    Self::BORDER.1 as usize,
                    0,
                    0,
                    chunk_size,
                )
            })
            .collect::<crate::error::Result<Vec<_>>>()
            .unwrap();

        let mut buffer_out: Vec<RowBuffer> = (0..numc)
            .map(|_| {
                RowBuffer::new(
                    T::OutputT::DATA_TYPE_ID,
                    0,
                    Self::SHIFT.1 as usize,
                    Self::SHIFT.0 as usize,
                    chunk_size << Self::SHIFT.0,
                )
            })
            .collect::<crate::error::Result<Vec<_>>>()
            .unwrap();

        let in_x0 = RowBuffer::x0_offset::<T::InputT>();
        let out_x0 = RowBuffer::x0_offset::<T::OutputT>();

        for y in 0..input_size.1 {
            for x in (0..input_size.0).step_by(chunk_size) {
                let border_x = Self::BORDER.0 as isize;
                let border_y = Self::BORDER.1 as isize;
                let xsize = input_size.0.min(x + chunk_size) - x;
                let xs = xsize as isize;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    for iy in -border_y..=border_y {
                        let imgy = mirror(y as isize + iy, input_size.1);
                        let in_row = input_buffers[c].row(imgy);
                        let buf_in_row = buffer_in[c].get_row_mut::<T::InputT>(imgy);
                        for ix in (-border_x..0).chain(xs..xs + border_x) {
                            let imgx = mirror(x as isize + ix, input_size.0);
                            buf_in_row[(in_x0 as isize + ix) as usize] =
                                T::InputT::from_f64(in_row[imgx]);
                        }
                        for ix in 0..xsize {
                            buf_in_row[in_x0 + ix] = T::InputT::from_f64(in_row[x + ix]);
                        }
                    }
                }

                {
                    let in_refs: SmallVec<&RowBuffer, 8, StackOnly> = buffer_in.iter().collect();
                    let input_rows = crate::render::Channels::from_row_buffers(
                        &in_refs,
                        in_x0,
                        y,
                        Self::BORDER.1 as usize,
                        input_size.1,
                    );

                    let mut output_rows = crate::render::ChannelsMut::from_row_buffers(
                        &mut buffer_out,
                        out_x0,
                        y << Self::SHIFT.1,
                        1 << Self::SHIFT.1,
                    );

                    self.process_row_chunk(
                        (x, y),
                        xsize,
                        &input_rows,
                        &mut output_rows,
                        state.as_deref_mut(),
                        false,
                    );
                }

                let stripe_xsize =
                    (xsize << Self::SHIFT.0).min(output_size.0 - (x << Self::SHIFT.0));
                let stripe_ysize =
                    (1usize << Self::SHIFT.1).min(output_size.1 - (y << Self::SHIFT.1));
                for c in 0..numc {
                    for iy in 0..stripe_ysize {
                        let out_row = output_buffers[c].row_mut((y << Self::SHIFT.1) + iy);
                        let buf_out_row = &buffer_out[c]
                            .get_row::<T::OutputT>((y << Self::SHIFT.1) + iy)
                            [out_x0..out_x0 + stripe_xsize];
                        for ix in 0..stripe_xsize {
                            out_row[(x << Self::SHIFT.0) + ix] = buf_out_row[ix].to_f64();
                        }
                    }
                }
            }
        }
    }
}
