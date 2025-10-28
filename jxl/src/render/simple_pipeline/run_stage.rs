use std::any::Any;

use crate::{
    image::{Image, ImageDataType},
    render::{
        RenderPipelineInOutStage, RenderPipelineInPlaceStage, RunInOutStage, RunInPlaceStage,
        internal::PipelineBuffer,
    },
    simd::round_up_size_to_two_cache_lines,
    util::tracing_wrappers::*,
};

impl PipelineBuffer for Image<f64> {
    type InPlaceExtraInfo = usize;
    type InOutExtraInfo = usize;
}

impl<T: RenderPipelineInPlaceStage> RunInPlaceStage<Image<f64>> for T {
    fn run_stage_on(
        &self,
        chunk_size: usize,
        buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut dyn Any>,
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
                vec![T::Type::default(); round_up_size_to_two_cache_lines::<T::Type>(chunk_size)];
                numc
            ];
        for y in 0..size.1 {
            for x in (0..size.0).step_by(chunk_size) {
                let xsize = size.0.min(x + chunk_size) - x;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    let in_rect = buffers[c].as_rect();
                    let in_row = in_rect.row(y);
                    for ix in 0..xsize {
                        buffer[c][ix] = T::Type::from_f64(in_row[x + ix]);
                    }
                }
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [_]).collect();
                self.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
                for c in 0..numc {
                    let mut out_rect = buffers[c].as_rect_mut();
                    let out_row = out_rect.row(y);
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
        output_buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut dyn Any>,
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
        let mut buffer_in = vec![
            vec![
                vec![
                    T::InputT::default();
                    // Double rounding make sure that we always have enough buffer for reading a whole SIMD lane.
                    round_up_size_to_two_cache_lines::<T::OutputT>(
                        round_up_size_to_two_cache_lines::<T::OutputT>(chunk_size)
                            + T::BORDER.0 as usize * 2
                    )
                ];
                T::BORDER.1 as usize * 2 + 1
            ];
            numc
        ];
        let mut buffer_out = vec![
            vec![
                vec![
                    T::OutputT::default();
                    round_up_size_to_two_cache_lines::<T::OutputT>(chunk_size)
                        << T::SHIFT.0
                ];
                1 << T::SHIFT.1
            ];
            numc
        ];

        let mirror = |mut v: i64, size: i64| {
            while v < 0 || v >= size {
                if v < 0 {
                    v = -v - 1;
                }
                if v >= size {
                    v = size + (size - v) - 1;
                }
            }
            v as usize
        };
        for y in 0..input_size.1 {
            for x in (0..input_size.0).step_by(chunk_size) {
                let border_x = Self::BORDER.0 as i64;
                let border_y = Self::BORDER.1 as i64;
                let xsize = input_size.0.min(x + chunk_size) - x;
                let xs = xsize as i64;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    let in_rect = input_buffers[c].as_rect();
                    for iy in -border_y..=border_y {
                        let imgy = mirror(y as i64 + iy, input_size.1 as i64);
                        let in_row = in_rect.row(imgy);
                        let buf_in_row = &mut buffer_in[c][(iy + border_y) as usize];
                        for ix in (-border_x..0).chain(xs..xs + border_x) {
                            let imgx = mirror(x as i64 + ix, input_size.0 as i64);
                            buf_in_row[(ix + border_x) as usize] =
                                T::InputT::from_f64(in_row[imgx]);
                        }
                        for ix in 0..xsize {
                            buf_in_row[ix + border_x as usize] =
                                T::InputT::from_f64(in_row[x + ix]);
                        }
                    }
                }
                let buffer_in_ref: Vec<Vec<_>> = buffer_in
                    .iter()
                    .map(|x| x.iter().map(|x| x as &[_]).collect())
                    .collect();
                let mut buffer_out_ref: Vec<Vec<_>> = buffer_out
                    .iter_mut()
                    .map(|x| x.iter_mut().map(|x| x as &mut [_]).collect::<Vec<_>>())
                    .collect();
                let in_rows: Vec<_> = buffer_in_ref.iter().map(|x| x as &[_]).collect();
                let mut out_rows: Vec<_> =
                    buffer_out_ref.iter_mut().map(|x| x as &mut [_]).collect();
                self.process_row_chunk(
                    (x, y),
                    xsize,
                    &in_rows,
                    &mut out_rows,
                    state.as_deref_mut(),
                );
                let stripe_xsize =
                    (xsize << Self::SHIFT.0).min(output_size.0 - (x << Self::SHIFT.0));
                let stripe_ysize =
                    (1usize << Self::SHIFT.1).min(output_size.1 - (y << Self::SHIFT.1));
                for c in 0..numc {
                    let mut out_rect = output_buffers[c].as_rect_mut();
                    for iy in 0..stripe_ysize {
                        let out_row = out_rect.row((y << Self::SHIFT.1) + iy);
                        for ix in 0..stripe_xsize {
                            out_row[(x << Self::SHIFT.0) + ix] = buffer_out[c][iy][ix].to_f64();
                        }
                    }
                }
            }
        }
    }
}
