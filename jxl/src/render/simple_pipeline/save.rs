// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{
    api::{JxlColorType, JxlDataFormat},
    error::Result,
    headers::Orientation,
    image::{DataTypeTag, Image, ImageDataType},
    util::tracing_wrappers::debug,
};

use super::RunStage;

pub struct SaveStage {
    channels: Vec<usize>,
    orientation: Orientation,
    output_buffer_index: usize,
    color_type: JxlColorType,
    data_format: JxlDataFormat,
}

#[allow(unused)]
impl SaveStage {
    pub(crate) fn new(
        channels: &[usize],
        orientation: Orientation,
        output_buffer_index: usize,
        color_type: JxlColorType,
        data_format: JxlDataFormat,
    ) -> SaveStage {
        Self {
            channels: channels.to_vec(),
            orientation,
            output_buffer_index,
            color_type,
            data_format,
        }
    }
}

impl std::fmt::Display for SaveStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "save channels {:?} (type {:?} {:?})",
            self.channels, self.color_type, self.data_format
        )
    }
}

impl SaveStage {
    /*
    fn process_row_chunk(
        &self,
        position: (usize, usize),
        mut xsize: usize,
        row: &mut [&[T]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let input = row[0];
        let mut buf = self.buf.lock().unwrap();
        let mut outbuf_rect = buf.as_rect_mut();
        let (out_w, out_h) = outbuf_rect.size();

        // Establish source dimensions based on the orientation.
        let (w_src, h_src) = if self.orientation.is_transposing() {
            (out_h, out_w) // Swapped
        } else {
            (out_w, out_h)
        };

        // Perform boundary checks against source dimensions.
        if position.1 >= h_src {
            return;
        }
        xsize = xsize.min(w_src - position.0);
        if xsize == 0 {
            return;
        }

        match self.orientation {
            // non-transposing cases
            Orientation::Identity => {
                let mut out_row = outbuf_rect
                    .rect(Rect {
                        origin: position,
                        size: (xsize, 1),
                    })
                    .unwrap();
                for (out_pixel, &in_pixel) in out_row.row(0).iter_mut().zip(input.iter()) {
                    *out_pixel = in_pixel * self.scale;
                }
            }
            Orientation::FlipHorizontal => {
                let mut out_row = outbuf_rect
                    .rect(Rect {
                        origin: (0, position.1),
                        size: (out_w, 1),
                    })
                    .unwrap();
                for (i, &in_pixel) in input.iter().enumerate().take(xsize) {
                    out_row.row(0)[out_w - 1 - position.0 - i] = in_pixel * self.scale;
                }
            }
            Orientation::FlipVertical => {
                let y_out = out_h - 1 - position.1;
                let mut out_row = outbuf_rect
                    .rect(Rect {
                        origin: (position.0, y_out),
                        size: (xsize, 1),
                    })
                    .unwrap();
                for (out_pixel, &in_pixel) in out_row.row(0).iter_mut().zip(input.iter()) {
                    *out_pixel = in_pixel * self.scale;
                }
            }
            Orientation::Rotate180 => {
                let y_out = out_h - 1 - position.1;
                let mut out_row = outbuf_rect
                    .rect(Rect {
                        origin: (0, y_out),
                        size: (out_w, 1),
                    })
                    .unwrap();
                for (i, &in_pixel) in input.iter().enumerate().take(xsize) {
                    out_row.row(0)[out_w - 1 - position.0 - i] = in_pixel * self.scale;
                }
            }

            // transposing cases
            Orientation::Transpose
            | Orientation::Rotate90
            | Orientation::AntiTranspose
            | Orientation::Rotate270 => {
                let y_src = position.1;
                for (i, &in_pixel) in input.iter().enumerate().take(xsize) {
                    let x_src = position.0 + i;

                    let (x_dest, y_dest) = match self.orientation {
                        Orientation::Transpose => (y_src, x_src),
                        Orientation::Rotate90 => (y_src, w_src - 1 - x_src),
                        Orientation::AntiTranspose => (h_src - 1 - y_src, w_src - 1 - x_src),
                        Orientation::Rotate270 => (h_src - 1 - y_src, x_src),
                        _ => unreachable!(),
                    };

                    // Final check to ensure we don't write out of the destination buffer bounds.
                    if x_dest < out_w && y_dest < out_h {
                        outbuf_rect.row(y_dest)[x_dest] = in_pixel * self.scale;
                    }
                }
            }
        }
    }
    */
}

// TODO(veluca): get rid of this, and make a enum { Save, Process }.
impl RunStage for SaveStage {
    fn run_stage_on(
        &self,
        _chunk_size: usize,
        _input_buffers: &[&Image<f64>],
        _output_buffers: &mut [&mut Image<f64>],
        _state: Option<&mut dyn Any>,
    ) {
        debug!("running save stage '{self}' in simple pipeline");
        /*
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        let size = input_buffers[0].size();
        for b in input_buffers.iter() {
            assert_eq!(size, b.size());
        }
        let mut buffer =
            vec![vec![T::default(); round_up_size_to_two_cache_lines::<T>(chunk_size)]; numc];
        for y in 0..size.1 {
            for x in (0..size.0).step_by(chunk_size) {
                let xsize = size.0.min(x + chunk_size) - x;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    let in_rect = input_buffers[c].as_rect();
                    let in_row = in_rect.row(y);
                    for ix in 0..xsize {
                        buffer[c][ix] = T::from_f64(in_row[x + ix]);
                    }
                }
                let mut row: Vec<_> = buffer.iter().map(|x| x as &[T]).collect();
                self.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
            }
        }
        */
    }

    fn init_local_state(&self) -> Result<Option<Box<dyn Any>>> {
        Ok(None)
    }
    fn shift(&self) -> (u8, u8) {
        (0, 0)
    }
    fn new_size(&self, size: (usize, usize)) -> (usize, usize) {
        size
    }
    fn uses_channel(&self, c: usize) -> bool {
        self.channels.contains(&c)
    }
    fn input_type(&self) -> DataTypeTag {
        // TODO(veluca): this should be data-dependent.
        f32::DATA_TYPE_ID
    }
    fn output_type(&self) -> DataTypeTag {
        self.input_type()
    }
}

#[cfg(test)]
mod test {
    /*
        use super::*;
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;
        use test_log::test;

        #[test]
        fn save_stage() -> Result<()> {
            let save_stage = SaveStage::<u8>::new(
                SaveStageType::Output,
                0,
                (128, 128),
                1,
                Orientation::Identity,
            )?;
            let mut rng = XorShiftRng::seed_from_u64(0);
            let src = Image::<u8>::new_random((128, 128), &mut rng)?;

            for i in 0..128 {
                save_stage.process_row_chunk((0, i), 128, &mut [src.as_rect().row(i)], None);
            }

            src.as_rect().check_equal(save_stage.buffer().as_rect());

            Ok(())
        }

        macro_rules! test_orientation {
            ($test_name:ident, $orientation:expr, $transform:expr) => {
                #[test]
                fn $test_name() -> Result<()> {
                    // Source dimensions
                    let (w, h) = (32, 16);
                    let mut rng = XorShiftRng::seed_from_u64(0);
                    let src = Image::<u8>::new_random((w, h), &mut rng)?;
                    let orientation = $orientation;

                    // SaveStage will create its buffer with the correct (possibly swapped) dimensions.
                    let save_stage =
                        SaveStage::<u8>::new(SaveStageType::Output, 0, (w, h), 1, orientation)?;

                    for y in 0..h {
                        save_stage.process_row_chunk((0, y), w, &mut [src.as_rect().row(y)], None);
                    }

                    let (out_w, out_h) = save_stage.buffer().size();

                    let mut expected = Image::<u8>::new((out_w, out_h))?;

                    // The transform is a closure: |x_dest, y_dest, w_src, h_src| -> (x_src, y_src)
                    let transform = $transform;

                    // Iterate over the DESTINATION image pixels.
                    for y_dest in 0..out_h {
                        for x_dest in 0..out_w {
                            // For each destination pixel, find its corresponding source pixel.
                            let (src_x, src_y) = transform(x_dest, y_dest, w, h);
                            expected.as_rect_mut().row(y_dest)[x_dest] =
                                src.as_rect().row(src_y)[src_x];
                        }
                    }

                    expected
                        .as_rect()
                        .check_equal(save_stage.buffer().as_rect());

                    Ok(())
                }
            };
        }
        test_orientation!(orientation_identity, Orientation::Identity, |x, y, _, _| (
            x, y
        ));

        test_orientation!(
            orientation_flip_horizontal,
            Orientation::FlipHorizontal,
            |x, y, w, _| (w - 1 - x, y)
        );

        test_orientation!(
            orientation_flip_vertical,
            Orientation::FlipVertical,
            |x, y, _, h| (x, h - 1 - y)
        );

        test_orientation!(
            orientation_rotate_180,
            Orientation::Rotate180,
            |x, y, w, h| (w - 1 - x, h - 1 - y)
        );

        // transposing orientations

        test_orientation!(
            orientation_transpose,
            Orientation::Transpose,
            |x_dest, y_dest, _, _| (y_dest, x_dest)
        );

        test_orientation!(
            orientation_rotate_90,
            Orientation::Rotate90,
            |x_dest, y_dest, w_src, _| (w_src - 1 - y_dest, x_dest)
        );

        test_orientation!(
            orientation_anti_transpose,
            Orientation::AntiTranspose,
            |x_dest, y_dest, w_src, h_src| (w_src - 1 - y_dest, h_src - 1 - x_dest)
        );

        test_orientation!(
            orientation_rotate_270,
            Orientation::Rotate270,
            |x_dest, y_dest, _, h_src| (y_dest, h_src - 1 - x_dest)
        );
    */
}
