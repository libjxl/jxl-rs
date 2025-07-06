// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Mutex;

use crate::{
    error::Result,
    headers::Orientation,
    image::{Image, ImageDataType, Rect},
    render::{RenderPipelineInspectStage, RenderPipelineStage},
};

pub enum SaveStageType {
    Output,
    Reference,
    Lf,
}

pub struct SaveStage<T: ImageDataType> {
    pub stage_type: SaveStageType,
    buf: Mutex<Image<T>>,
    channel: usize,
    // TODO(szabadka): Have a fixed scale per data-type and make the datatype conversions do
    // the scaling.
    scale: T,
    orientation: Orientation,
}

#[allow(unused)]
impl<T: ImageDataType> SaveStage<T> {
    pub(crate) fn new(
        stage_type: SaveStageType,
        channel: usize,
        size: (usize, usize),
        scale: T,
        orientation: Orientation,
    ) -> Result<SaveStage<T>> {
        let buf_size = if orientation.is_transposing() {
            (size.1, size.0)
        } else {
            size
        };
        Ok(SaveStage {
            stage_type,
            channel,
            buf: Mutex::new(Image::new(buf_size)?),
            scale,
            orientation,
        })
    }

    pub(crate) fn new_with_buffer(
        stage_type: SaveStageType,
        channel: usize,
        img: Image<T>,
        scale: T,
        orientation: Orientation,
    ) -> SaveStage<T> {
        SaveStage {
            stage_type,
            channel,
            buf: Mutex::new(img),
            scale,
            orientation,
        }
    }

    pub(crate) fn buffer(&self) -> impl std::ops::Deref<Target = Image<T>> {
        self.buf.lock().unwrap()
    }

    pub(crate) fn into_buffer(self) -> Image<T> {
        self.buf.into_inner().unwrap()
    }
}

impl<T: ImageDataType> std::fmt::Display for SaveStage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "save channel {} (type {:?}) scale {:?}",
            self.channel,
            T::DATA_TYPE_ID,
            self.scale,
        )
    }
}

impl<T: ImageDataType + std::ops::Mul<Output = T>> RenderPipelineStage for SaveStage<T> {
    type Type = RenderPipelineInspectStage<T>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

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
}

#[cfg(test)]
mod test {
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
}
