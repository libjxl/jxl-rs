// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    headers::Orientation,
    image::{Image, ImageDataType, Rect},
    render::{RenderPipelineInspectStage, RenderPipelineStage},
};

pub enum SaveStageType {
    Output,
    Reference,
}

pub struct SaveStage<T: ImageDataType> {
    pub stage_type: SaveStageType,
    buf: Image<T>,
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
            buf: Image::new(buf_size)?,
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
            buf: img,
            scale,
            orientation,
        }
    }

    pub(crate) fn buffer(&self) -> &Image<T> {
        &self.buf
    }

    pub(crate) fn into_buffer(self) -> Image<T> {
        self.buf
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

    fn process_row_chunk(&mut self, position: (usize, usize), mut xsize: usize, row: &mut [&[T]]) {
        let input = row[0];
        let mut outbuf_rect = self.buf.as_rect_mut();
        let (width, height) = outbuf_rect.size();

        // Single, corrected boundary check.
        match self.orientation {
            Orientation::Identity | Orientation::FlipHorizontal => {
                if position.1 >= height {
                    return;
                }
            }
            Orientation::FlipVertical | Orientation::Rotate180 => {
                if position.1 >= height {
                    return;
                }
            }
            _ => {} // No check needed for unimplemented orientations.
        }

        xsize = xsize.min(width - position.0);

        match self.orientation {
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
                        size: (width, 1),
                    })
                    .unwrap();
                for (i, &in_pixel) in input.iter().enumerate().take(xsize) {
                    out_row.row(0)[width - 1 - position.0 - i] = in_pixel * self.scale;
                }
            }
            Orientation::FlipVertical => {
                let y_out = height - 1 - position.1;
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
                let y_out = height - 1 - position.1;
                let mut out_row = outbuf_rect
                    .rect(Rect {
                        origin: (0, y_out),
                        size: (width, 1),
                    })
                    .unwrap();
                for (i, &in_pixel) in input.iter().enumerate().take(xsize) {
                    out_row.row(0)[width - 1 - position.0 - i] = in_pixel * self.scale;
                }
            }
            _ => {
                unimplemented!("Can only save images in Identity, Flip, or Rotate180 orientations, but got {:?}", self.orientation);
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
        let mut save_stage = SaveStage::<u8>::new(
            SaveStageType::Output,
            0,
            (128, 128),
            1,
            Orientation::Identity,
        )?;
        let mut rng = XorShiftRng::seed_from_u64(0);
        let src = Image::<u8>::new_random((128, 128), &mut rng)?;

        for i in 0..128 {
            save_stage.process_row_chunk((0, i), 128, &mut [src.as_rect().row(i)]);
        }

        src.as_rect().check_equal(save_stage.buffer().as_rect());

        Ok(())
    }

    macro_rules! test_orientation {
        ($test_name:ident, $orientation:expr, $transform:expr) => {
            #[test]
            fn $test_name() -> Result<()> {
                let (w, h) = (32, 16);
                let mut rng = XorShiftRng::seed_from_u64(0);
                let src = Image::<u8>::new_random((w, h), &mut rng)?;
                let orientation = $orientation;

                let mut save_stage =
                    SaveStage::<u8>::new(SaveStageType::Output, 0, (w, h), 1, orientation)?;

                for y in 0..h {
                    save_stage.process_row_chunk((0, y), w, &mut [src.as_rect().row(y)]);
                }

                // Create the expected result using the provided transform logic
                let mut expected = Image::<u8>::new((w, h))?;
                // The transform is a closure: |x, y, w, h| -> (src_x, src_y)
                let transform = $transform;
                for y in 0..h {
                    for x in 0..w {
                        let (src_x, src_y) = transform(x, y, w, h);
                        expected.as_rect_mut().row(y)[x] = src.as_rect().row(src_y)[src_x];
                    }
                }

                expected
                    .as_rect()
                    .check_equal(save_stage.buffer().as_rect());

                Ok(())
            }
        };
    }

    test_orientation!(test_identity, Orientation::Identity, |x, y, _, _| (x, y));

    test_orientation!(
        test_flip_horizontal,
        Orientation::FlipHorizontal,
        |x, y, w, _| (w - 1 - x, y)
    );

    test_orientation!(
        test_flip_vertical,
        Orientation::FlipVertical,
        |x, y, _, h| (x, h - 1 - y)
    );

    test_orientation!(test_rotate_180, Orientation::Rotate180, |x, y, w, h| (
        w - 1 - x,
        h - 1 - y
    ));
}
