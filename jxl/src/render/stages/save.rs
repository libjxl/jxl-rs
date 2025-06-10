// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
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
}

#[allow(unused)]
impl<T: ImageDataType> SaveStage<T> {
    pub(crate) fn new(
        stage_type: SaveStageType,
        channel: usize,
        size: (usize, usize),
        scale: T,
    ) -> Result<SaveStage<T>> {
        Ok(SaveStage {
            stage_type,
            channel,
            buf: Image::new(size)?,
            scale,
        })
    }

    pub(crate) fn new_with_buffer(
        stage_type: SaveStageType,
        channel: usize,
        img: Image<T>,
        scale: T,
    ) -> SaveStage<T> {
        SaveStage {
            stage_type,
            channel,
            buf: img,
            scale,
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
        let input = &mut row[0];
        // TODO(veluca): consider making `process_row_chunk` return a Result.
        let mut outbuf = self.buf.as_rect_mut();
        if position.1 >= outbuf.size().1 {
            return;
        }
        xsize = xsize.min(outbuf.size().0 - position.0);
        let mut outbuf = outbuf
            .rect(Rect {
                origin: position,
                size: (xsize, 1),
            })
            .expect("mismatch in image size");
        for ix in 0..xsize {
            outbuf.row(0)[ix] = input[ix] * self.scale;
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
        let mut save_stage = SaveStage::<u8>::new(SaveStageType::Output, 0, (128, 128), 1)?;
        let mut rng = XorShiftRng::seed_from_u64(0);
        let src = Image::<u8>::new_random((128, 128), &mut rng)?;

        for i in 0..128 {
            save_stage.process_row_chunk((0, i), 128, &mut [src.as_rect().row(i)]);
        }

        src.as_rect().check_equal(save_stage.buffer().as_rect());

        Ok(())
    }
}
