// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use half::f16;

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    error::Result,
    render::save::SaveStage,
};

use super::row_buffers::RowBuffer;

// Placeholder slow implementation.
impl SaveStage {
    // Takes as input only those channels that are *actually* saved.
    pub(super) fn save_lowmem(
        &self,
        data: &[&RowBuffer],
        buffers: &mut [Option<JxlOutputBuffer>],
        size: (usize, usize),
        image_y: usize,
        group_y0: usize,
    ) -> Result<()> {
        let Some(buf) = buffers[self.output_buffer_index].as_mut() else {
            return Ok(());
        };

        macro_rules! write_pixel {
            ($px: expr, $endianness: expr, $y: expr, $x: expr) => {
                let px = $px;
                let px_bytes = if $endianness == Endianness::LittleEndian {
                    px.to_le_bytes()
                } else {
                    px.to_be_bytes()
                };
                buf.write_bytes($y, $x, &px_bytes);
            };
        }

        // TODO(veluca): this is very slow, speed it up.
        for (c, d) in data.iter().enumerate() {
            let nc = self.channels.len();
            let (x0, y0) = self
                .orientation
                .display_pixel((0, image_y - group_y0), size);
            let (x1, y1) = self
                .orientation
                .display_pixel((1, image_y - group_y0), size);
            let dx = x1 as isize - x0 as isize;
            let dy = y1 as isize - y0 as isize;
            match self.data_format {
                JxlDataFormat::U8 { .. } => {
                    let src_row = d.get_row::<u8>(image_y);
                    for ix in 0..size.0 {
                        let px = src_row[RowBuffer::x0_offset::<u8>() + ix];
                        let y = y0 + (dy * ix as isize) as usize;
                        let x = x0 + (dx * ix as isize) as usize;
                        write_pixel!(px, Endianness::LittleEndian, y, x * nc + c);
                    }
                }
                JxlDataFormat::U16 { endianness, .. } => {
                    let src_row = d.get_row::<u16>(image_y);
                    for ix in 0..size.0 {
                        let px = src_row[RowBuffer::x0_offset::<u16>() + ix];
                        let y = y0 + (dy * ix as isize) as usize;
                        let x = x0 + (dx * ix as isize) as usize;
                        write_pixel!(px, endianness, y, (x * nc + c) * 2);
                    }
                }
                JxlDataFormat::F16 { endianness, .. } => {
                    let src_row = d.get_row::<f16>(image_y);
                    for ix in 0..size.0 {
                        let px = src_row[RowBuffer::x0_offset::<f16>() + ix];
                        let y = y0 + (dy * ix as isize) as usize;
                        let x = x0 + (dx * ix as isize) as usize;
                        write_pixel!(px, endianness, y, (x * nc + c) * 2);
                    }
                }
                JxlDataFormat::F32 { endianness, .. } => {
                    let src_row = d.get_row::<f32>(image_y);
                    for ix in 0..size.0 {
                        let px = src_row[RowBuffer::x0_offset::<f32>() + ix];
                        let y = y0 + (dy * ix as isize) as usize;
                        let x = x0 + (dx * ix as isize) as usize;
                        write_pixel!(px, endianness, y, (x * nc + c) * 4);
                    }
                }
            }
        }
        Ok(())
    }
}
