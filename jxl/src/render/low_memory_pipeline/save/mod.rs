// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    error::Result,
    headers::Orientation,
    render::save::SaveStage,
    util::f16,
};

use super::row_buffers::RowBuffer;

mod identity;

// Placeholder slow implementation.
impl SaveStage {
    // Takes as input only those channels that are *actually* saved.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn save_lowmem(
        &self,
        data: &[&RowBuffer],
        buffers: &mut [Option<JxlOutputBuffer>],
        group_size: (usize, usize),
        frame_y: usize,
        group_origin: (usize, usize),
        full_image_size: (usize, usize),
        frame_origin: (isize, isize),
    ) -> Result<()> {
        let Some(buf) = buffers[self.output_buffer_index].as_mut() else {
            return Ok(());
        };

        let group_y = frame_y - group_origin.1;

        let relative_full_image_start = (
            -frame_origin.0 - (group_origin.0 as isize),
            -frame_origin.1 - (group_origin.1 as isize),
        );

        let relative_full_image_end = (
            relative_full_image_start.0 + full_image_size.0 as isize,
            relative_full_image_start.1 + full_image_size.1 as isize,
        );

        let save_start = (
            relative_full_image_start.0.max(0) as usize,
            relative_full_image_start.1.max(0) as usize,
        );

        let save_end = (
            relative_full_image_end.0.clamp(0, group_size.0 as isize) as usize,
            relative_full_image_end.1.clamp(0, group_size.1 as isize) as usize,
        );

        // If the visible area were empty, we'd have gotten None for the buffer.
        assert!(save_start.0 < save_end.0);
        assert!(save_start.1 < save_end.1);

        if !(save_start.1..save_end.1).contains(&group_y) {
            // The current row is outside the visible area - skip rendering it.
            return Ok(());
        }

        let relative_y = group_y - save_start.1;

        let save_size = (save_end.0 - save_start.0, save_end.1 - save_start.1);

        // Fast path for identity orientation when not premultiplying
        // (premultiplication requires per-pixel alpha access which the fast path doesn't support)
        let num_fast = if self.premultiply_output {
            0
        } else {
            match self.orientation {
                Orientation::Identity => identity::store(
                    data,
                    frame_y,
                    save_start.0..save_end.0,
                    buf,
                    relative_y,
                    self.data_format,
                ),
                _ => 0,
            }
        };

        // TODO(veluca): this is very slow, implement more fast paths.

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

        // Determine if we need to premultiply and which channel index is alpha
        // When premultiplying, the last channel in data is the alpha channel
        let alpha_channel = if self.premultiply_output && data.len() > 1 {
            Some(data.len() - 1)
        } else {
            None
        };

        for (c, d) in data.iter().enumerate() {
            let nc = self.output_channels();
            let (x0, y0) = self.orientation.display_pixel((0, relative_y), save_size);
            let (x1, y1) = self.orientation.display_pixel((1, relative_y), save_size);
            let x0 = x0 as isize;
            let y0 = y0 as isize;
            let dx = x1 as isize - x0;
            let dy = y1 as isize - y0;

            // Check if this is a color channel that needs premultiplication
            // (all channels except the last one when alpha is present)
            let is_color_channel = alpha_channel.is_some() && c < data.len() - 1;

            match self.data_format {
                JxlDataFormat::U8 { .. } => {
                    let src_row = d.get_row::<u8>(frame_y);
                    let alpha_row = alpha_channel.map(|ac| data[ac].get_row::<u8>(frame_y));
                    for ix in (save_start.0..save_end.0).skip(num_fast) {
                        let mut px = src_row[RowBuffer::x0_offset::<u8>() + ix];
                        if is_color_channel {
                            let alpha = alpha_row.unwrap()[RowBuffer::x0_offset::<u8>() + ix];
                            // Premultiply: scale by alpha/255
                            px = ((px as u32 * alpha as u32 + 127) / 255) as u8;
                        }
                        let y = (y0 + (dy * (ix - save_start.0) as isize)) as usize;
                        let x = (x0 + (dx * (ix - save_start.0) as isize)) as usize;
                        write_pixel!(px, Endianness::LittleEndian, y, x * nc + c);
                    }
                }
                JxlDataFormat::U16 { endianness, .. } => {
                    let src_row = d.get_row::<u16>(frame_y);
                    let alpha_row = alpha_channel.map(|ac| data[ac].get_row::<u16>(frame_y));
                    for ix in (save_start.0..save_end.0).skip(num_fast) {
                        let mut px = src_row[RowBuffer::x0_offset::<u16>() + ix];
                        if is_color_channel {
                            let alpha = alpha_row.unwrap()[RowBuffer::x0_offset::<u16>() + ix];
                            // Premultiply: scale by alpha/65535
                            px = ((px as u64 * alpha as u64 + 32767) / 65535) as u16;
                        }
                        let y = (y0 + (dy * (ix - save_start.0) as isize)) as usize;
                        let x = (x0 + (dx * (ix - save_start.0) as isize)) as usize;
                        write_pixel!(px, endianness, y, (x * nc + c) * 2);
                    }
                }
                JxlDataFormat::F16 { endianness, .. } => {
                    let src_row = d.get_row::<u16>(frame_y);
                    let alpha_row = alpha_channel.map(|ac| data[ac].get_row::<u16>(frame_y));
                    for ix in (save_start.0..save_end.0).skip(num_fast) {
                        let mut px = src_row[RowBuffer::x0_offset::<u16>() + ix];
                        if is_color_channel {
                            let alpha = alpha_row.unwrap()[RowBuffer::x0_offset::<u16>() + ix];
                            // Premultiply f16: convert to f64, multiply, convert back
                            let px_f64 = f16::from_bits(px).to_f64();
                            let alpha_f64 = f16::from_bits(alpha).to_f64();
                            px = f16::from_f64(px_f64 * alpha_f64).to_bits();
                        }
                        let y = (y0 + (dy * (ix - save_start.0) as isize)) as usize;
                        let x = (x0 + (dx * (ix - save_start.0) as isize)) as usize;
                        write_pixel!(px, endianness, y, (x * nc + c) * 2);
                    }
                }
                JxlDataFormat::F32 { endianness, .. } => {
                    let src_row = d.get_row::<f32>(frame_y);
                    let alpha_row = alpha_channel.map(|ac| data[ac].get_row::<f32>(frame_y));
                    for ix in (save_start.0..save_end.0).skip(num_fast) {
                        let mut px = src_row[RowBuffer::x0_offset::<f32>() + ix];
                        if is_color_channel {
                            let alpha = alpha_row.unwrap()[RowBuffer::x0_offset::<f32>() + ix];
                            px *= alpha;
                        }
                        let y = (y0 + (dy * (ix - save_start.0) as isize)) as usize;
                        let x = (x0 + (dx * (ix - save_start.0) as isize)) as usize;
                        write_pixel!(px, endianness, y, (x * nc + c) * 4);
                    }
                }
            }
        }
        Ok(())
    }
}
