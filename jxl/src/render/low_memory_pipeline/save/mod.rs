// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::row_buffers::RowBuffer;
use crate::api::{Endianness, JxlDataFormat};
use crate::error::Result;
use crate::headers::Orientation;
use crate::image::ImageDataType;
use crate::render::buffer_splitter::OutputChannelRef;
use crate::render::save::{ChannelConversion, SaveStage};

mod identity;

impl SaveStage {
    // Takes as input only those channels that are *actually* saved.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn save_lowmem(
        &self,
        data: &[&RowBuffer],
        buffers: &mut [Option<OutputChannelRef>],
        group_size: (usize, usize),
        frame_y: usize,
        group_origin: (usize, usize),
        full_image_size: (usize, usize),
        frame_origin: (isize, isize),
        save_scratch: &mut Vec<u8>,
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
        let xlen = save_size.0;
        let nc = data.len();

        let out_channels = self.output_channels();
        let total_channels = out_channels.max(nc);
        let padded_len = xlen.div_ceil(64) * 64 + 64;
        let channel_stride = (padded_len * 4 + 63) & !63;
        let total_needed = 64 + total_channels * channel_stride;
        if save_scratch.len() < total_needed {
            save_scratch.resize(total_needed, 0);
        }
        let base_align = save_scratch.as_ptr().align_offset(64);

        macro_rules! write_pixel {
            ($px: expr, $endianness: expr, $y: expr, $x: expr) => {
                let px = $px;
                let px_bytes = if $endianness == Endianness::LittleEndian {
                    px.to_le_bytes()
                } else {
                    px.to_be_bytes()
                };
                buf.row_mut($y)[$x..][..px_bytes.len()].copy_from_slice(&px_bytes);
            };
        }

        match self.data_format {
            JxlDataFormat::U8 { .. } => {
                let mut u8_slices: [&[u8]; 4] = [&[]; 4];
                let mut scratch_chunks =
                    save_scratch[base_align..].chunks_exact_mut(channel_stride);
                for (c, d) in data.iter().enumerate() {
                    let out_scratch_chunk = scratch_chunks.next().unwrap();
                    let conv = self
                        .conversions
                        .get(c)
                        .copied()
                        .unwrap_or(ChannelConversion::None);
                    match conv {
                        ChannelConversion::None => {
                            let off = RowBuffer::x0_offset::<u8>() + save_start.0;
                            u8_slices[c] = &d.get_row::<u8>(frame_y)[off..off + xlen];
                        }
                        ChannelConversion::F32ToU8 {
                            bit_depth,
                            dither_channel,
                        } => {
                            let off = RowBuffer::x0_offset::<f32>() + save_start.0;
                            let in_slice = &d.get_row::<f32>(frame_y)[off..];
                            let out_scratch = &mut out_scratch_chunk[..xlen];
                            identity::f32_to_u8_simd(
                                in_slice,
                                out_scratch,
                                ((1 << bit_depth) - 1) as f32,
                                (group_origin.0 + save_start.0, frame_y),
                                dither_channel,
                            );
                            u8_slices[c] = out_scratch;
                        }
                        ChannelConversion::I16ToU8 { multiplier, max } => {
                            let off = RowBuffer::x0_offset::<i16>() + save_start.0;
                            let in_slice = &d.get_row::<i16>(frame_y)[off..];
                            let out_scratch = &mut out_scratch_chunk[..xlen];
                            identity::i16_to_u8_simd(
                                in_slice,
                                out_scratch,
                                multiplier as i16,
                                max as i16,
                            );
                            u8_slices[c] = out_scratch;
                        }
                        ChannelConversion::I32ToU8 { multiplier, max } => {
                            let off = RowBuffer::x0_offset::<i32>() + save_start.0;
                            let in_slice = &d.get_row::<i32>(frame_y)[off..];
                            let out_scratch = &mut out_scratch_chunk[..padded_len];
                            identity::i32_to_u8_simd_dispatch(
                                in_slice,
                                out_scratch,
                                multiplier,
                                max,
                                xlen,
                            );
                            u8_slices[c] = &out_scratch[..xlen];
                        }
                        _ => unreachable!("unsupported conversion to U8"),
                    }
                }

                let actual_nc = if self.fill_opaque_alpha && nc == 3 && out_channels == 4 {
                    let out_scratch_chunk = scratch_chunks.next().unwrap();
                    out_scratch_chunk[..xlen].fill(255);
                    u8_slices[3] = &out_scratch_chunk[..xlen];
                    4
                } else {
                    nc
                };

                let num_fast = match self.orientation {
                    Orientation::Identity => {
                        identity::store_u8(&u8_slices[..actual_nc], buf.row_mut(relative_y))
                    }
                    Orientation::FlipVertical => identity::store_u8(
                        &u8_slices[..actual_nc],
                        buf.row_mut(save_size.1 - 1 - relative_y),
                    ),
                    _ => 0,
                };

                if num_fast < xlen {
                    let (x0, y0) = self.orientation.display_pixel((0, relative_y), save_size);
                    let x0 = x0 as isize;
                    let y0 = y0 as isize;
                    let (dx, dy) = self.orientation.display_row_step();
                    for (c, slice) in u8_slices[..actual_nc].iter().enumerate() {
                        for (ix, &px) in slice.iter().enumerate().skip(num_fast) {
                            let y = (y0 + (dy * ix as isize)) as usize;
                            let x = (x0 + (dx * ix as isize)) as usize;
                            write_pixel!(px, Endianness::LittleEndian, y, x * actual_nc + c);
                        }
                    }
                }
            }
            JxlDataFormat::U16 { endianness, .. } | JxlDataFormat::F16 { endianness, .. } => {
                let mut u16_slices: [&[u16]; 4] = [&[]; 4];
                let mut scratch_chunks =
                    save_scratch[base_align..].chunks_exact_mut(channel_stride);
                for (c, d) in data.iter().enumerate() {
                    let out_scratch_chunk = scratch_chunks.next().unwrap();
                    let conv = self
                        .conversions
                        .get(c)
                        .copied()
                        .unwrap_or(ChannelConversion::None);
                    match conv {
                        ChannelConversion::None => {
                            let off = RowBuffer::x0_offset::<u16>() + save_start.0;
                            u16_slices[c] = &d.get_row::<u16>(frame_y)[off..off + xlen];
                        }
                        ChannelConversion::F32ToU16 { bit_depth } => {
                            let off = RowBuffer::x0_offset::<f32>() + save_start.0;
                            let in_slice = &d.get_row::<f32>(frame_y)[off..];
                            let out_scratch =
                                u16::cast_slice_mut(&mut out_scratch_chunk[..padded_len * 2]);
                            identity::f32_to_u16_simd_dispatch(
                                in_slice,
                                out_scratch,
                                ((1 << bit_depth) - 1) as f32,
                                xlen,
                            );
                            u16_slices[c] = &out_scratch[..xlen];
                        }
                        ChannelConversion::F32ToF16 { clamp_range } => {
                            let off = RowBuffer::x0_offset::<f32>() + save_start.0;
                            let in_slice = &d.get_row::<f32>(frame_y)[off..];
                            let out_scratch =
                                u16::cast_slice_mut(&mut out_scratch_chunk[..padded_len * 2]);
                            identity::f32_to_f16_simd_dispatch(
                                in_slice,
                                out_scratch,
                                clamp_range,
                                xlen,
                            );
                            u16_slices[c] = &out_scratch[..xlen];
                        }
                        _ => unreachable!("unsupported conversion to U16/F16"),
                    }
                }

                let actual_nc = if self.fill_opaque_alpha && nc == 3 && out_channels == 4 {
                    let out_scratch_chunk = scratch_chunks.next().unwrap();
                    let out_scratch =
                        u16::cast_slice_mut(&mut out_scratch_chunk[..padded_len * 2]);
                    let alpha_val = if matches!(self.data_format, JxlDataFormat::F16 { .. }) {
                        0x3C00
                    } else {
                        65535
                    };
                    out_scratch[..xlen].fill(alpha_val);
                    u16_slices[3] = &out_scratch[..xlen];
                    4
                } else {
                    nc
                };

                let is_native = endianness == Endianness::native();
                let num_fast = if is_native {
                    match self.orientation {
                        Orientation::Identity => {
                            identity::store_u16(&u16_slices[..actual_nc], buf.row_mut(relative_y))
                        }
                        Orientation::FlipVertical => identity::store_u16(
                            &u16_slices[..actual_nc],
                            buf.row_mut(save_size.1 - 1 - relative_y),
                        ),
                        _ => 0,
                    }
                } else {
                    0
                };

                if num_fast < xlen {
                    let (x0, y0) = self.orientation.display_pixel((0, relative_y), save_size);
                    let x0 = x0 as isize;
                    let y0 = y0 as isize;
                    let (dx, dy) = self.orientation.display_row_step();
                    for (c, slice) in u16_slices[..actual_nc].iter().enumerate() {
                        for (ix, &px) in slice.iter().enumerate().skip(num_fast) {
                            let y = (y0 + (dy * ix as isize)) as usize;
                            let x = (x0 + (dx * ix as isize)) as usize;
                            write_pixel!(px, endianness, y, (x * actual_nc + c) * 2);
                        }
                    }
                }
            }
            JxlDataFormat::F32 { endianness, .. } => {
                let mut f32_slices: [&[f32]; 4] = [&[]; 4];
                for (c, d) in data.iter().enumerate() {
                    let conv = self
                        .conversions
                        .get(c)
                        .copied()
                        .unwrap_or(ChannelConversion::None);
                    match conv {
                        ChannelConversion::None => {
                            let off = RowBuffer::x0_offset::<f32>() + save_start.0;
                            f32_slices[c] = &d.get_row::<f32>(frame_y)[off..off + xlen];
                        }
                        _ => unreachable!("unsupported conversion to F32"),
                    }
                }

                let actual_nc = if self.fill_opaque_alpha && nc == 3 && out_channels == 4 {
                    let mut scratch_chunks =
                        save_scratch[base_align..].chunks_exact_mut(channel_stride);
                    let out_scratch_chunk = scratch_chunks.next().unwrap();
                    let out_scratch =
                        f32::cast_slice_mut(&mut out_scratch_chunk[..padded_len * 4]);
                    out_scratch[..xlen].fill(1.0);
                    f32_slices[3] = &out_scratch[..xlen];
                    4
                } else {
                    nc
                };

                let is_native = endianness == Endianness::native();
                let num_fast = if is_native {
                    match self.orientation {
                        Orientation::Identity => {
                            identity::store_f32(&f32_slices[..actual_nc], buf.row_mut(relative_y))
                        }
                        Orientation::FlipVertical => identity::store_f32(
                            &f32_slices[..actual_nc],
                            buf.row_mut(save_size.1 - 1 - relative_y),
                        ),
                        _ => 0,
                    }
                } else {
                    0
                };

                if num_fast < xlen {
                    let (x0, y0) = self.orientation.display_pixel((0, relative_y), save_size);
                    let x0 = x0 as isize;
                    let y0 = y0 as isize;
                    let (dx, dy) = self.orientation.display_row_step();
                    for (c, slice) in f32_slices[..actual_nc].iter().enumerate() {
                        for (ix, &px) in slice.iter().enumerate().skip(num_fast) {
                            let y = (y0 + (dy * ix as isize)) as usize;
                            let x = (x0 + (dx * ix as isize)) as usize;
                            write_pixel!(px, endianness, y, (x * actual_nc + c) * 4);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
