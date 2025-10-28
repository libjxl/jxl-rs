// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use half::f16;

use crate::{
    api::JxlOutputBuffer,
    error::Result,
    image::{DataTypeTag, Image, ImageDataType},
    render::{
        internal::Stage,
        low_memory_pipeline::{
            helpers::{get_distinct_indices, mirror},
            run_stage::ExtraInfo,
        },
    },
    util::{ShiftRightCeil, tracing_wrappers::*},
};

use super::{LowMemoryRenderPipeline, row_buffers::RowBuffer};

fn apply_x_padding_impl<T: ImageDataType>(
    buf: &mut RowBuffer,
    y: usize,
    to_pad: Range<isize>,
    valid_pixels: Range<isize>,
) {
    let x0_offset = RowBuffer::x0_offset::<T>() as isize;
    let row = buf.get_row_mut::<T>(y);
    let num_valid = valid_pixels.clone().count();
    for x in to_pad {
        let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
        row[(x0_offset + x) as usize] = row[(x0_offset + sx) as usize];
    }
}

fn apply_x_padding(
    input_type: DataTypeTag,
    buf: &mut RowBuffer,
    y: usize,
    to_pad: Range<isize>,
    valid_pixels: Range<isize>,
) {
    // TODO(veluca): consider using a type-erased approach here (we only care about element
    // sizes).
    match input_type {
        DataTypeTag::U8 => apply_x_padding_impl::<u8>(buf, y, to_pad, valid_pixels),
        DataTypeTag::I8 => apply_x_padding_impl::<i8>(buf, y, to_pad, valid_pixels),
        DataTypeTag::U16 => apply_x_padding_impl::<u16>(buf, y, to_pad, valid_pixels),
        DataTypeTag::I16 => apply_x_padding_impl::<i16>(buf, y, to_pad, valid_pixels),
        DataTypeTag::F16 => apply_x_padding_impl::<f16>(buf, y, to_pad, valid_pixels),
        DataTypeTag::U32 => apply_x_padding_impl::<u32>(buf, y, to_pad, valid_pixels),
        DataTypeTag::I32 => apply_x_padding_impl::<i32>(buf, y, to_pad, valid_pixels),
        DataTypeTag::F32 => apply_x_padding_impl::<f32>(buf, y, to_pad, valid_pixels),
        DataTypeTag::F64 => apply_x_padding_impl::<f64>(buf, y, to_pad, valid_pixels),
    }
}

impl LowMemoryRenderPipeline {
    fn fill_buffer_row<T: ImageDataType>(
        &mut self,
        y: usize,
        y0: usize,
        c: usize,
        (gx, gy): (usize, usize),
    ) {
        let gys = 1
            << (self.shared.log_group_size - self.shared.channel_info[0][c].downsample.1 as usize);
        let extrax = self.input_border_pixels[0].0;

        let (input_y, igy) = if y < y0 {
            (y + gys - y0, gy - 1)
        } else if y >= y0 + gys {
            (y - y0 - gys, gy + 1)
        } else {
            (y - y0, gy)
        };

        let output_row = self.row_buffers[0][c].get_row_mut::<T>(y);
        let x0_offset = RowBuffer::x0_offset::<T>();

        let base_gid = igy * self.shared.group_count.0 + gx;

        // Previous group horizontally, if any.
        if gx > 0 && extrax != 0 {
            let input_buf = &self.input_buffers[base_gid - 1].data[c]
                .as_ref()
                .unwrap()
                .downcast_ref::<Image<T>>()
                .unwrap();
            let input_row = input_buf.as_rect().row(input_y);
            output_row[x0_offset - extrax..x0_offset]
                .copy_from_slice(&input_row[input_buf.size().0 - extrax..]);
        }
        let input_buf = &self.input_buffers[base_gid].data[c]
            .as_ref()
            .unwrap()
            .downcast_ref::<Image<T>>()
            .unwrap();
        let input_row = input_buf.as_rect().row(input_y);
        let gxs = input_buf.size().0;
        output_row[x0_offset..x0_offset + gxs].copy_from_slice(input_row);
        // Next group horizontally, if any.
        if gx + 1 < self.shared.group_count.0 && extrax != 0 {
            let input_buf = &self.input_buffers[base_gid + 1].data[c]
                .as_ref()
                .unwrap()
                .downcast_ref::<Image<T>>()
                .unwrap();
            let input_row = input_buf.as_rect().row(input_y);
            output_row[gxs + x0_offset..gxs + x0_offset + extrax]
                .copy_from_slice(&input_row[..extrax]);
        }
    }

    fn fill_initial_buffers(&mut self, c: usize, y: usize, y0: usize, (gx, gy): (usize, usize)) {
        // TODO(veluca): consider using a type-erased approach here (we only care about element
        // sizes).
        match self.shared.channel_info[0][c].ty {
            Some(DataTypeTag::F64) => self.fill_buffer_row::<f64>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::F32) => self.fill_buffer_row::<f32>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::I32) => self.fill_buffer_row::<i32>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::U32) => self.fill_buffer_row::<u32>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::I16) => self.fill_buffer_row::<i16>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::F16) => self.fill_buffer_row::<f16>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::U16) => self.fill_buffer_row::<u16>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::I8) => self.fill_buffer_row::<i8>(y, y0, c, (gx, gy)),
            Some(DataTypeTag::U8) => self.fill_buffer_row::<u8>(y, y0, c, (gx, gy)),
            None => {
                panic!("Channel info should be populated at this point");
            }
        }
    }

    // Renders a single group worth of data.
    #[instrument(skip(self, buffers))]
    pub(super) fn render_group(
        &mut self,
        (gx, gy): (usize, usize),
        buffers: &mut [Option<JxlOutputBuffer>],
    ) -> Result<()> {
        let gid = gy * self.shared.group_count.0 + gx;
        let (xsize, num_rows) = self.shared.group_size(gid);
        let (x0, y0) = self.shared.group_offset(gid);

        let num_channels = self.shared.num_channels();
        let mut num_extra_rows = 0;

        for c in 0..num_channels {
            num_extra_rows = num_extra_rows
                .max(self.input_border_pixels[c].1 << self.shared.channel_info[0][c].downsample.1);
        }
        for s in 0..self.shared.stages.len() {
            num_extra_rows = num_extra_rows
                .max(self.stage_output_border_pixels[s].1 << self.downsampling_for_stage[s].1);
        }

        // This follows the same implementation strategy as the C++ code in libjxl.
        // We pretend that every stage has a vertical shift of 0, i.e. it is as tall
        // as the final image.
        // We call each such row a "virtual" row, because it may or may not correspond
        // to an actual row of the current processing stage; actual processing happens
        // when vy % (1<<vshift) == 0.

        let vy0 = y0.saturating_sub(num_extra_rows);
        let vy1 = y0 + num_rows + num_extra_rows;

        for vy in vy0..vy1 {
            let mut current_origin = (0, 0);
            let mut current_size = self.shared.input_size;

            // Step 1: read input channels.
            for c in 0..num_channels {
                // Same logic as below, but adapted to the input stage.
                let dy = self.shared.channel_info[0][c].downsample.1;
                let scaled_y_border = self.input_border_pixels[c].1 << dy;
                let stage_vy = vy as isize - num_extra_rows as isize + scaled_y_border as isize;
                if stage_vy % (1 << dy) != 0 {
                    continue;
                }
                if stage_vy - (y0 as isize) < -(scaled_y_border as isize) {
                    continue;
                }
                let y = stage_vy >> dy;
                // Do not produce rows in out-of-bounds areas.
                if y < 0 || y >= self.shared.input_size.1.shrc(dy) as isize {
                    continue;
                }
                let y = y as usize;
                self.fill_initial_buffers(c, y, y0 >> dy, (gx, gy));
            }
            // Step 2: go through stages one by one.
            for (i, stage) in self.shared.stages.iter().enumerate() {
                let (dx, dy) = self.downsampling_for_stage[i];
                // The logic below uses *virtual* y coordinates, so we need to convert the border
                // amount appropriately.
                let scaled_y_border = self.stage_output_border_pixels[i].1 << dy;
                // I knew the reason behind this formula at some point, but now I don't.
                let stage_vy = vy as isize - num_extra_rows as isize + scaled_y_border as isize;
                if stage_vy % (1 << dy) != 0 {
                    continue;
                }
                if stage_vy - (y0 as isize) < -(scaled_y_border as isize) {
                    continue;
                }
                let y = stage_vy >> dy;
                let shifted_ysize = self.shared.input_size.1.shrc(dy);
                // Do not produce rows in out-of-bounds areas.
                if y < 0 || y >= shifted_ysize as isize {
                    continue;
                }
                let y = y as usize;

                let out_extra_x = self.stage_output_border_pixels[i].0;
                let shifted_xsize = xsize.shrc(self.downsampling_for_stage[i].0);

                match stage {
                    Stage::InPlace(s) => {
                        let buffer_indices: Vec<_> = (0..num_channels)
                            .filter(|c| s.uses_channel(*c))
                            .map(|x| self.stage_input_buffer_index[i][x])
                            .collect();
                        let mut buffers =
                            get_distinct_indices(&mut self.row_buffers, &buffer_indices);
                        s.run_stage_on(
                            ExtraInfo {
                                xsize: shifted_xsize,
                                current_row: y,
                                group_x0: x0 >> dx,
                                out_extra_x,
                                is_first_xgroup: gx == 0,
                                is_last_xgroup: gx + 1 == self.shared.group_count.0,
                                image_height: self.shared.input_size.1 >> dy,
                            },
                            &mut buffers,
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                    Stage::Save(s) => {
                        // Find buffers for channels that will be saved.
                        let input_data: Vec<_> = s
                            .channels
                            .iter()
                            .map(|c| {
                                let (si, ci) = self.stage_input_buffer_index[i][*c];
                                &self.row_buffers[si][ci]
                            })
                            .collect();
                        s.save_lowmem(
                            &input_data,
                            &mut *buffers,
                            (xsize >> dx, num_rows >> dy),
                            y,
                            (x0 >> dx, y0 >> dy),
                            current_size,
                            current_origin,
                        )?;
                    }
                    Stage::Extend(s) => {
                        current_size = s.image_size;
                        current_origin = s.frame_origin;
                    }
                    Stage::InOut(s) => {
                        let borderx = s.border().0 as usize;
                        let bordery = s.border().1 as isize;
                        // Apply x padding.
                        if gx == 0 && borderx != 0 {
                            for c in 0..num_channels {
                                if !s.uses_channel(c) {
                                    continue;
                                }
                                let (si, ci) = self.stage_input_buffer_index[i][c];
                                let buf = &mut self.row_buffers[si][ci];
                                for iy in -bordery..=bordery {
                                    let y = mirror(y as isize + iy, shifted_ysize);
                                    apply_x_padding(
                                        s.input_type(),
                                        buf,
                                        y,
                                        -(borderx as isize)..0,
                                        // Either xsize is the actual size of the image, or it is
                                        // much larger than borderx, so this works out either way.
                                        0..shifted_xsize as isize,
                                    );
                                }
                            }
                        }
                        if gx + 1 == self.shared.group_count.0 && borderx != 0 {
                            for c in 0..num_channels {
                                if !s.uses_channel(c) {
                                    continue;
                                }
                                let (si, ci) = self.stage_input_buffer_index[i][c];
                                let buf = &mut self.row_buffers[si][ci];
                                for iy in -bordery..=bordery {
                                    let y = mirror(y as isize + iy, shifted_ysize);
                                    apply_x_padding(
                                        s.input_type(),
                                        buf,
                                        y,
                                        shifted_xsize as isize..(shifted_xsize + borderx) as isize,
                                        // borderx..0 is either data from the neighbouring group or
                                        // data that was filled in by the iteration above.
                                        -(borderx as isize)..shifted_xsize as isize,
                                    );
                                }
                            }
                        }
                        let (inb, outb) = self.row_buffers.split_at_mut(i + 1);
                        // Prepare pointers to input and output buffers.
                        let input_data: Vec<_> = (0..num_channels)
                            .filter(|c| s.uses_channel(*c))
                            .map(|c| {
                                let (si, ci) = self.stage_input_buffer_index[i][c];
                                &inb[si][ci]
                            })
                            .collect();
                        let mut outb: Vec<_> = outb[0].iter_mut().collect();
                        s.run_stage_on(
                            ExtraInfo {
                                xsize: shifted_xsize,
                                current_row: y,
                                group_x0: x0 >> dx,
                                out_extra_x,
                                is_first_xgroup: gx == 0,
                                is_last_xgroup: gx + 1 == self.shared.group_count.0,
                                image_height: self.shared.input_size.1 >> dy,
                            },
                            &input_data,
                            &mut outb[..],
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // Renders a chunk of data outside the current frame.
    #[instrument(skip(self, buffers))]
    pub(super) fn render_outside_frame(
        &mut self,
        xrange: Range<usize>,
        yrange: Range<usize>,
        buffers: &mut [Option<JxlOutputBuffer>],
    ) -> Result<()> {
        let num_channels = self.shared.num_channels();
        let x0 = xrange.start;
        let y0 = yrange.start;
        let xsize = xrange.clone().count();
        let ysize = yrange.clone().count();
        // Significantly simplified version of render_group.
        for y in yrange.clone() {
            let extend = self.shared.extend_stage_index.unwrap();
            // Step 1: get padding from extend stage.
            for c in 0..num_channels {
                let (si, ci) = self.stage_input_buffer_index[extend][c];
                let buffer = &mut self.row_buffers[si][ci];
                let Stage::Extend(extend) = &self.shared.stages[extend] else {
                    unreachable!("extend stage is not an extend stage");
                };
                let row = &mut buffer.get_row_mut(y)[RowBuffer::x0_offset::<f32>()..];
                extend.process_row_chunk((x0, y), xsize, c, row);
            }
            // Step 2: go through remaining stages one by one.
            for (i, stage) in self.shared.stages.iter().enumerate().skip(extend + 1) {
                assert_eq!(self.downsampling_for_stage[i], (0, 0));

                match stage {
                    Stage::InPlace(s) => {
                        let buffer_indices: Vec<_> = (0..num_channels)
                            .filter(|c| s.uses_channel(*c))
                            .map(|x| self.stage_input_buffer_index[i][x])
                            .collect();
                        let mut buffers =
                            get_distinct_indices(&mut self.row_buffers, &buffer_indices);
                        s.run_stage_on(
                            ExtraInfo {
                                xsize,
                                current_row: y,
                                group_x0: x0,
                                out_extra_x: 0,
                                is_first_xgroup: false,
                                is_last_xgroup: false,
                                image_height: self.shared.input_size.1,
                            },
                            &mut buffers,
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                    Stage::Save(s) => {
                        // Find buffers for channels that will be saved.
                        let input_data: Vec<_> = s
                            .channels
                            .iter()
                            .map(|c| {
                                let (si, ci) = self.stage_input_buffer_index[i][*c];
                                &self.row_buffers[si][ci]
                            })
                            .collect();
                        s.save_lowmem(
                            &input_data,
                            &mut *buffers,
                            (xsize, ysize),
                            y,
                            (x0, y0),
                            (xrange.end, yrange.end), // this is not true, but works out correctly.
                            (0, 0),
                        )?;
                    }
                    Stage::Extend(_) => {
                        unreachable!("duplicate extend stage");
                    }
                    Stage::InOut(s) => {
                        assert_eq!(s.border(), (0, 0));
                        let (inb, outb) = self.row_buffers.split_at_mut(i + 1);
                        // Prepare pointers to input and output buffers.
                        let input_data: Vec<_> = (0..num_channels)
                            .filter(|c| s.uses_channel(*c))
                            .map(|c| {
                                let (si, ci) = self.stage_input_buffer_index[i][c];
                                &inb[si][ci]
                            })
                            .collect();
                        let mut outb: Vec<_> = outb[0].iter_mut().collect();
                        s.run_stage_on(
                            ExtraInfo {
                                xsize,
                                current_row: y,
                                group_x0: x0,
                                out_extra_x: 0,
                                is_first_xgroup: false,
                                is_last_xgroup: false,
                                image_height: self.shared.input_size.1,
                            },
                            &input_data,
                            &mut outb[..],
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
