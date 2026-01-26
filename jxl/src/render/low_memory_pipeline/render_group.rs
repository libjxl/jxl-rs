// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{array, ops::Range};

use crate::{
    api::JxlOutputBuffer,
    error::Result,
    image::{OwnedRawImage, Rect},
    render::{
        internal::{ChannelInfo, RenderPipelineShared, Stage},
        low_memory_pipeline::{
            InputBuffer,
            helpers::{get_distinct_indices, mirror},
            run_stage::ExtraInfo,
        },
    },
    util::{ShiftRightCeil, SmallVec, tracing_wrappers::*},
};

use super::{LowMemoryRenderPipeline, row_buffers::RowBuffer};

// Most images have at most 7 channels (RGBA + noise extra channels).
// 8 gives a bit extra leeway and makes the size a power of two.
pub(super) type ChannelVec<T> = SmallVec<T, 8>;

fn apply_x_padding(sz: usize, row: &mut [u8], to_pad: Range<isize>, valid_pixels: Range<isize>) {
    let x0_offset = RowBuffer::x0_byte_offset() as isize;
    let num_valid = valid_pixels.clone().count();
    match sz {
        1 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx) as usize;
                let to = (x0_offset + x) as usize;
                row[to] = row[from];
            }
        }
        2 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * 2) as usize;
                let to = (x0_offset + x * 2) as usize;
                row[to] = row[from];
                row[to + 1] = row[from + 1];
            }
        }
        4 => {
            for x in to_pad {
                let sx = mirror(x - valid_pixels.start, num_valid) as isize + valid_pixels.start;
                let from = (x0_offset + sx * 4) as usize;
                let to = (x0_offset + x * 4) as usize;
                row[to] = row[from];
                row[to + 1] = row[from + 1];
                row[to + 2] = row[from + 2];
                row[to + 3] = row[from + 3];
            }
        }
        _ => {
            unimplemented!("only 1, 2 or 4 byte data types supported");
        }
    }
}

// This struct pre-computes all the information needed to
// do the initial copies for a certain channel. This reduces
// the amount of computation that is repeated for every row.
struct BufferFiller<'a> {
    input_images: [&'a OwnedRawImage; 9],
    copy_src: [Range<usize>; 9],
    copy_dst: [Range<usize>; 9],
    group_yrange: Range<usize>,
    yoffsets: [isize; 3],
    to_pad: Range<isize>,
    valid_pixels: Range<isize>,
    ty_size: usize,
}

impl<'a> BufferFiller<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_buffers: &'a [InputBuffer],
        shared: &RenderPipelineShared<RowBuffer>,
        input_border_pixels: &[(usize, usize)],
        yrange: Range<usize>,
        c: usize,
        (x0, xsize): (usize, usize),
        (gx, gy): (usize, usize),
        (bx, by): (usize, usize),
    ) -> Self {
        let ChannelInfo {
            ty,
            downsample: (dx, dy),
        } = shared.channel_info[0][c];
        let ty = ty.expect("Channel info should be populated at this point");
        let group_ysize = 1 << (shared.log_group_size - dy as usize);
        let group_xsize = 1 << (shared.log_group_size - dx as usize);

        let x0 = x0 >> dx;
        let xsize = xsize >> dx;
        let yrange = (yrange.start >> dy)..yrange.end.shrc(dy);

        let group_y0 = gy * group_ysize;
        let group_y1 = group_y0 + group_ysize;
        let group_x0 = gx << (shared.log_group_size - dx as usize);
        let group_x1 = group_x0 + group_xsize;

        let copy_x0 = x0.saturating_sub(input_border_pixels[c].0);
        let copy_x1 = (x0 + xsize + input_border_pixels[c].0).min(shared.input_size.0.shrc(dx));

        assert!(copy_x1 >= group_x0);

        let pass = input_buffers[gy * shared.group_count.0 + gx].current_pass;

        let gxm1 = if copy_x0 < group_x0 { gx - 1 } else { gx };
        let gxp1 = if copy_x1 > group_x1 { gx + 1 } else { gx };
        let gym1 = if yrange.start < group_y0 { gy - 1 } else { gy };
        let gyp1 = if yrange.end > group_y1 { gy + 1 } else { gy };
        let gw = shared.group_count.0;

        let mut copy_src: [_; 9] = array::from_fn(|_| 0..0);
        let mut copy_dst: [_; 9] = array::from_fn(|_| 0..0);

        let next_group_xsize = shared.group_size(gy * gw + gxp1).0.shrc(dx);
        let border_x = copy_x1.saturating_sub(group_x1).min(next_group_xsize);
        let mut to_pad = 0..0;
        let mut valid_pixels = 0..0;
        if border_x + group_x1 < copy_x1 {
            let pad_from = (xsize + border_x) as isize;
            let pad_to = (xsize + copy_x1 - group_x1) as isize;
            to_pad = pad_from..pad_to;
            valid_pixels = 0..pad_from;
        }

        for (is_topbottom, off) in [(true, 0), (false, 3), (true, 6)] {
            let mut copy_byte_offset = RowBuffer::x0_byte_offset() - (x0 - copy_x0) * ty.size();

            // Previous group horizontally, if needed.
            if copy_x0 < group_x0 {
                let xs = if is_topbottom {
                    group_xsize
                } else {
                    4 * (bx >> dx)
                };

                let to_copy = (group_x0 - copy_x0) * ty.size();
                let src_byte_offset = xs * ty.size() - to_copy;

                copy_src[off] = src_byte_offset..src_byte_offset + to_copy;
                copy_dst[off] = copy_byte_offset..copy_byte_offset + to_copy;
                copy_byte_offset += to_copy;
            }
            let copy_start = copy_x0.saturating_sub(group_x0) * ty.size();
            let copy_end = (copy_x1.min(group_x1) - group_x0) * ty.size();
            let to_copy = copy_end - copy_start;
            copy_src[off + 1] = copy_start..copy_end;
            copy_dst[off + 1] = copy_byte_offset..copy_byte_offset + to_copy;
            copy_byte_offset += to_copy;
            // Next group horizontally, if needed.
            if copy_x1 > group_x1 {
                copy_src[off + 2] = 0..border_x * ty.size();
                copy_dst[off + 2] = copy_byte_offset..copy_byte_offset + border_x * ty.size();
            }
        }

        Self {
            group_yrange: group_y0..group_y1,
            yoffsets: [
                (by >> dy) as isize * 4 - group_y0 as isize,
                -(group_y0 as isize),
                -(group_y1 as isize),
            ],
            copy_src,
            copy_dst,
            to_pad,
            valid_pixels,
            ty_size: ty.size(),
            input_images: [
                input_buffers[gym1 * gw + gxm1].topbottom[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gym1 * gw + gx].topbottom[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gym1 * gw + gxp1].topbottom[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gy * gw + gxm1].leftright[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gy * gw + gx].data[c].as_ref().unwrap(),
                input_buffers[gy * gw + gxp1].leftright[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gyp1 * gw + gxm1].topbottom[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gyp1 * gw + gx].topbottom[pass][c]
                    .as_ref()
                    .unwrap(),
                input_buffers[gyp1 * gw + gxp1].topbottom[pass][c]
                    .as_ref()
                    .unwrap(),
            ],
        }
    }

    fn do_fill(&self, y: usize, row_buffer: &mut RowBuffer) {
        let output_row = row_buffer.get_row_mut::<u8>(y);
        let (yoff, off) = if y < self.group_yrange.start {
            (self.yoffsets[0], 0)
        } else if y >= self.group_yrange.end {
            (self.yoffsets[2], 6)
        } else {
            (self.yoffsets[1], 3)
        };
        let yy = (y as isize + yoff) as usize;
        for xx in 0..3 {
            if !self.copy_src[off + xx].is_empty() || xx == 1 {
                let to = &mut output_row[self.copy_dst[off + xx].clone()];
                let from = &self.input_images[off + xx].row(yy)[self.copy_src[off + xx].clone()];
                to.copy_from_slice(from);
            }
        }
        if !self.to_pad.is_empty() {
            apply_x_padding(
                self.ty_size,
                output_row,
                self.to_pad.clone(),
                self.valid_pixels.clone(),
            );
        }
    }
}

impl LowMemoryRenderPipeline {
    // Renders *parts* of group's worth of data.
    // In particular, renders the sub-rectangle given in `image_area`, where (1, 1) refers to
    // the center of the group, and 0 and 2 include data from the neighbouring group (if any).
    #[instrument(skip(self, buffers))]
    pub(super) fn render_group(
        &mut self,
        (gx, gy): (usize, usize),
        image_area: Rect,
        buffers: &mut [Option<JxlOutputBuffer>],
    ) -> Result<()> {
        let start_of_row = image_area.origin.0 == 0;
        let end_of_row = image_area.end().0 == self.shared.input_size.0;

        let Rect {
            origin: (x0, y0),
            size: (xsize, num_rows),
        } = image_area;

        let num_channels = self.shared.num_channels();

        let buffer_filler: ChannelVec<_> = (0..num_channels)
            .map(|c| {
                BufferFiller::new(
                    &self.input_buffers,
                    &self.shared,
                    &self.input_border_pixels,
                    y0..y0 + num_rows,
                    c,
                    (x0, xsize),
                    (gx, gy),
                    self.border_size,
                )
            })
            .collect();

        let num_extra_rows = self.border_size.1;

        // This follows the same implementation strategy as the C++ code in libjxl.
        // We pretend that every stage has a vertical shift of 0, i.e. it is as tall
        // as the final image.
        // We call each such row a "virtual" row, because it may or may not correspond
        // to an actual row of the current processing stage; actual processing happens
        // when vy % (1<<vshift) == 0.

        let vy0 = y0.saturating_sub(num_extra_rows);
        let vy1 = image_area.end().1 + num_extra_rows;

        for vy in vy0..vy1 {
            let mut current_origin = (0, 0);
            let mut current_size = self.shared.input_size;

            // Step 1: read input channels.
            for c in 0..num_channels {
                // Same logic as below, but adapted to the input stage.
                let (_, dy) = self.shared.channel_info[0][c].downsample;
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
                buffer_filler[c].do_fill(y, &mut self.row_buffers[0][c]);
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
                let shifted_xsize = xsize.shrc(dx);

                match stage {
                    Stage::InPlace(s) => {
                        let mut buffers = get_distinct_indices(
                            &mut self.row_buffers,
                            &self.sorted_buffer_indices[i],
                        );
                        s.run_stage_on(
                            ExtraInfo {
                                xsize: shifted_xsize,
                                current_row: y,
                                group_x0: x0 >> dx,
                                out_extra_x,
                                start_of_row,
                                end_of_row,
                                image_height: shifted_ysize,
                            },
                            &mut buffers,
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                    Stage::Save(s) => {
                        // Find buffers for channels that will be saved.
                        // Channel ordering is handled in stage_input_buffer_index construction.
                        let mut input_data: ChannelVec<_> = self.stage_input_buffer_index[i]
                            .iter()
                            .map(|(si, ci)| &self.row_buffers[*si][*ci])
                            .collect();
                        // Append opaque alpha buffer if fill_opaque_alpha is set
                        if let Some(ref alpha_buf) = self.opaque_alpha_buffers[i] {
                            input_data.push(alpha_buf);
                        }
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
                            for (si, ci) in self.stage_input_buffer_index[i].iter() {
                                for iy in -bordery..=bordery {
                                    let y = mirror(y as isize + iy, shifted_ysize);
                                    apply_x_padding(
                                        s.input_type().size(),
                                        self.row_buffers[*si][*ci].get_row_mut::<u8>(y),
                                        -(borderx as isize)..0,
                                        // Either xsize is the actual size of the image, or it is
                                        // much larger than borderx, so this works out either way.
                                        0..shifted_xsize as isize,
                                    );
                                }
                            }
                        }
                        if gx + 1 == self.shared.group_count.0 && borderx != 0 {
                            for (si, ci) in self.stage_input_buffer_index[i].iter() {
                                for iy in -bordery..=bordery {
                                    let y = mirror(y as isize + iy, shifted_ysize);
                                    apply_x_padding(
                                        s.input_type().size(),
                                        self.row_buffers[*si][*ci].get_row_mut::<u8>(y),
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
                        let input_data: ChannelVec<_> = self.stage_input_buffer_index[i]
                            .iter()
                            .map(|(si, ci)| &inb[*si][*ci])
                            .collect();
                        s.run_stage_on(
                            ExtraInfo {
                                xsize: shifted_xsize,
                                current_row: y,
                                group_x0: x0 >> dx,
                                out_extra_x,
                                start_of_row,
                                end_of_row,
                                image_height: shifted_ysize,
                            },
                            &input_data,
                            &mut outb[0][..],
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
                        let mut buffers = get_distinct_indices(
                            &mut self.row_buffers,
                            &self.sorted_buffer_indices[i],
                        );
                        s.run_stage_on(
                            ExtraInfo {
                                xsize,
                                current_row: y,
                                group_x0: x0,
                                out_extra_x: 0,
                                start_of_row: false,
                                end_of_row: false,
                                image_height: self.shared.input_size.1,
                            },
                            &mut buffers,
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                    Stage::Save(s) => {
                        // Find buffers for channels that will be saved.
                        // Channel ordering is handled in stage_input_buffer_index construction.
                        let mut input_data: ChannelVec<_> = self.stage_input_buffer_index[i]
                            .iter()
                            .map(|(si, ci)| &self.row_buffers[*si][*ci])
                            .collect();
                        // Append opaque alpha buffer if fill_opaque_alpha is set
                        if let Some(ref alpha_buf) = self.opaque_alpha_buffers[i] {
                            input_data.push(alpha_buf);
                        }
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
                        let input_data: ChannelVec<_> = self.stage_input_buffer_index[i]
                            .iter()
                            .map(|(si, ci)| &inb[*si][*ci])
                            .collect();
                        s.run_stage_on(
                            ExtraInfo {
                                xsize,
                                current_row: y,
                                group_x0: x0,
                                out_extra_x: 0,
                                start_of_row: false,
                                end_of_row: false,
                                image_height: self.shared.input_size.1,
                            },
                            &input_data,
                            &mut outb[0][..],
                            self.local_states[i].as_deref_mut(),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
