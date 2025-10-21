// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use half::f16;

use crate::{
    api::JxlOutputBuffer,
    error::Result,
    image::{DataTypeTag, Image, ImageDataType},
    render::{
        internal::Stage,
        low_memory_pipeline::{helpers::get_distinct_indices, run_stage::ExtraInfo},
    },
    util::tracing_wrappers::*,
};

use super::{LowMemoryRenderPipeline, row_buffers::RowBuffer};

impl LowMemoryRenderPipeline {
    fn fill_buffer_row<T: ImageDataType>(
        &mut self,
        y: usize,
        y0: usize,
        c: usize,
        (gx, gy): (usize, usize),
    ) {
        let gid = gy * self.shared.group_count.0 + gx;
        let buf = &mut self.row_buffers[0][c];
        let input_buf = &self.input_buffers[gid].data[c]
            .as_ref()
            .unwrap()
            .downcast_ref::<Image<T>>()
            .unwrap();
        let input_row = input_buf.as_rect().row(y - y0);
        let mut output_row = buf.advance_rows(1, 0);
        let start = RowBuffer::x0_offset::<T>();
        output_row[0][start..start + input_buf.size().0].copy_from_slice(input_row);
    }

    fn fill_initial_buffers(&mut self, y: usize, y0: usize, (gx, gy): (usize, usize)) {
        let num_channels = self.shared.num_channels();
        for c in 0..num_channels {
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
    }

    // Renders a single group worth of data.
    #[instrument(skip(self, buffers))]
    pub(super) fn render_group(
        &mut self,
        (gx, gy): (usize, usize),
        buffers: &mut [Option<JxlOutputBuffer>],
    ) -> Result<()> {
        if !self.bordeless {
            // TODO(veluca): implement this case.
            unimplemented!()
        }

        let gid = gy * self.shared.group_count.0 + gx;
        let (xsize, num_rows) = self.shared.group_size(gid);
        let (x0, y0) = self.shared.group_offset(gid);

        // Reset all buffers.
        for bufs in self.row_buffers.iter_mut() {
            for b in bufs.iter_mut() {
                // TODO(veluca): this is incorrect with borders or upsampling.
                b.reset(y0..y0 + num_rows);
            }
        }

        let num_channels = self.shared.num_channels();

        for y in y0..y0 + num_rows {
            // Step 1: read input channels.
            self.fill_initial_buffers(y, y0, (gx, gy));
            // Step 2: go through stages one by one.
            for (i, stage) in self.shared.stages.iter().enumerate() {
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
                                group_origin: (x0, y0),
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
                        s.save_lowmem(&input_data, &mut *buffers, (xsize, num_rows), y, y0)?;
                    }
                    Stage::Extend(s) => {
                        unimplemented!()
                    }
                    Stage::InOut(s) => {
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
                                group_origin: (x0, y0),
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
