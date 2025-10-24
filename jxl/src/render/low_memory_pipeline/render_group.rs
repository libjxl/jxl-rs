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
        /*
          // We pretend that every stage has a vertical shift of 0, i.e. it is as tall
          // as the final image.
          // We call each such row a "virtual" row, because it may or may not correspond
          // to an actual row of the current processing stage; actual processing happens
          // when vy % (1<<vshift) == 0.

          int num_extra_rows = *std::max_element(virtual_ypadding_for_output_.begin(),
                                                 virtual_ypadding_for_output_.end());

          for (int vy = -num_extra_rows;
               vy < static_cast<int>(image_area_rect.ysize()) + num_extra_rows; vy++) {
            for (size_t i = 0; i < first_trailing_stage_; i++) {
              int stage_vy = vy - num_extra_rows + virtual_ypadding_for_output_[i];

              if (stage_vy % (1 << channel_shifts_[i][anyc_[i]].second) != 0) {
                continue;
              }

              if (stage_vy < -virtual_ypadding_for_output_[i]) {
                continue;
              }

              int y = stage_vy >> channel_shifts_[i][anyc_[i]].second;

              ptrdiff_t image_y = static_cast<ptrdiff_t>(group_rect[i].y0()) + y;
              // Do not produce rows in out-of-bounds areas.
              if (image_y < 0 ||
                  image_y >= static_cast<ptrdiff_t>(image_rect_[i].ysize())) {
                continue;
              }

              // Get the input/output rows and potentially apply mirroring to the input.
              prepare_io_rows(y, i);

              // Produce output rows.
              JXL_RETURN_IF_ERROR(stages_[i]->ProcessRow(
                  input_rows[i], output_rows, xpadding_for_output_[i],
                  group_rect[i].xsize(), group_rect[i].x0(), image_y, thread_id));
            }

            // Process trailing stages, i.e. the final set of non-kInOut stages; they
            // all have the same input buffer and no need to use any mirroring.

            int y = vy - num_extra_rows;

            for (size_t c = 0; c < input_data.size(); c++) {
              // Skip pixels that are not part of the actual final image area.
              input_rows[first_trailing_stage_][c][0] =
                  rows.GetBuffer(stage_input_for_channel_[first_trailing_stage_][c], y,
                                 c) +
                  x_pixels_skip;
            }

            // Check that we are not outside of the bounds for the current rendering
            // rect. Not doing so might result in overwriting some rows that have been
            // written (or will be written) by other threads.
            if (y < 0 || y >= static_cast<ptrdiff_t>(image_area_rect.ysize())) {
              continue;
            }

            // Avoid running pipeline stages on pixels that are outside the full image
            // area. As trailing stages have no borders, this is a free optimization
            // (and may be necessary for correctness, as some stages assume coordinates
            // are within bounds).
            ptrdiff_t full_image_y = frame_y0 + image_area_rect.y0() + y;
            if (full_image_y < 0 ||
                full_image_y >= static_cast<ptrdiff_t>(full_image_ysize)) {
              continue;
            }

            for (size_t i = first_trailing_stage_; i < stages_.size(); i++) {
              // Before the first_image_dim_stage_, coordinates are relative to the
              // current frame.
              size_t x0 =
                  i < first_image_dim_stage_ ? full_image_x0 - frame_x0 : full_image_x0;
              size_t y0 =
                  i < first_image_dim_stage_ ? full_image_y - frame_y0 : full_image_y;
              JXL_RETURN_IF_ERROR(stages_[i]->ProcessRow(
                  input_rows[first_trailing_stage_], output_rows,
                  /*xextra=*/0, full_image_x1 - full_image_x0, x0, y0, thread_id));
            }
          }
        */

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
