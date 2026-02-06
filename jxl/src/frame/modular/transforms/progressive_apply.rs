// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Progressive rendering support for modular transforms.
//!
//! This module contains logic for non-destructively executing transforms with incomplete
//! data during progressive/flush mode, keeping decoder state intact while producing preview output.

use crate::{
    error::Result,
    frame::modular::{ModularBufferInfo, ModularChannel},
    headers::frame_header::FrameHeader,
    util::tracing_wrappers::*,
};

use super::{ModularGridKind, TransformStep, TransformStepChunk};

impl TransformStepChunk {
    /// Access the step field (for progressive rendering)
    pub fn step(&self) -> &TransformStep {
        &self.step
    }

    /// Access the grid_pos field (for progressive rendering)
    pub fn grid_pos(&self) -> (usize, usize) {
        self.grid_pos
    }

    /// Access the incomplete_deps field (for progressive rendering)
    pub fn incomplete_deps(&self) -> usize {
        self.incomplete_deps
    }

    /// Helper to check if input channel is available (in temp_outputs or buffers)
    fn has_input_channel(
        &self,
        buf_idx: usize,
        grid_idx: usize,
        buffers: &[ModularBufferInfo],
        temp_outputs: &[Option<ModularChannel>],
    ) -> bool {
        // Invariant: temp_outputs[buf_idx] is always for our current grid
        if temp_outputs[buf_idx].is_some() {
            return true;
        }
        buffers[buf_idx].buffer_grid[grid_idx]
            .data
            .borrow()
            .is_some()
    }

    /// Execute transform for progressive flush mode without modifying decoder state.
    ///
    /// Inputs are resolved from:
    /// 1. temp_outputs (temporary buffers from previous flush transforms) - can be consumed
    /// 2. buffers (real decoded buffers) - read-only, cannot be consumed
    ///
    /// Returns outputs as Vec<((buf_idx, grid_idx), channel)> for caller to store in temp_outputs
    ///
    /// Returns:
    /// - Ok(Some(vec)) = successfully executed, here are the outputs
    /// - Ok(None) = can't execute yet (not enough data), try again later
    /// - Err(...) = actual error occurred
    #[instrument(level = "trace", skip_all)]
    pub fn force_execute(
        &self, // Note: &self not &mut self - no state modification!
        _frame_header: &FrameHeader,
        buffers: &[ModularBufferInfo],
        temp_outputs: &[Option<ModularChannel>],
        out_grid_kind: ModularGridKind,
    ) -> Result<Option<Vec<(usize, ModularChannel)>>> {
        trace!(
            "force_execute: transform {:?} at grid_pos {:?}",
            self.step, self.grid_pos
        );

        // Execute transform based on type
        match &self.step {
            TransformStep::HSqueeze { buf_in, buf_out } => {
                let result = self.force_execute_hsqueeze(
                    _frame_header,
                    buffers,
                    temp_outputs,
                    out_grid_kind,
                    buf_in,
                    *buf_out,
                )?;
                // Empty vec means "can't execute yet" (input not available)
                Ok(if result.is_empty() {
                    None
                } else {
                    Some(result)
                })
            }
            TransformStep::VSqueeze { buf_in, buf_out } => {
                let result = self.force_execute_vsqueeze(
                    _frame_header,
                    buffers,
                    temp_outputs,
                    out_grid_kind,
                    buf_in,
                    *buf_out,
                )?;
                // Empty vec means "can't execute yet" (input not available)
                Ok(if result.is_empty() {
                    None
                } else {
                    Some(result)
                })
            }
            TransformStep::Rct {
                buf_in,
                buf_out,
                op,
                perm,
            } => {
                let result = self.force_execute_rct(
                    buffers,
                    temp_outputs,
                    buf_in,
                    buf_out,
                    *op,
                    *perm,
                )?;
                // Empty vec means "can't execute yet" (input not available)
                Ok(if result.is_empty() {
                    None
                } else {
                    Some(result)
                })
            }
            TransformStep::Palette { .. } => {
                // Palette transform in progressive mode not yet implemented
                trace!("Palette transform in progressive mode not yet implemented");
                Ok(None)
            }
        }
    }

    fn force_execute_hsqueeze(
        &self,
        frame_header: &FrameHeader,
        buffers: &[ModularBufferInfo],
        temp_outputs: &[Option<ModularChannel>],
        out_grid_kind: ModularGridKind,
        buf_in: &[usize; 2],
        buf_out: usize,
    ) -> Result<Vec<(usize, ModularChannel)>> {
        let buf_avg = &buffers[buf_in[0]];
        let buf_res = &buffers[buf_in[1]];
        let in_grid = buf_avg.get_grid_idx(out_grid_kind, self.grid_pos);
        let res_grid = buf_res.get_grid_idx(out_grid_kind, self.grid_pos);
        let out_grid = buffers[buf_out].get_grid_idx(out_grid_kind, self.grid_pos);

        // Check if required input channels are available
        if !self.has_input_channel(buf_in[0], in_grid, buffers, temp_outputs) {
            // Avg channel not available yet - can't execute
            return Ok(vec![]);
        }
        let has_res = self.has_input_channel(buf_in[1], res_grid, buffers, temp_outputs);

        let (gx, gy) = self.grid_pos;
        let output_size = buffers[buf_out].buffer_grid[out_grid].size;
        let mut output_channel = ModularChannel::new_with_shift(
            output_size,
            buffers[buf_out].info.shift,
            buffers[buf_out].info.bit_depth,
        )?;

        // Get the input rectangles
        let in_rect = buf_avg.get_grid_rect(frame_header, out_grid_kind, self.grid_pos);

        if !has_res {
            // Case 1: Only avg available - upsample with triangle filter (no cloning)
            // (AvgLeft + 3 * Avg)/4, (3 * Avg + AvgRight)/4
            trace!("HSqueeze progressive: res missing, upsampling avg with triangle filter");

            let out_width = output_channel.data.size().0;

            // Get input data from either temp_outputs or real buffer
            if let Some(temp_ch) = temp_outputs[buf_in[0]].as_ref() {
                let in_data = temp_ch.data.get_rect(in_rect);
                Self::process_hsqueeze_triangle_filter(&in_data, &mut output_channel, out_width);
            } else {
                let borrowed = buf_avg.buffer_grid[in_grid].data.borrow();
                let channel = borrowed
                    .as_ref()
                    .expect("avg input availability checked by has_input_channel");
                let in_data = channel.data.get_rect(in_rect);
                Self::process_hsqueeze_triangle_filter(&in_data, &mut output_channel, out_width);
            }

            return Ok(vec![(buf_out, output_channel)]);
        }

        // Case 2: Both avg and res available - run normal unsqueeze
        // Extract only the single columns we need from neighbors
        trace!("HSqueeze progressive: full unsqueeze with avg and res");

        // Get avg data (borrow from temp_outputs or real buffer)
        let _avg_guard;
        let avg_data = if let Some(temp_ch) = temp_outputs[buf_in[0]].as_ref() {
            temp_ch.data.get_rect(in_rect)
        } else {
            _avg_guard = buf_avg.buffer_grid[in_grid].data.borrow();
            let channel = _avg_guard.as_ref().expect("avg availability checked by has_input_channel");
            channel.data.get_rect(in_rect)
        };

        // Build edges: get next/prev columns from neighbors or fallback to current grid edges
        let next_col = (gx + 1 < buffers[buf_out].grid_shape.0).then(|| {
            let next_grid_pos = (gx + 1, gy);
            let next_in_grid = buf_avg.get_grid_idx(out_grid_kind, next_grid_pos);
            let next_rect = buf_avg.get_grid_rect(frame_header, out_grid_kind, next_grid_pos);
            let borrowed = buf_avg.buffer_grid[next_in_grid].data.borrow();
            borrowed.as_ref().map(|ch| {
                let data = ch.data.get_rect(next_rect);
                (0..data.size().1).map(|y| data.row(y)[0]).collect::<Vec<i32>>()
            })
        }).flatten().unwrap_or_else(|| {
            (0..avg_data.size().1).map(|y| avg_data.row(y)[avg_data.size().0 - 1]).collect()
        });

        let prev_col = (gx > 0).then(|| {
            let prev_grid_pos = (gx - 1, gy);
            let prev_out_grid = buffers[buf_out].get_grid_idx(out_grid_kind, prev_grid_pos);
            let prev_rect = buffers[buf_out].get_grid_rect(frame_header, out_grid_kind, prev_grid_pos);
            let borrowed = buffers[buf_out].buffer_grid[prev_out_grid].data.borrow();
            borrowed.as_ref().map(|ch| {
                let data = ch.data.get_rect(prev_rect);
                (0..data.size().1).map(|y| data.row(y)[data.size().0 - 1]).collect::<Vec<i32>>()
            })
        }).flatten().unwrap_or_else(|| {
            (0..avg_data.size().1).map(|y| avg_data.row(y)[0]).collect()
        });

        // Get res data (borrow from temp_outputs or real buffer)
        let res_rect = buf_res.get_grid_rect(frame_header, out_grid_kind, self.grid_pos);
        let _res_guard;
        let res_data = if let Some(temp_ch) = temp_outputs[buf_in[1]].as_ref() {
            temp_ch.data.get_rect(res_rect)
        } else {
            _res_guard = buf_res.buffer_grid[res_grid].data.borrow();
            let channel = _res_guard.as_ref().expect("res availability checked by has_input_channel");
            channel.data.get_rect(res_rect)
        };

        // Process unsqueeze with all data loaded and guards kept alive
        let out_width = output_channel.data.size().0;
        for y in 0..avg_data.size().1.min(res_data.size().1).min(output_channel.data.size().1) {
            let in_row = avg_data.row(y);
            let res_row = res_data.row(y);
            let out_row = output_channel.data.row_mut(y);
            let next_avg_edge = next_col[y];
            let prev_edge = prev_col[y];

            for x in 0..in_row.len().min(res_row.len()) {
                let avg = in_row[x];
                let res = res_row[x];

                // Get next_avg: normally the next position in current row, or edge value at boundary
                let next_avg = if x + 1 < in_row.len() {
                    in_row[x + 1]
                } else {
                    next_avg_edge
                };

                // Get prev: from previous output in current row, or edge value at left boundary
                let out_x = x * 2;
                let prev = if out_x > 0 {
                    out_row[out_x - 1]
                } else {
                    prev_edge
                };

                // Use the standard unsqueeze
                let (a, b) = super::squeeze::unsqueeze_scalar(avg, res, next_avg, prev);

                if out_x < out_width {
                    out_row[out_x] = a;
                }
                if out_x + 1 < out_width {
                    out_row[out_x + 1] = b;
                }
            }
        }

        Ok(vec![(buf_out, output_channel)])
    }

    /// Helper to process HSqueeze with triangle filter (avg-only, no res)
    fn process_hsqueeze_triangle_filter(
        in_data: &crate::image::ImageRect<'_, i32>,
        output_channel: &mut ModularChannel,
        out_width: usize,
    ) {
        for y in 0..in_data.size().1.min(output_channel.data.size().1) {
            let in_row = in_data.row(y);
            let out_row = output_channel.data.row_mut(y);

            for x in 0..in_data.size().0 {
                let avg = in_row[x];
                let avg_left = if x > 0 { in_row[x - 1] } else { avg };
                let avg_right = if x + 1 < in_data.size().0 {
                    in_row[x + 1]
                } else {
                    avg
                };

                let out_x = x * 2;
                if out_x < out_width {
                    out_row[out_x] = (avg_left + 3 * avg) / 4;
                }
                if out_x + 1 < out_width {
                    out_row[out_x + 1] = (3 * avg + avg_right) / 4;
                }
            }
        }
    }

    fn force_execute_vsqueeze(
        &self,
        frame_header: &FrameHeader,
        buffers: &[ModularBufferInfo],
        temp_outputs: &[Option<ModularChannel>],
        out_grid_kind: ModularGridKind,
        buf_in: &[usize; 2],
        buf_out: usize,
    ) -> Result<Vec<(usize, ModularChannel)>> {
        let buf_avg = &buffers[buf_in[0]];
        let buf_res = &buffers[buf_in[1]];
        let in_grid = buf_avg.get_grid_idx(out_grid_kind, self.grid_pos);
        let res_grid = buf_res.get_grid_idx(out_grid_kind, self.grid_pos);
        let out_grid = buffers[buf_out].get_grid_idx(out_grid_kind, self.grid_pos);

        // Check if required input channels are available
        if !self.has_input_channel(buf_in[0], in_grid, buffers, temp_outputs) {
            // Avg channel not available yet - can't execute
            return Ok(vec![]);
        }
        let has_res = self.has_input_channel(buf_in[1], res_grid, buffers, temp_outputs);

        let (gx, gy) = self.grid_pos;
        let output_size = buffers[buf_out].buffer_grid[out_grid].size;
        let mut output_channel = ModularChannel::new_with_shift(
            output_size,
            buffers[buf_out].info.shift,
            buffers[buf_out].info.bit_depth,
        )?;

        let in_rect = buf_avg.get_grid_rect(frame_header, out_grid_kind, self.grid_pos);

        if !has_res {
            // Case 1: Only avg available - upsample with triangle filter (no cloning)
            // Vertical: (AvgTop + 3 * Avg)/4, (3 * Avg + AvgBottom)/4
            trace!("VSqueeze progressive: res missing, upsampling avg with triangle filter");

            let out_width = output_channel.data.size().0;
            let out_height = output_channel.data.size().1;

            // Get input data from either temp_outputs or real buffer
            if let Some(temp_ch) = temp_outputs[buf_in[0]].as_ref() {
                let in_data = temp_ch.data.get_rect(in_rect);
                Self::process_vsqueeze_triangle_filter(&in_data, &mut output_channel, out_width, out_height);
            } else {
                let borrowed = buf_avg.buffer_grid[in_grid].data.borrow();
                let channel = borrowed
                    .as_ref()
                    .expect("avg availability checked by has_input_channel");
                let in_data = channel.data.get_rect(in_rect);
                Self::process_vsqueeze_triangle_filter(&in_data, &mut output_channel, out_width, out_height);
            }

            return Ok(vec![(buf_out, output_channel)]);
        }

        // Case 2: Both avg and res available - run normal unsqueeze
        // Extract only the single rows we need from neighbors
        trace!("VSqueeze progressive: full unsqueeze with avg and res");

        // Get avg data (borrow from temp_outputs or real buffer)
        let _avg_guard;
        let avg_data = if let Some(temp_ch) = temp_outputs[buf_in[0]].as_ref() {
            temp_ch.data.get_rect(in_rect)
        } else {
            _avg_guard = buf_avg.buffer_grid[in_grid].data.borrow();
            let channel = _avg_guard.as_ref().expect("avg availability checked by has_input_channel");
            channel.data.get_rect(in_rect)
        };

        // Build edges: get next/prev rows from neighbors or fallback to current grid edges
        let next_row = (gy + 1 < buffers[buf_out].grid_shape.1).then(|| {
            let next_grid_pos = (gx, gy + 1);
            let next_in_grid = buf_avg.get_grid_idx(out_grid_kind, next_grid_pos);
            let next_rect = buf_avg.get_grid_rect(frame_header, out_grid_kind, next_grid_pos);
            let borrowed = buf_avg.buffer_grid[next_in_grid].data.borrow();
            borrowed.as_ref().map(|ch| {
                let data = ch.data.get_rect(next_rect);
                data.row(0).to_vec()
            })
        }).flatten().unwrap_or_else(|| {
            avg_data.row(avg_data.size().1 - 1).to_vec()
        });

        let prev_row = (gy > 0).then(|| {
            let prev_grid_pos = (gx, gy - 1);
            let prev_out_grid = buffers[buf_out].get_grid_idx(out_grid_kind, prev_grid_pos);
            let prev_rect = buffers[buf_out].get_grid_rect(frame_header, out_grid_kind, prev_grid_pos);
            let borrowed = buffers[buf_out].buffer_grid[prev_out_grid].data.borrow();
            borrowed.as_ref().map(|ch| {
                let data = ch.data.get_rect(prev_rect);
                data.row(data.size().1 - 1).to_vec()
            })
        }).flatten().unwrap_or_else(|| {
            avg_data.row(0).to_vec()
        });

        // Get res data (borrow from temp_outputs or real buffer)
        let res_rect = buf_res.get_grid_rect(frame_header, out_grid_kind, self.grid_pos);
        let _res_guard;
        let res_data = if let Some(temp_ch) = temp_outputs[buf_in[1]].as_ref() {
            temp_ch.data.get_rect(res_rect)
        } else {
            _res_guard = buf_res.buffer_grid[res_grid].data.borrow();
            let channel = _res_guard.as_ref().expect("res availability checked by has_input_channel");
            channel.data.get_rect(res_rect)
        };

        // Process unsqueeze with all data loaded and guards kept alive
        let out_width = output_channel.data.size().0;
        let out_height = output_channel.data.size().1;

        for y in 0..avg_data.size().1.min(res_data.size().1) {
            let in_row = avg_data.row(y);
            let res_row = res_data.row(y);
            let out_y = y * 2;

            for x in 0..in_row.len().min(res_row.len()).min(out_width) {
                let avg = in_row[x];
                let res = res_row[x];
                let next_avg_edge = next_row[x];
                let prev_edge = prev_row[x];

                // Get next_avg: normally the next row in current column, or edge value at boundary
                let next_avg = if y + 1 < avg_data.size().1 {
                    avg_data.row(y + 1)[x]
                } else {
                    next_avg_edge
                };

                // Get prev: from previous output row, or edge value at top boundary
                let prev = if out_y > 0 {
                    output_channel.data.row(out_y - 1)[x]
                } else {
                    prev_edge
                };

                // Use the standard unsqueeze
                let (a, b) = super::squeeze::unsqueeze_scalar(avg, res, next_avg, prev);

                if out_y < out_height {
                    output_channel.data.row_mut(out_y)[x] = a;
                }
                if out_y + 1 < out_height {
                    output_channel.data.row_mut(out_y + 1)[x] = b;
                }
            }
        }

        Ok(vec![(buf_out, output_channel)])
    }

    /// Helper to process VSqueeze with triangle filter (avg-only, no res)
    fn process_vsqueeze_triangle_filter(
        in_data: &crate::image::ImageRect<'_, i32>,
        output_channel: &mut ModularChannel,
        out_width: usize,
        out_height: usize,
    ) {
        for y in 0..in_data.size().1 {
            let avg_top_row = if y > 0 {
                in_data.row(y - 1)
            } else {
                in_data.row(y)
            };
            let avg_row = in_data.row(y);
            let avg_bottom_row = if y + 1 < in_data.size().1 {
                in_data.row(y + 1)
            } else {
                in_data.row(y)
            };

            let out_y = y * 2;

            if out_y < out_height {
                let out_row_top = output_channel.data.row_mut(out_y);
                for x in 0..in_data.size().0.min(out_width) {
                    out_row_top[x] = (avg_top_row[x] + 3 * avg_row[x]) / 4;
                }
            }

            if out_y + 1 < out_height {
                let out_row_bottom = output_channel.data.row_mut(out_y + 1);
                for x in 0..in_data.size().0.min(out_width) {
                    out_row_bottom[x] = (3 * avg_row[x] + avg_bottom_row[x]) / 4;
                }
            }
        }
    }

    fn force_execute_rct(
        &self,
        buffers: &[ModularBufferInfo],
        temp_outputs: &[Option<ModularChannel>],
        buf_in: &[usize; 3],
        buf_out: &[usize; 3],
        op: super::RctOp,
        perm: super::RctPermutation,
    ) -> Result<Vec<(usize, ModularChannel)>> {
        // Check if all 3 input channels are available
        for &buf_idx in buf_in {
            let buf_info = &buffers[buf_idx];
            let grid_idx = buf_info.get_grid_idx(buf_info.grid_kind, self.grid_pos);
            if !self.has_input_channel(buf_idx, grid_idx, buffers, temp_outputs) {
                return Ok(vec![]); // Not enough data yet
            }
        }

        // Collect input channels - make clones since RCT is in-place
        let mut input_channels = Vec::new();
        for &buf_idx in buf_in {
            let buf_info = &buffers[buf_idx];
            let grid_idx = buf_info.get_grid_idx(buf_info.grid_kind, self.grid_pos);

            // Try temp_outputs first, then real buffer, and clone
            let cloned_channel = if let Some(temp_ch) = temp_outputs[buf_idx].as_ref() {
                temp_ch.try_clone()?
            } else {
                // For progressive mode, non-destructively access the buffer without modifying state
                let borrowed = buf_info.buffer_grid[grid_idx].data.borrow();
                let channel = borrowed
                    .as_ref()
                    .expect("input availability checked by has_input_channel");
                channel.try_clone()?
            };

            input_channels.push(cloned_channel);
        }

        // Execute RCT transform in-place on cloned channels
        // Need to use split_at_mut to get multiple mutable references
        let (r_slice, gb_slice) = input_channels.split_at_mut(1);
        let (g_slice, b_slice) = gb_slice.split_at_mut(1);
        super::rct::do_rct_step(
            &mut [&mut r_slice[0], &mut g_slice[0], &mut b_slice[0]],
            op,
            perm,
        );

        // Package outputs
        let mut results = Vec::new();
        for (i, channel) in input_channels.into_iter().enumerate() {
            let buf_idx = buf_out[i];
            results.push((buf_idx, channel));
        }

        Ok(results)
    }
}
