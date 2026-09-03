// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use super::{RctOp, RctPermutation};
use crate::error::Result;
use crate::frame::modular::buffers::{ModularChannel, with_buffers};
use crate::frame::modular::transforms::smooth_squeeze::smooth_upsample;
use crate::frame::modular::{
    DataStatus, FullModularImage, ModularBufferInfo, ModularGridKind, Predictor,
    TransformScratchSpace,
};
use crate::headers::frame_header::FrameHeader;
use crate::headers::modular::WeightedHeader;
use crate::image::{BufferRecycler, Image, ImageRect, Rect};
use crate::util::SmallVec;
use crate::util::sync::RwLockReadGuard;
use crate::util::sync::atomic::{AtomicUsize, Ordering};
use crate::util::tracing_wrappers::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frame::modular) struct SqueezeUpsample {
    pub source_buf: usize,
    pub source_grid: usize,
    pub shift_diff: (usize, usize),
}

#[derive(Debug, Clone)]
pub enum TransformStep {
    Rct {
        buf_in: [usize; 3],
        buf_out: [usize; 3],
        op: RctOp,
        perm: RctPermutation,
    },
    Palette {
        buf_in: usize,
        buf_pal: usize,
        buf_out: Vec<usize>,
        num_colors: usize,
        num_deltas: usize,
        predictor: Predictor,
        wp_header: WeightedHeader,
    },
    HSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
        upsample: Option<SqueezeUpsample>,
    },
    VSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
        upsample: Option<SqueezeUpsample>,
    },
    Output {
        buf_in: usize,
        rect: Option<Rect>,
        group: usize,
        channel: usize,
    },
}

#[derive(Debug)]
pub struct TransformStepChunk {
    pub(super) step: TransformStep,

    // Grid position this transform should produce.
    // Note that this is a lie for Palette with AverageAll or Weighted, as the transform with
    // position (0, y) will produce the entire row of blocks (*, y) (and there will be no
    // transforms with position (x, y) with x > 0).
    pub(super) grid_pos: (usize, usize),

    // Number of missing final dependencies for this transform.
    // Note that this is updated *before* actually computing other transforms.
    pub(super) missing_final_deps: usize,

    // Number of dependencies that are still missing *during this progressive
    // preview phase*.
    pub(super) missing_deps: AtomicUsize,
}

impl TransformStepChunk {
    pub(in crate::frame::modular) fn set_squeeze_upsample(
        &mut self,
        upsample: Option<SqueezeUpsample>,
    ) {
        match &mut self.step {
            TransformStep::HSqueeze { upsample: u, .. }
            | TransformStep::VSqueeze { upsample: u, .. } => {
                *u = upsample;
            }
            _ => {}
        }
    }
}

impl FullModularImage {
    pub(in crate::frame::modular) fn compute_squeeze_upsample(
        &self,
        t: usize,
    ) -> Option<SqueezeUpsample> {
        let chunk = &self.transform_steps[t];
        let (buf_in, buf_out) = match chunk.step {
            TransformStep::HSqueeze {
                buf_in, buf_out, ..
            }
            | TransformStep::VSqueeze {
                buf_in, buf_out, ..
            } => (buf_in, buf_out),
            _ => return None,
        };
        let out_kind = self.buffer_info[buf_out].grid_kind;
        let res_grid = self.buffer_info[buf_in[1]].get_grid_idx(out_kind, chunk.grid_pos);
        if self.buffer_info[buf_in[1]].buffer_grid[res_grid].data_status != DataStatus::Zero {
            return None;
        }
        let mut cur_buf = buf_in[0];
        let mut cur_grid = self.buffer_info[cur_buf].get_grid_idx(out_kind, chunk.grid_pos);
        while let Some(prev_t) = self.buffer_info[cur_buf].buffer_grid[cur_grid].produced_by_step {
            let (next_avg, next_res) = match self.transform_steps[prev_t].step {
                TransformStep::HSqueeze { buf_in: [a, r], .. }
                | TransformStep::VSqueeze { buf_in: [a, r], .. } => (a, r),
                _ => break,
            };
            let next_res_grid = self.buffer_info[next_res].get_grid_idx(out_kind, chunk.grid_pos);
            if self.buffer_info[next_res].buffer_grid[next_res_grid].data_status != DataStatus::Zero
            {
                break;
            }
            cur_buf = next_avg;
            cur_grid = self.buffer_info[cur_buf].get_grid_idx(out_kind, chunk.grid_pos);
        }
        let src_shift = self.buffer_info[cur_buf].info.shift.unwrap_or((0, 0));
        let dst_shift = self.buffer_info[buf_out].info.shift.unwrap_or((0, 0));
        let shift_diff = (
            src_shift.0.checked_sub(dst_shift.0).unwrap(),
            src_shift.1.checked_sub(dst_shift.1).unwrap(),
        );
        Some(SqueezeUpsample {
            source_buf: cur_buf,
            source_grid: cur_grid,
            shift_diff,
        })
    }
}

#[derive(Debug)]
enum SqueezeInfo {
    Upsample {
        upsample: SqueezeUpsample,
        out_rect: Rect,
    },
    Regular {
        in_avg: (usize, usize),
        avg_rect: Rect,
        in_res: (usize, usize),
        res_rect: Rect,
        in_next_avg: Option<(usize, usize)>,
        out_prev: Option<(usize, usize)>,
    },
}

fn borrow_channel(
    buffers: &[ModularBufferInfo],
    x: (usize, usize),
) -> RwLockReadGuard<'_, Option<ModularChannel>> {
    buffers[x.0].buffer_grid[x.1].data.try_read().unwrap()
}

fn borrow_topbottom(
    buffers: &[ModularBufferInfo],
    x: (usize, usize),
) -> RwLockReadGuard<'_, Option<Image<i32>>> {
    buffers[x.0].buffer_grid[x.1].topbottom.try_read().unwrap()
}

fn borrow_leftright(
    buffers: &[ModularBufferInfo],
    x: (usize, usize),
) -> RwLockReadGuard<'_, Option<Image<i32>>> {
    buffers[x.0].buffer_grid[x.1].leftright.try_read().unwrap()
}

impl SqueezeInfo {
    fn new(
        buffers: &[ModularBufferInfo],
        buf_in: [usize; 2],
        buf_out: usize,
        upsample: Option<SqueezeUpsample>,
        output_grid_pos: (usize, usize),
        frame_header: &FrameHeader,
        vertical: bool,
    ) -> Self {
        let (gx, gy) = output_grid_pos;
        let output_grid_kind = buffers[buf_out].grid_kind;
        if let Some(upsample) = upsample {
            let mut out_rect =
                buffers[buf_out].get_grid_rect(frame_header, output_grid_kind, output_grid_pos);
            out_rect.origin = if output_grid_kind == ModularGridKind::None {
                (0, 0)
            } else {
                let out_shift = buffers[buf_out].info.shift.unwrap_or((0, 0));
                let out_grid_dim = output_grid_kind.grid_dim(frame_header, out_shift);
                (gx * out_grid_dim.0, gy * out_grid_dim.1)
            };
            return Self::Upsample { upsample, out_rect };
        }
        let buf_avg = &buffers[buf_in[0]];
        let buf_res = &buffers[buf_in[1]];
        let in_grid = buf_avg.get_grid_idx(output_grid_kind, output_grid_pos);
        let res_grid = buf_res.get_grid_idx(output_grid_kind, output_grid_pos);
        let pos_next = if vertical {
            (gy < buffers[buf_out].grid_shape.1 - 1).then(|| (gx, gy + 1))
        } else {
            (gx < buffers[buf_out].grid_shape.0 - 1).then(|| (gx + 1, gy))
        };
        let pos_prev = if vertical {
            (gy > 0).then(|| (gx, gy - 1))
        } else {
            (gx > 0).then(|| (gx - 1, gy))
        };
        let mut avg_rect = buf_avg.get_grid_rect(frame_header, output_grid_kind, output_grid_pos);
        let next_avg_grid = if let Some(pos) = pos_next {
            let grid = buf_avg.get_grid_idx(output_grid_kind, pos);
            if grid == in_grid {
                if vertical {
                    avg_rect.size.1 += 1;
                } else {
                    avg_rect.size.0 += 1;
                }
                None
            } else {
                Some(grid)
            }
        } else {
            None
        };
        let prev_out_grid = pos_prev.map(|x| buffers[buf_out].get_grid_idx(output_grid_kind, x));
        Self::Regular {
            in_avg: (buf_in[0], in_grid),
            avg_rect,
            in_res: (buf_in[1], res_grid),
            res_rect: buf_res.get_grid_rect(frame_header, output_grid_kind, output_grid_pos),
            in_next_avg: next_avg_grid.map(|x| (buf_in[0], x)),
            out_prev: prev_out_grid.map(|x| (buf_out, x)),
        }
    }

    // The lifetimes prevent calling decrement_refs while buffers are still borrowed.
    fn borrow_inputs<'a>(
        &'a self,
        buffers: &'a [ModularBufferInfo],
        vertical: bool,
    ) -> SqueezeInputs<'a> {
        let (in_avg, in_res, in_next_avg, out_prev) = match self {
            Self::Regular {
                in_avg,
                in_res,
                in_next_avg,
                out_prev,
                ..
            } => (*in_avg, *in_res, *in_next_avg, *out_prev),
            Self::Upsample { .. } => unreachable!(),
        };
        let borrow_border = if vertical {
            borrow_topbottom
        } else {
            borrow_leftright
        };
        SqueezeInputs {
            in_avg: borrow_channel(buffers, in_avg),
            in_res: borrow_channel(buffers, in_res),
            in_next_border: in_next_avg.map(|x| borrow_border(buffers, x)),
            out_prev_border: out_prev.map(|x| borrow_border(buffers, x)),
        }
    }

    fn decrement_refs(
        self,
        buffers: &[ModularBufferInfo],
        is_final: bool,
        recycler: &BufferRecycler,
    ) {
        match self {
            Self::Regular { in_avg, in_res, .. } => {
                let buf_avg = &buffers[in_avg.0].buffer_grid[in_avg.1];
                let buf_res = &buffers[in_res.0].buffer_grid[in_res.1];
                buf_avg.mark_used(buf_avg.can_consume(is_final), recycler);
                buf_res.mark_used(buf_res.can_consume(is_final), recycler);
            }
            Self::Upsample { upsample: u, .. } => {
                assert!(!is_final);
                let buf = &buffers[u.source_buf].buffer_grid[u.source_grid];
                buf.mark_used(buf.can_consume(false), recycler);
            }
        }
    }

    fn borrow_upsample_view<'a>(
        &'a self,
        buffers: &'a [ModularBufferInfo],
        frame_header: &FrameHeader,
    ) -> TiledChannelView<'a> {
        let u = match self {
            Self::Upsample { upsample: u, .. } => *u,
            Self::Regular { .. } => unreachable!(),
        };
        let (b, g) = (u.source_buf, u.source_grid);
        let buf_info = &buffers[b];
        let (xs, ys) = buf_info.grid_shape;
        let (gx, gy) = (g % xs, g / xs);
        let grid_dim = if buf_info.grid_kind == ModularGridKind::None {
            None
        } else {
            let shift = buf_info.info.shift.unwrap_or((0, 0));
            Some(buf_info.grid_kind.grid_dim(frame_header, shift))
        };

        let mut tiles: [Option<TileGuard<'a>>; 9] = Default::default();
        tiles[4] = Some(TileGuard::Channel(borrow_channel(buffers, (b, g))));

        let borrow_border = |tile_x: isize, tile_y: isize, is_tb: bool| -> Option<TileGuard<'a>> {
            if tile_x >= 0 && tile_x < xs as isize && tile_y >= 0 && tile_y < ys as isize {
                let grid_idx = tile_y as usize * xs + tile_x as usize;
                let guard = if is_tb {
                    borrow_topbottom(buffers, (b, grid_idx))
                } else {
                    borrow_leftright(buffers, (b, grid_idx))
                };
                Some(TileGuard::Border(guard))
            } else {
                None
            }
        };

        for dx in 0..3 {
            let tx = gx as isize + dx as isize - 1;
            tiles[dx] = borrow_border(tx, gy as isize - 1, true);
            tiles[6 + dx] = borrow_border(tx, gy as isize + 1, true);
        }
        tiles[3] = borrow_border(gx as isize - 1, gy as isize, false);
        tiles[5] = borrow_border(gx as isize + 1, gy as isize, false);

        TiledChannelView {
            size: buf_info.info.size,
            grid_dim,
            center_grid_pos: (gx, gy),
            tiles,
        }
    }
}

struct SqueezeInputs<'a> {
    in_avg: RwLockReadGuard<'a, Option<ModularChannel>>,
    in_res: RwLockReadGuard<'a, Option<ModularChannel>>,
    in_next_border: Option<RwLockReadGuard<'a, Option<Image<i32>>>>,
    out_prev_border: Option<RwLockReadGuard<'a, Option<Image<i32>>>>,
}

impl<'a> SqueezeInputs<'a> {
    fn in_avg_rect(&self, rect: Rect) -> ImageRect<'_, i32> {
        self.in_avg.as_ref().unwrap().data.get_rect(rect)
    }

    fn in_res_rect(&self, rect: Rect) -> ImageRect<'_, i32> {
        self.in_res.as_ref().unwrap().data.get_rect(rect)
    }

    fn in_next_border(&self) -> Option<&Image<i32>> {
        self.in_next_border.as_ref().and_then(|g| g.as_ref())
    }

    fn out_prev_border(&self) -> Option<&Image<i32>> {
        self.out_prev_border.as_ref().and_then(|g| g.as_ref())
    }
}

enum TileGuard<'a> {
    Channel(RwLockReadGuard<'a, Option<ModularChannel>>),
    Border(RwLockReadGuard<'a, Option<Image<i32>>>),
}

impl<'a> std::ops::Deref for TileGuard<'a> {
    type Target = Image<i32>;

    fn deref(&self) -> &Self::Target {
        match self {
            TileGuard::Channel(g) => &g.as_ref().unwrap().data,
            TileGuard::Border(g) => g.as_ref().unwrap(),
        }
    }
}

pub(super) struct TiledChannelView<'a> {
    size: (usize, usize),
    grid_dim: Option<(usize, usize)>,
    center_grid_pos: (usize, usize),
    tiles: [Option<TileGuard<'a>>; 9],
}

impl<'a> TiledChannelView<'a> {
    #[inline(always)]
    fn top_border_row(ly: usize, top_tile_h: usize) -> usize {
        if top_tile_h <= 1 || ly == top_tile_h - 1 {
            3
        } else {
            2
        }
    }

    #[inline(always)]
    fn bottom_border_row(ly: usize) -> usize {
        if ly == 0 { 0 } else { 1 }
    }

    fn copy_tile_slice(
        &self,
        (dx, dy): (usize, usize),
        ly: usize,
        src_range: std::ops::Range<usize>,
        tile_w: usize,
        top_tile_h: usize,
        dest: &mut [i32],
    ) {
        let img = self.tiles[dy * 3 + dx].as_deref().unwrap();
        match (dy, dx) {
            (1, 1) => {
                dest.copy_from_slice(&img.row(ly)[src_range]);
            }
            (0, _) => {
                dest.copy_from_slice(&img.row(Self::top_border_row(ly, top_tile_h))[src_range]);
            }
            (2, _) => {
                dest.copy_from_slice(&img.row(Self::bottom_border_row(ly))[src_range]);
            }
            (1, 0) => {
                let offset_start = 4 - (tile_w - src_range.start);
                let offset_end = 4 - (tile_w - src_range.end);
                dest.copy_from_slice(&img.row(ly)[offset_start..offset_end]);
            }
            (1, 2) => {
                dest.copy_from_slice(&img.row(ly)[src_range]);
            }
            _ => unreachable!(),
        }
    }

    pub fn load_row_to_scratch(
        &self,
        yg: isize,
        xoff: usize,
        valid_len: usize,
        row_buf: &mut [i32],
    ) {
        let (w, h) = self.size;
        // Only the first `valid_len` values are actual convolution inputs; the rest
        // of the buffer only exists so that SIMD loads for the last output chunk
        // stay in bounds, and their values do not affect the output. Restrict the
        // buffer reads to the valid region: reading further could touch grid
        // positions outside the 3x3 neighbourhood that the transform's
        // dependencies guarantee to be safe to read, racing with their producers.
        let max_len = row_buf.len().min(valid_len);
        let clamped_y = if h == 1 {
            0
        } else if yg < 0 {
            (-yg - 1) as usize
        } else if yg as usize >= h {
            2 * h - 1 - yg as usize
        } else {
            yg as usize
        };

        let Some(grid_dim) = self.grid_dim else {
            let row = self.tiles[4].as_deref().unwrap().row(clamped_y);

            let left_clamp = (2 - xoff as isize).max(0) as usize;
            let right_clamp_start = (w as isize + 2 - xoff as isize).min(max_len as isize) as usize;

            if left_clamp < right_clamp_start {
                let src_start = (xoff as isize + left_clamp as isize - 2) as usize;
                let len = right_clamp_start - left_clamp;
                row_buf[left_clamp..right_clamp_start]
                    .copy_from_slice(&row[src_start..src_start + len]);
            }

            if left_clamp > 0 {
                row_buf[..left_clamp].fill(row[0]);
            }

            if right_clamp_start < max_len {
                row_buf[right_clamp_start..max_len].fill(row[w - 1]);
            }

            let last = row_buf[max_len - 1];
            row_buf[max_len..].fill(last);
            return;
        };

        let (grid_w, grid_h) = grid_dim;
        let (gx_center, gy_center) = self.center_grid_pos;
        let gy = clamped_y / grid_h;
        let ly = clamped_y % grid_h;
        let dy = (gy as isize - gy_center as isize + 1) as usize;

        let top_tile_h = if gy_center > 0 {
            (h - (gy_center - 1) * grid_h).min(grid_h)
        } else {
            grid_h
        };

        let global_x_start = xoff as isize - 2;
        let global_x_end = global_x_start + max_len as isize;

        let left_clamp = (-global_x_start).max(0) as usize;
        let right_clamp = (global_x_end - w as isize).max(0) as usize;

        let clamped_x_start = global_x_start.max(0) as usize;
        let clamped_x_end = global_x_end.min(w as isize) as usize;

        if clamped_x_start < clamped_x_end {
            let gx_start = clamped_x_start / grid_w;
            let gx_end = (clamped_x_end - 1) / grid_w;

            for gx in gx_start..=gx_end {
                let dx = (gx as isize - gx_center as isize + 1) as usize;
                let tile_x_start = gx * grid_w;
                let tile_w = (w - tile_x_start).min(grid_w);
                let intersect_start = clamped_x_start.max(tile_x_start);
                let intersect_end = clamped_x_end.min(tile_x_start + tile_w);

                if intersect_start < intersect_end {
                    let dest_start = left_clamp + (intersect_start - clamped_x_start);
                    let dest_end = dest_start + (intersect_end - intersect_start);
                    let src_start = intersect_start - tile_x_start;
                    let src_end = intersect_end - tile_x_start;
                    self.copy_tile_slice(
                        (dx, dy),
                        ly,
                        src_start..src_end,
                        tile_w,
                        top_tile_h,
                        &mut row_buf[dest_start..dest_end],
                    );
                }
            }
        }

        if left_clamp > 0 {
            let left_val = match dy {
                1 => self.tiles[4].as_deref().unwrap().row(ly)[0],
                0 => self.tiles[1]
                    .as_deref()
                    .unwrap()
                    .row(Self::top_border_row(ly, top_tile_h))[0],
                2 => self.tiles[7]
                    .as_deref()
                    .unwrap()
                    .row(Self::bottom_border_row(ly))[0],
                _ => 0,
            };
            row_buf[..left_clamp].fill(left_val);
        }

        if right_clamp > 0 {
            let gx_right = (w - 1) / grid_w;
            let lx_right = (w - 1) % grid_w;
            let dx = (gx_right as isize - gx_center as isize + 1) as usize;
            let right_val = match (dy, dx) {
                (1, 1) => self.tiles[4].as_deref().unwrap().row(ly)[lx_right],
                (1, 2) => self.tiles[5].as_deref().unwrap().row(ly)[3],
                (0, 1 | 2) => self.tiles[dx]
                    .as_deref()
                    .unwrap()
                    .row(Self::top_border_row(ly, top_tile_h))[lx_right],
                (2, 1 | 2) => self.tiles[6 + dx]
                    .as_deref()
                    .unwrap()
                    .row(Self::bottom_border_row(ly))[lx_right],
                _ => 0,
            };
            row_buf[max_len - right_clamp..max_len].fill(right_val);
        }

        let last = row_buf[max_len - 1];
        row_buf[max_len..].fill(last);
    }
}

impl TransformStepChunk {
    fn buf_out(&self) -> &[usize] {
        match &self.step {
            TransformStep::Rct { buf_out, .. } => buf_out,
            TransformStep::Palette { buf_out, .. } => buf_out,
            TransformStep::HSqueeze { buf_out, .. } | TransformStep::VSqueeze { buf_out, .. } => {
                std::slice::from_ref(buf_out)
            }
            TransformStep::Output { .. } => &[],
        }
    }

    // (group, channel)
    pub fn output_info(&self) -> Option<(usize, usize)> {
        match &self.step {
            TransformStep::Output { group, channel, .. } => Some((*group, *channel)),
            _ => None,
        }
    }

    // Returns true if this was the last remaining final dep.
    pub fn final_dep_ready(&mut self) -> bool {
        self.missing_final_deps = self.missing_final_deps.checked_sub(1).unwrap();
        self.missing_final_deps == 0
    }

    pub fn ready_for_final_render(&self) -> bool {
        self.missing_final_deps == 0
    }

    pub fn no_current_deps(&self) -> bool {
        self.missing_deps.load(Ordering::Relaxed) == 0
    }

    // Returns true if this was the last remaining current dep.
    pub fn current_dep_ready(&self) -> bool {
        let v = self.missing_deps.fetch_sub(1, Ordering::Relaxed);
        assert_ne!(v, 0);
        v == 1
    }

    pub fn add_current_dep(&mut self) {
        self.missing_deps.fetch_add(1, Ordering::Relaxed);
    }

    // Runs this transform. This function *will* crash if the transform is not ready.
    #[instrument(level = "trace", skip_all)]
    pub fn do_run(
        &self,
        frame_header: &FrameHeader,
        buffers: &[ModularBufferInfo],
        transform_scratch_space: &mut TransformScratchSpace,
        recycler: &BufferRecycler,
        pass_to_pipeline: &dyn Fn(usize, usize, bool, Image<i32>) -> Result<()>,
    ) -> Result<()> {
        let is_final = self.missing_final_deps == 0;
        let buf_out = self.buf_out();

        // *INPUT* values for Output transforms.
        let (out_grid_kind, out_grid, out_size) =
            if let TransformStep::Output { buf_in, .. } = self.step {
                let b = buf_in;
                (
                    buffers[b].grid_kind,
                    buffers[b].get_grid_idx(buffers[b].grid_kind, self.grid_pos),
                    buffers[b].info.size,
                )
            } else {
                let b = buf_out[0];
                (
                    buffers[b].grid_kind,
                    buffers[b].get_grid_idx(buffers[b].grid_kind, self.grid_pos),
                    buffers[b].info.size,
                )
            };
        for bo in buf_out {
            assert_eq!(out_grid_kind, buffers[*bo].grid_kind);
            assert_eq!(out_size, buffers[*bo].info.size);
        }

        match &self.step {
            TransformStep::Rct {
                buf_in,
                buf_out,
                op,
                perm,
            } => {
                for i in 0..3 {
                    assert_eq!(out_grid_kind, buffers[buf_in[i]].grid_kind);
                    assert_eq!(out_size, buffers[buf_in[i]].info.size);
                    // Optimistically move the buffers to the output if possible.
                    // If not, creates buffers in the output that are a copy of the input buffers.
                    // This should be rare.
                    let b_in = &buffers[buf_in[i]].buffer_grid[out_grid];
                    let b_out = &buffers[buf_out[i]].buffer_grid[out_grid];
                    if b_in.data_status == DataStatus::Zero && !b_in.has_buffer() {
                        b_out.ensure_buffer(&buffers[buf_out[i]].info, recycler)?;
                    } else {
                        *b_out.data.try_write().unwrap() =
                            Some(b_in.get_buffer(b_in.can_consume(is_final), recycler)?);
                    }
                }
                with_buffers(buffers, buf_out, out_grid, recycler, |mut bufs| {
                    super::rct::do_rct_step(&mut bufs, *op, *perm);
                    Ok(())
                })?;
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                num_colors,
                num_deltas,
                predictor,
                ..
            } if !predictor.requires_full_row() => {
                assert_eq!(out_grid_kind, buffers[*buf_in].grid_kind);
                assert_eq!(out_size, buffers[*buf_in].info.size);

                with_buffers(buffers, buf_out, out_grid, recycler, |_| Ok(()))?;
                if out_size.0 != 0 {
                    let img_pal = borrow_channel(buffers, (*buf_pal, 0));
                    let grid_shape = buffers[buf_out[0]].grid_shape;
                    let grid_x = out_grid % grid_shape.0;
                    let grid_y = out_grid / grid_shape.0;
                    let has_left = *predictor != Predictor::Zero && grid_x > 0;
                    let has_top = *predictor != Predictor::Zero && grid_y > 0;
                    let borrow_border_guards = |grid_idx: usize, is_lr: bool| {
                        buf_out
                            .iter()
                            .map(|i| {
                                if is_lr {
                                    borrow_leftright(buffers, (*i, grid_idx))
                                } else {
                                    borrow_topbottom(buffers, (*i, grid_idx))
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    let stride = grid_shape.0;
                    let left_guards = has_left
                        .then(|| borrow_border_guards(grid_y * stride + (grid_x - 1), true));
                    let top_guards = has_top
                        .then(|| borrow_border_guards((grid_y - 1) * stride + grid_x, false));
                    let topleft_guards = (has_left && has_top)
                        .then(|| borrow_border_guards((grid_y - 1) * stride + (grid_x - 1), false));
                    let left_refs = left_guards
                        .as_ref()
                        .map(|v| v.iter().map(|x| x.as_ref().unwrap()).collect::<Vec<_>>());
                    let top_refs = top_guards
                        .as_ref()
                        .map(|v| v.iter().map(|x| x.as_ref().unwrap()).collect::<Vec<_>>());
                    let topleft_refs = topleft_guards
                        .as_ref()
                        .map(|v| v.iter().map(|x| x.as_ref().unwrap()).collect::<Vec<_>>());

                    let mut guards = vec![];
                    for i in buf_out {
                        let b = &buffers[*i].buffer_grid[out_grid];
                        guards.push(b.data.try_write().unwrap());
                    }
                    let mut out_buf_refs: Vec<&mut ModularChannel> =
                        guards.iter_mut().map(|g| g.as_mut().unwrap()).collect();
                    if matches!(predictor, Predictor::Zero)
                        && buffers[*buf_in].buffer_grid[out_grid].data_status == DataStatus::Zero
                    {
                        super::palette::zero_palette_step_one_group(
                            img_pal.as_ref().unwrap(),
                            &mut out_buf_refs,
                            *num_colors,
                            *num_deltas,
                        );
                    } else {
                        let img_in = borrow_channel(buffers, (*buf_in, out_grid));
                        super::palette::do_palette_step_one_group(
                            img_in.as_ref().unwrap(),
                            img_pal.as_ref().unwrap(),
                            &mut out_buf_refs,
                            left_refs.as_deref(),
                            top_refs.as_deref(),
                            topleft_refs.as_deref(),
                            *num_colors,
                            *num_deltas,
                            *predictor,
                            &mut transform_scratch_space.palette_row_scratch,
                        );
                    }
                }
                let buf_in_grid = &buffers[*buf_in].buffer_grid[out_grid];
                let buf_pal_grid = &buffers[*buf_pal].buffer_grid[0];
                buf_in_grid.mark_used(buf_in_grid.can_consume(is_final), recycler);
                buf_pal_grid.mark_used(buf_pal_grid.can_consume(is_final), recycler);
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                num_colors,
                num_deltas,
                predictor,
                wp_header,
            } => {
                assert_eq!(out_grid_kind, buffers[*buf_in].grid_kind);
                assert_eq!(out_size, buffers[*buf_in].info.size);
                let grid_shape = buffers[buf_out[0]].grid_shape;
                assert_eq!(out_grid % grid_shape.0, 0);
                let grid_y = out_grid / grid_shape.0;
                for grid_x in 0..grid_shape.0 {
                    // Ensure that the output buffers are present.
                    // TODO(szabadka): Extend the callback to support many grid points.
                    with_buffers(buffers, buf_out, out_grid + grid_x, recycler, |_| Ok(()))?;
                }
                if out_size.0 != 0 {
                    let mut in_bufs = vec![];
                    for grid_x in 0..grid_shape.0 {
                        let grid = grid_y * grid_shape.0 + grid_x;
                        in_bufs.push(borrow_channel(buffers, (*buf_in, grid)));
                    }
                    let in_buf_refs: Vec<&ModularChannel> =
                        in_bufs.iter().map(|x| x.as_ref().unwrap()).collect();
                    let img_pal = borrow_channel(buffers, (*buf_pal, 0));
                    // The previous row of output grids is only read as prediction
                    // context, so take read locks on it: other transforms (e.g.
                    // Output) may read it concurrently.
                    let mut prev_guards = vec![];
                    let mut prev_aux_guards = vec![];
                    if grid_y > 0 {
                        for i in buf_out {
                            for grid_x in 0..grid_shape.0 {
                                let grid = (grid_y - 1) * grid_shape.0 + grid_x;
                                prev_guards.push(borrow_topbottom(buffers, (*i, grid)));
                            }
                            if *predictor == Predictor::Weighted {
                                let grid = (grid_y - 1) * grid_shape.0;
                                prev_aux_guards.push(
                                    buffers[*i].buffer_grid[grid]
                                        .auxiliary_data
                                        .try_read()
                                        .unwrap(),
                                );
                            }
                        }
                    }
                    let prev_refs: Vec<&Image<i32>> =
                        prev_guards.iter().map(|g| g.as_ref().unwrap()).collect();
                    let prev_aux_refs: Vec<Option<&Image<i32>>> =
                        prev_aux_guards.iter().map(|g| g.as_ref()).collect();
                    let mut guards = vec![];
                    for i in buf_out {
                        for grid_x in 0..grid_shape.0 {
                            let grid = grid_y * grid_shape.0 + grid_x;
                            let b = &buffers[*i].buffer_grid[grid];
                            guards.push(b.data.try_write().unwrap());
                        }
                    }
                    let mut out_buf_refs: Vec<&mut ModularChannel> =
                        guards.iter_mut().map(|g| g.as_mut().unwrap()).collect();
                    let mut aux_out: Vec<_> = if *predictor == Predictor::Weighted {
                        buf_out
                            .iter()
                            .map(|i| {
                                buffers[*i].buffer_grid[grid_y * grid_shape.0]
                                    .auxiliary_data
                                    .try_write()
                                    .unwrap()
                            })
                            .collect()
                    } else {
                        vec![]
                    };
                    super::palette::do_palette_step_group_row(
                        &in_buf_refs,
                        img_pal.as_ref().unwrap(),
                        &mut out_buf_refs,
                        (grid_y > 0).then_some(&prev_refs[..]),
                        (grid_y > 0 && *predictor == Predictor::Weighted)
                            .then_some(&prev_aux_refs[..]),
                        &mut aux_out,
                        grid_shape.0,
                        *num_colors,
                        *num_deltas,
                        *predictor,
                        wp_header,
                        &mut transform_scratch_space.palette_row_scratch,
                    )?;
                }
                let buf_pal_grid = &buffers[*buf_pal].buffer_grid[0];
                buf_pal_grid.mark_used(buf_pal_grid.can_consume(is_final), recycler);
                for grid_x in 0..grid_shape.0 {
                    let buf_in_grid = &buffers[*buf_in].buffer_grid[out_grid + grid_x];
                    buf_in_grid.mark_used(buf_in_grid.can_consume(is_final), recycler);
                }
            }
            TransformStep::HSqueeze {
                buf_in,
                buf_out,
                upsample,
            }
            | TransformStep::VSqueeze {
                buf_in,
                buf_out,
                upsample,
            } => {
                let vertical = matches!(self.step, TransformStep::VSqueeze { .. });
                let info = SqueezeInfo::new(
                    buffers,
                    *buf_in,
                    *buf_out,
                    *upsample,
                    self.grid_pos,
                    frame_header,
                    vertical,
                );
                trace!(
                    "{} {:?} -> {:?}, grid {out_grid} grid pos {:?}: {info:?}",
                    if vertical { "VSqueeze" } else { "HSqueeze" },
                    buf_in,
                    buf_out,
                    self.grid_pos
                );
                with_buffers(buffers, &[*buf_out], out_grid, recycler, |mut bufs| {
                    if bufs.is_empty() {
                        return Ok(());
                    }
                    match info {
                        SqueezeInfo::Upsample {
                            upsample: u,
                            out_rect,
                        } => {
                            assert!(!is_final);
                            assert_eq!(bufs.len(), 1);
                            let view = info.borrow_upsample_view(buffers, frame_header);
                            let scratch = &mut transform_scratch_space.smooth_upsample_scratch;
                            let dither = buffers[*buf_out].info.shift.unwrap_or((0, 0)) == (0, 0)
                                && !buffers[*buf_out].info.followed_by_palette;
                            smooth_upsample(
                                &view,
                                u.shift_diff,
                                dither,
                                out_rect,
                                &mut bufs[0].data,
                                scratch,
                            );
                        }
                        SqueezeInfo::Regular {
                            avg_rect, res_rect, ..
                        } => {
                            let inputs = info.borrow_inputs(buffers, vertical);
                            if vertical {
                                super::squeeze::do_vsqueeze_step(
                                    &inputs.in_avg_rect(avg_rect),
                                    &inputs.in_res_rect(res_rect),
                                    inputs.in_next_border(),
                                    inputs.out_prev_border(),
                                    &mut bufs,
                                );
                            } else {
                                super::squeeze::do_hsqueeze_step(
                                    &inputs.in_avg_rect(avg_rect),
                                    &inputs.in_res_rect(res_rect),
                                    inputs.in_next_border(),
                                    inputs.out_prev_border(),
                                    &mut bufs,
                                );
                            }
                        }
                    }
                    Ok(())
                })?;
                info.decrement_refs(buffers, is_final, recycler);
            }
            TransformStep::Output {
                buf_in,
                rect,
                group,
                channel,
            } => {
                debug!("Rendering channel {channel:?}, rect {rect:?}, group {group}");
                let buf = &buffers[*buf_in].buffer_grid[out_grid];
                if buf.data_status == DataStatus::Zero && !buf.has_buffer() {
                    let zero = Image::new(rect.map(|x| x.size).unwrap_or(buf.size))?;
                    pass_to_pipeline(*channel, *group, is_final, zero)?;
                } else {
                    let modular_buf = buf.get_buffer(buf.can_consume(is_final), recycler)?;
                    if let Some(rect) = rect {
                        let mut cropped = Image::new(rect.size)?;
                        let src_view = modular_buf.data.get_rect(*rect);
                        for y in 0..rect.size.1 {
                            cropped.row_mut(y).copy_from_slice(src_view.row(y));
                        }
                        pass_to_pipeline(*channel, *group, is_final, cropped)?;
                    } else {
                        pass_to_pipeline(*channel, *group, is_final, modular_buf.data)?;
                    }
                }
            }
        };

        for &(buf, grid) in self.outputs(buffers).iter() {
            buffers[buf].buffer_grid[grid].extract_needed_borders(recycler)?;
        }

        if is_final {
            for dep in self.dependencies(buffers, frame_header).iter() {
                buffers[dep.buffer].buffer_grid[dep.grid].mark_final_use_done(recycler);
            }
        }

        Ok(())
    }

    // Iterates over the list of outputs for this transform.
    // Except for palette, we only output 1 (squeeze) or 3 (RCT) buffers.
    // For non-delta palette, in most cases we output 1, 3 or 4 channels.
    pub fn outputs(&self, buffers: &[ModularBufferInfo]) -> SmallVec<(usize, usize), 4> {
        let buf_out = self.buf_out();
        let b = buf_out.first().copied().unwrap_or(0);
        let out_grid_kind = buffers[b].grid_kind;
        let out_grid = buffers[b].get_grid_idx(out_grid_kind, self.grid_pos);
        let grid_offset_up = match &self.step {
            TransformStep::Palette {
                buf_out, predictor, ..
            } if predictor.requires_full_row() => buffers[buf_out[0]].grid_shape.0,
            TransformStep::Output { .. } => 0,
            _ => 1,
        };

        buf_out
            .iter()
            .flat_map(move |x| (0..grid_offset_up).map(move |y| (*x, out_grid + y)))
            .collect()
    }
}

#[derive(Debug)]
pub struct TransformDependency {
    pub buffer: usize,
    pub grid: usize,
    pub order_only: bool,
}

impl TransformDependency {
    fn new(buffer: usize, grid: usize) -> Self {
        Self {
            buffer,
            grid,
            order_only: false,
        }
    }

    fn new_order_only(buffer: usize, grid: usize) -> Self {
        Self {
            buffer,
            grid,
            order_only: true,
        }
    }
}

impl TransformStepChunk {
    // List of input buffers for this transform.
    // We use a stack-size-9 SmallVec because upsampling squeezes touch 9 buffers total (and that is the maximum for
    // non-delta-palette transforms)
    pub fn dependencies(
        &self,
        buffers: &[ModularBufferInfo],
        frame_header: &FrameHeader,
    ) -> SmallVec<TransformDependency, 9> {
        match &self.step {
            TransformStep::Rct { buf_in, .. } => {
                let b = buf_in[0];
                let grid_idx = buffers[b].get_grid_idx(buffers[b].grid_kind, self.grid_pos);
                buf_in
                    .iter()
                    .map(|x| TransformDependency::new(*x, grid_idx))
                    .collect()
            }
            TransformStep::Output { buf_in, .. } => {
                let b = *buf_in;
                let grid_idx = buffers[b].get_grid_idx(buffers[b].grid_kind, self.grid_pos);
                std::iter::once(TransformDependency::new(b, grid_idx)).collect()
            }
            TransformStep::Palette {
                buf_in,
                buf_out,
                buf_pal,
                predictor,
                ..
            } if !predictor.requires_full_row() => {
                let b = *buf_in;
                let grid_idx = buffers[b].get_grid_idx(buffers[b].grid_kind, self.grid_pos);
                let mut ans: SmallVec<TransformDependency, 9> = SmallVec::new();
                ans.push(TransformDependency::new(b, grid_idx));
                ans.push(TransformDependency::new(*buf_pal, 0));
                let grid_shape = buffers[b].grid_shape;
                if *predictor != Predictor::Zero {
                    let (gx, gy) = (self.grid_pos.0 as isize, self.grid_pos.1 as isize);
                    let (xs, ys) = (grid_shape.0 as isize, grid_shape.1 as isize);
                    for (dx, dy) in [(0, -1), (-1, 0), (-1, -1)] {
                        let (nx, ny) = (gx + dx, gy + dy);
                        if nx >= 0 && nx < xs && ny >= 0 && ny < ys {
                            let prev_grid = ny as usize * grid_shape.0 + nx as usize;
                            for out in buf_out {
                                ans.push(TransformDependency::new_order_only(*out, prev_grid));
                            }
                        }
                    }
                }
                ans
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                ..
            } => {
                let b = *buf_in;
                let mut ans = SmallVec::new();
                let grid_shape = buffers[b].grid_shape;
                let grid_idx = buffers[b].get_grid_idx(buffers[b].grid_kind, self.grid_pos);
                ans.push(TransformDependency::new(*buf_pal, 0));
                for grid_x in 0..grid_shape.0 {
                    ans.push(TransformDependency::new(b, grid_idx + grid_x));
                }
                if let Some(prev) = self.grid_pos.1.checked_sub(1) {
                    let prev_grid = prev * grid_shape.0;
                    for out in buf_out {
                        for grid_x in 0..grid_shape.0 {
                            ans.push(TransformDependency::new_order_only(
                                *out,
                                prev_grid + grid_x,
                            ));
                        }
                    }
                }

                ans
            }
            TransformStep::VSqueeze {
                buf_in,
                buf_out,
                upsample,
            }
            | TransformStep::HSqueeze {
                buf_in,
                buf_out,
                upsample,
            } => {
                let info = SqueezeInfo::new(
                    buffers,
                    *buf_in,
                    *buf_out,
                    *upsample,
                    self.grid_pos,
                    frame_header,
                    matches!(self.step, TransformStep::VSqueeze { .. }),
                );
                let mut ans = SmallVec::new();
                match info {
                    SqueezeInfo::Regular {
                        in_avg,
                        in_res,
                        in_next_avg,
                        out_prev,
                        ..
                    } => {
                        ans.push(TransformDependency::new(in_avg.0, in_avg.1));
                        ans.push(TransformDependency::new(in_res.0, in_res.1));
                        if let Some(na) = in_next_avg {
                            ans.push(TransformDependency::new_order_only(na.0, na.1));
                        }
                        if let Some(op) = out_prev {
                            ans.push(TransformDependency::new_order_only(op.0, op.1));
                        }
                    }
                    SqueezeInfo::Upsample { upsample: u, .. } => {
                        let (xs, ys) = buffers[u.source_buf].grid_shape;
                        let (gx, gy) = (u.source_grid % xs, u.source_grid / xs);
                        ans.push(TransformDependency::new(u.source_buf, u.source_grid));
                        for dy in -1..=1 {
                            let ny = gy as isize + dy;
                            if ny < 0 || ny >= ys as isize {
                                continue;
                            }
                            for dx in -1..=1 {
                                let nx = gx as isize + dx;
                                if nx < 0 || nx >= xs as isize || (dx == 0 && dy == 0) {
                                    continue;
                                }
                                ans.push(TransformDependency::new_order_only(
                                    u.source_buf,
                                    ny as usize * xs + nx as usize,
                                ));
                            }
                        }
                    }
                }
                ans
            }
        }
    }
}
