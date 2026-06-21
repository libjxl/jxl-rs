// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use crate::{
    error::Result,
    frame::modular::{
        BUFFER_STATUS_ZERO_FILLED, ModularBufferInfo, ModularGridKind, Predictor,
        TransformScratchSpace,
        buffers::{ModularChannel, with_buffers},
        transforms::squeeze::{smooth_2d_unsqueeze, smooth_h_unsqueeze, smooth_v_unsqueeze},
    },
    headers::{frame_header::FrameHeader, modular::WeightedHeader},
    util::{AtomicRef, AtomicRefMut, tracing_wrappers::*},
};
use std::ops::{Deref, DerefMut};

use super::{RctOp, RctPermutation};

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
        // If buf_in[0] was obtained via VSqueeze, the two source buffers
        // for that transform.
        buf_in_avg: Option<[usize; 2]>,
    },
    VSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
        // If buf_in[0] was obtained via HSqueeze, the two source buffers
        // for that transform.
        buf_in_avg: Option<[usize; 2]>,
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

    // List of (buffer, grid) that this transform depends on.
    pub(in super::super) deps: Vec<(usize, usize)>,

    // Processing layer that this transform belongs to. Layer 0 are transforms
    // that only depend on coded channels, layer 1 are transforms that only
    // depend on coded channels and layer 0 outputs, etc. Since transforms
    // in the same layer have no inter-dependencies, they can be run at the
    // same time.
    pub(in super::super) layer: usize,
}

impl TransformStepChunk {
    fn buf_out(&self) -> &[usize] {
        match &self.step {
            TransformStep::Rct { buf_out, .. } => buf_out,
            TransformStep::Palette { buf_out, .. } => buf_out,
            TransformStep::HSqueeze { buf_out, .. } | TransformStep::VSqueeze { buf_out, .. } => {
                std::slice::from_ref(buf_out)
            }
        }
    }

    // Runs this transform. This function *will* crash if the transform is not ready.
    #[instrument(level = "trace", skip_all)]
    pub fn do_run(
        &self,
        frame_header: &FrameHeader,
        buffers: &[ModularBufferInfo],
        is_final: bool,
        tranform_scratch_space: &mut TransformScratchSpace,
    ) -> Result<()> {
        let buf_out = self.buf_out();
        let out_grid_kind = buffers[buf_out[0]].grid_kind;
        let out_grid = buffers[buf_out[0]].get_grid_idx(out_grid_kind, self.grid_pos);
        let out_size = buffers[buf_out[0]].info.size;
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
                    *buffers[buf_out[i]].buffer_grid[out_grid].data.borrow_mut() =
                        Some(buffers[buf_in[i]].buffer_grid[out_grid].get_buffer(is_final)?);
                }
                with_buffers(buffers, buf_out, out_grid, |mut bufs| {
                    super::rct::do_rct_step(&mut bufs, *op, *perm);
                    Ok(())
                })?;
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                ..
            } if buffers[*buf_in].info.size.0 == 0 => {
                // Nothing to do, just bookkeeping.
                buffers[*buf_in].buffer_grid[out_grid].mark_used(is_final);
                buffers[*buf_pal].buffer_grid[0].mark_used(is_final);
                with_buffers(buffers, buf_out, out_grid, |_| Ok(()))?;
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

                {
                    let img_in =
                        AtomicRef::map(buffers[*buf_in].buffer_grid[out_grid].data.borrow(), |x| {
                            x.as_ref().unwrap()
                        });
                    let img_pal =
                        AtomicRef::map(buffers[*buf_pal].buffer_grid[0].data.borrow(), |x| {
                            x.as_ref().unwrap()
                        });
                    // Ensure that the output buffers are present.
                    // TODO(szabadka): Extend the callback to support many grid points.
                    with_buffers(buffers, buf_out, out_grid, |_| Ok(()))?;
                    let grid_shape = buffers[buf_out[0]].grid_shape;
                    let grid_x = out_grid % grid_shape.0;
                    let grid_y = out_grid / grid_shape.0;
                    let border = if *predictor == Predictor::Zero { 0 } else { 1 };
                    let grid_x0 = grid_x.saturating_sub(border);
                    let grid_y0 = grid_y.saturating_sub(border);
                    let grid_x1 = grid_x + 1;
                    let grid_y1 = grid_y + 1;
                    let mut out_bufs = vec![];
                    for i in buf_out {
                        for gy in grid_y0..grid_y1 {
                            for gx in grid_x0..grid_x1 {
                                let grid = gy * grid_shape.0 + gx;
                                let buf = &buffers[*i];
                                let b = &buf.buffer_grid[grid];
                                let data = b.data.borrow_mut();
                                out_bufs.push(AtomicRefMut::map(data, |x| x.as_mut().unwrap()));
                            }
                        }
                    }
                    let mut out_buf_refs: Vec<&mut ModularChannel> =
                        out_bufs.iter_mut().map(|x| x.deref_mut()).collect();
                    super::palette::do_palette_step_one_group(
                        &img_in,
                        &img_pal,
                        &mut out_buf_refs,
                        grid_x - grid_x0,
                        grid_y - grid_y0,
                        grid_x1 - grid_x0,
                        grid_y1 - grid_y0,
                        *num_colors,
                        *num_deltas,
                        *predictor,
                    );
                }
                buffers[*buf_in].buffer_grid[out_grid].mark_used(is_final);
                buffers[*buf_pal].buffer_grid[0].mark_used(is_final);
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
                {
                    assert_eq!(out_grid % grid_shape.0, 0);
                    let grid_y = out_grid / grid_shape.0;
                    let grid_y0 = grid_y.saturating_sub(1);
                    let grid_y1 = grid_y + 1;
                    let mut in_bufs = vec![];
                    for grid_x in 0..grid_shape.0 {
                        let grid = grid_y * grid_shape.0 + grid_x;
                        in_bufs.push(AtomicRef::map(
                            buffers[*buf_in].buffer_grid[grid].data.borrow(),
                            |x| x.as_ref().unwrap(),
                        ));
                        // Ensure that the output buffers are present.
                        // TODO(szabadka): Extend the callback to support many grid points.
                        with_buffers(buffers, buf_out, out_grid + grid_x, |_| Ok(()))?;
                    }
                    let in_buf_refs: Vec<&ModularChannel> =
                        in_bufs.iter().map(|x| x.deref()).collect();
                    let img_pal =
                        AtomicRef::map(buffers[*buf_pal].buffer_grid[0].data.borrow(), |x| {
                            x.as_ref().unwrap()
                        });
                    let mut out_bufs = vec![];
                    for i in buf_out {
                        for grid_y in grid_y0..grid_y1 {
                            for grid_x in 0..grid_shape.0 {
                                let grid = grid_y * grid_shape.0 + grid_x;
                                let buf = &buffers[*i];
                                let b = &buf.buffer_grid[grid];
                                let data = b.data.borrow_mut();
                                out_bufs.push(AtomicRefMut::map(data, |x| x.as_mut().unwrap()));
                            }
                        }
                    }
                    let mut out_buf_refs: Vec<&mut ModularChannel> =
                        out_bufs.iter_mut().map(|x| x.deref_mut()).collect();
                    super::palette::do_palette_step_group_row(
                        &in_buf_refs,
                        &img_pal,
                        &mut out_buf_refs,
                        grid_y - grid_y0,
                        grid_shape.0,
                        *num_colors,
                        *num_deltas,
                        *predictor,
                        wp_header,
                    )?;
                }
                buffers[*buf_pal].buffer_grid[0].mark_used(is_final);
                for grid_x in 0..grid_shape.0 {
                    buffers[*buf_in].buffer_grid[out_grid + grid_x].mark_used(is_final);
                }
            }
            TransformStep::HSqueeze {
                buf_in,
                buf_out,
                buf_in_avg,
            } => {
                let buf_avg = &buffers[buf_in[0]];
                let buf_res = &buffers[buf_in[1]];
                let in_grid = buf_avg.get_grid_idx(out_grid_kind, self.grid_pos);
                let res_grid = buf_res.get_grid_idx(out_grid_kind, self.grid_pos);
                let zero_res =
                    buf_res.buffer_grid[res_grid].get_status() == BUFFER_STATUS_ZERO_FILLED;
                let double_zero_res = zero_res
                    && buf_in_avg.is_some_and(|x| {
                        let avg2_res_grid =
                            buffers[x[1]].get_grid_idx(out_grid_kind, self.grid_pos);
                        buffers[x[1]].buffer_grid[avg2_res_grid].get_status()
                            == BUFFER_STATUS_ZERO_FILLED
                    });
                {
                    trace!(
                        "HSqueeze {:?} -> {:?}, grid {out_grid} grid pos {:?}",
                        buf_in, buf_out, self.grid_pos
                    );
                    let (gx, gy) = self.grid_pos;
                    let mut out_rect =
                        buffers[*buf_out].get_grid_rect(frame_header, out_grid_kind, (gx, gy));
                    out_rect.origin = if out_grid_kind == ModularGridKind::None {
                        (0, 0)
                    } else {
                        let out_shift = buffers[*buf_out].info.shift.unwrap_or((0, 0));
                        let out_grid_dim = out_grid_kind.grid_dim(frame_header, out_shift);
                        (gx * out_grid_dim.0, gy * out_grid_dim.1)
                    };
                    let in_avg = AtomicRef::map(buf_avg.buffer_grid[in_grid].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    let has_next = gx + 1 < buffers[*buf_out].grid_shape.0;
                    let gx_next = if has_next { gx + 1 } else { gx };
                    let next_avg_grid = buf_avg.get_grid_idx(out_grid_kind, (gx_next, gy));
                    let in_next_avg =
                        AtomicRef::map(buf_avg.buffer_grid[next_avg_grid].data.borrow(), |x| {
                            x.as_ref().unwrap()
                        });
                    let in_next_avg_rect = if has_next {
                        Some(in_next_avg.data.get_rect(buf_avg.get_grid_rect(
                            frame_header,
                            out_grid_kind,
                            (gx_next, gy),
                        )))
                    } else {
                        None
                    };
                    let in_res = AtomicRef::map(buf_res.buffer_grid[res_grid].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    let out_prev = if gx == 0 {
                        None
                    } else {
                        let prev_out_grid =
                            buffers[*buf_out].get_grid_idx(out_grid_kind, (gx - 1, gy));
                        Some(AtomicRef::map(
                            buffers[*buf_out].buffer_grid[prev_out_grid].data.borrow(),
                            |x| x.as_ref().unwrap(),
                        ))
                    };

                    with_buffers(buffers, &[*buf_out], out_grid, |mut bufs| {
                        if bufs.is_empty() {
                            return Ok(());
                        }
                        if double_zero_res {
                            assert_eq!(bufs.len(), 1);
                            smooth_2d_unsqueeze(
                                &buffers[buf_in_avg.unwrap()[0]],
                                frame_header,
                                out_rect,
                                &mut bufs[0].data,
                                &mut tranform_scratch_space.smooth_unsqueeze_buffer,
                            );
                            return Ok(());
                        }
                        if zero_res {
                            assert_eq!(bufs.len(), 1);
                            smooth_h_unsqueeze(
                                buf_avg,
                                frame_header,
                                out_rect,
                                &mut bufs[0].data,
                                &mut tranform_scratch_space.smooth_unsqueeze_buffer,
                            );
                            return Ok(());
                        }
                        super::squeeze::do_hsqueeze_step(
                            &in_avg.data.get_rect(buf_avg.get_grid_rect(
                                frame_header,
                                out_grid_kind,
                                (gx, gy),
                            )),
                            &in_res.data.get_rect(buf_res.get_grid_rect(
                                frame_header,
                                out_grid_kind,
                                (gx, gy),
                            )),
                            &in_next_avg_rect,
                            &out_prev,
                            &mut bufs,
                        );
                        Ok(())
                    })?;
                }
                buffers[buf_in[0]].buffer_grid[in_grid].mark_used(is_final);
                buffers[buf_in[1]].buffer_grid[res_grid].mark_used(is_final);
            }
            TransformStep::VSqueeze {
                buf_in,
                buf_out,
                buf_in_avg,
            } => {
                let buf_avg = &buffers[buf_in[0]];
                let buf_res = &buffers[buf_in[1]];
                let in_grid = buf_avg.get_grid_idx(out_grid_kind, self.grid_pos);
                let res_grid = buf_res.get_grid_idx(out_grid_kind, self.grid_pos);
                let zero_res =
                    buf_res.buffer_grid[res_grid].get_status() == BUFFER_STATUS_ZERO_FILLED;
                let double_zero_res = zero_res
                    && buf_in_avg.is_some_and(|x| {
                        let avg2_res_grid =
                            buffers[x[1]].get_grid_idx(out_grid_kind, self.grid_pos);
                        buffers[x[1]].buffer_grid[avg2_res_grid].get_status()
                            == BUFFER_STATUS_ZERO_FILLED
                    });
                {
                    trace!(
                        "VSqueeze {:?} -> {:?} grid: {out_grid:?} grid pos: {:?}",
                        buf_in, buf_out, self.grid_pos
                    );
                    let (gx, gy) = self.grid_pos;
                    let mut out_rect =
                        buffers[*buf_out].get_grid_rect(frame_header, out_grid_kind, (gx, gy));
                    out_rect.origin = if out_grid_kind == ModularGridKind::None {
                        (0, 0)
                    } else {
                        let out_shift = buffers[*buf_out].info.shift.unwrap_or((0, 0));
                        let out_grid_dim = out_grid_kind.grid_dim(frame_header, out_shift);
                        (gx * out_grid_dim.0, gy * out_grid_dim.1)
                    };
                    let in_avg = AtomicRef::map(buf_avg.buffer_grid[in_grid].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    let has_next = gy + 1 < buffers[*buf_out].grid_shape.1;
                    let gy_next = if has_next { gy + 1 } else { gy };
                    let next_avg_grid = buf_avg.get_grid_idx(out_grid_kind, (gx, gy_next));
                    let in_next_avg =
                        AtomicRef::map(buf_avg.buffer_grid[next_avg_grid].data.borrow(), |x| {
                            x.as_ref().unwrap()
                        });
                    let in_next_avg_rect = if has_next {
                        Some(in_next_avg.data.get_rect(buf_avg.get_grid_rect(
                            frame_header,
                            out_grid_kind,
                            (gx, gy_next),
                        )))
                    } else {
                        None
                    };
                    let in_res = AtomicRef::map(buf_res.buffer_grid[res_grid].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    let out_prev = if gy == 0 {
                        None
                    } else {
                        let prev_out_grid =
                            buffers[*buf_out].get_grid_idx(out_grid_kind, (gx, gy - 1));
                        Some(AtomicRef::map(
                            buffers[*buf_out].buffer_grid[prev_out_grid].data.borrow(),
                            |x| x.as_ref().unwrap(),
                        ))
                    };
                    let avg_grid_rect =
                        buf_avg.get_grid_rect(frame_header, out_grid_kind, (gx, gy));
                    let res_grid_rect =
                        buf_res.get_grid_rect(frame_header, out_grid_kind, (gx, gy));

                    with_buffers(buffers, &[*buf_out], out_grid, |mut bufs| {
                        if bufs.is_empty() {
                            return Ok(());
                        }
                        if double_zero_res {
                            assert_eq!(bufs.len(), 1);
                            smooth_2d_unsqueeze(
                                &buffers[buf_in_avg.unwrap()[0]],
                                frame_header,
                                out_rect,
                                &mut bufs[0].data,
                                &mut tranform_scratch_space.smooth_unsqueeze_buffer,
                            );
                            return Ok(());
                        }
                        if zero_res {
                            assert_eq!(bufs.len(), 1);
                            smooth_v_unsqueeze(
                                buf_avg,
                                frame_header,
                                out_rect,
                                &mut bufs[0].data,
                                &mut tranform_scratch_space.smooth_unsqueeze_buffer,
                            );
                            return Ok(());
                        }
                        super::squeeze::do_vsqueeze_step(
                            &in_avg.data.get_rect(avg_grid_rect),
                            &in_res.data.get_rect(res_grid_rect),
                            &in_next_avg_rect,
                            &out_prev,
                            &mut bufs,
                        );
                        Ok(())
                    })?;
                }
                buffers[buf_in[0]].buffer_grid[in_grid].mark_used(is_final);
                buffers[buf_in[1]].buffer_grid[res_grid].mark_used(is_final);
            }
        };

        Ok(())
    }

    // Iterates over the list of outputs for this transform.
    pub fn outputs(&self, buffers: &[ModularBufferInfo]) -> impl Iterator<Item = (usize, usize)> {
        let buf_out = self.buf_out();
        let out_grid_kind = buffers[buf_out[0]].grid_kind;
        let out_grid = buffers[buf_out[0]].get_grid_idx(out_grid_kind, self.grid_pos);
        let grid_offset_up = match &self.step {
            TransformStep::Palette {
                buf_in,
                buf_out,
                predictor,
                ..
            } if buffers[*buf_in].info.size.0 != 0 && predictor.requires_full_row() => {
                buffers[buf_out[0]].grid_shape.0
            }
            _ => 1,
        };

        buf_out
            .iter()
            .flat_map(move |x| (0..grid_offset_up).map(move |y| (*x, out_grid + y)))
    }
}
