// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{cell::Ref, fmt::Debug};

use crate::{
    error::Result,
    frame::modular::{borrowed_buffers::with_buffers, ModularBufferInfo, Predictor},
};

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
    },
    HSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
    },
    VSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
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
    // Number of inputs that are not yet available.
    pub(super) incomplete_deps: usize,
}

impl TransformStepChunk {
    // Marks that one dependency of this transform is ready, and potentially runs the transform,
    // returning the new buffers that are now ready.
    pub fn dep_ready(&mut self, buffers: &mut [ModularBufferInfo]) -> Result<Vec<(usize, usize)>> {
        self.incomplete_deps = self.incomplete_deps.checked_sub(1).unwrap();
        if self.incomplete_deps > 0 {
            return Ok(vec![]);
        }
        let buf_out: &[usize] = match &self.step {
            TransformStep::Rct { buf_out, .. } => buf_out,
            TransformStep::Palette { buf_out, .. } => buf_out,
            TransformStep::HSqueeze { buf_out, .. } | TransformStep::VSqueeze { buf_out, .. } => {
                &[*buf_out]
            }
        };

        let grid = buffers[buf_out[0]].get_grid_idx(self.grid_pos);
        for bo in buf_out {
            assert_eq!(buffers[buf_out[0]].grid_kind, buffers[*bo].grid_kind);
            assert_eq!(buffers[buf_out[0]].info.size, buffers[*bo].info.size);
        }

        match &self.step {
            TransformStep::Rct {
                buf_in,
                buf_out,
                op,
                perm,
            } => {
                for i in 0..3 {
                    assert_eq!(buffers[buf_out[0]].grid_kind, buffers[buf_in[i]].grid_kind);
                    assert_eq!(buffers[buf_out[0]].info.size, buffers[buf_in[i]].info.size);
                    // Optimistically move the buffers to the output if possible.
                    // If not, creates buffers in the output that are a copy of the input buffers.
                    // This should be rare.
                    *buffers[buf_out[i]].buffer_grid[grid].data.borrow_mut() =
                        Some(buffers[buf_in[i]].buffer_grid[grid].get_buffer()?);
                }
                with_buffers(buffers, buf_out, grid, |bufs| {
                    super::rct::do_rct_step(bufs, *op, *perm);
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
            } if *predictor != Predictor::Weighted && *predictor != Predictor::AverageAll => {
                assert_eq!(buffers[buf_out[0]].grid_kind, buffers[*buf_in].grid_kind);
                assert_eq!(buffers[buf_out[0]].info.size, buffers[*buf_in].info.size);

                {
                    let img_in = Ref::map(buffers[*buf_in].buffer_grid[grid].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    let img_pal = Ref::map(buffers[*buf_pal].buffer_grid[0].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    with_buffers(buffers, buf_out, grid, |bufs| {
                        super::palette::do_palette_step_general(
                            &img_in,
                            &img_pal,
                            bufs,
                            *num_colors,
                            *num_deltas,
                            *predictor,
                        );
                        Ok(())
                    })?;
                }
                buffers[*buf_in].buffer_grid[grid].mark_used();
                buffers[*buf_pal].buffer_grid[0].mark_used();
            }
            _ => {
                todo!()
            }
        };

        Ok(buf_out.iter().map(|x| (*x, grid)).collect())
    }
}
