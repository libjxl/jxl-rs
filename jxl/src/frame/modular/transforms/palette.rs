// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
use crate::{
    error::Result,
    frame::modular::{ModularBufferInfo, ModularGridKind, Predictor},
    image::{Image, ImageRect},
};

use super::TransformStep;

const MAX_PALETTE_LOOKUP_TABLE_SIZE: usize = 1 << 16;

const RGB_CHANNELS: usize = 3;

// 5x5x5 color cube for the larger cube.
const LARGE_CUBE: usize = 5;

// Smaller interleaved color cube to fill the holes of the larger cube.
const SMALL_CUBE: usize = 4;
const SMALL_CUBE_BITS: usize = 2;
// SMALL_CUBE ** 3
const LARGE_CUBE_OFFSET: usize = SMALL_CUBE * SMALL_CUBE * SMALL_CUBE;
const IMPLICIT_PALETTE_SIZE: usize = LARGE_CUBE_OFFSET + LARGE_CUBE * LARGE_CUBE * LARGE_CUBE;

fn scale<const DENOM: usize>(value: usize, bit_depth: usize) -> i32 {
    // return (value * ((1 << bit_depth) - 1)) / DENOM;
    // We only call this function with SMALL_CUBE or LARGE_CUBE - 1 as DENOM,
    // allowing us to avoid a division here.
    const {
        assert!(DENOM == 4, "denom must be 4");
    }
    ((value * ((1 << bit_depth) - 1)) >> 2) as i32
}

// The purpose of this function is solely to extend the interpretation of
// palette indices to implicit values. If index < nb_deltas, indicating that the
// result is a delta palette entry, it is the responsibility of the caller to
// treat it as such.
fn get_palette_value(
    palette: &ImageRect<i32>,
    index: isize,
    c: usize,
    palette_size: usize,
    bit_depth: usize,
) -> i32 {
    if index < 0 {
        const DELTA_PALETTE: [[i32; 3]; 72] = [
            [0, 0, 0],
            [4, 4, 4],
            [11, 0, 0],
            [0, 0, -13],
            [0, -12, 0],
            [-10, -10, -10],
            [-18, -18, -18],
            [-27, -27, -27],
            [-18, -18, 0],
            [0, 0, -32],
            [-32, 0, 0],
            [-37, -37, -37],
            [0, -32, -32],
            [24, 24, 45],
            [50, 50, 50],
            [-45, -24, -24],
            [-24, -45, -45],
            [0, -24, -24],
            [-34, -34, 0],
            [-24, 0, -24],
            [-45, -45, -24],
            [64, 64, 64],
            [-32, 0, -32],
            [0, -32, 0],
            [-32, 0, 32],
            [-24, -45, -24],
            [45, 24, 45],
            [24, -24, -45],
            [-45, -24, 24],
            [80, 80, 80],
            [64, 0, 0],
            [0, 0, -64],
            [0, -64, -64],
            [-24, -24, 45],
            [96, 96, 96],
            [64, 64, 0],
            [45, -24, -24],
            [34, -34, 0],
            [112, 112, 112],
            [24, -45, -45],
            [45, 45, -24],
            [0, -32, 32],
            [24, -24, 45],
            [0, 96, 96],
            [45, -24, 24],
            [24, -45, -24],
            [-24, -45, 24],
            [0, -64, 0],
            [96, 0, 0],
            [128, 128, 128],
            [64, 0, 64],
            [144, 144, 144],
            [96, 96, 0],
            [-36, -36, 36],
            [45, -24, -45],
            [45, -45, -24],
            [0, 0, -96],
            [0, 128, 128],
            [0, 96, 0],
            [45, 24, -45],
            [-128, 0, 0],
            [24, -45, 24],
            [-45, 24, -45],
            [64, 0, -64],
            [64, -64, -64],
            [96, 0, 96],
            [45, -45, 24],
            [24, 45, -45],
            [64, 64, -64],
            [128, 128, 0],
            [0, 0, -128],
            [-24, 45, -45],
        ];
        if c >= RGB_CHANNELS {
            return 0;
        }
        // Do not open the brackets, otherwise INT32_MIN negation could overflow.
        let mut index = -(index + 1) as usize;
        index %= 1 + 2 * (DELTA_PALETTE.len() - 1);
        const MULTIPLIER: [i32; 2] = [-1, 1];
        let mut result = DELTA_PALETTE[(index + 1) >> 1][c] * MULTIPLIER[index & 1];
        if bit_depth > 8 {
            result *= 1 << (bit_depth - 8);
        }
        result
    } else {
        let mut index = index as usize;
        if palette_size <= index && index < palette_size + LARGE_CUBE_OFFSET {
            if c >= RGB_CHANNELS {
                return 0;
            }
            index -= palette_size;
            index >>= c * SMALL_CUBE_BITS;
            scale::<SMALL_CUBE>(index % SMALL_CUBE, bit_depth)
                + (1 << (0.max(bit_depth as isize - 3)))
        } else if palette_size + LARGE_CUBE_OFFSET <= index {
            if c >= RGB_CHANNELS {
                return 0;
            }
            index -= palette_size + LARGE_CUBE_OFFSET;
            // TODO(eustas): should we take care of ambiguity created by
            //               index >= LARGE_CUBE ** 3 ?
            match c {
                0 => (),
                1 => {
                    index /= LARGE_CUBE;
                }
                2 => {
                    index /= LARGE_CUBE * LARGE_CUBE;
                }
                _ => (),
            }
            scale::<{ LARGE_CUBE - 1 }>(index % LARGE_CUBE, bit_depth)
        } else {
            palette.row(c)[index]
        }
    }
}

pub fn do_palette_step(
    step: TransformStep,
    buffers: &mut [ModularBufferInfo],
    (gx, gy): (usize, usize),
) -> Result<Vec<(usize, usize)>> {
    let TransformStep::Palette {
        buf_in,
        buf_pal,
        buf_out,
        num_colors,
        num_deltas,
        chan_index,
        predictor,
    } = step
    else {
        unreachable!("non-palette step passed to palette transform")
    };
    let grid = buffers[buf_in].grid_shape.0 * gy + gx;
    assert_eq!(buffers[buf_in].grid_kind, buffers[buf_out].grid_kind);
    assert_eq!(buffers[buf_in].info.size, buffers[buf_out].info.size);
    assert!(buffers[buf_out].buffer_grid[grid].data.is_none());

    assert_eq!(buffers[buf_pal].grid_kind, ModularGridKind::None);
    assert_eq!(buffers[buf_pal].grid_shape, (1, 1));

    let buf_size = buffers[buf_in].buffer_grid[grid]
        .data
        .as_ref()
        .unwrap()
        .size();
    buffers[buf_out].buffer_grid[grid].data = Some(Image::new(buf_size)?);

    let (w, h) = buf_size;
    let bit_depth = 8; // TODO(sboukortt): plumb the actual bit depth

    if w == 0 {
        // Nothing to do.
        // Avoid touching "empty" channels with non-zero height.
    } else if num_deltas == 0 && predictor == Predictor::Zero {
        for y in 0..h {
            for x in 0..w {
                let index = buffers[buf_in].buffer_grid[grid]
                    .data
                    .as_ref()
                    .unwrap()
                    .as_rect()
                    .row(y)[x];
                let palette_value = {
                    let palette = buffers[buf_pal].buffer_grid[0]
                        .data
                        .as_ref()
                        .unwrap()
                        .as_rect();
                    get_palette_value(
                        &palette,
                        index as isize,
                        /*c=*/ chan_index,
                        /*palette_size=*/ num_colors,
                        /*bit_depth=*/ bit_depth,
                    )
                };
                buffers[buf_out].buffer_grid[grid]
                    .data
                    .as_mut()
                    .unwrap()
                    .as_rect_mut()
                    .row(y)[x] = palette_value;
            }
        }
    } else {
        todo!();
    }

    buffers[buf_in].buffer_grid[grid].mark_used();

    Ok(vec![(buf_out, grid)])
}
