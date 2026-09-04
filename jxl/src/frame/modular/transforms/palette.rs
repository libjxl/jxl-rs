// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::frame::modular::predict::{PredictionData, WeightedPredictorState};
use crate::frame::modular::{ModularChannel, Predictor};
use crate::headers::bit_depth::BitDepth;
use crate::headers::modular::WeightedHeader;
use crate::image::{Image, ImageRect, ImageRectMut};
use crate::util::sync::{OnceLock, RwLockWriteGuard};

const RGB_CHANNELS: usize = 3;

// 5x5x5 color cube for the larger cube.
const LARGE_CUBE: usize = 5;
const LARGE_CUBE_ENTRIES: usize = LARGE_CUBE * LARGE_CUBE * LARGE_CUBE;

// Smaller interleaved color cube to fill the holes of the larger cube.
const SMALL_CUBE: usize = 4;
const SMALL_CUBE_BITS: usize = 2;
// SMALL_CUBE ** 3
const LARGE_CUBE_OFFSET: usize = SMALL_CUBE * SMALL_CUBE * SMALL_CUBE;

const DELTA_PALETTE_ENTRIES: usize = 143;

#[derive(Debug)]
struct ImplicitPalette {
    deltas: [[i32; DELTA_PALETTE_ENTRIES]; RGB_CHANNELS],
    small_cube: [[i32; LARGE_CUBE_OFFSET]; RGB_CHANNELS],
    large_cube: [[i32; LARGE_CUBE_ENTRIES]; RGB_CHANNELS],
}

impl ImplicitPalette {
    fn new(bit_depth: usize) -> Self {
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
        const MULTIPLIER: [i32; 2] = [-1, 1];

        let mut deltas = [[0; DELTA_PALETTE_ENTRIES]; RGB_CHANNELS];
        for (c, deltas_c) in deltas.iter_mut().enumerate() {
            for (idx, slot) in deltas_c.iter_mut().enumerate() {
                let mut result = DELTA_PALETTE[(idx + 1) >> 1][c] * MULTIPLIER[idx & 1];
                if bit_depth > 8 {
                    result *= 1 << (bit_depth - 8);
                }
                *slot = result;
            }
        }

        let scale = |value: usize, bit_depth: usize| ((value * ((1 << bit_depth) - 1)) / 4) as i32;

        let mut small_cube = [[0; LARGE_CUBE_OFFSET]; RGB_CHANNELS];
        for (c, cube_c) in small_cube.iter_mut().enumerate() {
            for (idx, slot) in cube_c.iter_mut().enumerate() {
                let shifted = idx >> (c * SMALL_CUBE_BITS);
                *slot =
                    scale(shifted % SMALL_CUBE, bit_depth) + (1 << (0.max(bit_depth as isize - 3)));
            }
        }

        let mut large_cube = [[0; LARGE_CUBE_ENTRIES]; RGB_CHANNELS];
        for (c, cube_c) in large_cube.iter_mut().enumerate() {
            for (idx, slot) in cube_c.iter_mut().enumerate() {
                let val = match c {
                    0 => idx,
                    1 => idx / LARGE_CUBE,
                    2 => idx / (LARGE_CUBE * LARGE_CUBE),
                    _ => unreachable!(),
                };
                *slot = scale(val % LARGE_CUBE, bit_depth);
            }
        }

        ImplicitPalette {
            deltas,
            small_cube,
            large_cube,
        }
    }
}

struct Palette<'a> {
    implicit: &'static ImplicitPalette,
    explicit: &'a [i32],
    c: usize,
}

impl<'a> Palette<'a> {
    fn new(bit_depth: &BitDepth, c: usize, buf: &'a ModularChannel) -> Self {
        static IMPLICIT_PALETTES: [OnceLock<ImplicitPalette>; 25] = [const { OnceLock::new() }; 25];
        let bit_depth = bit_depth.bits_per_sample().min(24) as usize;
        let pal_rect = ImageRect::<i32>::from_raw(buf.data.as_rect());
        Self {
            implicit: IMPLICIT_PALETTES[bit_depth].get_or_init(|| ImplicitPalette::new(bit_depth)),
            explicit: if pal_rect.size().0 > 0 {
                pal_rect.row(c)
            } else {
                &[]
            },
            c,
        }
    }

    #[inline(always)]
    fn get(&self, index: isize) -> i32 {
        let Self {
            implicit,
            explicit,
            c,
        } = self;
        let uindex = index as usize;
        if index >= 0 && uindex < explicit.len() {
            explicit[uindex]
        } else if *c >= RGB_CHANNELS {
            0
        } else if index < 0 {
            let idx = (-(index + 1) as usize) % DELTA_PALETTE_ENTRIES;
            implicit.deltas[*c][idx]
        } else {
            let cube_idx = uindex - self.explicit.len();
            if cube_idx < LARGE_CUBE_OFFSET {
                implicit.small_cube[*c][cube_idx]
            } else {
                implicit.large_cube[*c][(cube_idx - LARGE_CUBE_OFFSET) % LARGE_CUBE_ENTRIES]
            }
        }
    }
}

pub(super) struct PaletteStep<'a, 'b> {
    pub buf_in: &'a [&'b ModularChannel],
    pub buf_pal: &'b ModularChannel,
    pub buf_out: &'a mut [&'b mut ModularChannel],
    pub num_deltas: usize,
    pub predictor: Predictor,
    pub wp_header: &'a WeightedHeader,
    pub grid_xsize: usize,
    pub buf_left: Option<&'a [ImageRect<'b, i32>]>,
    pub buf_top: Option<&'a [ImageRect<'b, i32>]>,
    pub buf_topleft: Option<&'a [ImageRect<'b, i32>]>,
    pub prev_aux: Option<&'a [Option<&'b Image<i32>>]>,
    pub aux_out: &'a mut [RwLockWriteGuard<'b, Option<Image<i32>>>],
}

impl<'a, 'b> PaletteStep<'a, 'b> {
    pub fn run(self, scratch: &mut [Vec<i32>; 3]) -> Result<()> {
        let PaletteStep {
            buf_in,
            buf_pal,
            buf_out,
            num_deltas,
            predictor,
            wp_header,
            grid_xsize,
            buf_left,
            buf_top,
            buf_topleft,
            prev_aux,
            aux_out,
        } = self;
        let (w0, h) = buf_in[0].size();
        if w0 == 0 || h == 0 {
            return Ok(());
        }

        let num_c = buf_out.len() / grid_xsize;

        if predictor == Predictor::Zero {
            assert_eq!(grid_xsize, 1);
            assert_eq!(buf_in.len(), 1);
            let in_rect = ImageRect::<i32>::from_raw(buf_in[0].data.as_rect());
            for (c, out_buf) in buf_out.iter_mut().enumerate() {
                let palette = Palette::new(&out_buf.bit_depth, c, buf_pal);
                let mut out_rect = ImageRectMut::<i32>::from_raw(out_buf.data.as_rect_mut());
                for y in 0..h {
                    let index_row = in_rect.row(y);
                    let out_row = out_rect.row(y);
                    for (out, &index) in out_row.iter_mut().zip(index_row.iter()) {
                        *out = palette.get(index as isize);
                    }
                }
            }
            return Ok(());
        }

        let total_w: usize = buf_out[..grid_xsize].iter().map(|b| b.size().0).sum();
        let left_offset = if buf_left.is_some() { 2 } else { 0 };
        let row_len = total_w + left_offset;
        for s in scratch.iter_mut() {
            s.resize(row_len, 0);
        }

        for c in 0..num_c {
            let out_row_idx = c * grid_xsize;
            let palette = Palette::new(&buf_out[out_row_idx].bit_depth, c, buf_pal);
            let mut wp_state = if predictor == Predictor::Weighted {
                let mut state = WeightedPredictorState::new(wp_header, total_w);
                if let Some(Some(aux_img)) = prev_aux.and_then(|aux| aux.get(c)) {
                    state.restore_state(aux_img);
                }
                Some(state)
            } else {
                None
            };

            if let Some(prev) = buf_top {
                let mut x_offset = 0;
                for grid_x in 0..grid_xsize {
                    let prev_img = &prev[out_row_idx + grid_x];
                    let w = prev_img.size().0;
                    scratch[1][left_offset + x_offset..left_offset + x_offset + w]
                        .copy_from_slice(prev_img.row(3));
                    scratch[2][left_offset + x_offset..left_offset + x_offset + w]
                        .copy_from_slice(prev_img.row(2));
                    x_offset += w;
                }
                if let Some(left_border) = buf_left {
                    let tl = if let Some(tl_border) = buf_topleft {
                        *tl_border[c].row(3).last().unwrap()
                    } else {
                        left_border[c].row(0)[3]
                    };
                    scratch[1][1] = tl;
                    scratch[1][0] = tl;
                }
            }

            for y in 0..h {
                // y+2 is not the correct y value, but it suffices for things to work.
                let effective_y = if buf_top.is_some() { y + 2 } else { y };

                if let Some(left_border) = buf_left {
                    let left_img = &left_border[c];
                    scratch[0][1] = left_img.row(y)[3];
                    scratch[0][0] = left_img.row(y)[2];
                }

                let [row_cur, row_top, row_toptop] = scratch;

                let mut gx = 0;
                for (grid_x, index_buf) in buf_in.iter().enumerate().take(grid_xsize) {
                    let in_rect = ImageRect::<i32>::from_raw(index_buf.data.as_rect());
                    let index_img = in_rect.row(y);
                    let out_idx = out_row_idx + grid_x;
                    let mut out_rect =
                        ImageRectMut::<i32>::from_raw(buf_out[out_idx].data.as_rect_mut());
                    let out_row = out_rect.row(y);
                    for (x, &index) in index_img.iter().enumerate() {
                        let palette_entry = palette.get(index as isize);
                        let x_scratch = left_offset + gx;
                        let prediction_data = PredictionData::get_rows(
                            row_cur,
                            row_top,
                            row_toptop,
                            x_scratch,
                            effective_y,
                        );
                        let val = if let Some(wp) = &mut wp_state {
                            let (pred, _) = wp.predict_and_property((gx, y & 1), &prediction_data);
                            let val = if index < num_deltas as i32 {
                                (pred + palette_entry as i64) as i32
                            } else {
                                palette_entry
                            };
                            wp.update_errors(val, (gx, y & 1));
                            val
                        } else if index < num_deltas as i32 {
                            let pred = predictor.predict_one(prediction_data, /*wp_pred=*/ 0);
                            (pred + palette_entry as i64) as i32
                        } else {
                            palette_entry
                        };
                        out_row[x] = val;
                        row_cur[x_scratch] = val;
                        gx += 1;
                    }
                }

                scratch.rotate_right(1);
            }

            if let (Some(wp), Some(aux)) = (wp_state, aux_out.get_mut(c)) {
                let mut wp_image = Image::<i32>::new((total_w + 1, 5))?;
                wp.save_state(&mut wp_image);
                **aux = Some(wp_image);
            }
        }

        Ok(())
    }
}

pub fn zero_palette_step_one_group(buf_pal: &ModularChannel, buf_out: &mut [&mut ModularChannel]) {
    let (_w, h) = buf_out[0].size();
    for (c, out) in buf_out.iter_mut().enumerate() {
        let palette = Palette::new(&out.bit_depth, c, buf_pal);
        let palette_entry = palette.get(0);
        let mut out_rect = ImageRectMut::<i32>::from_raw(out.data.as_rect_mut());
        for y in 0..h {
            out_rect.row(y).fill(palette_entry);
        }
    }
}
