// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    frame::modular::{
        ModularChannel, Predictor,
        predict::{PredictionData, WeightedPredictorState},
    },
    headers::modular::WeightedHeader,
    image::Image,
};

const RGB_CHANNELS: usize = 3;

// 5x5x5 color cube for the larger cube.
const LARGE_CUBE: usize = 5;

// Smaller interleaved color cube to fill the holes of the larger cube.
const SMALL_CUBE: usize = 4;
const SMALL_CUBE_BITS: usize = 2;
// SMALL_CUBE ** 3
const LARGE_CUBE_OFFSET: usize = SMALL_CUBE * SMALL_CUBE * SMALL_CUBE;

#[inline(always)]
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
/// Look up palette value. `pal_row` is the pre-fetched palette row for channel `c`.
#[inline(always)]
fn get_palette_value_with_row(
    pal_row: &[i32],
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
            pal_row[index]
        }
    }
}

#[inline(always)]
pub fn do_palette_step_general(
    buf_in: &ModularChannel,
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    wp_header: &WeightedHeader,
) {
    let (w, h) = buf_in.data.size();
    let palette = &buf_pal.data;
    let bit_depth = buf_in.bit_depth.bits_per_sample().min(24) as usize;

    if w == 0 {
        // Nothing to do.
        // Avoid touching "empty" channels with non-zero height.
    } else if num_deltas == 0 && predictor == Predictor::Zero {
        for (chan_index, out) in buf_out.iter_mut().enumerate() {
            let pal_row = palette.row(chan_index);
            for y in 0..h {
                let row_index = buf_in.data.row(y);
                let row_out = out.data.row_mut(y);
                #[allow(unsafe_code)]
                for x in 0..w {
                    let index = row_index[x];
                    let idx = index as usize;
                    if idx < num_colors {
                        // SAFETY: idx < num_colors <= pal_row.len()
                        row_out[x] = unsafe { *pal_row.get_unchecked(idx) };
                    } else {
                        row_out[x] = get_palette_value_with_row(
                            pal_row,
                            index as isize,
                            chan_index,
                            num_colors,
                            bit_depth,
                        );
                    }
                }
            }
        }
    } else if predictor == Predictor::Weighted {
        let w = buf_in.data.size().0;
        for (chan_index, out) in buf_out.iter_mut().enumerate() {
            let pal_row = palette.row(chan_index);
            let mut wp_state = WeightedPredictorState::new(wp_header, w);
            for y in 0..h {
                let idx = buf_in.data.row(y);
                for (x, &index) in idx.iter().enumerate() {
                    let palette_entry = get_palette_value_with_row(
                        pal_row,
                        index as isize,
                        /*c=*/ chan_index,
                        /*palette_size=*/ num_colors + num_deltas,
                        /*bit_depth=*/ bit_depth,
                    );
                    let val = if index < num_deltas as i32 {
                        let prediction_data = PredictionData::get(&out.data, x, y);
                        let (wp_pred, _) =
                            wp_state.predict_and_property((x, y), w, &prediction_data);
                        let pred = predictor.predict_one(prediction_data, wp_pred);
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    out.data.row_mut(y)[x] = val;
                    wp_state.update_errors(val, (x, y), w);
                }
            }
        }
    } else {
        for (chan_index, out) in buf_out.iter_mut().enumerate() {
            let pal_row = palette.row(chan_index);
            for y in 0..h {
                let idx = buf_in.data.row(y);
                for (x, &index) in idx.iter().enumerate() {
                    let palette_entry = get_palette_value_with_row(
                        pal_row,
                        index as isize,
                        /*c=*/ chan_index,
                        /*palette_size=*/ num_colors + num_deltas,
                        /*bit_depth=*/ bit_depth,
                    );
                    let val = if index < num_deltas as i32 {
                        let pred = predictor
                            .predict_one(PredictionData::get(&out.data, x, y), /*wp_pred=*/ 0);
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    out.data.row_mut(y)[x] = val;
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_prediction_data(
    buf: &mut [&mut ModularChannel],
    idx: usize,
    grid_x: usize,
    grid_y: usize,
    grid_xsize: usize,
    x: usize,
    y: usize,
    xsize: usize,
    ysize: usize,
) -> PredictionData {
    PredictionData::get_with_neighbors(
        &buf[idx].data,
        if grid_x > 0 {
            Some(&buf[idx - 1].data)
        } else {
            None
        },
        if grid_y > 0 {
            Some(&buf[idx - grid_xsize].data)
        } else {
            None
        },
        if grid_x > 0 && grid_y > 0 {
            Some(&buf[idx - grid_xsize - 1].data)
        } else {
            None
        },
        if grid_x + 1 < grid_xsize {
            Some(&buf[idx + 1].data)
        } else {
            None
        },
        if grid_x + 1 < grid_xsize && grid_y > 0 {
            Some(&buf[idx - grid_xsize + 1].data)
        } else {
            None
        },
        x,
        y,
        xsize,
        ysize,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_one_group(
    buf_in: &ModularChannel,
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    grid_x: usize,
    grid_y: usize,
    grid_xsize: usize,
    grid_ysize: usize,
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
) {
    let h = buf_in.data.size().1;
    let palette = &buf_pal.data;
    let bit_depth = buf_in.bit_depth.bits_per_sample().min(24) as usize;
    let num_c = buf_out.len() / (grid_xsize * grid_ysize);
    let (xsize, ysize) = buf_out[0].data.size();

    let palette_size = num_colors + num_deltas;

    if num_deltas == 0 {
        // Fast path: no delta palette entries, just direct lookups.
        // Avoids prediction data computation entirely.
        for c in 0..num_c {
            let pal_row = palette.row(c);
            let out_idx = c * grid_ysize * grid_xsize + grid_y * grid_xsize + grid_x;
            for y in 0..h {
                let index_img = buf_in.data.row(y);
                let out_row = buf_out[out_idx].data.row_mut(y);
                #[allow(unsafe_code)]
                for (x, &index) in index_img.iter().enumerate() {
                    // Fast path: direct palette lookup for valid indices (common case).
                    // Skip the multi-branch get_palette_value_with_row for the hot path.
                    let idx = index as usize;
                    if idx < palette_size {
                        // SAFETY: idx < palette_size <= pal_row.len() (palette is at least
                        // palette_size wide, validated during palette transform setup).
                        out_row[x] = unsafe { *pal_row.get_unchecked(idx) };
                    } else {
                        // Rare case: implicit color cube or negative index
                        out_row[x] = get_palette_value_with_row(
                            pal_row,
                            index as isize,
                            c,
                            palette_size,
                            bit_depth,
                        );
                    }
                }
            }
        }
    } else {
        for c in 0..num_c {
            let pal_row = palette.row(c);
            let out_idx = c * grid_ysize * grid_xsize + grid_y * grid_xsize + grid_x;
            for y in 0..h {
                let index_img = buf_in.data.row(y);
                for (x, &index) in index_img.iter().enumerate() {
                    let palette_entry = get_palette_value_with_row(
                        pal_row,
                        index as isize,
                        c,
                        palette_size,
                        bit_depth,
                    );
                    let val = if index < num_deltas as i32 {
                        // Fast path for interior pixels: avoid expensive get_with_neighbors
                        // when we can read directly from the same output buffer.
                        let pred = if x > 0 && y > 0 && x + 1 < xsize {
                            predictor.predict_one(
                                PredictionData::get(&buf_out[out_idx].data, x, y),
                                /*wp_pred=*/ 0,
                            )
                        } else {
                            predictor.predict_one(
                                get_prediction_data(
                                    buf_out, out_idx, grid_x, grid_y, grid_xsize, x, y, xsize,
                                    ysize,
                                ),
                                /*wp_pred=*/ 0,
                            )
                        };
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    buf_out[out_idx].data.row_mut(y)[x] = val;
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_group_row(
    buf_in: &[&ModularChannel],
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    grid_y: usize,
    grid_xsize: usize,
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    wp_header: &WeightedHeader,
) -> Result<()> {
    let palette = &buf_pal.data;
    let h = buf_in[0].data.size().1;
    let bit_depth = buf_in[0].bit_depth.bits_per_sample().min(24) as usize;
    let grid_ysize = grid_y + 1;
    let num_c = buf_out.len() / (grid_xsize * grid_ysize);
    let total_w = buf_out[0..grid_xsize]
        .iter()
        .map(|buf| buf.data.size().0)
        .sum();
    let (xsize, ysize) = buf_out[0].data.size();

    let palette_size = num_colors + num_deltas;
    if predictor == Predictor::Weighted {
        for c in 0..num_c {
            let pal_row = palette.row(c);
            let mut wp_state = WeightedPredictorState::new(wp_header, total_w);
            let out_row_idx = c * grid_ysize * grid_xsize + grid_y * grid_xsize;
            if grid_y > 0 {
                let prev_row_idx = out_row_idx - grid_y * grid_xsize;
                wp_state.restore_state(
                    buf_out[prev_row_idx].auxiliary_data.as_ref().unwrap(),
                    total_w,
                );
            }
            for y in 0..h {
                for (grid_x, index_buf) in buf_in.iter().enumerate().take(grid_xsize) {
                    let index_img = index_buf.data.row(y);
                    let out_idx = out_row_idx + grid_x;
                    for (x, &index) in index_img.iter().enumerate() {
                        let palette_entry = get_palette_value_with_row(
                            pal_row,
                            index as isize,
                            c,
                            palette_size,
                            bit_depth,
                        );
                        let val = if index < num_deltas as i32 {
                            // Fast path for interior pixels: avoid expensive get_with_neighbors
                            let prediction_data = if x > 0 && y > 0 && x + 1 < xsize {
                                PredictionData::get(&buf_out[out_idx].data, x, y)
                            } else {
                                get_prediction_data(
                                    buf_out, out_idx, grid_x, grid_y, grid_xsize, x, y, xsize,
                                    ysize,
                                )
                            };
                            let (pred, _) = wp_state.predict_and_property(
                                (grid_x * xsize + x, y & 1),
                                total_w,
                                &prediction_data,
                            );
                            (pred + palette_entry as i64) as i32
                        } else {
                            palette_entry
                        };
                        buf_out[out_idx].data.row_mut(y)[x] = val;
                        wp_state.update_errors(val, (grid_x * xsize + x, y & 1), total_w);
                    }
                }
            }
            let mut wp_image = Image::<i32>::new((total_w + 2, 1))?;
            wp_state.save_state(&mut wp_image, total_w);
            buf_out[out_row_idx].auxiliary_data = Some(wp_image);
        }
    } else {
        for c in 0..num_c {
            let pal_row = palette.row(c);
            for y in 0..h {
                for (grid_x, index_buf) in buf_in.iter().enumerate().take(grid_xsize) {
                    let index_img = index_buf.data.row(y);
                    let out_idx = c * grid_ysize * grid_xsize + grid_y * grid_xsize + grid_x;
                    for (x, &index) in index_img.iter().enumerate() {
                        let palette_entry = get_palette_value_with_row(
                            pal_row,
                            index as isize,
                            c,
                            palette_size,
                            bit_depth,
                        );
                        let val = if index < num_deltas as i32 {
                            // Fast path for interior pixels: avoid expensive get_with_neighbors
                            let pred = if x > 0 && y > 0 && x + 1 < xsize {
                                predictor.predict_one(
                                    PredictionData::get(&buf_out[out_idx].data, x, y),
                                    /*wp_pred=*/ 0,
                                )
                            } else {
                                predictor.predict_one(
                                    get_prediction_data(
                                        buf_out, out_idx, grid_x, grid_y, grid_xsize, x, y, xsize,
                                        ysize,
                                    ),
                                    /*wp_pred=*/ 0,
                                )
                            };
                            (pred + palette_entry as i64) as i32
                        } else {
                            palette_entry
                        };
                        buf_out[out_idx].data.row_mut(y)[x] = val;
                    }
                }
            }
        }
    }
    Ok(())
}
