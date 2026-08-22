// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::frame::modular::buffers::{ModularChannel, ModularData};
use crate::frame::modular::predict::{PredictionData, WeightedPredictorState};
use crate::frame::modular::Predictor;
use crate::headers::modular::WeightedHeader;
use crate::image::Image;
use crate::util::sync::RwLockWriteGuard;

const RGB_CHANNELS: usize = 3;

// 5x5x5 color cube for the larger cube.
const LARGE_CUBE: usize = 5;

// Smaller interleaved color cube to fill the holes of the larger cube.
const SMALL_CUBE: usize = 4;
const SMALL_CUBE_BITS: usize = 2;
// SMALL_CUBE ** 3
const LARGE_CUBE_OFFSET: usize = SMALL_CUBE * SMALL_CUBE * SMALL_CUBE;

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
    palette: &ModularData,
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
            palette.get_pixel_i32(index, c)
        }
    }
}

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
            for y in 0..h {
                for x in 0..w {
                    let index = buf_in.data.get_pixel_i32(x, y);
                    let palette_value = get_palette_value(
                        palette,
                        index as isize,
                        /*c=*/ chan_index,
                        /*palette_size=*/ num_colors,
                        /*bit_depth=*/ bit_depth,
                    );
                    match &mut out.data {
                        ModularData::I32(img) => img.row_mut(y)[x] = palette_value,
                        ModularData::I16(img) => img.row_mut(y)[x] = palette_value as i16,
                    }
                }
            }
        }
    } else if predictor == Predictor::Weighted {
        let w = buf_in.data.size().0;
        for (chan_index, out) in buf_out.iter_mut().enumerate() {
            let mut wp_state = WeightedPredictorState::new(wp_header, w);
            for y in 0..h {
                for x in 0..w {
                    let index = buf_in.data.get_pixel_i32(x, y);
                    let palette_entry = get_palette_value(
                        palette,
                        index as isize,
                        /*c=*/ chan_index,
                        /*palette_size=*/ num_colors + num_deltas,
                        /*bit_depth=*/ bit_depth,
                    );
                    let prediction_data = match &out.data {
                        ModularData::I32(img) => PredictionData::get(img, x, y),
                        ModularData::I16(img) => PredictionData::get_rows(
                            &img.row(y).iter().map(|&v| v as i32).collect::<Vec<_>>(),
                            &img.row(y.saturating_sub(1)).iter().map(|&v| v as i32).collect::<Vec<_>>(),
                            &img.row(y.saturating_sub(2)).iter().map(|&v| v as i32).collect::<Vec<_>>(),
                            x,
                            y,
                        ),
                    };
                    let (wp_pred, _) = wp_state.predict_and_property((x, y), &prediction_data);
                    let pred = predictor.predict_one(prediction_data, wp_pred);
                    let val = if index < num_deltas as i32 {
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    match &mut out.data {
                        ModularData::I32(img) => img.row_mut(y)[x] = val,
                        ModularData::I16(img) => img.row_mut(y)[x] = val as i16,
                    }
                    wp_state.update_errors(val, (x, y));
                }
            }
        }
    } else {
        for (chan_index, out) in buf_out.iter_mut().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    let index = buf_in.data.get_pixel_i32(x, y);
                    let palette_entry = get_palette_value(
                        palette,
                        index as isize,
                        /*c=*/ chan_index,
                        /*palette_size=*/ num_colors + num_deltas,
                        /*bit_depth=*/ bit_depth,
                    );
                    let pred_data = match &out.data {
                        ModularData::I32(img) => PredictionData::get(img, x, y),
                        ModularData::I16(img) => PredictionData::get_rows(
                            &img.row(y).iter().map(|&v| v as i32).collect::<Vec<_>>(),
                            &img.row(y.saturating_sub(1)).iter().map(|&v| v as i32).collect::<Vec<_>>(),
                            &img.row(y.saturating_sub(2)).iter().map(|&v| v as i32).collect::<Vec<_>>(),
                            x,
                            y,
                        ),
                    };
                    let val = if index < num_deltas as i32 {
                        let pred = predictor.predict_one(pred_data, /*wp_pred=*/ 0);
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    match &mut out.data {
                        ModularData::I32(img) => img.row_mut(y)[x] = val,
                        ModularData::I16(img) => img.row_mut(y)[x] = val as i16,
                    }
                }
            }
        }
    }
}

fn stage_padded_top_row(row_top: &mut [i32], src: &[i32], topleft: Option<i32>) {
    let w = src.len();
    row_top[1..=w].copy_from_slice(src);
    row_top[0] = topleft.unwrap_or(row_top[1]);
    row_top[w + 1] = row_top[w];
}

#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_one_group(
    buf_in: &ModularChannel,
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    buf_left: Option<&[&ModularData]>,
    buf_top: Option<&[&ModularData]>,
    buf_topleft: Option<&[&ModularData]>,
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    scratch: &mut [Vec<i32>; 2],
) {
    let (w, h) = buf_in.data.size();
    let palette = &buf_pal.data;
    let bit_depth = buf_in.bit_depth.bits_per_sample().min(24) as usize;
    let num_c = buf_out.len();

    let row_top = &mut scratch[0];
    row_top.resize(w + 2, 0);

    for c in 0..num_c {
        for y in 0..h {
            let has_top = y > 0 || buf_top.is_some();
            if y > 0 {
                match &buf_out[c].data {
                    ModularData::I32(img) => {
                        let prev_row = img.row(y - 1);
                        let topleft = buf_left.map(|l| l[c].get_pixel_i32(3, y - 1));
                        stage_padded_top_row(row_top, &prev_row[..w], topleft);
                    }
                    ModularData::I16(img) => {
                        let prev_row = img.row(y - 1);
                        let topleft = buf_left.map(|l| l[c].get_pixel_i32(3, y - 1));
                        for i in 0..w {
                            row_top[i + 1] = prev_row[i] as i32;
                        }
                        row_top[0] = topleft.unwrap_or(row_top[1]);
                        row_top[w + 1] = row_top[w];
                    }
                }
            } else if let Some(top) = buf_top {
                let top_img = top[c];
                let topleft = buf_topleft.map(|tl| tl[c].get_pixel_i32(tl[c].size().0 - 1, 3));
                for i in 0..w {
                    row_top[i + 1] = top_img.get_pixel_i32(i, 3);
                }
                row_top[0] = topleft.unwrap_or(row_top[1]);
                row_top[w + 1] = row_top[w];
            }

            let mut left = if let Some(l) = buf_left {
                l[c].get_pixel_i32(3, y)
            } else if has_top {
                row_top[1]
            } else {
                0
            };
            let mut leftleft = if let Some(l) = buf_left {
                l[c].get_pixel_i32(2, y)
            } else {
                left
            };

            for x in 0..w {
                let index = buf_in.data.get_pixel_i32(x, y);
                let palette_entry = get_palette_value(
                    palette,
                    index as isize,
                    c,
                    /*palette_size=*/ num_colors + num_deltas,
                    /*bit_depth=*/ bit_depth,
                );
                let val = if index < num_deltas as i32 {
                    let (top, topleft, topright) = if has_top {
                        (row_top[x + 1], row_top[x], row_top[x + 2])
                    } else {
                        (left, left, left)
                    };
                    let leftleft_val = if x > 1 || buf_left.is_some() {
                        leftleft
                    } else {
                        left
                    };
                    let pred_data = PredictionData {
                        left,
                        top,
                        toptop: top,
                        topleft,
                        topright,
                        leftleft: leftleft_val,
                        toprightright: topright,
                    };
                    let pred = predictor.predict_one(pred_data, /*wp_pred=*/ 0);
                    (pred + palette_entry as i64) as i32
                } else {
                    palette_entry
                };
                match &mut buf_out[c].data {
                    ModularData::I32(img) => img.row_mut(y)[x] = val,
                    ModularData::I16(img) => img.row_mut(y)[x] = val as i16,
                }
                leftleft = left;
                left = val;
            }
        }
    }
}

pub fn zero_palette_step_one_group(
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    num_colors: usize,
    num_deltas: usize,
) {
    let (_w, h) = buf_out[0].data.size();
    let palette = &buf_pal.data;
    let bit_depth = buf_out[0].bit_depth.bits_per_sample().min(24) as usize;

    for (c, out) in buf_out.iter_mut().enumerate() {
        let palette_entry = get_palette_value(
            palette,
            0,
            c,
            /*palette_size=*/ num_colors + num_deltas,
            /*bit_depth=*/ bit_depth,
        );
        match &mut out.data {
            ModularData::I32(img) => {
                for y in 0..h {
                    img.row_mut(y).fill(palette_entry);
                }
            }
            ModularData::I16(img) => {
                for y in 0..h {
                    img.row_mut(y).fill(palette_entry as i16);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_group_row(
    buf_in: &[&ModularChannel],
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    buf_prev: Option<&[&ModularData]>,
    prev_aux: Option<&[Option<&Image<i32>>]>,
    aux_out: &mut [RwLockWriteGuard<Option<Image<i32>>>],
    grid_xsize: usize,
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    wp_header: &WeightedHeader,
    scratch: &mut [Vec<i32>; 2],
) -> Result<()> {
    let (_, h) = buf_in[0].data.size();
    let palette = &buf_pal.data;
    let bit_depth = buf_in[0].bit_depth.bits_per_sample().min(24) as usize;
    let num_c = buf_out.len() / grid_xsize;

    let total_w: usize = buf_out[..grid_xsize]
        .iter()
        .map(|buf| buf.data.size().0)
        .sum();

    scratch[0].resize(total_w, 0);
    scratch[1].resize(total_w, 0);

    for c in 0..num_c {
        let out_row_idx = c * grid_xsize;
        let mut wp_state = if predictor == Predictor::Weighted {
            let mut state = WeightedPredictorState::new(wp_header, total_w);
            if let Some(aux) = prev_aux {
                state.restore_state(aux[c].unwrap());
            }
            Some(state)
        } else {
            None
        };

        if let Some(prev) = buf_prev {
            let mut x_offset = 0;
            for grid_x in 0..grid_xsize {
                let prev_img = prev[out_row_idx + grid_x];
                let w = prev_img.size().0;
                match prev_img {
                    ModularData::I32(img) => {
                        let r3 = img.row(3);
                        let r2 = img.row(2);
                        scratch[0][x_offset..x_offset + w].copy_from_slice(&r3[..w]);
                        scratch[1][x_offset..x_offset + w].copy_from_slice(&r2[..w]);
                    }
                    ModularData::I16(img) => {
                        let r3 = img.row(3);
                        let r2 = img.row(2);
                        for i in 0..w {
                            scratch[0][x_offset + i] = r3[i] as i32;
                            scratch[1][x_offset + i] = r2[i] as i32;
                        }
                    }
                }
                x_offset += w;
            }
        }

        for y in 0..h {
            let has_top = y > 0 || buf_prev.is_some();
            let has_toptop = y > 1 || buf_prev.is_some();

            let (top_row, curr_row) = if y % 2 == 0 {
                let (s0, s1) = scratch.split_at_mut(1);
                (&s0[0], &mut s1[0])
            } else {
                let (s0, s1) = scratch.split_at_mut(1);
                (&s1[0], &mut s0[0])
            };

            let mut left = if has_top { top_row[0] } else { 0 };
            let mut leftleft = left;

            let mut gx = 0;
            for (grid_x, index_buf) in buf_in.iter().enumerate().take(grid_xsize) {
                let out_idx = out_row_idx + grid_x;
                let cur_w = index_buf.data.size().0;
                match (&index_buf.data, &mut buf_out[out_idx].data) {
                    (ModularData::I32(index_img), ModularData::I32(out_img)) => {
                        let in_row = index_img.row(y);
                        let out_row = out_img.row_mut(y);
                        for x in 0..cur_w {
                            let index = in_row[x];
                            let palette_entry = get_palette_value(
                                palette,
                                index as isize,
                                c,
                                num_colors + num_deltas,
                                bit_depth,
                            );
                            let (top, topleft, topright, toprightright) = if has_top {
                                (
                                    top_row[gx],
                                    if gx > 0 { top_row[gx - 1] } else { left },
                                    if gx + 1 < total_w {
                                        top_row[gx + 1]
                                    } else {
                                        top_row[gx]
                                    },
                                    if gx + 2 < total_w {
                                        top_row[gx + 2]
                                    } else if gx + 1 < total_w {
                                        top_row[gx + 1]
                                    } else {
                                        top_row[gx]
                                    },
                                )
                            } else {
                                (left, left, left, left)
                            };
                            let toptop = if has_toptop { curr_row[gx] } else { top };
                            let leftleft_val = if gx > 1 { leftleft } else { left };
                            let prediction_data = PredictionData {
                                left,
                                top,
                                toptop,
                                topleft,
                                topright,
                                leftleft: leftleft_val,
                                toprightright,
                            };
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
                            curr_row[gx] = val;
                            leftleft = left;
                            left = val;
                            gx += 1;
                        }
                    }
                    (ModularData::I16(index_img), ModularData::I16(out_img)) => {
                        let in_row = index_img.row(y);
                        let out_row = out_img.row_mut(y);
                        for x in 0..cur_w {
                            let index = in_row[x] as i32;
                            let palette_entry = get_palette_value(
                                palette,
                                index as isize,
                                c,
                                num_colors + num_deltas,
                                bit_depth,
                            );
                            let (top, topleft, topright, toprightright) = if has_top {
                                (
                                    top_row[gx],
                                    if gx > 0 { top_row[gx - 1] } else { left },
                                    if gx + 1 < total_w {
                                        top_row[gx + 1]
                                    } else {
                                        top_row[gx]
                                    },
                                    if gx + 2 < total_w {
                                        top_row[gx + 2]
                                    } else if gx + 1 < total_w {
                                        top_row[gx + 1]
                                    } else {
                                        top_row[gx]
                                    },
                                )
                            } else {
                                (left, left, left, left)
                            };
                            let toptop = if has_toptop { curr_row[gx] } else { top };
                            let leftleft_val = if gx > 1 { leftleft } else { left };
                            let prediction_data = PredictionData {
                                left,
                                top,
                                toptop,
                                topleft,
                                topright,
                                leftleft: leftleft_val,
                                toprightright,
                            };
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
                            out_row[x] = val as i16;
                            curr_row[gx] = val;
                            leftleft = left;
                            left = val;
                            gx += 1;
                        }
                    }
                    _ => unreachable!("mismatched buffer types in palette"),
                }
            }
        }

        if let Some(wp) = wp_state {
            let mut wp_image = Image::<i32>::new((total_w + 1, 5))?;
            wp.save_state(&mut wp_image);
            *aux_out[c] = Some(wp_image);
        }
    }
    Ok(())
}
