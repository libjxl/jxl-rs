// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::frame::modular::predict::{PredictionData, WeightedPredictorState};
use crate::frame::modular::{ModularChannel, Predictor};
use crate::headers::modular::WeightedHeader;
use crate::image::{Image, ImageRect, ImageRectMut, OwnedRawImage};
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
    palette: &ModularChannel,
    is_16bit: bool,
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
        } else if is_16bit {
            ImageRect::<i16>::from_raw(palette.data.as_rect()).row(c)[index] as i32
        } else {
            ImageRect::<i32>::from_raw(palette.data.as_rect()).row(c)[index]
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_general(
    buf_in: &ModularChannel,
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    wp_header: &WeightedHeader,
    is_16bit: bool,
) {
    let (w, h) = buf_in.size(is_16bit);
    let bit_depth = buf_in.bit_depth.bits_per_sample().min(24) as usize;

    if w == 0 {
        // Nothing to do.
        // Avoid touching "empty" channels with non-zero height.
        return;
    }

    if num_deltas == 0 && predictor == Predictor::Zero {
        let mut row_scratch = vec![0i32; w];
        for (chan_index, out) in buf_out.iter_mut().enumerate() {
            for y in 0..h {
                if is_16bit {
                    let in_rect = ImageRect::<i16>::from_raw(buf_in.data.as_rect());
                    for (dst, &src) in row_scratch.iter_mut().zip(in_rect.row(y)) {
                        *dst = src as i32;
                    }
                } else {
                    let in_rect = ImageRect::<i32>::from_raw(buf_in.data.as_rect());
                    row_scratch.copy_from_slice(in_rect.row(y));
                }
                for index in &mut row_scratch {
                    *index = get_palette_value(
                        buf_pal,
                        is_16bit,
                        *index as isize,
                        chan_index,
                        num_colors,
                        bit_depth,
                    );
                }
                if is_16bit {
                    let mut out_rect = ImageRectMut::<i16>::from_raw(out.data.as_rect_mut());
                    for (dst, &src) in out_rect.row(y).iter_mut().zip(&row_scratch) {
                        *dst = src as i16;
                    }
                } else {
                    let mut out_rect = ImageRectMut::<i32>::from_raw(out.data.as_rect_mut());
                    out_rect.row(y).copy_from_slice(&row_scratch);
                }
            }
        }
        return;
    }

    let mut rows = [vec![0i32; w], vec![0i32; w], vec![0i32; w]];
    let mut in_row = vec![0i32; w];
    for (chan_index, out) in buf_out.iter_mut().enumerate() {
        for r in &mut rows {
            r.fill(0);
        }
        let mut wp_state = if predictor == Predictor::Weighted {
            Some(WeightedPredictorState::new(wp_header, w))
        } else {
            None
        };
        for y in 0..h {
            if is_16bit {
                let in_rect = ImageRect::<i16>::from_raw(buf_in.data.as_rect());
                for (dst, &src) in in_row.iter_mut().zip(in_rect.row(y)) {
                    *dst = src as i32;
                }
            } else {
                let in_rect = ImageRect::<i32>::from_raw(buf_in.data.as_rect());
                in_row.copy_from_slice(in_rect.row(y));
            }
            for (x, &index) in in_row.iter().enumerate() {
                let palette_entry = get_palette_value(
                    buf_pal,
                    is_16bit,
                    index as isize,
                    chan_index,
                    num_colors + num_deltas,
                    bit_depth,
                );
                let prediction_data = PredictionData::get_rows(&rows[0], &rows[1], &rows[2], x, y);
                let val = if let Some(wp) = &mut wp_state {
                    let (wp_pred, _) = wp.predict_and_property((x, y), &prediction_data);
                    let val = if index < num_deltas as i32 {
                        let pred = predictor.predict_one(prediction_data, wp_pred);
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    wp.update_errors(val, (x, y));
                    val
                } else if index < num_deltas as i32 {
                    let pred = predictor.predict_one(prediction_data, /*wp_pred=*/ 0);
                    (pred + palette_entry as i64) as i32
                } else {
                    palette_entry
                };
                rows[0][x] = val;
            }
            if is_16bit {
                let mut out_rect = ImageRectMut::<i16>::from_raw(out.data.as_rect_mut());
                for (dst, &src) in out_rect.row(y).iter_mut().zip(&rows[0]) {
                    *dst = src as i16;
                }
            } else {
                let mut out_rect = ImageRectMut::<i32>::from_raw(out.data.as_rect_mut());
                out_rect.row(y).copy_from_slice(&rows[0]);
            }
            rows.rotate_right(1);
        }
    }
}

fn stage_padded_top_row(row_top: &mut [i32], src: &[i32], topleft: Option<i32>) {
    let w = src.len();
    row_top[1..=w].copy_from_slice(src);
    row_top[0] = topleft.unwrap_or(row_top[1]);
    row_top[w + 1] = row_top[w];
}

fn stage_padded_top_row_i16(row_top: &mut [i32], src: &[i16], topleft: Option<i32>) {
    let w = src.len();
    for (dest, &s) in row_top[1..=w].iter_mut().zip(src) {
        *dest = s as i32;
    }
    row_top[0] = topleft.unwrap_or(row_top[1]);
    row_top[w + 1] = row_top[w];
}

#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_one_group(
    buf_in: &ModularChannel,
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    buf_left: Option<&[&OwnedRawImage]>,
    buf_top: Option<&[&OwnedRawImage]>,
    buf_topleft: Option<&[&OwnedRawImage]>,
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    is_16bit: bool,
    scratch: &mut [Vec<i32>; 2],
) {
    let (w, h) = buf_in.size(is_16bit);
    let bit_depth = buf_in.bit_depth.bits_per_sample().min(24) as usize;
    let num_c = buf_out.len();

    let (row_top, row_out) = scratch.split_at_mut(1);
    let row_top = &mut row_top[0];
    let row_out = &mut row_out[0];
    row_top.resize(w + 2, 0);
    row_out.resize(w, 0);

    for c in 0..num_c {
        for y in 0..h {
            let has_top = y > 0 || buf_top.is_some();
            if y > 0 {
                let topleft = if is_16bit {
                    buf_left
                        .map(|l| ImageRect::<i16>::from_raw(l[c].as_rect()).row(y - 1)[3] as i32)
                } else {
                    buf_left.map(|l| ImageRect::<i32>::from_raw(l[c].as_rect()).row(y - 1)[3])
                };
                if is_16bit {
                    let prev_rect = ImageRect::<i16>::from_raw(buf_out[c].data.as_rect());
                    let prev_row = prev_rect.row(y - 1);
                    stage_padded_top_row_i16(row_top, &prev_row[..w], topleft);
                } else {
                    let prev_rect = ImageRect::<i32>::from_raw(buf_out[c].data.as_rect());
                    let prev_row = prev_rect.row(y - 1);
                    stage_padded_top_row(row_top, &prev_row[..w], topleft);
                }
            } else if let Some(top) = buf_top {
                let topleft = if is_16bit {
                    buf_topleft.map(|tl| {
                        let rect = ImageRect::<i16>::from_raw(tl[c].as_rect());
                        let row = rect.row(3);
                        row[row.len() - 1] as i32
                    })
                } else {
                    buf_topleft.map(|tl| {
                        let rect = ImageRect::<i32>::from_raw(tl[c].as_rect());
                        let row = rect.row(3);
                        row[row.len() - 1]
                    })
                };
                if is_16bit {
                    let top_rect = ImageRect::<i16>::from_raw(top[c].as_rect());
                    let top_row = top_rect.row(3);
                    stage_padded_top_row_i16(row_top, &top_row[..w], topleft);
                } else {
                    let top_rect = ImageRect::<i32>::from_raw(top[c].as_rect());
                    let top_row = top_rect.row(3);
                    stage_padded_top_row(row_top, &top_row[..w], topleft);
                }
            }

            let mut left = if let Some(l) = buf_left {
                if is_16bit {
                    ImageRect::<i16>::from_raw(l[c].as_rect()).row(y)[3] as i32
                } else {
                    ImageRect::<i32>::from_raw(l[c].as_rect()).row(y)[3]
                }
            } else if has_top {
                row_top[1]
            } else {
                0
            };
            let mut leftleft = if let Some(l) = buf_left {
                if is_16bit {
                    ImageRect::<i16>::from_raw(l[c].as_rect()).row(y)[2] as i32
                } else {
                    ImageRect::<i32>::from_raw(l[c].as_rect()).row(y)[2]
                }
            } else {
                left
            };

            if is_16bit {
                let in_rect = ImageRect::<i16>::from_raw(buf_in.data.as_rect());
                for (dst, &src) in row_out[..w].iter_mut().zip(in_rect.row(y)) {
                    *dst = src as i32;
                }
            } else {
                let in_rect = ImageRect::<i32>::from_raw(buf_in.data.as_rect());
                row_out[..w].copy_from_slice(in_rect.row(y));
            }

            for x in 0..w {
                let index = row_out[x];
                let palette_entry = get_palette_value(
                    buf_pal,
                    is_16bit,
                    index as isize,
                    c,
                    num_colors + num_deltas,
                    bit_depth,
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
                row_out[x] = val;
                leftleft = left;
                left = val;
            }

            if is_16bit {
                let mut out_rect = ImageRectMut::<i16>::from_raw(buf_out[c].data.as_rect_mut());
                for (dst, &src) in out_rect.row(y).iter_mut().zip(&row_out[..w]) {
                    *dst = src as i16;
                }
            } else {
                let mut out_rect = ImageRectMut::<i32>::from_raw(buf_out[c].data.as_rect_mut());
                out_rect.row(y).copy_from_slice(&row_out[..w]);
            }
        }
    }
}

pub fn zero_palette_step_one_group(
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    num_colors: usize,
    num_deltas: usize,
    is_16bit: bool,
) {
    let bit_depth = buf_out[0].bit_depth.bits_per_sample().min(24) as usize;

    for (c, out) in buf_out.iter_mut().enumerate() {
        let palette_entry = get_palette_value(
            buf_pal,
            is_16bit,
            0,
            c,
            /*palette_size=*/ num_colors + num_deltas,
            /*bit_depth=*/ bit_depth,
        );
        let (_w, h) = out.size(is_16bit);
        if is_16bit {
            let mut out_rect = ImageRectMut::<i16>::from_raw(out.data.as_rect_mut());
            for y in 0..h {
                out_rect.row(y).fill(palette_entry as i16);
            }
        } else {
            let mut out_rect = ImageRectMut::<i32>::from_raw(out.data.as_rect_mut());
            for y in 0..h {
                out_rect.row(y).fill(palette_entry);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn do_palette_step_group_row(
    buf_in: &[&ModularChannel],
    buf_pal: &ModularChannel,
    buf_out: &mut [&mut ModularChannel],
    buf_prev: Option<&[&OwnedRawImage]>,
    prev_aux: Option<&[Option<&Image<i32>>]>,
    aux_out: &mut [RwLockWriteGuard<Option<Image<i32>>>],
    grid_xsize: usize,
    num_colors: usize,
    num_deltas: usize,
    predictor: Predictor,
    wp_header: &WeightedHeader,
    is_16bit: bool,
    scratch: &mut [Vec<i32>; 2],
) -> Result<()> {
    let (_, h) = buf_in[0].size(is_16bit);
    let bit_depth = buf_in[0].bit_depth.bits_per_sample().min(24) as usize;
    let num_c = buf_out.len() / grid_xsize;

    let total_w: usize = buf_out[..grid_xsize]
        .iter()
        .map(|buf| buf.size(is_16bit).0)
        .sum();

    scratch[0].resize(total_w, 0);
    scratch[1].resize(total_w, 0);

    let mut in_indices = Vec::new();

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
                let w = prev_img.byte_size().0 / if is_16bit { 2 } else { 4 };
                if is_16bit {
                    let prev_rect = ImageRect::<i16>::from_raw(prev_img.as_rect());
                    for (d, &s) in scratch[0][x_offset..x_offset + w]
                        .iter_mut()
                        .zip(prev_rect.row(3))
                    {
                        *d = s as i32;
                    }
                    for (d, &s) in scratch[1][x_offset..x_offset + w]
                        .iter_mut()
                        .zip(prev_rect.row(2))
                    {
                        *d = s as i32;
                    }
                } else {
                    let prev_rect = ImageRect::<i32>::from_raw(prev_img.as_rect());
                    scratch[0][x_offset..x_offset + w].copy_from_slice(prev_rect.row(3));
                    scratch[1][x_offset..x_offset + w].copy_from_slice(prev_rect.row(2));
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
                let w = index_buf.size(is_16bit).0;
                let gx_start = gx;

                in_indices.resize(w, 0);
                if is_16bit {
                    let in_rect = ImageRect::<i16>::from_raw(index_buf.data.as_rect());
                    for (d, &s) in in_indices.iter_mut().zip(in_rect.row(y)) {
                        *d = s as i32;
                    }
                } else {
                    let in_rect = ImageRect::<i32>::from_raw(index_buf.data.as_rect());
                    in_indices.copy_from_slice(in_rect.row(y));
                }

                for &index in &in_indices {
                    let palette_entry = get_palette_value(
                        buf_pal,
                        is_16bit,
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
                        let (wp_pred, _) = wp.predict_and_property((gx, y), &prediction_data);
                        let val = if index < num_deltas as i32 {
                            let pred = predictor.predict_one(prediction_data, wp_pred);
                            (pred + palette_entry as i64) as i32
                        } else {
                            palette_entry
                        };
                        wp.update_errors(val, (gx, y));
                        val
                    } else if index < num_deltas as i32 {
                        let pred = predictor.predict_one(prediction_data, /*wp_pred=*/ 0);
                        (pred + palette_entry as i64) as i32
                    } else {
                        palette_entry
                    };
                    curr_row[gx] = val;
                    leftleft = left;
                    left = val;
                    gx += 1;
                }

                if is_16bit {
                    let mut out_rect =
                        ImageRectMut::<i16>::from_raw(buf_out[out_idx].data.as_rect_mut());
                    for (d, &s) in out_rect
                        .row(y)
                        .iter_mut()
                        .zip(&curr_row[gx_start..gx_start + w])
                    {
                        *d = s as i16;
                    }
                } else {
                    let mut out_rect =
                        ImageRectMut::<i32>::from_raw(buf_out[out_idx].data.as_rect_mut());
                    out_rect
                        .row(y)
                        .copy_from_slice(&curr_row[gx_start..gx_start + w]);
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
