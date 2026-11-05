// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

use crate::error::{Error, Result};
use crate::headers::modular::WeightedHeader;
use crate::image::Image;

#[repr(u8)]
#[derive(Debug, FromPrimitive, Clone, Copy, PartialEq, Eq, Default)]
pub enum Predictor {
    #[default]
    Zero = 0,
    West = 1,
    North = 2,
    AverageWestAndNorth = 3,
    Select = 4,
    Gradient = 5,
    Weighted = 6,
    NorthEast = 7,
    NorthWest = 8,
    WestWest = 9,
    AverageWestAndNorthWest = 10,
    AverageNorthAndNorthWest = 11,
    AverageNorthAndNorthEast = 12,
    AverageAll = 13,
}

impl Predictor {
    pub fn requires_full_row(&self) -> bool {
        matches!(
            self,
            Predictor::Weighted
                | Predictor::NorthEast
                | Predictor::AverageNorthAndNorthEast
                | Predictor::AverageAll
        )
    }
}

impl TryFrom<u32> for Predictor {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        Self::from_u32(value).ok_or(Error::InvalidPredictor(value))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PredictionData {
    pub left: i32,
    pub top: i32,
    pub toptop: i32,
    pub topleft: i32,
    pub topright: i32,
    pub leftleft: i32,
    pub toprightright: i32,
}

impl PredictionData {
    #[inline(always)]
    pub fn update_for_interior_row(
        self,
        cur: i32,
        toprightright: i32,
        toptop: i32,
    ) -> PredictionData {
        Self {
            left: cur,
            top: self.topright,
            toptop,
            topleft: self.top,
            topright: self.toprightright,
            leftleft: self.left,
            toprightright,
        }
    }

    #[inline]
    pub fn get_rows(row: &[i32], row_top: &[i32], row_toptop: &[i32], x: usize, y: usize) -> Self {
        let left = if x > 0 {
            row[x - 1]
        } else if y > 0 {
            row_top[0]
        } else {
            0
        };
        let top = if y > 0 { row_top[x] } else { left };
        let topleft = if x > 0 && y > 0 { row_top[x - 1] } else { left };
        let topright = if x + 1 < row.len() && y > 0 {
            row_top[x + 1]
        } else {
            top
        };
        let leftleft = if x > 1 { row[x - 2] } else { left };
        let toptop = if y > 1 { row_toptop[x] } else { top };
        let toprightright = if x + 2 < row.len() && y > 0 {
            row_top[x + 2]
        } else {
            topright
        };
        Self {
            left,
            top,
            toptop,
            topleft,
            topright,
            leftleft,
            toprightright,
        }
    }
}

pub fn clamped_gradient(left: i64, top: i64, topleft: i64) -> i64 {
    // Same code/logic as libjxl.
    let min = left.min(top);
    let max = left.max(top);
    let grad = left + top - topleft;
    let grad_clamp_max = if topleft < min { max } else { grad };
    if topleft > max { min } else { grad_clamp_max }
}

impl Predictor {
    pub const NUM_PREDICTORS: u32 = Predictor::AverageAll as u32 + 1;

    #[inline]
    pub fn predict_one(
        &self,
        PredictionData {
            left,
            top,
            toptop,
            topleft,
            topright,
            leftleft,
            toprightright,
        }: PredictionData,
        wp_pred: i64,
    ) -> i64 {
        match self {
            Predictor::Zero => 0,
            Predictor::West => left as i64,
            Predictor::North => top as i64,
            Predictor::Select => Self::select(left as i64, top as i64, topleft as i64),
            Predictor::Gradient => clamped_gradient(left as i64, top as i64, topleft as i64),
            Predictor::Weighted => wp_pred,
            Predictor::WestWest => leftleft as i64,
            Predictor::NorthEast => topright as i64,
            Predictor::NorthWest => topleft as i64,
            Predictor::AverageWestAndNorth => (top as i64 + left as i64) / 2,
            Predictor::AverageWestAndNorthWest => (left as i64 + topleft as i64) / 2,
            Predictor::AverageNorthAndNorthWest => (top as i64 + topleft as i64) / 2,
            Predictor::AverageNorthAndNorthEast => (top as i64 + topright as i64) / 2,
            Predictor::AverageAll => {
                (6 * top as i64 - 2 * toptop as i64
                    + 7 * left as i64
                    + leftleft as i64
                    + toprightright as i64
                    + 3 * topright as i64
                    + 8)
                    / 16
            }
        }
    }

    fn select(left: i64, top: i64, topleft: i64) -> i64 {
        let p = left + top - topleft;
        if (p - left).abs() < (p - top).abs() {
            left
        } else {
            top
        }
    }
}

const NUM_PREDICTORS: usize = 4;
const PRED_EXTRA_BITS: i64 = 3;
const PREDICTION_ROUND: i64 = ((1 << PRED_EXTRA_BITS) >> 1) - 1;
// Allows to approximate division by a number from 1 to 64.
//  for (int i = 0; i < 64; i++) divlookup[i] = (1 << 24) / (i + 1);
const DIVLOOKUP: [u32; 64] = [
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288, 508400,
    493447, 479349, 466033, 453438, 441505, 430185, 419430, 409200, 399457, 390167, 381300, 372827,
    364722, 356962, 349525, 342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593, 294337,
    289262, 284359, 279620, 275036, 270600, 266305, 262144,
];

// 1-based lookup table for dividing by `weight_sum` (which is in 1..=31):
// `WEIGHT_SUM_DIVLOOKUP[i] = (1 << 24) / i` for i in 1..32, with 0 at index 0.
const WEIGHT_SUM_DIVLOOKUP: [u32; 32] = [
    0, 16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200,
];

#[inline(always)]
fn add_bits(x: i32) -> i64 {
    (x as i64) << PRED_EXTRA_BITS
}

#[inline(always)]
fn ilog2_shifted<const SHIFT: u32>(x: u64) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        ((x | (1 << SHIFT)).ilog2() - SHIFT) as usize
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        (((x >> SHIFT) | 1).ilog2()) as usize
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
struct WpRowEntry {
    pred_errors: [u32; NUM_PREDICTORS],
    // Note: we store errors in positions [1..=xsize], and error[0] == 0.
    error: i32,
}

#[derive(Debug)]
pub struct WeightedPredictorState<'a> {
    prediction: [i64; NUM_PREDICTORS],
    pred: i64,
    buffers: &'a mut [[WpRowEntry; 2]],
    w: [u8; 4],
    p1c: u8,
    p2c: u8,
    p3c: [u8; 5],
}

#[derive(Debug, Default)]
pub struct WeightedPredictorScratch {
    buffers: Vec<[WpRowEntry; 2]>,
}

impl<'a> WeightedPredictorState<'a> {
    pub fn new(
        wp_header: &WeightedHeader,
        xsize: usize,
        scratch: &'a mut WeightedPredictorScratch,
    ) -> WeightedPredictorState<'a> {
        let row_len = xsize.checked_add(1).unwrap();
        scratch.buffers.clear();
        scratch.buffers.resize(row_len, [WpRowEntry::default(); 2]);
        WeightedPredictorState {
            prediction: [0; NUM_PREDICTORS],
            pred: 0,
            buffers: &mut scratch.buffers[..row_len],
            // These casts are lossless because w fits in 4 bits and p fits in 5 by
            // bitstream constraints.
            w: [
                wp_header.w0 as u8,
                wp_header.w1 as u8,
                wp_header.w2 as u8,
                wp_header.w3 as u8,
            ],
            p1c: wp_header.p1c as u8,
            p2c: wp_header.p2c as u8,
            p3c: [
                wp_header.p3ca as u8,
                wp_header.p3cb as u8,
                wp_header.p3cc as u8,
                wp_header.p3cd as u8,
                wp_header.p3ce as u8,
            ],
        }
    }

    pub fn save_state(&self, wp_image: &mut Image<i32>) {
        let [d_err, d0, d1, d2, d3] = wp_image.distinct_rows_mut([0, 1, 2, 3, 4]);
        for (((((d_err, d0), d1), d2), d3), s) in d_err
            .iter_mut()
            .zip(d0.iter_mut())
            .zip(d1.iter_mut())
            .zip(d2.iter_mut())
            .zip(d3.iter_mut())
            .zip(self.buffers.iter())
        {
            *d_err = s[0].error;
            *d0 = s[0].pred_errors[0] as i32;
            *d1 = s[0].pred_errors[1] as i32;
            *d2 = s[0].pred_errors[2] as i32;
            *d3 = s[0].pred_errors[3] as i32;
        }
    }

    pub fn restore_state(&mut self, wp_image: &Image<i32>) {
        let s_err = wp_image.row(0);
        let s0 = wp_image.row(1);
        let s1 = wp_image.row(2);
        let s2 = wp_image.row(3);
        let s3 = wp_image.row(4);
        for (d, (&s_err, (&s0, (&s1, (&s2, &s3))))) in self.buffers.iter_mut().zip(
            s_err
                .iter()
                .zip(s0.iter().zip(s1.iter().zip(s2.iter().zip(s3.iter())))),
        ) {
            d[0].error = s_err;
            d[0].pred_errors = [s0 as u32, s1 as u32, s2 as u32, s3 as u32];
        }
    }

    // Note: optimizations in this function are a bit finnicky.
    #[inline(always)]
    pub fn predict_and_property(
        &mut self,
        pos: (usize, usize),
        data: &PredictionData,
    ) -> (i64, i32) {
        let x = pos.0 & (isize::MAX as usize);
        assert!(x + 1 < self.buffers.len());
        let xsize = self.buffers.len() - 1;

        let cur_row = (pos.1 & 1) ^ 1;
        let prev_row = pos.1 & 1;

        let pos_ne = (x + 1).min(xsize - 1);
        let pos_nw = x.saturating_sub(1);

        let err_n = &self.buffers[x][prev_row].pred_errors;
        let err_ne = &self.buffers[pos_ne][prev_row].pred_errors;
        let err_nw = &self.buffers[pos_nw][prev_row].pred_errors;

        let err0 = err_n[0].wrapping_add(err_ne[0]).wrapping_add(err_nw[0]);
        let err1 = err_n[1].wrapping_add(err_ne[1]).wrapping_add(err_nw[1]);
        let err2 = err_n[2].wrapping_add(err_ne[2]).wrapping_add(err_nw[2]);
        let err3 = err_n[3].wrapping_add(err_ne[3]).wrapping_add(err_nw[3]);

        let shift0 = ilog2_shifted::<5>(err0 as u64 + 1);
        let shift1 = ilog2_shifted::<5>(err1 as u64 + 1);
        let shift2 = ilog2_shifted::<5>(err2 as u64 + 1);
        let shift3 = ilog2_shifted::<5>(err3 as u64 + 1);

        let div0 = DIVLOOKUP[((err0 as usize) >> shift0) & 63];
        let div1 = DIVLOOKUP[((err1 as usize) >> shift1) & 63];
        let div2 = DIVLOOKUP[((err2 as usize) >> shift2) & 63];
        let div3 = DIVLOOKUP[((err3 as usize) >> shift3) & 63];

        let w0 = (4 + ((self.w[0] as u32 * div0) >> shift0)) as u64;
        let w1 = (4 + ((self.w[1] as u32 * div1) >> shift1)) as u64;
        let w2 = (4 + ((self.w[2] as u32 * div2) >> shift2)) as u64;
        let w3 = (4 + ((self.w[3] as u32 * div3) >> shift3)) as u64;

        // Note: this might access what's morally equivalent to position -1,
        // but that value is guaranteed to be 0.
        let te_w = self.buffers[x][cur_row].error as i64;
        let te_n = self.buffers[x + 1][prev_row].error as i64;
        let te_nw = self.buffers[pos_nw + 1][prev_row].error as i64;
        let sum_wn = te_n + te_w;
        let te_ne = self.buffers[pos_ne + 1][prev_row].error as i64;

        let mut p = te_w;
        if te_n.abs() > p.abs() {
            p = te_n;
        }
        if te_nw.abs() > p.abs() {
            p = te_nw;
        }
        if te_ne.abs() > p.abs() {
            p = te_ne;
        }

        let n = add_bits(data.top);
        let w = add_bits(data.left);
        let ne = add_bits(data.topright);
        let nw = add_bits(data.topleft);
        let nn = add_bits(data.toptop);

        let p0 = w + ne - n;
        let p1 = n - (((sum_wn + te_ne) * self.p1c as i64) >> 5);
        let p2 = w - (((sum_wn + te_nw) * self.p2c as i64) >> 5);
        let p3 = n
            - ((te_nw * (self.p3c[0] as i64)
                + (te_n * (self.p3c[1] as i64))
                + (te_ne * (self.p3c[2] as i64))
                + ((nn - n) * (self.p3c[3] as i64))
                + ((nw - w) * (self.p3c[4] as i64)))
                >> 5);

        let log_weight_m4 = ilog2_shifted::<4>(w0 + w1 + w2 + w3);

        let w0s = (w0 >> log_weight_m4) as i64;
        let w1s = (w1 >> log_weight_m4) as i64;
        let w2s = (w2 >> log_weight_m4) as i64;
        let w3s = (w3 >> log_weight_m4) as i64;

        let weight_sum = w0s + w1s + w2s + w3s;

        let sum = (weight_sum >> 1) - 1 + w0s * p0 + w1s * p1 + w2s * p2 + w3s * p3;

        let div_sum = WEIGHT_SUM_DIVLOOKUP[(weight_sum as usize) & 31] as i64;
        let mut pred = (sum * div_sum) >> 24;

        if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
            let mx = w.max(ne.max(n));
            let mn = w.min(ne.min(n));
            pred = mn.max(mx.min(pred));
        }
        self.prediction = [p0, p1, p2, p3];
        self.pred = pred;
        ((pred + PREDICTION_ROUND) >> PRED_EXTRA_BITS, p as i32)
    }

    #[inline(always)]
    pub fn update_errors(&mut self, correct_val: i32, pos: (usize, usize)) {
        let x = pos.0 & (isize::MAX as usize);
        assert!(x + 1 < self.buffers.len());

        let cur_row = (pos.1 & 1) ^ 1;
        let prev_row = pos.1 & 1;

        let val = add_bits(correct_val);
        self.buffers[x + 1][cur_row].error = (self.pred - val) as i32;

        // Compute errors for all predictors
        let err0 =
            (((self.prediction[0] - val).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32;
        let err1 =
            (((self.prediction[1] - val).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32;
        let err2 =
            (((self.prediction[2] - val).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32;
        let err3 =
            (((self.prediction[3] - val).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32;

        self.buffers[x][cur_row].pred_errors = [err0, err1, err2, err3];

        let prev_errors = &mut self.buffers[x + 1][prev_row].pred_errors;
        prev_errors[0] = prev_errors[0].wrapping_add(err0);
        prev_errors[1] = prev_errors[1].wrapping_add(err1);
        prev_errors[2] = prev_errors[2].wrapping_add(err2);
        prev_errors[3] = prev_errors[3].wrapping_add(err3);
    }
}

#[cfg(test)]
mod tests {
    use super::{PredictionData, WeightedPredictorState};
    use crate::headers::modular::{GroupHeader, WeightedHeader};

    struct SimpleRandom {
        out: i64,
    }

    impl SimpleRandom {
        fn new() -> SimpleRandom {
            SimpleRandom { out: 1 }
        }
        fn next(&mut self) -> i64 {
            self.out = self.out * 48271 % 0x7fffffff;
            self.out
        }
    }

    fn step(
        rng: &mut SimpleRandom,
        state: &mut WeightedPredictorState,
        xsize: usize,
        ysize: usize,
    ) -> (i64, i32) {
        let pos = (rng.next() as usize % xsize, rng.next() as usize % ysize);
        let res = state.predict_and_property(
            pos,
            &PredictionData {
                top: rng.next() as i32 % 256,
                left: rng.next() as i32 % 256,
                topright: rng.next() as i32 % 256,
                topleft: rng.next() as i32 % 256,
                toptop: rng.next() as i32 % 256,
                leftleft: 0,
                toprightright: 0,
            },
        );
        state.update_errors((rng.next() % 256) as i32, pos);
        res
    }

    #[test]
    fn predict_and_update_errors() {
        let mut rng = SimpleRandom::new();
        let header = GroupHeader {
            use_global_tree: false,
            wp_header: WeightedHeader {
                all_default: true,
                p1c: rng.next() as u32 % 32,
                p2c: rng.next() as u32 % 32,
                p3ca: rng.next() as u32 % 32,
                p3cb: rng.next() as u32 % 32,
                p3cc: rng.next() as u32 % 32,
                p3cd: rng.next() as u32 % 32,
                p3ce: rng.next() as u32 % 32,
                w0: rng.next() as u32 % 16,
                w1: rng.next() as u32 % 16,
                w2: rng.next() as u32 % 16,
                w3: rng.next() as u32 % 16,
            },
            transforms: Vec::new(),
        };
        let xsize = 8;
        let ysize = 8;
        let mut scratch = super::WeightedPredictorScratch::default();
        let mut state = WeightedPredictorState::new(&header.wp_header, xsize, &mut scratch);
        // The golden number results are generated by using the libjxl predictor with the same input numbers.
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (135i64, 0i32));
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (110i64, -60i32));
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (165i64, 0i32));
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (153i64, -60i32));
    }
}
