// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::array::from_fn;

use crate::{
    error::{Error, Result},
    headers::modular::WeightedHeader,
    image::{Image, ImageRect},
    util::floor_log2_nonzero,
};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

#[repr(u8)]
#[derive(Debug, FromPrimitive, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
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
    pub fn get(rect: ImageRect<i32>, x: usize, y: usize) -> Self {
        let left = if x > 0 {
            rect.row(y)[x - 1]
        } else if y > 0 {
            rect.row(y - 1)[0]
        } else {
            0
        };
        let top = if y > 0 { rect.row(y - 1)[x] } else { left };
        let topleft = if x > 0 && y > 0 {
            rect.row(y - 1)[x - 1]
        } else {
            left
        };
        let topright = if x + 1 < rect.size().0 && y > 0 {
            rect.row(y - 1)[x + 1]
        } else {
            top
        };
        let leftleft = if x > 1 { rect.row(y)[x - 2] } else { left };
        let toptop = if y > 1 { rect.row(y - 2)[x] } else { top };
        let toprightright = if x + 2 < rect.size().0 && y > 0 {
            rect.row(y - 1)[x + 2]
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

    #[allow(clippy::too_many_arguments)]
    pub fn get_with_neighbors(
        rect: ImageRect<i32>,
        rect_left: Option<ImageRect<i32>>,
        rect_top: Option<ImageRect<i32>>,
        rect_top_left: Option<ImageRect<i32>>,
        rect_right: Option<ImageRect<i32>>,
        rect_top_right: Option<ImageRect<i32>>,
        x: usize,
        y: usize,
        xsize: usize,
        ysize: usize,
    ) -> Self {
        let left = if x > 0 {
            rect.row(y)[x - 1]
        } else if let Some(l) = rect_left {
            l.row(y)[xsize - 1]
        } else if y > 0 {
            rect.row(y - 1)[0]
        } else if let Some(t) = rect_top {
            t.row(ysize - 1)[0]
        } else {
            0
        };
        let top = if y > 0 {
            rect.row(y - 1)[x]
        } else if let Some(t) = rect_top {
            t.row(ysize - 1)[x]
        } else {
            left
        };
        let topleft = if x > 0 {
            if y > 0 {
                rect.row(y - 1)[x - 1]
            } else if let Some(t) = rect_top {
                t.row(ysize - 1)[x - 1]
            } else {
                left
            }
        } else if y > 0 {
            if let Some(l) = rect_left {
                l.row(y - 1)[xsize - 1]
            } else {
                left
            }
        } else if let Some(tl) = rect_top_left {
            tl.row(ysize - 1)[xsize - 1]
        } else {
            left
        };
        let topright = if x + 1 < rect.size().0 {
            if y > 0 {
                rect.row(y - 1)[x + 1]
            } else if let Some(t) = rect_top {
                t.row(ysize - 1)[x + 1]
            } else {
                top
            }
        } else if y > 0 {
            if let Some(r) = rect_right {
                r.row(y - 1)[0]
            } else {
                top
            }
        } else if let Some(tr) = rect_top_right {
            tr.row(ysize - 1)[0]
        } else {
            top
        };
        let leftleft = if x > 1 {
            rect.row(y)[x - 2]
        } else if let Some(l) = rect_left {
            l.row(y)[xsize + x - 2]
        } else {
            left
        };
        let toptop = if y > 1 {
            rect.row(y - 2)[x]
        } else if let Some(t) = rect_top {
            t.row(ysize + y - 2)[x]
        } else {
            top
        };
        let toprightright = if x + 2 < rect.size().0 {
            if y > 0 {
                rect.row(y - 1)[x + 2]
            } else if let Some(t) = rect_top {
                t.row(ysize - 1)[x + 2]
            } else {
                topright
            }
        } else if y > 0 {
            if let Some(r) = rect_right {
                r.row(y - 1)[x + 2 - rect.size().0]
            } else {
                topright
            }
        } else if let Some(tr) = rect_top_right {
            tr.row(ysize - 1)[x + 2 - rect.size().0]
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

fn add_bits(x: i32) -> i64 {
    (x as i64) << PRED_EXTRA_BITS
}

fn error_weight(x: u32, maxweight: u32) -> u32 {
    let shift = floor_log2_nonzero(x + 1) as i32 - 5;
    if shift < 0 {
        4u32 + maxweight * DIVLOOKUP[x as usize]
    } else {
        4u32 + ((maxweight * DIVLOOKUP[x as usize >> shift]) >> shift)
    }
}

fn weighted_average(pixels: &[i64; NUM_PREDICTORS], weights: &mut [u32; NUM_PREDICTORS]) -> i64 {
    let log_weight = floor_log2_nonzero(weights.iter().fold(0u32, |sum, el| sum + *el));
    let weight_sum = weights.iter_mut().fold(0, |sum, el| {
        *el >>= log_weight - 4;
        sum + *el
    });
    let sum = weights
        .iter()
        .enumerate()
        .fold(((weight_sum >> 1) - 1) as i64, |sum, (i, weight)| {
            sum + pixels[i] * *weight as i64
        });
    (sum * DIVLOOKUP[(weight_sum - 1) as usize] as i64) >> 24
}

#[derive(Debug)]
pub struct WeightedPredictorState<'a> {
    prediction: [i64; NUM_PREDICTORS],
    pred: i64,
    pred_errors: [Vec<u32>; NUM_PREDICTORS],
    error: Vec<i32>,
    wp_header: &'a WeightedHeader,
}

impl<'a> WeightedPredictorState<'a> {
    pub fn new(wp_header: &'a WeightedHeader, xsize: usize) -> WeightedPredictorState<'a> {
        let num_errors = (xsize + 2) * 2;
        WeightedPredictorState {
            prediction: [0; NUM_PREDICTORS],
            pred: 0,
            pred_errors: from_fn(|_| vec![0; num_errors]),
            error: vec![0; num_errors],
            wp_header,
        }
    }

    pub fn save_state(&self, wp_image: &mut Image<i32>, xsize: usize) {
        wp_image
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&self.error[xsize + 2..]);
    }

    pub fn restore_state(&mut self, wp_image: &Image<i32>, xsize: usize) {
        self.error[xsize + 2..].copy_from_slice(wp_image.as_rect().row(0));
    }

    pub fn update_errors(&mut self, correct_val: i32, pos: (usize, usize), xsize: usize) {
        let (cur_row, prev_row) = if pos.1 & 1 != 0 {
            (0, xsize + 2)
        } else {
            (xsize + 2, 0)
        };
        let val = add_bits(correct_val);
        self.error[cur_row + pos.0] = (self.pred - val) as i32;
        for (i, pred_err) in self.pred_errors.iter_mut().enumerate() {
            let err =
                (((self.prediction[i] - val).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32;
            pred_err[cur_row + pos.0] = err;
            let idx = prev_row + pos.0 + 1;
            pred_err[idx] = pred_err[idx].wrapping_add(err);
        }
    }

    pub fn predict_and_property(
        &mut self,
        pos: (usize, usize),
        xsize: usize,
        data: &PredictionData,
    ) -> (i64, i32) {
        let (cur_row, prev_row) = if pos.1 & 1 != 0 {
            (0, xsize + 2)
        } else {
            (xsize + 2, 0)
        };
        let pos_n = prev_row + pos.0;
        let pos_ne = if pos.0 < xsize - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if pos.0 > 0 { pos_n - 1 } else { pos_n };
        let mut weights = [0u32; NUM_PREDICTORS];
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = error_weight(
                self.pred_errors[i][pos_n]
                    .wrapping_add(self.pred_errors[i][pos_ne])
                    .wrapping_add(self.pred_errors[i][pos_nw]),
                self.wp_header.w(i).unwrap(),
            );
        }
        let n = add_bits(data.top);
        let w = add_bits(data.left);
        let ne = add_bits(data.topright);
        let nw = add_bits(data.topleft);
        let nn = add_bits(data.toptop);

        let te_w = if pos.0 == 0 {
            0
        } else {
            self.error[cur_row + pos.0 - 1] as i64
        };
        let te_n = self.error[pos_n] as i64;
        let te_nw = self.error[pos_nw] as i64;
        let sum_wn = te_n + te_w;
        let te_ne = self.error[pos_ne] as i64;

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

        self.prediction[0] = w + ne - n;
        self.prediction[1] = n - (((sum_wn + te_ne) * self.wp_header.p1c as i64) >> 5);
        self.prediction[2] = w - (((sum_wn + te_nw) * self.wp_header.p2c as i64) >> 5);
        self.prediction[3] = n
            - ((te_nw * (self.wp_header.p3ca as i64)
                + (te_n * (self.wp_header.p3cb as i64))
                + (te_ne * (self.wp_header.p3cc as i64))
                + ((nn - n) * (self.wp_header.p3cd as i64))
                + ((nw - w) * (self.wp_header.p3ce as i64)))
                >> 5);

        self.pred = weighted_average(&self.prediction, &mut weights);

        if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
            let mx = w.max(ne.max(n));
            let mn = w.min(ne.min(n));
            self.pred = mn.max(mx.min(self.pred));
        }
        ((self.pred + PREDICTION_ROUND) >> PRED_EXTRA_BITS, p as i32)
    }
}

#[cfg(test)]
mod tests {
    use crate::headers::modular::{GroupHeader, WeightedHeader};

    use super::{PredictionData, WeightedPredictorState};

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
            xsize,
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
        state.update_errors((rng.next() % 256) as i32, pos, xsize);
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
        let mut state = WeightedPredictorState::new(&header.wp_header, xsize);
        // The golden number results are generated by using the libjxl predictor with the same input numbers.
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (135i64, 0i32));
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (110i64, -60i32));
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (165i64, 0i32));
        assert_eq!(step(&mut rng, &mut state, xsize, ysize), (153i64, -60i32));
    }
}
