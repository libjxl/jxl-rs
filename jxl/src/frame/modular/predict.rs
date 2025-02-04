// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    headers::modular::GroupHeader,
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

impl TryFrom<u32> for Predictor {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        Self::from_u32(value).ok_or(Error::InvalidPredictor(value))
    }
}

impl Predictor {
    pub const NUM_PREDICTORS: u32 = Predictor::AverageAll as u32 + 1;
}

#[derive(Debug)]
pub struct WeightedPredictorState;

impl WeightedPredictorState {
    pub fn new(_header: &GroupHeader) -> Self {
        // TODO(veluca): implement the weighted predictor.
        Self
    }

    pub fn predict_and_property(&self) -> (i64, i32) {
        // TODO(veluca): implement the weighted predictor.
        (0, 0)
    }
}
