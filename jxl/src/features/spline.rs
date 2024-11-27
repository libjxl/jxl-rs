// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    error::{Error, Result},
    util::tracing_wrappers::*,
};
const MAX_NUM_CONTROL_POINTS: u32 = 1 << 20;
const MAX_NUM_CONTROL_POINTS_PER_PIXEL_RATIO: u32 = 2;
const NUM_SPLINES_CONTEXTS: usize = 6;
#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < 1e-3 && (self.y - other.y).abs() < 1e-3
    }
}

pub struct Spline {
    control_points: Vec<Point>,
    // X, Y, B.
    color_dct: [[f32; 32]; 3],
    // Splines are drawn by normalized Gaussian splatting. This controls the
    // Gaussian's parameter along the spline.
    sigma_dct: [f32; 32],
}

#[derive(Debug, Default)]
pub struct QuantizedSpline {
    // Double delta-encoded.
    control_points_: Vec<(i64, i64)>,
    color_dct_: [[i32; 32]; 3],
    sigma_dct_: [i32; 32],
}

#[derive(Debug, Clone, Copy, Default)]
struct SplineSegment {
    center_x: f32,
    center_y: f32,
    maximum_distance: f32,
    inv_sigma: f32,
    sigma_over_4_times_intensity: f32,
    color: [f32; 3],
}

#[derive(Debug, Default)]
pub struct Splines {
    quantization_adjustment_: i32,
    splines_: Vec<QuantizedSpline>,
    starting_points_: Vec<Point>,
    segments_: Vec<SplineSegment>,
    segment_indices_: Vec<usize>,
    segment_y_start_: Vec<usize>,
}

impl Splines {
    pub fn new(
        quantization_adjustment: i32,
        splines: Vec<QuantizedSpline>,
        starting_points: Vec<Point>,
    ) -> Self {
        Splines {
            quantization_adjustment_: quantization_adjustment,
            splines_: splines,
            starting_points_: starting_points,
            segments_: Vec::new(),
            segment_indices_: Vec::new(),
            segment_y_start_: Vec::new(),
        }
    }

    fn has_any(&self) -> bool {
        !self.splines_.is_empty()
    }

    fn clear(&mut self) {
        self.splines_.clear();
        self.starting_points_.clear();
        self.segments_.clear();
        self.segment_indices_.clear();
        self.segment_y_start_.clear();
    }
    #[instrument(level = "debug", skip(br), ret, err)]
    pub fn read(br: &mut BitReader, num_pixels: u32) -> Result<Splines> {
        trace!(pos = br.total_bits_read());
        let splines_histograms = Histograms::decode(NUM_SPLINES_CONTEXTS, br, true)?;
        let mut splines_reader = splines_histograms.make_reader(br)?;
        // TODO: check what is right context
        let num_splines = splines_reader.read(br, 1)?;
        let max_control_points =
            MAX_NUM_CONTROL_POINTS.min(num_pixels / MAX_NUM_CONTROL_POINTS_PER_PIXEL_RATIO);
        if num_splines > max_control_points {
            return Err(Error::SplinesTooMany(num_splines, max_control_points));
        }
        todo!("complete Splines::read")
    }

    fn quantized_splines(&self) -> &Vec<QuantizedSpline> {
        &self.splines_
    }

    fn starting_points(&self) -> &Vec<Point> {
        &self.starting_points_
    }

    fn get_quantization_adjustment(&self) -> i32 {
        self.quantization_adjustment_
    }
}
