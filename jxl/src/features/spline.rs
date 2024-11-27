// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::{unpack_signed, Histograms},
    error::{Error, Result},
    util::tracing_wrappers::*,
};
const MAX_NUM_CONTROL_POINTS: u32 = 1 << 20;
const MAX_NUM_CONTROL_POINTS_PER_PIXEL_RATIO: u32 = 2;

const QUANTIZATION_ADJUSTMENT_CONTEXT: usize = 0;
const STARTING_POSITION_CONTEXT: usize = 1;
const NUM_SPLINES_CONTEXT: usize = 2;
const NUM_CONTROL_POINTS_CONTEXT: usize = 3;
const CONTROL_POINTS_CONTEXT: usize = 4;
const DCT_CONTEXT: usize = 5;
const NUM_SPLINE_CONTEXTS: usize = 6;

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
    quantization_adjustment: i32,
    splines: Vec<QuantizedSpline>,
    starting_points: Vec<Point>,
    segments: Vec<SplineSegment>,
    segment_indices: Vec<usize>,
    segment_y_start: Vec<usize>,
}

impl Splines {
    pub fn new(
        quantization_adjustment: i32,
        splines: Vec<QuantizedSpline>,
        starting_points: Vec<Point>,
    ) -> Self {
        Splines {
            quantization_adjustment: quantization_adjustment,
            splines: splines,
            starting_points: starting_points,
            segments: Vec::new(),
            segment_indices: Vec::new(),
            segment_y_start: Vec::new(),
        }
    }

    fn has_any(&self) -> bool {
        !self.splines.is_empty()
    }

    #[instrument(level = "debug", skip(br), ret, err)]
    pub fn read(br: &mut BitReader, num_pixels: u32) -> Result<Splines> {
        trace!(pos = br.total_bits_read());
        let splines_histograms = Histograms::decode(NUM_SPLINE_CONTEXTS, br, true)?;
        let mut splines_reader = splines_histograms.make_reader(br)?;
        let num_splines = 1 + splines_reader.read(br, NUM_SPLINES_CONTEXT)?;
        let max_control_points =
            MAX_NUM_CONTROL_POINTS.min(num_pixels / MAX_NUM_CONTROL_POINTS_PER_PIXEL_RATIO);
        if num_splines > max_control_points {
            return Err(Error::SplinesTooMany(num_splines, max_control_points));
        }

        let mut starting_points = Vec::new();
        let mut last_x = 0;
        let mut last_y = 0;
        for i in 0..num_splines {
            let unsigned_x = splines_reader.read(br, STARTING_POSITION_CONTEXT)?;
            let unsigned_y = splines_reader.read(br, STARTING_POSITION_CONTEXT)?;

            let (x, y) = if i != 0 {
                (
                    unpack_signed(unsigned_x) + last_x,
                    unpack_signed(unsigned_y) + last_y,
                )
            } else {
                (unsigned_x as i32, unsigned_y as i32)
            };

            // TODO: validate spline position here
            starting_points.push(Point {
                x: x as f32,
                y: y as f32,
            });

            last_x = x;
            last_y = y;
        }

        let quantization_adjustment = splines_reader.read_signed(br, QUANTIZATION_ADJUSTMENT_CONTEXT)?;
        todo!("complete Splines::read");
        Ok(Splines {quantization_adjustment, splines: Vec::new(), starting_points, segments: Vec::new(), segment_indices: Vec::new(), segment_y_start: Vec::new()})
    }
}
