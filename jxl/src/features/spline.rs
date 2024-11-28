// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::{unpack_signed, Histograms, Reader},
    error::{Error, Result},
    util::tracing_wrappers::*,
};
const MAX_NUM_CONTROL_POINTS: u32 = 1 << 20;
const MAX_NUM_CONTROL_POINTS_PER_PIXEL_RATIO: u32 = 2;
const DELTA_LIMIT: i64 = 1 << 30;
const SPLINE_POS_LIMIT: i32 = 1 << 23;

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
    control_points: Vec<(i64, i64)>,
    color_dct: [[i32; 32]; 3],
    sigma_dct: [i32; 32],
}

impl QuantizedSpline {
    #[instrument(level = "debug", skip(br), ret, err)]
    pub fn read(
        br: &mut BitReader,
        splines_reader: &mut Reader,
        max_control_points: u32,
        total_num_control_points: &mut u32,
    ) -> Result<QuantizedSpline> {
        let num_control_points = splines_reader.read(br, NUM_CONTROL_POINTS_CONTEXT)?;
        *total_num_control_points += num_control_points;
        if *total_num_control_points > max_control_points {
            return Err(Error::SplinesTooManyControlPoints(
                *total_num_control_points,
                max_control_points,
            ));
        }
        let mut control_points = Vec::with_capacity(num_control_points as usize);
        for _ in 0..num_control_points {
            let x = splines_reader.read_signed(br, CONTROL_POINTS_CONTEXT)? as i64;
            let y = splines_reader.read_signed(br, CONTROL_POINTS_CONTEXT)? as i64;
            control_points.push((x, y));
            // Add check that double deltas are not outrageous (not in spec).
            let max_delta_delta = x.abs().max(y.abs());
            if max_delta_delta >= DELTA_LIMIT {
                return Err(Error::SplinesDeltaLimit(max_delta_delta, DELTA_LIMIT));
            }
        }
        // Decode DCTs and populate the QuantizedSpline struct
        let mut color_dct = [[0; 32]; 3];
        let mut sigma_dct = [0; 32];

        let mut decode_dct = |dct: &mut [i32; 32]| -> Result<()> {
            for value in dct.iter_mut() {
                *value = splines_reader.read_signed(br, DCT_CONTEXT)?;
            }
            Ok(())
        };

        for channel in &mut color_dct {
            decode_dct(channel)?;
        }
        decode_dct(&mut sigma_dct)?;

        Ok(QuantizedSpline {
            control_points,
            color_dct,
            sigma_dct,
        })
    }
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
}

impl Splines {
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
            // It is not in spec, but reasonable limit to avoid overflows.
            let max_coordinate = x.abs().max(y.abs());
            if max_coordinate >= SPLINE_POS_LIMIT {
                return Err(Error::SplinesCoordinatesLimit(
                    max_coordinate,
                    SPLINE_POS_LIMIT,
                ));
            }

            starting_points.push(Point {
                x: x as f32,
                y: y as f32,
            });

            last_x = x;
            last_y = y;
        }

        let quantization_adjustment =
            splines_reader.read_signed(br, QUANTIZATION_ADJUSTMENT_CONTEXT)?;

        let mut splines = Vec::new();
        let mut num_control_points = 0u32;
        for _ in 0..num_splines {
            splines.push(QuantizedSpline::read(
                br,
                &mut splines_reader,
                max_control_points,
                &mut num_control_points,
            )?);
        }
        splines_reader.check_final_state()?;
        Ok(Splines {
            quantization_adjustment,
            splines,
            starting_points,
        })
    }
}
