// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]
use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::{unpack_signed, Histograms, Reader},
    error::{Error, Result},
    util::{tracing_wrappers::*, CeilLog2},
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
    pub x: f32,
    pub y: f32,
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
    // The estimated area in pixels covered by the spline.
    estimated_area_reached: u64,
}

#[derive(Debug, Default, Clone)]
pub struct QuantizedSpline {
    // Double delta-encoded.
    pub control_points: Vec<(i64, i64)>,
    pub color_dct: [[i32; 32]; 3],
    pub sigma_dct: [i32; 32],
}

fn inv_adjusted_quant(adjustment: i32) -> f32 {
    if adjustment >= 0 {
        1.0 / (1.0 + 0.125 * adjustment as f32)
    } else {
        1.0 - 0.125 * adjustment as f32
    }
}

fn validate_spline_point_pos<T: num_traits::ToPrimitive>(x: T, y: T) -> Result<()> {
    let xi = x.to_i32().unwrap();
    let yi = y.to_i32().unwrap();
    let ok_range = -(1i32 << 23)..(1i32 << 23);
    if !ok_range.contains(&xi) {
        return Err(Error::SplinesPointOutOfRange(xi, ok_range));
    }
    if !ok_range.contains(&yi) {
        return Err(Error::SplinesPointOutOfRange(yi, ok_range));
    }
    Ok(())
}

const CHANNEL_WEIGHT: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];

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

    pub fn dequantize(
        &self,
        starting_point: &Point,
        quantization_adjustment: i32,
        y_to_x: f32,
        y_to_b: f32,
        image_size: u64,
    ) -> Result<Spline> {
        let area_limit = (1024 * image_size + (1u64 << 32)).min(1u64 << 42);

        let mut result = Spline {
            control_points: Vec::with_capacity(self.control_points.len() + 1),
            color_dct: [[0.0; 32]; 3],
            sigma_dct: [0.0; 32],
            estimated_area_reached: 0,
        };

        let px = starting_point.x.round();
        let py = starting_point.y.round();
        //TODO: validate_spline_point_pos(px, py)?;

        let mut current_x = px as i32;
        let mut current_y = py as i32;
        result
            .control_points
            .push(Point::new(current_x as f32, current_y as f32));

        let mut current_delta_x = 0i32;
        let mut current_delta_y = 0i32;
        let mut manhattan_distance = 0u64;

        for &(dx, dy) in &self.control_points {
            current_delta_x += dx as i32;
            current_delta_y += dy as i32;
            validate_spline_point_pos(current_delta_x, current_delta_y).unwrap();

            manhattan_distance +=
                current_delta_x.unsigned_abs() as u64 + current_delta_y.unsigned_abs() as u64;

            if manhattan_distance > area_limit {
                return Err(Error::SplinesDistanceTooLarge(
                    manhattan_distance,
                    area_limit,
                ));
            }

            current_x += current_delta_x;
            current_y += current_delta_y;
            validate_spline_point_pos(current_x, current_y).unwrap();

            result
                .control_points
                .push(Point::new(current_x as f32, current_y as f32));
        }

        let inv_quant = inv_adjusted_quant(quantization_adjustment);

        for (c, weight) in CHANNEL_WEIGHT.iter().enumerate().take(3) {
            for i in 0..32 {
                let inv_dct_factor = if i == 0 { FRAC_1_SQRT_2 } else { 1.0 };
                result.color_dct[c][i] =
                    self.color_dct[c][i] as f32 * inv_dct_factor * weight * inv_quant;
            }
        }

        for i in 0..32 {
            result.color_dct[0][i] += y_to_x * result.color_dct[1][i];
            result.color_dct[2][i] += y_to_b * result.color_dct[1][i];
        }

        let mut width_estimate = 0;
        let mut color = [0u64; 3];

        for (c, color_val) in color.iter_mut().enumerate() {
            for i in 0..32 {
                *color_val += (inv_quant * self.color_dct[c][i] as f32).abs().ceil() as u64;
            }
        }

        color[0] += y_to_x.abs().ceil() as u64 * color[1];
        color[2] += y_to_b.abs().ceil() as u64 * color[1];

        let max_color = color[0].max(color[1]).max(color[2]);
        let logcolor = 1u64.max((1u64 + max_color).ceil_log2());

        let weight_limit =
            (((area_limit as f32 / logcolor as f32) / manhattan_distance.max(1) as f32).sqrt())
                .ceil();

        for i in 0..32 {
            let inv_dct_factor = if i == 0 { FRAC_1_SQRT_2 } else { 1.0 };
            result.sigma_dct[i] =
                self.sigma_dct[i] as f32 * inv_dct_factor * CHANNEL_WEIGHT[3] * inv_quant;

            let weight_f = (inv_quant * self.sigma_dct[i] as f32).abs().ceil();
            let weight = weight_limit.min(weight_f.max(1.0)) as u64;
            width_estimate += weight * weight * logcolor;
        }

        result.estimated_area_reached = width_estimate * manhattan_distance;

        // TODO: move this check to the outside, using the returned estimated_area_reached,
        // making the caller keep track of the total estimated_area_readed
        //if result.total_estimated_area_reached > area_limit {
        //    return Err(Error::SplinesAreaTooLarge(
        //        *total_estimated_area_reached,
        //        area_limit,
        //    ));
        //}

        Ok(result)
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

#[derive(Debug, Default, Clone)]
pub struct Splines {
    pub quantization_adjustment: i32,
    pub splines: Vec<QuantizedSpline>,
    pub starting_points: Vec<Point>,
    segments: Vec<SplineSegment>,
    segment_indices: Vec<usize>,
    segment_y_start: Vec<usize>,
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
            ..Splines::default()
        })
    }
}
