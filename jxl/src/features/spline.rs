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
        validate_spline_point_pos(px, py)?;

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
            validate_spline_point_pos(current_delta_x, current_delta_y)?;

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
            validate_spline_point_pos(current_x, current_y)?;

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

#[cfg(test)]
mod test_splines {
    use crate::{error::Error, util::test::assert_all_almost_eq};

    use super::{Point, QuantizedSpline, Spline};

    #[test]
    fn dequantize() -> Result<(), Error> {
        // Golden data generated by libjxl.
        let quantized_and_dequantized = [
            (
                QuantizedSpline {
                    control_points: vec![
                        (109, 105),
                        (-247, -261),
                        (168, 427),
                        (-46, -360),
                        (-61, 181),
                    ],
                    color_dct: [
                        [
                            12223, 9452, 5524, 16071, 1048, 17024, 14833, 7690, 21952, 2405, 2571,
                            2190, 1452, 2500, 18833, 1667, 5857, 21619, 1310, 20000, 10429, 11667,
                            7976, 18786, 12976, 18548, 14786, 12238, 8667, 3405, 19929, 8429,
                        ],
                        [
                            177, 712, 127, 999, 969, 356, 105, 12, 1132, 309, 353, 415, 1213, 156,
                            988, 524, 316, 1100, 64, 36, 816, 1285, 183, 889, 839, 1099, 79, 1316,
                            287, 105, 689, 841,
                        ],
                        [
                            780, -201, -38, -695, -563, -293, -88, 1400, -357, 520, 979, 431, -118,
                            590, -971, -127, 157, 206, 1266, 204, -320, -223, 704, -687, -276,
                            -716, 787, -1121, 40, 292, 249, -10,
                        ],
                    ],
                    sigma_dct: [
                        139, 65, 133, 5, 137, 272, 88, 178, 71, 256, 254, 82, 126, 252, 152, 53,
                        281, 15, 8, 209, 285, 156, 73, 56, 36, 287, 86, 244, 270, 94, 224, 156,
                    ],
                },
                Spline {
                    control_points: vec![
                        Point { x: 109.0, y: 54.0 },
                        Point { x: 218.0, y: 159.0 },
                        Point { x: 80.0, y: 3.0 },
                        Point { x: 110.0, y: 274.0 },
                        Point { x: 94.0, y: 185.0 },
                        Point { x: 17.0, y: 277.0 },
                    ],
                    color_dct: [
                        [
                            36.3005, 39.6984, 23.2008, 67.4982, 4.4016, 71.5008, 62.2986, 32.298,
                            92.1984, 10.101, 10.7982, 9.198, 6.0984, 10.5, 79.0986, 7.0014,
                            24.5994, 90.7998, 5.502, 84.0, 43.8018, 49.0014, 33.4992, 78.9012,
                            54.4992, 77.9016, 62.1012, 51.3996, 36.4014, 14.301, 83.7018, 35.4018,
                        ],
                        [
                            9.38684, 53.4, 9.525, 74.925, 72.675, 26.7, 7.875, 0.9, 84.9, 23.175,
                            26.475, 31.125, 90.975, 11.7, 74.1, 39.3, 23.7, 82.5, 4.8, 2.7, 61.2,
                            96.375, 13.725, 66.675, 62.925, 82.425, 5.925, 98.7, 21.525, 7.875,
                            51.675, 63.075,
                        ],
                        [
                            47.9949, 39.33, 6.865, 26.275, 33.265, 6.19, 1.715, 98.9, 59.91,
                            59.575, 95.005, 61.295, 82.715, 53.0, 6.13, 30.41, 34.69, 96.92, 93.42,
                            16.98, 38.8, 80.765, 63.005, 18.585, 43.605, 32.305, 61.015, 20.23,
                            24.325, 28.315, 69.105, 62.375,
                        ],
                    ],
                    sigma_dct: [
                        32.7593, 21.6645, 44.3289, 1.6665, 45.6621, 90.6576, 29.3304, 59.3274,
                        23.6643, 85.3248, 84.6582, 27.3306, 41.9958, 83.9916, 50.6616, 17.6649,
                        93.6573, 4.9995, 2.6664, 69.6597, 94.9905, 51.9948, 24.3309, 18.6648,
                        11.9988, 95.6571, 28.6638, 81.3252, 89.991, 31.3302, 74.6592, 51.9948,
                    ],
                    estimated_area_reached: 19843491681,
                },
            ),
            (
                QuantizedSpline {
                    control_points: vec![
                        (24, -32),
                        (-178, -7),
                        (226, 151),
                        (121, -172),
                        (-184, 39),
                        (-201, -182),
                        (301, 404),
                    ],
                    color_dct: [
                        [
                            5051, 6881, 5238, 1571, 9952, 19762, 2048, 13524, 16405, 2310, 1286,
                            4714, 16857, 21429, 12500, 15524, 1857, 5595, 6286, 17190, 15405,
                            20738, 310, 16071, 10952, 16286, 15571, 8452, 6929, 3095, 9905, 5690,
                        ],
                        [
                            899, 1059, 836, 388, 1291, 247, 235, 203, 1073, 747, 1283, 799, 356,
                            1281, 1231, 561, 477, 720, 309, 733, 1013, 477, 779, 1183, 32, 1041,
                            1275, 367, 88, 1047, 321, 931,
                        ],
                        [
                            -78, 244, -883, 943, -682, 752, 107, 262, -75, 557, -202, -575, -231,
                            -731, -605, 732, 682, 650, 592, -14, -1035, 913, -188, -95, 286, -574,
                            -509, 67, 86, -1056, 592, 380,
                        ],
                    ],
                    sigma_dct: [
                        308, 8, 125, 7, 119, 237, 209, 60, 277, 215, 126, 186, 90, 148, 211, 136,
                        188, 142, 140, 124, 272, 140, 274, 165, 24, 209, 76, 254, 185, 83, 11, 141,
                    ],
                },
                Spline {
                    control_points: vec![
                        Point { x: 172.0, y: 309.0 },
                        Point { x: 196.0, y: 277.0 },
                        Point { x: 42.0, y: 238.0 },
                        Point { x: 114.0, y: 350.0 },
                        Point { x: 307.0, y: 290.0 },
                        Point { x: 316.0, y: 269.0 },
                        Point { x: 124.0, y: 66.0 },
                        Point { x: 233.0, y: 267.0 },
                    ],
                    color_dct: [
                        [
                            15.0007, 28.9002, 21.9996, 6.5982, 41.7984, 83.0004, 8.6016, 56.8008,
                            68.901, 9.702, 5.4012, 19.7988, 70.7994, 90.0018, 52.5, 65.2008,
                            7.7994, 23.499, 26.4012, 72.198, 64.701, 87.0996, 1.302, 67.4982,
                            45.9984, 68.4012, 65.3982, 35.4984, 29.1018, 12.999, 41.601, 23.898,
                        ],
                        [
                            47.6767, 79.425, 62.7, 29.1, 96.825, 18.525, 17.625, 15.225, 80.475,
                            56.025, 96.225, 59.925, 26.7, 96.075, 92.325, 42.075, 35.775, 54.0,
                            23.175, 54.975, 75.975, 35.775, 58.425, 88.725, 2.4, 78.075, 95.625,
                            27.525, 6.6, 78.525, 24.075, 69.825,
                        ],
                        [
                            43.8159, 96.505, 0.889999, 95.11, 49.085, 71.165, 25.115, 33.565,
                            75.225, 95.015, 82.085, 19.675, 10.53, 44.905, 49.975, 93.315, 83.515,
                            99.5, 64.615, 53.995, 3.52501, 99.685, 45.265, 82.075, 22.42, 37.895,
                            59.995, 32.215, 12.62, 4.605, 65.515, 96.425,
                        ],
                    ],
                    sigma_dct: [
                        72.589, 2.6664, 41.6625, 2.3331, 39.6627, 78.9921, 69.6597, 19.998,
                        92.3241, 71.6595, 41.9958, 61.9938, 29.997, 49.3284, 70.3263, 45.3288,
                        62.6604, 47.3286, 46.662, 41.3292, 90.6576, 46.662, 91.3242, 54.9945,
                        7.9992, 69.6597, 25.3308, 84.6582, 61.6605, 27.6639, 3.6663, 46.9953,
                    ],
                    estimated_area_reached: 25829781306,
                },
            ),
            (
                QuantizedSpline {
                    control_points: vec![
                        (157, -89),
                        (-244, 41),
                        (-58, 168),
                        (429, -185),
                        (-361, 198),
                        (230, -269),
                        (-416, 203),
                        (167, 65),
                        (460, -344),
                    ],
                    color_dct: [
                        [
                            5691, 15429, 1000, 2524, 5595, 4048, 18881, 1357, 14381, 3952, 22595,
                            15167, 20857, 2500, 905, 14548, 5452, 19500, 19143, 9643, 10929, 6048,
                            9476, 7143, 11952, 21524, 6643, 22310, 15500, 11476, 5310, 10452,
                        ],
                        [
                            470, 880, 47, 1203, 1295, 211, 475, 8, 907, 528, 325, 1145, 769, 1035,
                            633, 905, 57, 72, 1216, 780, 1, 696, 47, 637, 843, 580, 1144, 477, 669,
                            479, 256, 643,
                        ],
                        [
                            1169, -301, 1041, -725, -43, -22, 774, 134, -822, 499, 456, -287, -713,
                            -776, 76, 449, 750, 580, -207, -643, 956, -426, 377, -64, 101, -250,
                            -164, 259, 169, -240, 430, -22,
                        ],
                    ],
                    sigma_dct: [
                        354, 5, 75, 56, 140, 226, 84, 187, 151, 70, 257, 288, 137, 99, 100, 159,
                        79, 176, 59, 210, 278, 68, 171, 65, 230, 263, 69, 199, 107, 107, 170, 202,
                    ],
                },
                Spline {
                    control_points: vec![
                        Point { x: 100.0, y: 186.0 },
                        Point { x: 257.0, y: 97.0 },
                        Point { x: 170.0, y: 49.0 },
                        Point { x: 25.0, y: 169.0 },
                        Point { x: 309.0, y: 104.0 },
                        Point { x: 232.0, y: 237.0 },
                        Point { x: 385.0, y: 101.0 },
                        Point { x: 122.0, y: 168.0 },
                        Point { x: 26.0, y: 300.0 },
                        Point { x: 390.0, y: 88.0 },
                    ],
                    color_dct: [
                        [
                            16.9014, 64.8018, 4.2, 10.6008, 23.499, 17.0016, 79.3002, 5.6994,
                            60.4002, 16.5984, 94.899, 63.7014, 87.5994, 10.5, 3.801, 61.1016,
                            22.8984, 81.9, 80.4006, 40.5006, 45.9018, 25.4016, 39.7992, 30.0006,
                            50.1984, 90.4008, 27.9006, 93.702, 65.1, 48.1992, 22.302, 43.8984,
                        ],
                        [
                            24.9255, 66.0, 3.525, 90.225, 97.125, 15.825, 35.625, 0.6, 68.025,
                            39.6, 24.375, 85.875, 57.675, 77.625, 47.475, 67.875, 4.275, 5.4, 91.2,
                            58.5, 0.075, 52.2, 3.525, 47.775, 63.225, 43.5, 85.8, 35.775, 50.175,
                            35.925, 19.2, 48.225,
                        ],
                        [
                            82.7881, 44.93, 76.395, 39.475, 94.115, 14.285, 89.805, 9.98, 10.485,
                            74.53, 56.295, 65.785, 7.765, 23.305, 52.795, 99.305, 56.775, 46.0,
                            76.71, 13.49, 66.995, 22.38, 29.915, 43.295, 70.295, 26.0, 74.32,
                            53.905, 62.005, 19.125, 49.3, 46.685,
                        ],
                    ],
                    sigma_dct: [
                        83.4303, 1.6665, 24.9975, 18.6648, 46.662, 75.3258, 27.9972, 62.3271,
                        50.3283, 23.331, 85.6581, 95.9904, 45.6621, 32.9967, 33.33, 52.9947,
                        26.3307, 58.6608, 19.6647, 69.993, 92.6574, 22.6644, 56.9943, 21.6645,
                        76.659, 87.6579, 22.9977, 66.3267, 35.6631, 35.6631, 56.661, 67.3266,
                    ],
                    estimated_area_reached: 47263284396,
                },
            ),
        ];
        for (quantized, want_dequantized) in quantized_and_dequantized {
            let got_dequantized = quantized.dequantize(
                &want_dequantized.control_points[0],
                0,
                0.0,
                1.0,
                2u64 << 30,
            )?;
            assert_eq!(
                got_dequantized.control_points.len(),
                want_dequantized.control_points.len()
            );
            assert_all_almost_eq!(
                got_dequantized
                    .control_points
                    .iter()
                    .map(|p| p.x)
                    .collect::<Vec<f32>>(),
                want_dequantized
                    .control_points
                    .iter()
                    .map(|p| p.x)
                    .collect::<Vec<f32>>(),
                1e-6,
            );
            assert_all_almost_eq!(
                got_dequantized
                    .control_points
                    .iter()
                    .map(|p| p.y)
                    .collect::<Vec<f32>>(),
                want_dequantized
                    .control_points
                    .iter()
                    .map(|p| p.y)
                    .collect::<Vec<f32>>(),
                1e-6,
            );
            for index in 0..got_dequantized.color_dct.len() {
                assert_all_almost_eq!(
                    got_dequantized.color_dct[index],
                    want_dequantized.color_dct[index],
                    1e-4,
                );
            }
            assert_all_almost_eq!(got_dequantized.sigma_dct, want_dequantized.sigma_dct, 1e-4);
            assert_eq!(
                got_dequantized.estimated_area_reached,
                want_dequantized.estimated_area_reached,
            );
        }
        Ok(())
    }
}
