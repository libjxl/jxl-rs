// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::headers::extra_channels::{ExtraChannel, ExtraChannelInfo};

use super::patches::{PatchBlendMode, PatchBlending};

const K_SMALL_ALPHA: f32 = 1e-6;

#[inline]
fn maybe_clamp(v: f32, clamp: bool) -> f32 {
    if clamp { v.clamp(0.0, 1.0) } else { v }
}

fn perform_alpha_blending_layers<T: AsRef<[f32]>, V: AsMut<[f32]>>(
    bg: &[T],
    fg: &[T],
    alpha_is_premultiplied: bool,
    clamp: bool,
    out: &mut [V],
) {
    if alpha_is_premultiplied {
        for x in 0..out[0].as_mut().len() {
            let fga = maybe_clamp(fg[3].as_ref()[x], clamp);
            out[0].as_mut()[x] = fg[0].as_ref()[x] + bg[0].as_ref()[x] * (1.0 - fga);
            out[1].as_mut()[x] = fg[1].as_ref()[x] + bg[1].as_ref()[x] * (1.0 - fga);
            out[2].as_mut()[x] = fg[2].as_ref()[x] + bg[2].as_ref()[x] * (1.0 - fga);
            out[3].as_mut()[x] = 1.0 - (1.0 - fga) * (1.0 - bg[3].as_ref()[x]);
        }
    } else {
        for x in 0..out[0].as_mut().len() {
            let fga = maybe_clamp(fg[3].as_ref()[x], clamp);
            let new_a = 1.0 - (1.0 - fga) * (1.0 - bg[3].as_ref()[x]);
            let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
            out[0].as_mut()[x] = (fg[0].as_ref()[x] * fga
                + bg[0].as_ref()[x] * bg[3].as_ref()[x] * (1.0 - fga))
                * rnew_a;
            out[1].as_mut()[x] = (fg[1].as_ref()[x] * fga
                + bg[1].as_ref()[x] * bg[3].as_ref()[x] * (1.0 - fga))
                * rnew_a;
            out[2].as_mut()[x] = (fg[2].as_ref()[x] * fga
                + bg[2].as_ref()[x] * bg[3].as_ref()[x] * (1.0 - fga))
                * rnew_a;
            out[3].as_mut()[x] = new_a;
        }
    }
}

pub fn perform_alpha_blending(
    bg: &[f32],
    bga: &[f32],
    fg: &[f32],
    fga: &[f32],
    alpha_is_premultiplied: bool,
    clamp: bool,
    out: &mut [f32],
) {
    if std::ptr::addr_eq(bg, bga) && std::ptr::addr_eq(fg, fga) {
        for x in 0..out.len() {
            let fa = maybe_clamp(fga[x], clamp);
            out[x] = 1.0 - (1.0 - fa) * (1.0 - bga[x]);
        }
    } else if alpha_is_premultiplied {
        for x in 0..out.len() {
            let fa = maybe_clamp(fga[x], clamp);
            out[x] = fg[x] + bg[x] * (1.0 - fa);
        }
    } else {
        for x in 0..out.len() {
            let fa = maybe_clamp(fga[x], clamp);
            let new_a = 1.0 - (1.0 - fa) * (1.0 - bga[x]);
            let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
            out[x] = (fg[x] * fa + bg[x] * bga[x] * (1.0 - fa)) * rnew_a;
        }
    }
}

pub fn perform_alpha_weighted_add(
    bg: &[f32],
    fg: &[f32],
    fga: &[f32],
    clamp_alpha: bool,
    out: &mut [f32],
) {
    if std::ptr::addr_eq(fg, fga) {
        out.copy_from_slice(bg);
    } else if clamp_alpha {
        for x in 0..out.len() {
            out[x] = bg[x] + fg[x] * fga[x].clamp(0.0, 1.0);
        }
    } else {
        for x in 0..out.len() {
            out[x] = bg[x] + fg[x] * fga[x];
        }
    }
}

pub fn perform_mul_blending(bg: &[f32], fg: &[f32], clamp_alpha: bool, out: &mut [f32]) {
    if clamp_alpha {
        for x in 0..out.len() {
            out[x] = bg[x] * fg[x].clamp(0.0, 1.0);
        }
    } else {
        for x in 0..out.len() {
            out[x] = bg[x] * fg[x];
        }
    }
}

pub fn premultiply_alpha(r: &mut [f32], g: &mut [f32], b: &mut [f32], a: &[f32]) {
    for x in 0..a.len() {
        let multiplier = a[x].max(K_SMALL_ALPHA);
        r[x] *= multiplier;
        g[x] *= multiplier;
        b[x] *= multiplier;
    }
}

pub fn unpremultiply_alpha(r: &mut [f32], g: &mut [f32], b: &mut [f32], a: &[f32]) {
    for x in 0..a.len() {
        let multiplier = 1.0 / a[x].max(K_SMALL_ALPHA);
        r[x] *= multiplier;
        g[x] *= multiplier;
        b[x] *= multiplier;
    }
}

fn add<T: AsRef<[f32]>, V: AsMut<[f32]>>(bg: &[T], fg: &[T], out: &mut [V]) {
    for c in 0..3 {
        for (x, v) in out[c].as_mut().iter_mut().enumerate() {
            *v = bg[c].as_ref()[x] + fg[c].as_ref()[x];
        }
    }
}

fn blend_weighted<T: AsRef<[f32]>, V: AsMut<[f32]>>(
    bg: &[T],
    fg: &[T],
    alpha_is_premultiplied: bool,
    alpha: usize,
    clamp: bool,
    out: &mut [V],
) {
    perform_alpha_blending_layers(
        &[
            bg[0].as_ref(),
            bg[1].as_ref(),
            bg[2].as_ref(),
            bg[3 + alpha].as_ref(),
        ],
        &[
            fg[0].as_ref(),
            fg[1].as_ref(),
            fg[2].as_ref(),
            fg[3 + alpha].as_ref(),
        ],
        alpha_is_premultiplied,
        clamp,
        out,
    );
}

fn add_weighted<T: AsRef<[f32]>, V: AsMut<[f32]>>(
    bg: &[T],
    fg: &[T],
    alpha: usize,
    clamp: bool,
    out: &mut [V],
) {
    for c in 0..3 {
        perform_alpha_weighted_add(
            bg[c].as_ref(),
            fg[c].as_ref(),
            fg[3 + alpha].as_ref(),
            clamp,
            out[c].as_mut(),
        );
    }
}

fn copy<T: AsRef<[f32]>, V: AsMut<[f32]>>(src: &[T], out: &mut [V]) {
    for p in 0..3 {
        out[p].as_mut().copy_from_slice(src[p].as_ref());
    }
}

pub fn perform_blending<T: AsRef<[f32]>, V: AsMut<[f32]>>(
    bg: &[T],
    fg: &[T],
    color_blending: &PatchBlending,
    ec_blending: &[PatchBlending],
    extra_channel_info: &[ExtraChannelInfo],
    out: &mut [V],
) {
    let has_alpha = extra_channel_info
        .iter()
        .any(|info| info.ec_type == ExtraChannel::Alpha);
    let num_ec = extra_channel_info.len();

    let mut tmp = vec![vec![0.0; out[0].as_mut().len()]; 3 + num_ec];

    for i in 0..num_ec {
        match ec_blending[i].mode {
            PatchBlendMode::Add => {
                for x in 0..tmp[3 + i].len() {
                    tmp[3 + i][x] = bg[3 + i].as_ref()[x] + fg[3 + i].as_ref()[x];
                }
            }
            PatchBlendMode::BlendAbove => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_blending(
                    bg[3 + i].as_ref(),
                    bg[3 + alpha].as_ref(),
                    fg[3 + i].as_ref(),
                    fg[3 + alpha].as_ref(),
                    extra_channel_info[alpha].alpha_associated(),
                    ec_blending[i].clamp,
                    &mut tmp[3 + i],
                );
            }
            PatchBlendMode::BlendBelow => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_blending(
                    fg[3 + i].as_ref(),
                    fg[3 + alpha].as_ref(),
                    bg[3 + i].as_ref(),
                    bg[3 + alpha].as_ref(),
                    extra_channel_info[alpha].alpha_associated(),
                    ec_blending[i].clamp,
                    &mut tmp[3 + i],
                );
            }
            PatchBlendMode::AlphaWeightedAddAbove => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_weighted_add(
                    bg[3 + i].as_ref(),
                    fg[3 + i].as_ref(),
                    fg[3 + alpha].as_ref(),
                    ec_blending[i].clamp,
                    &mut tmp[3 + i],
                );
            }
            PatchBlendMode::AlphaWeightedAddBelow => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_weighted_add(
                    fg[3 + i].as_ref(),
                    bg[3 + i].as_ref(),
                    bg[3 + alpha].as_ref(),
                    ec_blending[i].clamp,
                    &mut tmp[3 + i],
                );
            }
            PatchBlendMode::Mul => {
                perform_mul_blending(
                    bg[3 + i].as_ref(),
                    fg[3 + i].as_ref(),
                    ec_blending[i].clamp,
                    &mut tmp[3 + i],
                );
            }
            PatchBlendMode::Replace => {
                tmp[3 + i].copy_from_slice(fg[3 + i].as_ref());
            }
            PatchBlendMode::None => {
                tmp[3 + i].copy_from_slice(bg[3 + i].as_ref());
            }
        }
    }

    let alpha = color_blending.alpha_channel;

    match color_blending.mode {
        PatchBlendMode::Add => {
            add(bg, fg, &mut tmp);
        }
        PatchBlendMode::AlphaWeightedAddAbove => {
            if has_alpha {
                add_weighted(bg, fg, alpha, color_blending.clamp, &mut tmp);
            } else {
                add(bg, fg, &mut tmp);
            }
        }
        PatchBlendMode::AlphaWeightedAddBelow => {
            if has_alpha {
                add_weighted(fg, bg, alpha, color_blending.clamp, &mut tmp);
            } else {
                add(fg, bg, &mut tmp);
            }
        }
        PatchBlendMode::BlendAbove => {
            if has_alpha {
                blend_weighted(
                    bg,
                    fg,
                    extra_channel_info[alpha].alpha_associated(),
                    alpha,
                    color_blending.clamp,
                    &mut tmp,
                );
            } else {
                copy(fg, &mut tmp);
            }
        }
        PatchBlendMode::BlendBelow => {
            if has_alpha {
                blend_weighted(
                    fg,
                    bg,
                    extra_channel_info[alpha].alpha_associated(),
                    alpha,
                    color_blending.clamp,
                    &mut tmp,
                );
            } else {
                copy(bg, &mut tmp);
            }
        }
        PatchBlendMode::Mul => {
            for p in 0..3 {
                perform_mul_blending(
                    bg[p].as_ref(),
                    fg[p].as_ref(),
                    color_blending.clamp,
                    &mut tmp[p],
                );
            }
        }
        PatchBlendMode::Replace => copy(fg, &mut tmp),
        PatchBlendMode::None => copy(bg, &mut tmp),
    }
    for i in 0..(3 + num_ec) {
        out[i].as_mut().copy_from_slice(&tmp[i]);
    }
}

#[cfg(test)]
mod tests {
    fn clamp(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    const MAX_DELTA: f32 = 1e-6;

    mod perform_alpha_blending_layers_tests {
        use super::{super::*, *};
        use crate::util::test::assert_all_almost_eq;
        use test_log::test;

        #[test]
        fn test_premultiplied_clamped_one_pixel() {
            let bg_r_data = [0.1f32];
            let bg_g_data = [0.2f32];
            let bg_b_data = [0.3f32];
            let bg_a_data = [0.8f32];
            let fg_r_data = [0.7f32];
            let fg_g_data = [0.6f32];
            let fg_b_data = [0.5f32];
            let fg_a_data = [1.5f32]; // fg_a will be clamped to 1.0

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let mut out_channels = vec![
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
            ];

            let alpha_is_premultiplied = true;
            let clamp_alpha = true;
            let num_pixels = 1;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                };
                expected_r[x] = fg_channels[0][x] + bg_channels[0][x] * (1.0 - fga);
                expected_g[x] = fg_channels[1][x] + bg_channels[1][x] * (1.0 - fga);
                expected_b[x] = fg_channels[2][x] + bg_channels[2][x] * (1.0 - fga);
                expected_a[x] = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]);
            }

            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA);
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA);
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA);
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA);
        }

        #[test]
        fn test_premultiplied_not_clamped_one_pixel() {
            let bg_r_data = [0.1f32];
            let bg_g_data = [0.2f32];
            let bg_b_data = [0.3f32];
            let bg_a_data = [0.8f32];
            let fg_r_data = [0.7f32];
            let fg_g_data = [0.6f32];
            let fg_b_data = [0.5f32];
            let fg_a_data = [1.5f32]; // fg_a is > 1.0, not clamped

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let mut out_channels = vec![
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
            ];

            let alpha_is_premultiplied = true;
            let clamp_alpha = false;
            let num_pixels = 1;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                };
                expected_r[x] = fg_channels[0][x] + bg_channels[0][x] * (1.0 - fga);
                expected_g[x] = fg_channels[1][x] + bg_channels[1][x] * (1.0 - fga);
                expected_b[x] = fg_channels[2][x] + bg_channels[2][x] * (1.0 - fga);
                expected_a[x] = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]);
            }

            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA);
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA);
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA);
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA);
        }

        #[test]
        fn test_straight_alpha_clamped_one_pixel() {
            let bg_r_data = [0.1f32];
            let bg_g_data = [0.2f32];
            let bg_b_data = [0.3f32];
            let bg_a_data = [0.8f32];
            let fg_r_data = [0.7f32];
            let fg_g_data = [0.6f32];
            let fg_b_data = [0.5f32];
            let fg_a_data = [-0.5f32]; // fg_a will be clamped to 0.0

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let mut out_channels = vec![
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
            ];

            let alpha_is_premultiplied = false;
            let clamp_alpha = true;
            let num_pixels = 1;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                };
                let new_a = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]);
                let rnew_a = if new_a > 1e-9 { 1.0 / new_a } else { 0.0 }; // Avoid division by zero, match C++ `> 0`

                expected_r[x] = (fg_channels[0][x] * fga
                    + bg_channels[0][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_g[x] = (fg_channels[1][x] * fga
                    + bg_channels[1][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_b[x] = (fg_channels[2][x] * fga
                    + bg_channels[2][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_a[x] = new_a;
            }

            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA);
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA);
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA);
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA);
        }

        #[test]
        fn test_straight_alpha_not_clamped_one_pixel() {
            let bg_r_data = [0.1f32];
            let bg_g_data = [0.2f32];
            let bg_b_data = [0.3f32];
            let bg_a_data = [0.8f32];
            let fg_r_data = [0.7f32];
            let fg_g_data = [0.6f32];
            let fg_b_data = [0.5f32];
            let fg_a_data = [-0.5f32]; // fg_a is < 0.0, not clamped

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let mut out_channels = vec![
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
            ];

            let alpha_is_premultiplied = false;
            let clamp_alpha = false;
            let num_pixels = 1;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                };
                let new_a = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]);
                let rnew_a = if new_a.abs() > 1e-9 { 1.0 / new_a } else { 0.0 }; // Allow for new_a potentially being negative if fga is very large

                expected_r[x] = (fg_channels[0][x] * fga
                    + bg_channels[0][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_g[x] = (fg_channels[1][x] * fga
                    + bg_channels[1][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_b[x] = (fg_channels[2][x] * fga
                    + bg_channels[2][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_a[x] = new_a;
            }

            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA);
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA);
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA);
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA);
        }

        #[test]
        fn test_premultiplied_clamped_multiple_pixels() {
            let bg_r_data = [0.1, 0.9];
            let bg_g_data = [0.2, 0.8];
            let bg_b_data = [0.3, 0.7];
            let bg_a_data = [0.8, 0.2];
            let fg_r_data = [0.7, 0.3];
            let fg_g_data = [0.6, 0.4];
            let fg_b_data = [0.5, 0.5];
            let fg_a_data = [1.5, -0.2]; // Mixed clamping

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let num_pixels = 2;
            let mut out_channels = vec![
                vec![0.0f32; num_pixels],
                vec![0.0f32; num_pixels],
                vec![0.0f32; num_pixels],
                vec![0.0f32; num_pixels],
            ];

            let alpha_is_premultiplied = true;
            let clamp_alpha = true;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                };
                expected_r[x] = fg_channels[0][x] + bg_channels[0][x] * (1.0 - fga);
                expected_g[x] = fg_channels[1][x] + bg_channels[1][x] * (1.0 - fga);
                expected_b[x] = fg_channels[2][x] + bg_channels[2][x] * (1.0 - fga);
                expected_a[x] = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]);
            }

            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA);
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA);
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA);
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA);
        }

        #[test]
        fn test_straight_alpha_new_a_zero() {
            // fg_a (clamped or not) = 0, bg_a = 0  => new_a = 1 - (1-0)*(1-0) = 0
            let bg_r_data = [0.1f32];
            let bg_g_data = [0.2f32];
            let bg_b_data = [0.3f32];
            let bg_a_data = [0.0f32];
            let fg_r_data = [0.7f32];
            let fg_g_data = [0.6f32];
            let fg_b_data = [0.5f32];
            let fg_a_data = [0.0f32];

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let mut out_channels = vec![
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
            ];

            let alpha_is_premultiplied = false;
            let clamp_alpha = true; // or false, result is same for fg_a = 0.0
            let num_pixels = 1;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                }; // fga = 0.0
                let new_a = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]); // new_a = 0.0
                let rnew_a = if new_a > 1e-9 { 1.0 / new_a } else { 0.0 }; // rnew_a = 0.0

                // Numerator: (fg_r[x] * 0.0 + bg_r[x] * bg_a[x] * (1.0 - 0.0))
                //            = bg_r[x] * bg_a[x]
                // In this case, bg_a[x] is 0.0, so numerator is 0.0
                // Result is 0.0 * 0.0 = 0.0
                expected_r[x] = (fg_channels[0][x] * fga
                    + bg_channels[0][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_g[x] = (fg_channels[1][x] * fga
                    + bg_channels[1][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_b[x] = (fg_channels[2][x] * fga
                    + bg_channels[2][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_a[x] = new_a;
            }

            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA); // Expected [0.0]
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA); // Expected [0.0]
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA); // Expected [0.0]
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA); // Expected [0.0]
        }

        #[test]
        fn test_straight_alpha_fg_fully_opaque() {
            // fg_a (clamped or not) = 1.0 => new_a = 1.0 - (1.0 - 1.0) * (1.0 - bg_a[x]) = 1.0
            let bg_r_data = [0.1f32];
            let bg_g_data = [0.2f32];
            let bg_b_data = [0.3f32];
            let bg_a_data = [0.5f32];
            let fg_r_data = [0.7f32];
            let fg_g_data = [0.6f32];
            let fg_b_data = [0.5f32];
            let fg_a_data = [1.0f32];

            let bg_channels = vec![
                bg_r_data.to_vec(),
                bg_g_data.to_vec(),
                bg_b_data.to_vec(),
                bg_a_data.to_vec(),
            ];
            let fg_channels = vec![
                fg_r_data.to_vec(),
                fg_g_data.to_vec(),
                fg_b_data.to_vec(),
                fg_a_data.to_vec(),
            ];
            let mut out_channels = vec![
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
                vec![0.0f32; 1],
            ];

            let alpha_is_premultiplied = false;
            let clamp_alpha = true; // or false, result is same for fg_a = 1.0
            let num_pixels = 1;

            perform_alpha_blending_layers(
                &bg_channels,
                &fg_channels,
                alpha_is_premultiplied,
                clamp_alpha,
                &mut out_channels,
            );

            let mut expected_r = vec![0.0f32; num_pixels];
            let mut expected_g = vec![0.0f32; num_pixels];
            let mut expected_b = vec![0.0f32; num_pixels];
            let mut expected_a = vec![0.0f32; num_pixels];

            for x in 0..num_pixels {
                let fga = if clamp_alpha {
                    clamp(fg_channels[3][x])
                } else {
                    fg_channels[3][x]
                }; // fga = 1.0
                let new_a = 1.0 - (1.0 - fga) * (1.0 - bg_channels[3][x]); // new_a = 1.0
                let rnew_a = if new_a > 1e-9 { 1.0 / new_a } else { 0.0 }; // rnew_a = 1.0

                // Numerator: (fg_r[x] * 1.0 + bg_r[x] * bg_a[x] * (1.0 - 1.0))
                //            = fg_r[x]
                // Result is fg_r[x] * 1.0 = fg_r[x]
                expected_r[x] = (fg_channels[0][x] * fga
                    + bg_channels[0][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_g[x] = (fg_channels[1][x] * fga
                    + bg_channels[1][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_b[x] = (fg_channels[2][x] * fga
                    + bg_channels[2][x] * bg_channels[3][x] * (1.0 - fga))
                    * rnew_a;
                expected_a[x] = new_a;
            }
            // Expected R,G,B should be fg_r, fg_g, fg_b; Expected A should be 1.0
            assert_all_almost_eq!(out_channels[0], expected_r, MAX_DELTA); // Expected [0.7]
            assert_all_almost_eq!(out_channels[1], expected_g, MAX_DELTA); // Expected [0.6]
            assert_all_almost_eq!(out_channels[2], expected_b, MAX_DELTA); // Expected [0.5]
            assert_all_almost_eq!(out_channels[3], expected_a, MAX_DELTA); // Expected [1.0]
        }

        #[test]
        fn test_bg_equals_bga_and_fg_equals_fga_clamp_true() {
            let bg_data = [0.1, 0.8, 0.5]; // bg and bga are the same
            let fg_data = [0.9, 0.2, 1.7]; // fg and fga are the same, will be clamped
            let num_pixels = bg_data.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            let alpha_is_premultiplied = false; // This flag is irrelevant in this branch

            for i in 0..num_pixels {
                let fa = clamp(fg_data[i]);
                let bga_val = bg_data[i];
                expected_out[i] = 1.0 - (1.0 - fa) * (1.0 - bga_val);
            }

            perform_alpha_blending(
                &bg_data,
                &bg_data, // bga is same as bg
                &fg_data,
                &fg_data, // fga is same as fg
                alpha_is_premultiplied,
                true,
                &mut out,
            );

            assert_all_almost_eq!(out, expected_out, 1e-6);
        }
    }

    mod perform_alpha_blending_tests {
        use super::{super::*, *};
        use crate::util::test::assert_all_almost_eq;
        use test_log::test;

        #[test]
        fn test_bg_equals_bga_fg_equals_fga_clamp_true() {
            let bg_bga = [0.1, 0.8, 1.2]; // Background color and alpha are the same
            let fg_fga = [0.9, 0.2, -0.3]; // Foreground color and alpha are the same
            let num_pixels = bg_bga.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            // Calculate expected output
            for x in 0..num_pixels {
                let fa = clamp(fg_fga[x]);
                // The spec is unclear about which alphas should be clamped, and the C++ version
                // only clamps fg.
                expected_out[x] = 1.0 - (1.0 - fa) * (1.0 - bg_bga[x]);
            }

            perform_alpha_blending(
                &bg_bga, // bg
                &bg_bga, // bga (same pointer simulated by same slice)
                &fg_fga, // fg
                &fg_fga, // fga (same pointer simulated by same slice)
                false,   // alpha_is_premultiplied (doesn't matter for this branch)
                true,    // clamp
                &mut out,
            );

            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_bg_equals_bga_fg_equals_fga_clamp_false() {
            let bg_bga = [0.1, 0.8, 0.5];
            let fg_fga = [0.9, 0.2, 0.6];
            let num_pixels = bg_bga.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = fg_fga[x]; // No clamping
                let bga_val = bg_bga[x]; // No clamping
                expected_out[x] = 1.0 - (1.0 - fa) * (1.0 - bga_val);
            }

            perform_alpha_blending(
                &bg_bga, &bg_bga, &fg_fga, &fg_fga, false, false, // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_premultiplied_clamp_true() {
            let bg = [0.1, 0.2, 0.3];
            let bga = [0.8, 0.7, 0.9];
            let fg = [0.4, 0.5, 0.6]; // Premultiplied, so fg <= fga often
            let fga = [0.9, 0.6, 1.2]; // fga[2] > 1.0, will be clamped
            let num_pixels = bg.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = clamp(fga[x]);
                // The spec is unclear about which alphas should be clamped, and the C++ version
                // only clamps fg.
                expected_out[x] = fg[x] + bg[x] * (1.0 - fa);
            }

            perform_alpha_blending(
                &bg, &bga, &fg, &fga, true, // alpha_is_premultiplied
                true, // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_premultiplied_clamp_false() {
            let bg = [0.1, 0.2, 0.3];
            let bga = [0.8, 0.7, 0.9];
            let fg = [0.4, 0.5, 0.6];
            let fga = [0.9, 0.6, 1.5]; // No clamping, fga values are within [0,1] but could be anything
            let num_pixels = bg.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = fga[x]; // No clamping
                expected_out[x] = fg[x] + bg[x] * (1.0 - fa);
            }

            perform_alpha_blending(
                &bg, &bga, &fg, &fga, true,  // alpha_is_premultiplied
                false, // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_not_premultiplied_clamp_true() {
            let bg = [0.1, 0.2, 0.3]; // Straight background color
            let bga = [0.8, 0.7, 1.3]; // Background alpha, bga[2] > 1.0 (C++ doesn't clamp bga here)
            let fg = [0.4, 0.5, 0.6]; // Straight foreground color
            let fga = [0.9, 0.6, -0.2]; // Foreground alpha, fga[2] < 0.0, will be clamped
            let num_pixels = bg.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = clamp(fga[x]);
                // C++ uses bga[x] directly, without clamping it in this path.
                let current_bga = bga[x];
                let new_a = 1.0 - (1.0 - fa) * (1.0 - current_bga);
                let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
                expected_out[x] = (fg[x] * fa + bg[x] * current_bga * (1.0 - fa)) * rnew_a;
            }

            perform_alpha_blending(
                &bg, &bga, &fg, &fga, false, // alpha_is_premultiplied
                true,  // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_not_premultiplied_clamp_false() {
            let bg = [0.1, 0.2, 0.3];
            let bga = [0.8, 0.7, 0.9];
            let fg = [0.4, 0.5, 0.6];
            let fga = [0.9, 0.6, -0.5]; // No clamping
            let num_pixels = bg.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = fga[x]; // No clamping
                let current_bga = bga[x]; // No clamping
                let new_a = 1.0 - (1.0 - fa) * (1.0 - current_bga);
                let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
                expected_out[x] = (fg[x] * fa + bg[x] * current_bga * (1.0 - fa)) * rnew_a;
            }

            perform_alpha_blending(
                &bg, &bga, &fg, &fga, false, // alpha_is_premultiplied
                false, // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_not_premultiplied_clamp_false_new_a_zero() {
            // Test with the scenario where new_a can be zero:
            let bg_n_zero = [0.1, 0.9, 0.3]; // bg color for the zero alpha case
            let bga_n_zero = [0.8, 0.0, 0.9]; // bga is 0 for the second pixel
            let fg_n_zero = [0.4, 0.9, 0.6]; // fg color for the zero alpha case
            let fga_n_zero = [0.9, 0.0, 0.5]; // fa is 0 for the second pixel
            let mut out_n_zero = vec![0.0; 3];
            let mut expected_out_n_zero = vec![0.0; 3];

            for x in 0..3 {
                let fa = fga_n_zero[x];
                let current_bga = bga_n_zero[x];
                let new_a = 1.0 - (1.0 - fa) * (1.0 - current_bga);
                let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
                expected_out_n_zero[x] =
                    (fg_n_zero[x] * fa + bg_n_zero[x] * current_bga * (1.0 - fa)) * rnew_a;
            }

            perform_alpha_blending(
                &bg_n_zero,
                &bga_n_zero,
                &fg_n_zero,
                &fga_n_zero,
                false, // alpha_is_premultiplied
                false, // clamp
                &mut out_n_zero,
            );
            assert_all_almost_eq!(out_n_zero, expected_out_n_zero, MAX_DELTA);
        }

        #[test]
        fn test_all_same_slice_data_optimized_path_clamp_true() {
            let data = [0.1, 0.8, 1.2, 0.9, 0.2, -0.3];
            let common_bg_bga = &[data[0], data[1], data[2]]; // Represents bg and bga
            let common_fg_fga = &[data[3], data[4], data[5]]; // Represents fg and fga
            let num_pixels = common_bg_bga.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = clamp(common_fg_fga[x]); // fg_fga[x] is fga[x] in this path
                let bga_val = common_bg_bga[x]; // bg_bga[x] is bga[x] in this path
                expected_out[x] = 1.0 - (1.0 - fa) * (1.0 - bga_val);
            }

            // Call Rust function: bg and bga are the same slice, fg and fga are the same slice
            perform_alpha_blending(
                common_bg_bga, // bg
                common_bg_bga, // bga
                common_fg_fga, // fg
                common_fg_fga, // fga
                false,         // alpha_is_premultiplied (irrelevant for this branch)
                true,          // clamp
                &mut out,
            );

            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        #[test]
        fn test_all_same_slice_data_optimized_path_clamp_false() {
            let data = [0.1, 0.8, 0.5, 0.9, 0.2, 0.6];
            let common_bg_bga = &[data[0], data[1], data[2]];
            let common_fg_fga = &[data[3], data[4], data[5]];
            let num_pixels = common_bg_bga.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for x in 0..num_pixels {
                let fa = common_fg_fga[x];
                let bga_val = common_bg_bga[x];
                expected_out[x] = 1.0 - (1.0 - fa) * (1.0 - bga_val);
            }

            perform_alpha_blending(
                common_bg_bga,
                common_bg_bga,
                common_fg_fga,
                common_fg_fga,
                false,
                false, // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        // Consider edge case values for alpha that might cause issues.
        #[test]
        fn test_not_premultiplied_edge_alphas_clamp_true() {
            let bg = [0.5, 0.5, 0.5, 0.5];
            let bga = [1.0, 0.0, 0.5, 1.1]; // bga[3] > 1.0 (C++ doesn't clamp bga here)
            let fg = [0.2, 0.2, 0.2, 0.2];
            let fga = [1.0, 0.0, -0.1, 1.2]; // fga[2] < 0 will be clamped to 0, fga[3] > 1 will be clamped to 1
            let num_pixels = bg.len();
            let mut out = vec![0.0; num_pixels];
            let mut expected_out = vec![0.0; num_pixels];

            for i in 0..4 {
                expected_out[i] = {
                    let fa = clamp(fga[i]);
                    let c_bga = bga[i];
                    let new_a = 1.0 - (1.0 - fa) * (1.0 - c_bga);
                    let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
                    (fg[i] * fa + bg[i] * c_bga * (1.0 - fa)) * rnew_a
                };
            }
            perform_alpha_blending(
                &bg, &bga, &fg, &fga, false, // alpha_is_premultiplied
                true,  // clamp
                &mut out,
            );
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }

        // Test with empty slices
        #[test]
        fn test_empty_slices() {
            let bg: [f32; 0] = [];
            let bga: [f32; 0] = [];
            let fg: [f32; 0] = [];
            let fga: [f32; 0] = [];
            let mut out: [f32; 0] = [];
            let expected_out: [f32; 0] = [];

            // Test all branches with empty inputs
            perform_alpha_blending(&bg, &bga, &fg, &fga, false, true, &mut out); // bg==bga, fg==fga implicit, clamp=true
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);

            perform_alpha_blending(&bg, &bga, &fg, &fga, false, false, &mut out); // bg==bga, fg==fga implicit, clamp=false
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);

            let bga_diff = [0.0]; // To make pointers different for the non-optimized path
            perform_alpha_blending(&bg, &bga_diff[0..0], &fg, &fga, true, true, &mut out); // premultiplied, clamp=true
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);

            perform_alpha_blending(&bg, &bga_diff[0..0], &fg, &fga, true, false, &mut out); // premultiplied, clamp=false
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);

            perform_alpha_blending(&bg, &bga_diff[0..0], &fg, &fga, false, true, &mut out); // not premultiplied, clamp=true
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);

            perform_alpha_blending(&bg, &bga_diff[0..0], &fg, &fga, false, false, &mut out); // not premultiplied, clamp=false
            assert_all_almost_eq!(out, expected_out, MAX_DELTA);
        }
    }

    mod simple_blending_functions_test {
        use super::{super::*, *};
        use crate::util::test::assert_all_almost_eq;
        use test_log::test;

        const K_SMALL_ALPHA: f32 = 1.0f32 / (1u32 << 26) as f32; // Equivalent to C++ kSmallAlpha

        // --- Tests for perform_alpha_weighted_add ---

        #[test]
        fn test_alpha_weighted_add_fg_is_fga_equivalent() {
            let bg = &[0.1f32, 0.5f32, 1.0f32];
            let shared_fg_fga = &[0.2f32, 0.3f32, 0.4f32];
            let mut out = vec![0.0f32; bg.len()];

            // Expected behavior: out = bg, regardless of the clamp_alpha value
            let expected_out: Vec<f32> = bg.to_vec();

            // Test with clamp_alpha = false
            perform_alpha_weighted_add(bg, shared_fg_fga, shared_fg_fga, false, &mut out);
            assert_all_almost_eq!(&out, &expected_out, MAX_DELTA);

            // Test with clamp_alpha = true (should still be out = bg)
            let mut out_clamped = vec![0.0f32; bg.len()];
            perform_alpha_weighted_add(bg, shared_fg_fga, shared_fg_fga, true, &mut out_clamped);
            assert_all_almost_eq!(&out_clamped, &expected_out, MAX_DELTA);
        }

        #[test]
        fn test_alpha_weighted_add_clamp_alpha() {
            let bg = &[0.1f32, 0.5f32, 0.2f32];
            let fg = &[0.2f32, 0.3f32, 0.8f32];
            let fga = &[-0.5f32, 0.5f32, 1.5f32]; // Values that will be clamped
            let mut out = vec![0.0f32; bg.len()];
            let clamp_alpha_param = true;

            let mut expected_out = vec![0.0f32; bg.len()];
            for i in 0..bg.len() {
                expected_out[i] = bg[i] + fg[i] * clamp(fga[i]);
            }

            perform_alpha_weighted_add(bg, fg, fga, clamp_alpha_param, &mut out);
            assert_all_almost_eq!(&out, &expected_out, MAX_DELTA);
        }

        #[test]
        fn test_alpha_weighted_add_no_clamp_alpha() {
            let bg = &[0.1f32, 0.5f32, 0.2f32];
            let fg = &[0.2f32, 0.3f32, 0.8f32];
            let fga = &[-0.5f32, 0.5f32, 1.5f32]; // Values will not be clamped
            let mut out = vec![0.0f32; bg.len()];
            let clamp_alpha_param = false;

            let mut expected_out = vec![0.0f32; bg.len()];
            for i in 0..bg.len() {
                expected_out[i] = bg[i] + fg[i] * fga[i];
            }

            perform_alpha_weighted_add(bg, fg, fga, clamp_alpha_param, &mut out);
            assert_all_almost_eq!(&out, &expected_out, MAX_DELTA);
        }

        #[test]
        fn test_alpha_weighted_add_empty_slices() {
            let bg_empty: &[f32] = &[];
            let fg_empty: &[f32] = &[];
            let fga_empty: &[f32] = &[];
            let mut out_empty: Vec<f32> = vec![];
            let expected_empty: &[f32] = &[];

            perform_alpha_weighted_add(bg_empty, fg_empty, fga_empty, true, &mut out_empty);
            assert_all_almost_eq!(&out_empty, expected_empty, MAX_DELTA);

            perform_alpha_weighted_add(bg_empty, fg_empty, fga_empty, false, &mut out_empty);
            assert_all_almost_eq!(&out_empty, expected_empty, MAX_DELTA);

            // Test fg == fga equivalent case for empty slices
            perform_alpha_weighted_add(bg_empty, fg_empty, fg_empty, false, &mut out_empty);
            assert_all_almost_eq!(&out_empty, expected_empty, MAX_DELTA);
        }

        // --- Tests for perform_mul_blending ---

        #[test]
        fn test_mul_blending_clamp_fg() {
            let bg = &[0.1f32, 0.5f32, 1.0f32, 0.8f32];
            let fg = &[-0.5f32, 0.5f32, 1.5f32, 0.2f32]; // Some values will be clamped
            let mut out = vec![0.0f32; bg.len()];
            // In C++, this clamps `fg[x]`. So `clamp_alpha` in Rust fn should control this.
            let clamp_fg_param = true;

            let mut expected_out = vec![0.0f32; bg.len()];
            for i in 0..bg.len() {
                expected_out[i] = bg[i] * clamp(fg[i]);
            }

            perform_mul_blending(bg, fg, clamp_fg_param, &mut out);
            assert_all_almost_eq!(&out, &expected_out, MAX_DELTA);
        }

        #[test]
        fn test_mul_blending_no_clamp_fg() {
            let bg = &[0.1f32, 0.5f32, 1.0f32, 0.8f32];
            let fg = &[-0.5f32, 0.5f32, 1.5f32, 0.2f32]; // Values will not be clamped
            let mut out = vec![0.0f32; bg.len()];
            let clamp_fg_param = false;

            let mut expected_out = vec![0.0f32; bg.len()];
            for i in 0..bg.len() {
                expected_out[i] = bg[i] * fg[i];
            }

            perform_mul_blending(bg, fg, clamp_fg_param, &mut out);
            assert_all_almost_eq!(&out, &expected_out, MAX_DELTA);
        }

        #[test]
        fn test_mul_blending_empty_slices() {
            let bg_empty: &[f32] = &[];
            let fg_empty: &[f32] = &[];
            let mut out_empty: Vec<f32> = vec![];
            let expected_empty: &[f32] = &[];

            perform_mul_blending(bg_empty, fg_empty, true, &mut out_empty);
            assert_all_almost_eq!(&out_empty, expected_empty, MAX_DELTA);

            perform_mul_blending(bg_empty, fg_empty, false, &mut out_empty);
            assert_all_almost_eq!(&out_empty, expected_empty, MAX_DELTA);
        }

        // --- Tests for premultiply_alpha ---

        #[test]
        fn test_premultiply_alpha_various_cases() {
            let mut r_actual = vec![0.8f32, 0.6f32, 0.4f32, 0.2f32, 1.0f32, 0.7f32];
            let mut g_actual = vec![0.7f32, 0.5f32, 0.3f32, 0.1f32, 0.9f32, 0.6f32];
            let mut b_actual = vec![0.9f32, 0.8f32, 0.2f32, 0.0f32, 0.5f32, 0.5f32];
            let a_input = &[
                0.5f32,
                K_SMALL_ALPHA,
                0.0f32,
                1.0f32,
                K_SMALL_ALPHA - 1e-9f32,
                K_SMALL_ALPHA + 1e-9f32,
            ];
            // Cases: alpha > kSA, alpha == kSA, alpha == 0 (<kSA), alpha == 1.0, alpha < kSA (non-zero), alpha > kSA (just barely)

            let mut expected_r = r_actual.clone();
            let mut expected_g = g_actual.clone();
            let mut expected_b = b_actual.clone();

            for i in 0..a_input.len() {
                let multiplier = K_SMALL_ALPHA.max(a_input[i]);
                expected_r[i] *= multiplier;
                expected_g[i] *= multiplier;
                expected_b[i] *= multiplier;
            }

            premultiply_alpha(&mut r_actual, &mut g_actual, &mut b_actual, a_input);
            assert_all_almost_eq!(&r_actual, &expected_r, MAX_DELTA);
            assert_all_almost_eq!(&g_actual, &expected_g, MAX_DELTA);
            assert_all_almost_eq!(&b_actual, &expected_b, MAX_DELTA);
        }

        #[test]
        fn test_premultiply_alpha_empty_slices() {
            let mut r_empty: Vec<f32> = vec![];
            let mut g_empty: Vec<f32> = vec![];
            let mut b_empty: Vec<f32> = vec![];
            let a_empty: &[f32] = &[];

            let expected_r_empty: &[f32] = &[];
            let expected_g_empty: &[f32] = &[];
            let expected_b_empty: &[f32] = &[];

            premultiply_alpha(&mut r_empty, &mut g_empty, &mut b_empty, a_empty);
            assert_all_almost_eq!(&r_empty, expected_r_empty, MAX_DELTA);
            assert_all_almost_eq!(&g_empty, expected_g_empty, MAX_DELTA);
            assert_all_almost_eq!(&b_empty, expected_b_empty, MAX_DELTA);
        }

        // --- Tests for unpremultiply_alpha ---

        #[test]
        fn test_unpremultiply_alpha_various_cases() {
            // Initial (non-premultiplied) values
            let initial_rgb_alphas = [
                (0.8f32, 0.7f32, 0.9f32, 0.5f32),
                (0.4f32, 0.3f32, 0.2f32, 1e-5f32),
                (0.2f32, 0.1f32, 0.0f32, 1.0f32),
            ];

            let mut r_premultiplied_actual = Vec::new();
            let mut g_premultiplied_actual = Vec::new();
            let mut b_premultiplied_actual = Vec::new();
            let mut a_input = Vec::new();

            let mut expected_r_unpremultiplied = Vec::new();
            let mut expected_g_unpremultiplied = Vec::new();
            let mut expected_b_unpremultiplied = Vec::new();

            for (r_init, g_init, b_init, alpha_val) in initial_rgb_alphas.iter().cloned() {
                let premul_multiplier = K_SMALL_ALPHA.max(alpha_val);
                r_premultiplied_actual.push(r_init * premul_multiplier);
                g_premultiplied_actual.push(g_init * premul_multiplier);
                b_premultiplied_actual.push(b_init * premul_multiplier);
                a_input.push(alpha_val);

                // Expected values after unpremultiplication should be the initial values
                // We calculate them using the C++ logic for unpremultiplication from the premultiplied values.
                let unpremul_multiplier = 1.0f32 / K_SMALL_ALPHA.max(alpha_val);
                expected_r_unpremultiplied.push((r_init * premul_multiplier) * unpremul_multiplier);
                expected_g_unpremultiplied.push((g_init * premul_multiplier) * unpremul_multiplier);
                expected_b_unpremultiplied.push((b_init * premul_multiplier) * unpremul_multiplier);
            }

            unpremultiply_alpha(
                &mut r_premultiplied_actual,
                &mut g_premultiplied_actual,
                &mut b_premultiplied_actual,
                &a_input,
            );

            assert_all_almost_eq!(
                &r_premultiplied_actual,
                &expected_r_unpremultiplied,
                MAX_DELTA
            );
            assert_all_almost_eq!(
                &g_premultiplied_actual,
                &expected_g_unpremultiplied,
                MAX_DELTA
            );
            assert_all_almost_eq!(
                &b_premultiplied_actual,
                &expected_b_unpremultiplied,
                MAX_DELTA
            );

            // Also verify they are close to the original initial values directly
            let initial_r_values: Vec<f32> =
                initial_rgb_alphas.iter().map(|(r, _, _, _)| *r).collect();
            let initial_g_values: Vec<f32> =
                initial_rgb_alphas.iter().map(|(_, g, _, _)| *g).collect();
            let initial_b_values: Vec<f32> =
                initial_rgb_alphas.iter().map(|(_, _, b, _)| *b).collect();
            assert_all_almost_eq!(&r_premultiplied_actual, &initial_r_values, MAX_DELTA);
            assert_all_almost_eq!(&g_premultiplied_actual, &initial_g_values, MAX_DELTA);
            assert_all_almost_eq!(&b_premultiplied_actual, &initial_b_values, MAX_DELTA);
        }

        #[test]
        fn test_unpremultiply_alpha_empty_slices() {
            let mut r_empty: Vec<f32> = vec![];
            let mut g_empty: Vec<f32> = vec![];
            let mut b_empty: Vec<f32> = vec![];
            let a_empty: &[f32] = &[];

            let expected_r_empty: &[f32] = &[];
            let expected_g_empty: &[f32] = &[];
            let expected_b_empty: &[f32] = &[];

            unpremultiply_alpha(&mut r_empty, &mut g_empty, &mut b_empty, a_empty);
            assert_all_almost_eq!(&r_empty, expected_r_empty, MAX_DELTA);
            assert_all_almost_eq!(&g_empty, expected_g_empty, MAX_DELTA);
            assert_all_almost_eq!(&b_empty, expected_b_empty, MAX_DELTA);
        }
    }

    mod perform_blending_tests {
        use super::{super::*, *};
        use crate::{headers::bit_depth::BitDepth, util::test::assert_all_almost_eq};
        use test_log::test;

        const ABS_DELTA: f32 = 1e-6;

        // Helper for expected value calculations based on C++ logic

        // Alpha compositing formula: Ao = As + Ab * (1 - As)
        // Used for kBlend modes for the alpha channel itself.
        fn expected_alpha_blend(fg_a: f32, bg_a: f32) -> f32 {
            fg_a + bg_a * (1.0 - fg_a)
        }

        // Color compositing for kBlend, premultiplied alpha: Co = Cs_premult + Cb_premult * (1 - As)
        fn expected_color_blend_premultiplied(c_fg: f32, c_bg: f32, fg_a: f32) -> f32 {
            c_fg + c_bg * (1.0 - fg_a)
        }

        // Color compositing for kBlend, non-premultiplied alpha: Co = (Cs * As + Cb * Ab * (1 - As)) / Ao_blend
        fn expected_color_blend_non_premultiplied(
            c_fg: f32,
            fg_a: f32, // Foreground color and its alpha
            c_bg: f32,
            bg_a: f32,            // Background color and its alpha
            alpha_blend_out: f32, // The resulting alpha from expected_alpha_blend(fg_a, bg_a)
        ) -> f32 {
            if alpha_blend_out.abs() < ABS_DELTA {
                // Avoid division by zero
                0.0
            } else {
                (c_fg * fg_a + c_bg * bg_a * (1.0 - fg_a)) / alpha_blend_out
            }
        }

        // For kAlphaWeightedAdd modes: Co = Cb + Cs * As
        fn expected_alpha_weighted_add(c_bg: f32, c_fg: f32, fg_a: f32) -> f32 {
            c_bg + c_fg * fg_a
        }

        // For kMul mode: Co = Cb * Cs
        fn expected_mul_blend(c_bg: f32, c_fg: f32) -> f32 {
            c_bg * c_fg
        }

        #[test]
        fn test_color_replace_fg_over_bg() {
            let bg_r = [0.1];
            let bg_g = [0.2];
            let bg_b = [0.3];
            let fg_r = [0.7];
            let fg_g = [0.8];
            let fg_b = [0.9];

            let bg_channels: [&[f32]; 3] = [&bg_r, &bg_g, &bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_channels: [&mut [f32]; 3] = [&mut out_r, &mut out_g, &mut out_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Replace,
                alpha_channel: 0, // Not used for Replace
                clamp: false,
            };

            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            // Expected: output color is fg color
            assert_all_almost_eq!(&out_r, &fg_r, ABS_DELTA);
            assert_all_almost_eq!(&out_g, &fg_g, ABS_DELTA);
            assert_all_almost_eq!(&out_b, &fg_b, ABS_DELTA);
        }

        #[test]
        fn test_color_add() {
            let bg_r = [0.1];
            let bg_g = [0.2];
            let bg_b = [0.3];
            let fg_r = [0.7];
            let fg_g = [0.6];
            let fg_b = [0.5];

            let bg_channels: [&[f32]; 3] = [&bg_r, &bg_g, &bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_channels: [&mut [f32]; 3] = [&mut out_r, &mut out_g, &mut out_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Add,
                alpha_channel: 0, // Not used
                clamp: false,
            };
            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            let expected_r = [bg_r[0] + fg_r[0]];
            let expected_g = [bg_g[0] + fg_g[0]];
            let expected_b = [bg_b[0] + fg_b[0]];

            assert_all_almost_eq!(&out_r, &expected_r, ABS_DELTA);
            assert_all_almost_eq!(&out_g, &expected_g, ABS_DELTA);
            assert_all_almost_eq!(&out_b, &expected_b, ABS_DELTA);
        }

        #[test]
        fn test_color_blend_above_premultiplied_alpha() {
            // BG: R=0.1, G=0.2, B=0.3, A=0.8 (premultiplied)
            // FG: R=0.4, G=0.3, B=0.2, A=0.5 (premultiplied)
            let bg_r = [0.1];
            let bg_g = [0.2];
            let bg_b = [0.3];
            let bg_a = [0.8];
            let fg_r = [0.4];
            let fg_g = [0.3];
            let fg_b = [0.2];
            let fg_a = [0.5];

            let bg_channels: [&[f32]; 4] = [&bg_r, &bg_g, &bg_b, &bg_a];
            let fg_channels: [&[f32]; 4] = [&fg_r, &fg_g, &fg_b, &fg_a];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_a = [0.0];
            let mut out_channels: [&mut [f32]; 4] =
                [&mut out_r, &mut out_g, &mut out_b, &mut out_a];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::BlendAbove,
                alpha_channel: 0, // EC0 is the alpha channel
                clamp: false,
            };
            // EC0 (alpha) blending rule.
            // For BlendAbove color mode, the alpha channel itself is also blended using source-over.
            // So this ec_blending rule for alpha will be effectively overridden by color blending's alpha calculation.
            let ec_blending = [PatchBlending {
                mode: PatchBlendMode::Replace, // Arbitrary, will be overwritten by color blend
                alpha_channel: 0,
                clamp: false,
            }];
            let extra_channel_info = [ExtraChannelInfo::new(
                false,
                ExtraChannel::Alpha,
                BitDepth::f32(), // Assuming f32
                0,
                "alpha".to_string(),
                true, // alpha_associated = true (premultiplied)
                None,
                None,
            )];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            let fga = fg_a[0]; // Not clamped
            let bga = bg_a[0];

            // Expected alpha: Ao = Afg + Abg * (1 - Afg)
            let expected_a_val = expected_alpha_blend(fga, bga);
            // Expected color: Co = Cfg_premult + Cbg_premult * (1 - Afg)
            let expected_r_val = expected_color_blend_premultiplied(fg_r[0], bg_r[0], fga);
            let expected_g_val = expected_color_blend_premultiplied(fg_g[0], bg_g[0], fga);
            let expected_b_val = expected_color_blend_premultiplied(fg_b[0], bg_b[0], fga);

            assert_all_almost_eq!(&out_a, &[expected_a_val], ABS_DELTA);
            assert_all_almost_eq!(&out_r, &[expected_r_val], ABS_DELTA);
            assert_all_almost_eq!(&out_g, &[expected_g_val], ABS_DELTA);
            assert_all_almost_eq!(&out_b, &[expected_b_val], ABS_DELTA);
        }

        #[test]
        fn test_color_blend_above_non_premultiplied_alpha() {
            // BG: R=0.1, G=0.2, B=0.3 (unpremult), A=0.8
            // FG: R=0.7, G=0.6, B=0.5 (unpremult), A=0.5
            let bg_r = [0.1];
            let bg_g = [0.2];
            let bg_b = [0.3];
            let bg_a = [0.8];
            let fg_r = [0.7];
            let fg_g = [0.6];
            let fg_b = [0.5];
            let fg_a = [0.5];

            let bg_channels: [&[f32]; 4] = [&bg_r, &bg_g, &bg_b, &bg_a];
            let fg_channels: [&[f32]; 4] = [&fg_r, &fg_g, &fg_b, &fg_a];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_a = [0.0];
            let mut out_channels: [&mut [f32]; 4] =
                [&mut out_r, &mut out_g, &mut out_b, &mut out_a];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::BlendAbove,
                alpha_channel: 0, // EC0
                clamp: false,
            };
            let ec_blending = [PatchBlending {
                // This will be overwritten for the alpha channel by color blending
                mode: PatchBlendMode::Replace,
                alpha_channel: 0,
                clamp: false,
            }];
            let extra_channel_info = [ExtraChannelInfo::new(
                false,
                ExtraChannel::Alpha,
                BitDepth::f32(),
                0,
                "alpha".to_string(),
                false, // alpha_associated = false (non-premultiplied)
                None,
                None,
            )];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            let fga = fg_a[0];
            let bga = bg_a[0];

            // Expected alpha: Ao = Afg + Abg * (1 - Afg)
            let expected_a_val = expected_alpha_blend(fga, bga);
            // Expected color: Co = (Cfg_unpremult * Afg + Cbg_unpremult * Abg * (1 - Afg)) / Ao_blend
            let expected_r_val =
                expected_color_blend_non_premultiplied(fg_r[0], fga, bg_r[0], bga, expected_a_val);
            let expected_g_val =
                expected_color_blend_non_premultiplied(fg_g[0], fga, bg_g[0], bga, expected_a_val);
            let expected_b_val =
                expected_color_blend_non_premultiplied(fg_b[0], fga, bg_b[0], bga, expected_a_val);

            assert_all_almost_eq!(&out_a, &[expected_a_val], ABS_DELTA);
            assert_all_almost_eq!(&out_r, &[expected_r_val], ABS_DELTA);
            assert_all_almost_eq!(&out_g, &[expected_g_val], ABS_DELTA);
            assert_all_almost_eq!(&out_b, &[expected_b_val], ABS_DELTA);
        }

        #[test]
        fn test_color_alpha_weighted_add_above() {
            let bg_r = [0.1];
            let bg_g = [0.2];
            let bg_b = [0.3];
            let bg_a = [0.8]; // bg alpha used by ec_blending
            let fg_r = [0.7];
            let fg_g = [0.6];
            let fg_b = [0.5];
            let fg_a = [0.5]; // fg alpha used for weighting

            let bg_channels: [&[f32]; 4] = [&bg_r, &bg_g, &bg_b, &bg_a];
            let fg_channels: [&[f32]; 4] = [&fg_r, &fg_g, &fg_b, &fg_a];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_a = [0.0];
            let mut out_channels: [&mut [f32]; 4] =
                [&mut out_r, &mut out_g, &mut out_b, &mut out_a];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::AlphaWeightedAddAbove,
                alpha_channel: 0, // EC0 is alpha
                clamp: false,
            };
            // For AlphaWeightedAdd color mode, the alpha channel (EC0) value is determined by its ec_blending rule.
            // Let's make EC0 blend itself using BlendAbove mode.
            let ec_blending = [PatchBlending {
                mode: PatchBlendMode::BlendAbove, // Alpha channel EC0 blends itself
                alpha_channel: 0,                 //  using itself as alpha reference
                clamp: false,
            }];
            let extra_channel_info = [ExtraChannelInfo::new(
                false,
                ExtraChannel::Alpha,
                BitDepth::f32(),
                0,
                "alpha".to_string(),
                true, // alpha_associated (doesn't directly affect AWA color, but affects EC0's own blend)
                None,
                None,
            )];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            let fga_for_weighting = fg_a[0]; // Not clamped as color_blending.clamp is false

            // Expected color: Co = Cbg + Cfg * Afg_for_weighting
            let expected_r_val = expected_alpha_weighted_add(bg_r[0], fg_r[0], fga_for_weighting);
            let expected_g_val = expected_alpha_weighted_add(bg_g[0], fg_g[0], fga_for_weighting);
            let expected_b_val = expected_alpha_weighted_add(bg_b[0], fg_b[0], fga_for_weighting);

            // Expected alpha (EC0): Blended according to ec_blending[0].
            // Mode is BlendAbove, alpha_channel is 0 (itself).
            // C++: PerformAlphaBlending(bg[3+0], bg[3+0], fg[3+0], fg[3+0], tmp.Row(3+0), ...)
            // This means it's the "alpha channel blends itself" case: Ao = Afg + Abg * (1 - Afg)
            // fg_alpha_for_ec0 = fg_a[0], bg_alpha_for_ec0 = bg_a[0]. ec_blending[0].clamp is false.
            let expected_a_val = expected_alpha_blend(fg_a[0], bg_a[0]);

            assert_all_almost_eq!(&out_r, &[expected_r_val], ABS_DELTA);
            assert_all_almost_eq!(&out_g, &[expected_g_val], ABS_DELTA);
            assert_all_almost_eq!(&out_b, &[expected_b_val], ABS_DELTA);
            assert_all_almost_eq!(&out_a, &[expected_a_val], ABS_DELTA);
        }

        #[test]
        fn test_color_mul_with_clamp() {
            let bg_r = [0.5];
            let bg_g = [0.8];
            let bg_b = [1.0];
            let fg_r = [1.5];
            let fg_g = [-0.2];
            let fg_b = [0.5]; // fg values will be clamped

            let bg_channels: [&[f32]; 3] = [&bg_r, &bg_g, &bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_channels: [&mut [f32]; 3] = [&mut out_r, &mut out_g, &mut out_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Mul,
                alpha_channel: 0, // Not used
                clamp: true,      // Clamp fg values
            };
            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            let expected_r = [expected_mul_blend(bg_r[0], clamp(fg_r[0]))]; // 0.5 * 1.0 = 0.5
            let expected_g = [expected_mul_blend(bg_g[0], clamp(fg_g[0]))]; // 0.8 * 0.0 = 0.0
            let expected_b = [expected_mul_blend(bg_b[0], clamp(fg_b[0]))]; // 1.0 * 0.5 = 0.5

            assert_all_almost_eq!(&out_r, &expected_r, ABS_DELTA);
            assert_all_almost_eq!(&out_g, &expected_g, ABS_DELTA);
            assert_all_almost_eq!(&out_b, &expected_b, ABS_DELTA);
        }

        #[test]
        fn test_ec_blend_data_with_separate_alpha_premultiplied() {
            // Color: Replace FG over BG (to keep it simple)
            // EC0: Data channel
            // EC1: Alpha channel for EC0
            let bg_r = [0.1];
            let bg_g = [0.1];
            let bg_b = [0.1];
            let bg_ec0 = [0.2];
            let bg_ec1_alpha = [0.9]; // EC1 is alpha for EC0

            let fg_r = [0.5];
            let fg_g = [0.5];
            let fg_b = [0.5];
            let fg_ec0 = [0.6];
            let fg_ec1_alpha = [0.4];

            let bg_channels: [&[f32]; 5] = [&bg_r, &bg_g, &bg_b, &bg_ec0, &bg_ec1_alpha];
            let fg_channels: [&[f32]; 5] = [&fg_r, &fg_g, &fg_b, &fg_ec0, &fg_ec1_alpha];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_ec0 = [0.0];
            let mut out_ec1_alpha = [0.0];
            let mut out_channels: [&mut [f32]; 5] = [
                &mut out_r,
                &mut out_g,
                &mut out_b,
                &mut out_ec0,
                &mut out_ec1_alpha,
            ];

            let color_blending = PatchBlending {
                // Simple color replace
                mode: PatchBlendMode::Replace,
                alpha_channel: 0,
                clamp: false,
            };

            let ec_blending = [
                PatchBlending {
                    // EC0 (data) uses EC1 as alpha
                    mode: PatchBlendMode::BlendAbove,
                    alpha_channel: 1, // EC1 is alpha for EC0
                    clamp: false,
                },
                PatchBlending {
                    // EC1 (alpha) blends itself
                    mode: PatchBlendMode::BlendAbove,
                    alpha_channel: 1, // EC1 uses itself as alpha
                    clamp: false,
                },
            ];
            let extra_channel_info = [
                ExtraChannelInfo::new(
                    false,
                    ExtraChannel::Unknown,
                    BitDepth::f32(),
                    0,
                    "ec0".to_string(),
                    false,
                    None,
                    None,
                ), // EC0 data
                ExtraChannelInfo::new(
                    false,
                    ExtraChannel::Alpha,
                    BitDepth::f32(),
                    0,
                    "alpha_for_ec0".to_string(),
                    true, // EC1 is premultiplied alpha
                    None,
                    None,
                ), // EC1 alpha
            ];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            // Expected Color (Replace)
            assert_all_almost_eq!(&out_r, &fg_r, ABS_DELTA);
            assert_all_almost_eq!(&out_g, &fg_g, ABS_DELTA);
            assert_all_almost_eq!(&out_b, &fg_b, ABS_DELTA);

            // EC1 (Alpha channel for EC0) blending: BlendAbove, uses itself as alpha.
            // fg_alpha = fg_ec1_alpha[0], bg_alpha = bg_ec1_alpha[0]
            let expected_out_ec1_alpha = expected_alpha_blend(fg_ec1_alpha[0], bg_ec1_alpha[0]);
            assert_all_almost_eq!(&out_ec1_alpha, &[expected_out_ec1_alpha], ABS_DELTA);

            // EC0 (Data channel) blending: BlendAbove, uses EC1 as alpha.
            // fg_alpha_for_ec0 = fg_ec1_alpha[0] (not clamped as ec_blending[0].clamp is false)
            // is_premultiplied = extra_channel_info[ec_blending[0].alpha_channel (is 1)].alpha_associated = true.
            // Formula: out = fg_data + bg_data * (1.f - fg_alpha_of_data)
            let expected_out_ec0 =
                expected_color_blend_premultiplied(fg_ec0[0], bg_ec0[0], fg_ec1_alpha[0]);
            assert_all_almost_eq!(&out_ec0, &[expected_out_ec0], ABS_DELTA);
        }

        #[test]
        fn test_no_alpha_channel_blend_above_falls_back_to_copy_fg() {
            let bg_r = [0.1];
            let bg_g = [0.2];
            let bg_b = [0.3];
            let fg_r = [0.7];
            let fg_g = [0.8];
            let fg_b = [0.9];

            let bg_channels: [&[f32]; 3] = [&bg_r, &bg_g, &bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let mut out_r = [0.0];
            let mut out_g = [0.0];
            let mut out_b = [0.0];
            let mut out_channels: [&mut [f32]; 3] = [&mut out_r, &mut out_g, &mut out_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::BlendAbove,
                alpha_channel: 0, // Irrelevant as no alpha EIs
                clamp: false,
            };

            let ec_blending: [PatchBlending; 0] = [];
            // No ExtraChannelInfo means has_alpha will be false.
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            // Expected: output color is fg color due to fallback
            assert_all_almost_eq!(&out_r, &fg_r, ABS_DELTA);
            assert_all_almost_eq!(&out_g, &fg_g, ABS_DELTA);
            assert_all_almost_eq!(&out_b, &fg_b, ABS_DELTA);
        }

        #[test]
        fn test_empty_pixels() {
            let bg_r: [f32; 0] = [];
            let bg_g: [f32; 0] = [];
            let bg_b: [f32; 0] = [];
            let fg_r: [f32; 0] = [];
            let fg_g: [f32; 0] = [];
            let fg_b: [f32; 0] = [];

            let bg_channels: [&[f32]; 3] = [&bg_r, &bg_g, &bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let mut out_r: [f32; 0] = [];
            let mut out_g: [f32; 0] = [];
            let mut out_b: [f32; 0] = [];
            let mut out_channels: [&mut [f32]; 3] = [&mut out_r, &mut out_g, &mut out_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Replace,
                alpha_channel: 0,
                clamp: false,
            };
            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            perform_blending(
                &bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
                &mut out_channels,
            );

            // Expect output slices to also be empty and unchanged.
            assert_eq!(out_r.len(), 0);
            assert_eq!(out_g.len(), 0);
            assert_eq!(out_b.len(), 0);
        }
    }
}
