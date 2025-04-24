// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    headers::extra_channels::{ExtraChannel, ExtraChannelInfo},
    util::TwoDArray,
};

use super::patches::{PatchBlendMode, PatchBlending};

const K_SMALL_ALPHA: f32 = 1e-6;

#[inline]
fn maybe_clamp(v: f32, clamp: bool) -> f32 {
    if clamp {
        v.clamp(0.0, 1.0)
    } else {
        v
    }
}

fn perform_alpha_blending_layers(
    bg: &[&[f32]],
    fg: &[&[f32]],
    alpha: usize,
    alpha_is_premultiplied: bool,
    clamp: bool,
    out: &mut [&mut [f32]],
) {
    if alpha_is_premultiplied {
        for x in 0..out[0].len() {
            let fga = maybe_clamp(fg[3 + alpha][x], clamp);
            out[0][x] = fg[0][x] + bg[0][x] * (1.0 - fga);
            out[1][x] = fg[1][x] + bg[1][x] * (1.0 - fga);
            out[2][x] = fg[2][x] + bg[2][x] * (1.0 - fga);
            out[3 + alpha][x] = 1.0 - (1.0 - fga) * (1.0 - bg[3 + alpha][x]);
        }
    } else {
        for x in 0..out[0].len() {
            let fga = maybe_clamp(fg[3 + alpha][x], clamp);
            let new_a = 1.0 - (1.0 - fga) * (1.0 - bg[3 + alpha][x]);
            let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
            out[0][x] = (fg[0][x] * fga + bg[0][x] * bg[3 + alpha][x] * (1.0 - fga)) * rnew_a;
            out[1][x] = (fg[1][x] * fga + bg[1][x] * bg[3 + alpha][x] * (1.0 - fga)) * rnew_a;
            out[2][x] = (fg[2][x] * fga + bg[2][x] * bg[3 + alpha][x] * (1.0 - fga)) * rnew_a;
            out[3 + alpha][x] = new_a;
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

fn add(bg: &[&[f32]], fg: &[&[f32]], out: &mut [&mut [f32]]) {
    for c in 0..3 {
        for (x, v) in out[c].iter_mut().enumerate() {
            *v = bg[c][x] + fg[c][x];
        }
    }
}

fn blend_weighted(
    bg: &[&[f32]],
    fg: &[&[f32]],
    alpha_is_premultiplied: bool,
    alpha: usize,
    clamp: bool,
    out: &mut [&mut [f32]],
) {
    perform_alpha_blending_layers(
        &[bg[0], bg[1], bg[2], bg[3 + alpha]],
        &[fg[0], fg[1], fg[2], fg[3 + alpha]],
        alpha,
        alpha_is_premultiplied,
        clamp,
        out,
    );
}

fn add_weighted(bg: &[&[f32]], fg: &[&[f32]], alpha: usize, clamp: bool, out: &mut [&mut [f32]]) {
    for c in 0..3 {
        perform_alpha_weighted_add(bg[c], fg[c], fg[3 + alpha], clamp, out[c]);
    }
}

fn copy(src: &[&[f32]], out: &mut [&mut [f32]]) {
    for p in 0..3 {
        out[p].copy_from_slice(src[p]);
    }
}

pub fn perform_blending(
    bg: &[&[f32]],
    fg: &[&[f32]],
    color_blending: &PatchBlending,
    ec_blending: &[PatchBlending],
    extra_channel_info: &[ExtraChannelInfo],
    out: &mut [&mut [f32]],
) {
    let has_alpha = extra_channel_info
        .iter()
        .any(|info| info.ec_type == ExtraChannel::Alpha);
    let num_ec = extra_channel_info.len();

    let mut tmp_ary = TwoDArray::blank(3 + num_ec, out.len());
    let mut tmp = tmp_ary.as_mut_refs();

    for i in 0..num_ec {
        match ec_blending[i].mode {
            PatchBlendMode::Add => {
                for x in 0..tmp.len() {
                    tmp[3 + i][x] = bg[3 + i][x] + fg[3 + i][x];
                }
            }
            PatchBlendMode::BlendAbove => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_blending(
                    bg[3 + i],
                    bg[3 + alpha],
                    fg[3 + i],
                    fg[3 + alpha],
                    extra_channel_info[alpha].alpha_associated(),
                    ec_blending[i].clamp,
                    tmp[3 + i],
                );
            }
            PatchBlendMode::BlendBelow => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_blending(
                    fg[3 + i],
                    fg[3 + alpha],
                    bg[3 + i],
                    bg[3 + alpha],
                    extra_channel_info[alpha].alpha_associated(),
                    ec_blending[i].clamp,
                    tmp[3 + i],
                );
            }
            PatchBlendMode::AlphaWeightedAddAbove => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_weighted_add(
                    bg[3 + i],
                    fg[3 + i],
                    fg[3 + alpha],
                    ec_blending[i].clamp,
                    tmp[3 + i],
                );
            }
            PatchBlendMode::AlphaWeightedAddBelow => {
                let alpha = ec_blending[i].alpha_channel;
                perform_alpha_weighted_add(
                    fg[3 + i],
                    bg[3 + i],
                    bg[3 + alpha],
                    ec_blending[i].clamp,
                    tmp[3 + i],
                );
            }
            PatchBlendMode::Mul => {
                perform_mul_blending(bg[3 + i], fg[3 + i], ec_blending[i].clamp, tmp[3 + i]);
            }
            PatchBlendMode::Replace => {
                tmp[3 + i].copy_from_slice(fg[3 + i]);
            }
            PatchBlendMode::None => {
                tmp[3 + i].copy_from_slice(bg[3 + i]);
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
                perform_mul_blending(bg[p], fg[p], color_blending.clamp, tmp[p]);
            }
        }
        PatchBlendMode::Replace => copy(fg, &mut tmp),
        PatchBlendMode::None => copy(bg, &mut tmp),
    }
    for i in 0..(3 + num_ec) {
        out[i].copy_from_slice(tmp[i]);
    }
}
