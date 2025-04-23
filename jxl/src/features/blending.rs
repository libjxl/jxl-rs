// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::headers::extra_channels::{ExtraChannel, ExtraChannelInfo};

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

fn perform_alpha_blending_layers<T: AsRef<[f32]>, V: AsMut<[f32]>>(
    bg: &[T],
    fg: &[T],
    alpha: usize,
    alpha_is_premultiplied: bool,
    clamp: bool,
    out: &mut [V],
) {
    if alpha_is_premultiplied {
        for x in 0..out[0].as_mut().len() {
            let fga = maybe_clamp(fg[3 + alpha].as_ref()[x], clamp);
            out[0].as_mut()[x] = fg[0].as_ref()[x] + bg[0].as_ref()[x] * (1.0 - fga);
            out[1].as_mut()[x] = fg[1].as_ref()[x] + bg[1].as_ref()[x] * (1.0 - fga);
            out[2].as_mut()[x] = fg[2].as_ref()[x] + bg[2].as_ref()[x] * (1.0 - fga);
            out[3 + alpha].as_mut()[x] = 1.0 - (1.0 - fga) * (1.0 - bg[3 + alpha].as_ref()[x]);
        }
    } else {
        for x in 0..out[0].as_mut().len() {
            let fga = maybe_clamp(fg[3 + alpha].as_ref()[x], clamp);
            let new_a = 1.0 - (1.0 - fga) * (1.0 - bg[3 + alpha].as_ref()[x]);
            let rnew_a = if new_a > 0.0 { 1.0 / new_a } else { 0.0 };
            out[0].as_mut()[x] = (fg[0].as_ref()[x] * fga
                + bg[0].as_ref()[x] * bg[3 + alpha].as_ref()[x] * (1.0 - fga))
                * rnew_a;
            out[1].as_mut()[x] = (fg[1].as_ref()[x] * fga
                + bg[1].as_ref()[x] * bg[3 + alpha].as_ref()[x] * (1.0 - fga))
                * rnew_a;
            out[2].as_mut()[x] = (fg[2].as_ref()[x] * fga
                + bg[2].as_ref()[x] * bg[3 + alpha].as_ref()[x] * (1.0 - fga))
                * rnew_a;
            out[3 + alpha].as_mut()[x] = new_a;
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
        alpha,
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

    let mut tmp = vec![vec![0.0; out.len()]; 3 + num_ec];

    for i in 0..num_ec {
        match ec_blending[i].mode {
            PatchBlendMode::Add => {
                for x in 0..tmp.len() {
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
