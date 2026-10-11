// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use jxl_simd::{F16SimdMask, F16SimdVec, SimdDescriptor, simd_function};

use super::patches::{PatchBlendMode, PatchBlending};
use crate::headers::extra_channels::{ExtraChannel, ExtraChannelInfo};

const MAX_F16_LANES: usize = 16;

/// Which layer is placed on top: `Above` blends fg over bg, `Below` blends bg
/// over fg. The top layer's alpha is the one that is clamped and drives the blend.
#[derive(Copy, Clone, PartialEq, Eq)]
enum BlendOrder {
    Above,
    Below,
}

#[derive(Copy, Clone)]
struct BlendConfig {
    clamp: bool,
    alpha_associated: bool,
    order: BlendOrder,
}

#[inline(always)]
fn load_f16_vec<D: SimdDescriptor>(d: D, s: &[crate::util::f16], x: usize) -> D::F16Vec {
    let lanes = D::F16Vec::LEN;
    let s_u16 = crate::util::f16::slice_to_bits(s);
    if x + lanes <= s_u16.len() {
        D::F16Vec::load_f16_bits(d, &s_u16[x..x + lanes])
    } else {
        let mut buf = [0u16; MAX_F16_LANES];
        buf[..s_u16.len() - x].copy_from_slice(&s_u16[x..]);
        D::F16Vec::load_f16_bits(d, &buf[..lanes])
    }
}

#[inline(always)]
fn store_f16_vec<D: SimdDescriptor>(_d: D, v: D::F16Vec, s: &mut [crate::util::f16], x: usize) {
    let lanes = D::F16Vec::LEN;
    let s_u16 = crate::util::f16::slice_to_bits_mut(s);
    if x + lanes <= s_u16.len() {
        v.store_f16_bits(&mut s_u16[x..x + lanes]);
    } else {
        let mut buf = [0u16; MAX_F16_LANES];
        v.store_f16_bits(&mut buf[..lanes]);
        let rem = s_u16.len() - x;
        s_u16[x..].copy_from_slice(&buf[..rem]);
    }
}

simd_function!(
    blend,
    d: D,
    #[allow(clippy::too_many_arguments)]
    fn blend_impl(
        bg_color: &mut [&mut [crate::util::f16]],
        bg_alpha: &mut [crate::util::f16],
        bg_alpha_old: &[crate::util::f16],
        fg_color: &[&[crate::util::f16]],
        fg_alpha: &[crate::util::f16],
        xsize: usize,
        cfg: BlendConfig,
    ) {
        let BlendConfig { clamp, alpha_associated, order } = cfg;
        let fg_on_top = order == BlendOrder::Above;

        let lanes = D::F16Vec::LEN;
        let one = D::F16Vec::splat_f32(d, 1.0);
        let zero = D::F16Vec::zero(d);

        let [bg_c0, bg_c1, bg_c2] = bg_color else { unreachable!() };
        let bg_c = [&mut bg_c0[..xsize], &mut bg_c1[..xsize], &mut bg_c2[..xsize]];
        let bg_alpha = &mut bg_alpha[..xsize];
        let bg_alpha_old = &bg_alpha_old[..xsize];
        let fg_c = [&fg_color[0][..xsize], &fg_color[1][..xsize], &fg_color[2][..xsize]];
        let fg_alpha = &fg_alpha[..xsize];

        let maybe_clamp_vec = |v: D::F16Vec| if clamp { v.max(zero).min(one) } else { v };

        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            let fg_a = load_f16_vec(d, fg_alpha, x);
            let bg_a = load_f16_vec(d, bg_alpha_old, x);
            let (top_a, bottom_a) = if fg_on_top { (fg_a, bg_a) } else { (bg_a, fg_a) };
            let top_a = maybe_clamp_vec(top_a);
            let one_minus_top_a = one - top_a;
            let new_a = one - one_minus_top_a * (one - bottom_a);
            let reciprocal_a = new_a.gt(zero).if_then_else_f16(one / new_a, zero);
            for c in 0..3 {
                let bg_v = load_f16_vec(d, bg_c[c], x);
                let fg_v = load_f16_vec(d, fg_c[c], x);
                let (top_c, bottom_c) = if fg_on_top { (fg_v, bg_v) } else { (bg_v, fg_v) };
                let out = if alpha_associated {
                    top_c + bottom_c * one_minus_top_a
                } else {
                    (top_c * top_a + bottom_c * bottom_a * one_minus_top_a) * reciprocal_a
                };
                store_f16_vec(d, out, bg_c[c], x);
            }
            store_f16_vec(d, new_a, bg_alpha, x);
        }
    }
);

simd_function!(
    blend_alpha,
    d: D,
    fn blend_alpha_impl(
        bg_alpha: &mut [crate::util::f16],
        fg_alpha: &[crate::util::f16],
        xsize: usize,
        cfg: BlendConfig,
    ) {
        let clamp = cfg.clamp;
        let fg_on_top = cfg.order == BlendOrder::Above;
        let lanes = D::F16Vec::LEN;
        let one = D::F16Vec::splat_f32(d, 1.0);
        let zero = D::F16Vec::zero(d);

        let bg_alpha = &mut bg_alpha[..xsize];
        let fg_alpha = &fg_alpha[..xsize];
        let maybe_clamp_vec = |v: D::F16Vec| if clamp { v.max(zero).min(one) } else { v };

        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            let fg_a = load_f16_vec(d, fg_alpha, x);
            let bg_a = load_f16_vec(d, bg_alpha, x);
            let (top_a, bottom_a) = if fg_on_top { (fg_a, bg_a) } else { (bg_a, fg_a) };
            let top_a = maybe_clamp_vec(top_a);
            store_f16_vec(d, one - (one - top_a) * (one - bottom_a), bg_alpha, x);
        }
    }
);

simd_function!(
    blend_ec,
    d: D,
    fn blend_ec_impl(
        bg_ec: &mut [crate::util::f16],
        bg_alpha_old: &[crate::util::f16],
        fg_ec: &[crate::util::f16],
        fg_alpha: &[crate::util::f16],
        xsize: usize,
        cfg: BlendConfig,
    ) {
        let BlendConfig {
            clamp,
            alpha_associated,
            order,
        } = cfg;
        let fg_on_top = order == BlendOrder::Above;

        let lanes = D::F16Vec::LEN;
        let one = D::F16Vec::splat_f32(d, 1.0);
        let zero = D::F16Vec::zero(d);

        let bg_ec = &mut bg_ec[..xsize];
        let bg_alpha_old = &bg_alpha_old[..xsize];
        let fg_ec = &fg_ec[..xsize];
        let fg_alpha = &fg_alpha[..xsize];

        let maybe_clamp_vec = |v: D::F16Vec| if clamp { v.max(zero).min(one) } else { v };

        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            let fg_a = load_f16_vec(d, fg_alpha, x);
            let bg_v = load_f16_vec(d, bg_ec, x);
            let fg_v = load_f16_vec(d, fg_ec, x);
            let (top_c, bottom_c) = if fg_on_top {
                (fg_v, bg_v)
            } else {
                (bg_v, fg_v)
            };
            if alpha_associated {
                let top_a = if fg_on_top {
                    maybe_clamp_vec(fg_a)
                } else {
                    maybe_clamp_vec(load_f16_vec(d, bg_alpha_old, x))
                };
                let out = top_c + bottom_c * (one - top_a);
                store_f16_vec(d, out, bg_ec, x);
            } else {
                let bg_a = load_f16_vec(d, bg_alpha_old, x);
                let (top_a, bottom_a) = if fg_on_top {
                    (fg_a, bg_a)
                } else {
                    (bg_a, fg_a)
                };
                let top_a = maybe_clamp_vec(top_a);
                let one_minus_top_a = one - top_a;
                let new_a = one - one_minus_top_a * (one - bottom_a);
                let reciprocal_a = new_a.gt(zero).if_then_else_f16(one / new_a, zero);
                let out = (top_c * top_a + bottom_c * bottom_a * one_minus_top_a) * reciprocal_a;
                store_f16_vec(d, out, bg_ec, x);
            }
        }
    }
);

simd_function!(
    add,
    d: D,
    fn add_impl(dst: &mut [crate::util::f16], src: &[crate::util::f16], xsize: usize) {
        let lanes = D::F16Vec::LEN;
        let dst = &mut dst[..xsize];
        let src = &src[..xsize];
        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            store_f16_vec(d, load_f16_vec(d, dst, x) + load_f16_vec(d, src, x), dst, x);
        }
    }
);

simd_function!(
    mul,
    d: D,
    fn mul_impl(
        dst: &mut [crate::util::f16],
        src: &[crate::util::f16],
        xsize: usize,
        clamp: bool,
    ) {
        let lanes = D::F16Vec::LEN;
        let one = D::F16Vec::splat_f32(d, 1.0);
        let zero = D::F16Vec::zero(d);
        let dst = &mut dst[..xsize];
        let src = &src[..xsize];
        let maybe_clamp_vec = |v: D::F16Vec| if clamp { v.max(zero).min(one) } else { v };
        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            let s = maybe_clamp_vec(load_f16_vec(d, src, x));
            store_f16_vec(d, load_f16_vec(d, dst, x) * s, dst, x);
        }
    }
);

// Above: dst = dst + src * weight (weight is the newly composited layer's own alpha, in f16).
simd_function!(
    alpha_weighted_add_above,
    d: D,
    fn alpha_weighted_add_above_impl(
        dst: &mut [crate::util::f16],
        src: &[crate::util::f16],
        weight: &[crate::util::f16],
        xsize: usize,
        clamp: bool,
    ) {
        let lanes = D::F16Vec::LEN;
        let one = D::F16Vec::splat_f32(d, 1.0);
        let zero = D::F16Vec::zero(d);
        let dst = &mut dst[..xsize];
        let src = &src[..xsize];
        let weight = &weight[..xsize];
        let maybe_clamp_vec = |v: D::F16Vec| if clamp { v.max(zero).min(one) } else { v };
        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            let w = maybe_clamp_vec(load_f16_vec(d, weight, x));
            let dst_v = load_f16_vec(d, dst, x);
            let src_v = load_f16_vec(d, src, x);
            store_f16_vec(d, dst_v + src_v * w, dst, x);
        }
    }
);

// Below: dst = src + dst * weight (weight is the old background's alpha, in f16).
simd_function!(
    alpha_weighted_add_below,
    d: D,
    fn alpha_weighted_add_below_impl(
        dst: &mut [crate::util::f16],
        src: &[crate::util::f16],
        weight: &[crate::util::f16],
        xsize: usize,
        clamp: bool,
    ) {
        let lanes = D::F16Vec::LEN;
        let one = D::F16Vec::splat_f32(d, 1.0);
        let zero = D::F16Vec::zero(d);
        let dst = &mut dst[..xsize];
        let src = &src[..xsize];
        let weight = &weight[..xsize];
        let maybe_clamp_vec = |v: D::F16Vec| if clamp { v.max(zero).min(one) } else { v };
        for k in 0..xsize.div_ceil(lanes) {
            let x = k * lanes;
            let w = maybe_clamp_vec(load_f16_vec(d, weight, x));
            let dst_v = load_f16_vec(d, dst, x);
            let src_v = load_f16_vec(d, src, x);
            store_f16_vec(d, src_v + dst_v * w, dst, x);
        }
    }
);

/// Blend `fg` onto `bg` in place.
pub fn perform_blending(
    bg: &mut [&mut [crate::util::f16]],
    fg: &[&[crate::util::f16]],
    color_blending: &PatchBlending,
    ec_blending: &[PatchBlending],
    extra_channel_info: &[ExtraChannelInfo],
    tmp: &mut Vec<crate::util::f16>,
) {
    // TODO(veluca): in many cases, the copy to temporary space seems redundant. Investigate more.
    let num_ec = extra_channel_info.len();
    let xsize = bg[0].len();

    // Fast path: if color is None (keep bg) and all ec are None, nothing to do.
    if color_blending.mode == PatchBlendMode::None
        && ec_blending.iter().all(|b| b.mode == PatchBlendMode::None)
    {
        return;
    }

    // Fast path: Replace color + Replace/None ec -> copy fg directly to bg, no tmp needed.
    if color_blending.mode == PatchBlendMode::Replace {
        let all_simple = ec_blending[..num_ec]
            .iter()
            .all(|b| b.mode == PatchBlendMode::Replace || b.mode == PatchBlendMode::None);
        if all_simple {
            for c in 0..3 {
                bg[c][..xsize].copy_from_slice(&fg[c][..xsize]);
            }
            for i in 0..num_ec {
                match ec_blending[i].mode {
                    PatchBlendMode::Replace => {
                        bg[3 + i][..xsize].copy_from_slice(&fg[3 + i][..xsize]);
                    }
                    PatchBlendMode::None => {} // keep bg
                    _ => unreachable!(),
                }
            }
            return;
        }
    }

    let has_alpha = extra_channel_info
        .iter()
        .any(|info| info.ec_type == ExtraChannel::Alpha);

    let needed_scratch = num_ec * xsize;

    if tmp.len() < needed_scratch {
        tmp.resize(needed_scratch, crate::util::f16::ZERO);
    }

    for i in 0..num_ec {
        tmp[i * xsize..][..xsize].copy_from_slice(&bg[3 + i][..xsize]);
    }
    let old_ec: &[crate::util::f16] = tmp;
    let old_alpha = |a: usize| &old_ec[a * xsize..][..xsize];

    for i in 0..num_ec {
        let alpha = ec_blending[i].alpha_channel;
        let clamp = ec_blending[i].clamp;
        let alpha_associated = extra_channel_info[alpha].alpha_associated();

        let (_bg_color, bg_ec) = bg.split_at_mut(3);
        let ec_out = &mut bg_ec[i][..xsize];

        match ec_blending[i].mode {
            PatchBlendMode::Add => {
                add(ec_out, &fg[3 + i][..xsize], xsize);
            }
            PatchBlendMode::BlendAbove => {
                if i == alpha {
                    blend_alpha(
                        ec_out,
                        &fg[3 + alpha][..xsize],
                        xsize,
                        BlendConfig {
                            clamp,
                            alpha_associated,
                            order: BlendOrder::Above,
                        },
                    );
                } else {
                    blend_ec(
                        ec_out,
                        old_alpha(alpha),
                        &fg[3 + i][..xsize],
                        &fg[3 + alpha][..xsize],
                        xsize,
                        BlendConfig {
                            clamp,
                            alpha_associated,
                            order: BlendOrder::Above,
                        },
                    );
                }
            }
            PatchBlendMode::BlendBelow => {
                if i == alpha {
                    blend_alpha(
                        ec_out,
                        &fg[3 + i][..xsize],
                        xsize,
                        BlendConfig {
                            clamp,
                            alpha_associated,
                            order: BlendOrder::Below,
                        },
                    );
                } else {
                    blend_ec(
                        ec_out,
                        old_alpha(alpha),
                        &fg[3 + i][..xsize],
                        &fg[3 + alpha][..xsize],
                        xsize,
                        BlendConfig {
                            clamp,
                            alpha_associated,
                            order: BlendOrder::Below,
                        },
                    );
                }
            }
            PatchBlendMode::AlphaWeightedAddAbove => {
                if i == alpha {
                    // ec_out is already bg[3 + i]
                } else {
                    alpha_weighted_add_above(
                        ec_out,
                        &fg[3 + i][..xsize],
                        &fg[3 + alpha][..xsize],
                        xsize,
                        clamp,
                    );
                }
            }
            PatchBlendMode::AlphaWeightedAddBelow => {
                if i == alpha {
                    ec_out.copy_from_slice(&fg[3 + i][..xsize]);
                } else {
                    alpha_weighted_add_below(
                        ec_out,
                        &fg[3 + i][..xsize],
                        old_alpha(alpha),
                        xsize,
                        clamp,
                    );
                }
            }
            PatchBlendMode::Mul => {
                mul(ec_out, &fg[3 + i][..xsize], xsize, clamp);
            }
            PatchBlendMode::Replace => {
                ec_out.copy_from_slice(&fg[3 + i][..xsize]);
            }
            PatchBlendMode::None => {
                // ec_out is already bg[3 + i]
            }
        }
    }

    let alpha = color_blending.alpha_channel;
    let clamp = color_blending.clamp;

    let (bg_color, bg_ec) = bg.split_at_mut(3);

    match color_blending.mode {
        PatchBlendMode::Add => {
            for c in 0..3 {
                add(bg_color[c], &fg[c][..xsize], xsize);
            }
        }
        PatchBlendMode::AlphaWeightedAddAbove => {
            for c in 0..3 {
                if !has_alpha {
                    add(bg_color[c], &fg[c][..xsize], xsize);
                } else {
                    alpha_weighted_add_above(
                        bg_color[c],
                        &fg[c][..xsize],
                        &fg[3 + alpha][..xsize],
                        xsize,
                        clamp,
                    );
                }
            }
        }
        PatchBlendMode::AlphaWeightedAddBelow => {
            for c in 0..3 {
                if !has_alpha {
                    add(bg_color[c], &fg[c][..xsize], xsize);
                } else {
                    alpha_weighted_add_below(
                        bg_color[c],
                        &fg[c][..xsize],
                        old_alpha(alpha),
                        xsize,
                        clamp,
                    );
                }
            }
        }
        PatchBlendMode::BlendAbove => {
            if !has_alpha {
                for c in 0..3 {
                    bg_color[c][..xsize].copy_from_slice(&fg[c][..xsize]);
                }
            } else {
                blend(
                    bg_color,
                    bg_ec[alpha],
                    old_alpha(alpha),
                    &fg[..3],
                    fg[3 + alpha],
                    xsize,
                    BlendConfig {
                        clamp,
                        alpha_associated: extra_channel_info[alpha].alpha_associated(),
                        order: BlendOrder::Above,
                    },
                );
            }
        }
        PatchBlendMode::BlendBelow => {
            if !has_alpha {
                // already bg[c]
            } else {
                blend(
                    bg_color,
                    bg_ec[alpha],
                    old_alpha(alpha),
                    &fg[..3],
                    fg[3 + alpha],
                    xsize,
                    BlendConfig {
                        clamp,
                        alpha_associated: extra_channel_info[alpha].alpha_associated(),
                        order: BlendOrder::Below,
                    },
                );
            }
        }
        PatchBlendMode::Mul => {
            for c in 0..3 {
                mul(bg_color[c], &fg[c][..xsize], xsize, clamp);
            }
        }
        PatchBlendMode::Replace => {
            for c in 0..3 {
                bg_color[c][..xsize].copy_from_slice(&fg[c][..xsize]);
            }
        }
        PatchBlendMode::None => {
            // already bg_color[c]
        }
    }
}

#[cfg(test)]
mod tests {
    fn clamp(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    mod perform_blending_tests {
        use test_log::test;

        use super::super::*;
        use super::*;
        use crate::headers::bit_depth::BitDepth;
        use crate::tests::assert_close;

        const ABS_DELTA: f32 = 2e-3;

        /// Test-only wrapper that allocates a tmp scratch buffer per call. Production
        /// callers are expected to supply their own (reusable) tmp.
        fn blend(
            bg: &mut [&mut [f32]],
            fg: &[&[f32]],
            color_blending: &PatchBlending,
            ec_blending: &[PatchBlending],
            extra_channel_info: &[ExtraChannelInfo],
        ) {
            let mut bg_f16: Vec<Vec<crate::util::f16>> = bg
                .iter()
                .map(|s| s.iter().copied().map(crate::util::f16::from_f32).collect())
                .collect();
            let mut bg_slices: Vec<&mut [crate::util::f16]> =
                bg_f16.iter_mut().map(|v| v.as_mut_slice()).collect();
            let fg_f16: Vec<Vec<crate::util::f16>> = fg
                .iter()
                .map(|s| s.iter().copied().map(crate::util::f16::from_f32).collect())
                .collect();
            let fg_slices: Vec<&[crate::util::f16]> = fg_f16.iter().map(|v| v.as_slice()).collect();
            perform_blending(
                &mut bg_slices,
                &fg_slices,
                color_blending,
                ec_blending,
                extra_channel_info,
                &mut vec![],
            );
            for (dst_ch, src_ch) in bg.iter_mut().zip(bg_f16.iter()) {
                for (d, s) in dst_ch.iter_mut().zip(src_ch.iter()) {
                    *d = s.to_f32();
                }
            }
        }

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
            let mut bg_r = [0.1];
            let mut bg_g = [0.2];
            let mut bg_b = [0.3];
            let fg_r = [0.7];
            let fg_g = [0.8];
            let fg_b = [0.9];

            let mut bg_channels: [&mut [f32]; 3] = [&mut bg_r, &mut bg_g, &mut bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Replace,
                alpha_channel: 0, // Not used for Replace
                clamp: false,
            };

            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            // Expected: output color is fg color
            assert_close!(all, &bg_r, &fg_r, ABS_DELTA);
            assert_close!(all, &bg_g, &fg_g, ABS_DELTA);
            assert_close!(all, &bg_b, &fg_b, ABS_DELTA);
        }

        #[test]
        fn test_color_add() {
            let mut bg_r = [0.1];
            let mut bg_g = [0.2];
            let mut bg_b = [0.3];
            let fg_r = [0.7];
            let fg_g = [0.6];
            let fg_b = [0.5];
            let expected_r = [bg_r[0] + fg_r[0]];
            let expected_g = [bg_g[0] + fg_g[0]];
            let expected_b = [bg_b[0] + fg_b[0]];

            let mut bg_channels: [&mut [f32]; 3] = [&mut bg_r, &mut bg_g, &mut bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Add,
                alpha_channel: 0, // Not used
                clamp: false,
            };
            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            assert_close!(all, &bg_r, &expected_r, ABS_DELTA);
            assert_close!(all, &bg_g, &expected_g, ABS_DELTA);
            assert_close!(all, &bg_b, &expected_b, ABS_DELTA);
        }

        #[test]
        fn test_color_blend_above_premultiplied_alpha() {
            // BG: R=0.1, G=0.2, B=0.3, A=0.8 (premultiplied)
            // FG: R=0.4, G=0.3, B=0.2, A=0.5 (premultiplied)
            let mut bg_r = [0.1];
            let mut bg_g = [0.2];
            let mut bg_b = [0.3];
            let mut bg_a = [0.8];
            let fg_r = [0.4];
            let fg_g = [0.3];
            let fg_b = [0.2];
            let fg_a = [0.5];
            let fga = fg_a[0]; // Not clamped
            let bga = bg_a[0];

            // Expected alpha: Ao = Afg + Abg * (1 - Afg)
            let expected_a_val = expected_alpha_blend(fga, bga);
            // Expected color: Co = Cfg_premult + Cbg_premult * (1 - Afg)
            let expected_r_val = expected_color_blend_premultiplied(fg_r[0], bg_r[0], fga);
            let expected_g_val = expected_color_blend_premultiplied(fg_g[0], bg_g[0], fga);
            let expected_b_val = expected_color_blend_premultiplied(fg_b[0], bg_b[0], fga);

            let mut bg_channels: [&mut [f32]; 4] = [&mut bg_r, &mut bg_g, &mut bg_b, &mut bg_a];
            let fg_channels: [&[f32]; 4] = [&fg_r, &fg_g, &fg_b, &fg_a];

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

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            assert_close!(all, &bg_a, &[expected_a_val], ABS_DELTA);
            assert_close!(all, &bg_r, &[expected_r_val], ABS_DELTA);
            assert_close!(all, &bg_g, &[expected_g_val], ABS_DELTA);
            assert_close!(all, &bg_b, &[expected_b_val], ABS_DELTA);
        }

        #[test]
        fn test_color_blend_above_non_premultiplied_alpha() {
            // BG: R=0.1, G=0.2, B=0.3 (unpremult), A=0.8
            // FG: R=0.7, G=0.6, B=0.5 (unpremult), A=0.5
            let mut bg_r = [0.1];
            let mut bg_g = [0.2];
            let mut bg_b = [0.3];
            let mut bg_a = [0.8];
            let fg_r = [0.7];
            let fg_g = [0.6];
            let fg_b = [0.5];
            let fg_a = [0.5];
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

            let mut bg_channels: [&mut [f32]; 4] = [&mut bg_r, &mut bg_g, &mut bg_b, &mut bg_a];
            let fg_channels: [&[f32]; 4] = [&fg_r, &fg_g, &fg_b, &fg_a];

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

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            assert_close!(all, &bg_a, &[expected_a_val], ABS_DELTA);
            assert_close!(all, &bg_r, &[expected_r_val], ABS_DELTA);
            assert_close!(all, &bg_g, &[expected_g_val], ABS_DELTA);
            assert_close!(all, &bg_b, &[expected_b_val], ABS_DELTA);
        }

        #[test]
        fn test_color_alpha_weighted_add_above() {
            let mut bg_r = [0.1];
            let mut bg_g = [0.2];
            let mut bg_b = [0.3];
            let mut bg_a = [0.8]; // bg alpha used by ec_blending
            let fg_r = [0.7];
            let fg_g = [0.6];
            let fg_b = [0.5];
            let fg_a = [0.5]; // fg alpha used for weighting
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

            let mut bg_channels: [&mut [f32]; 4] = [&mut bg_r, &mut bg_g, &mut bg_b, &mut bg_a];
            let fg_channels: [&[f32]; 4] = [&fg_r, &fg_g, &fg_b, &fg_a];

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

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            assert_close!(all, &bg_r, &[expected_r_val], ABS_DELTA);
            assert_close!(all, &bg_g, &[expected_g_val], ABS_DELTA);
            assert_close!(all, &bg_b, &[expected_b_val], ABS_DELTA);
            assert_close!(all, &bg_a, &[expected_a_val], ABS_DELTA);
        }

        #[test]
        fn test_color_mul_with_clamp() {
            let mut bg_r = [0.5];
            let mut bg_g = [0.8];
            let mut bg_b = [1.0];
            let fg_r = [1.5];
            let fg_g = [-0.2];
            let fg_b = [0.5]; // fg values will be clamped
            let expected_r = [expected_mul_blend(bg_r[0], clamp(fg_r[0]))]; // 0.5 * 1.0 = 0.5
            let expected_g = [expected_mul_blend(bg_g[0], clamp(fg_g[0]))]; // 0.8 * 0.0 = 0.0
            let expected_b = [expected_mul_blend(bg_b[0], clamp(fg_b[0]))]; // 1.0 * 0.5 = 0.5

            let mut bg_channels: [&mut [f32]; 3] = [&mut bg_r, &mut bg_g, &mut bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Mul,
                alpha_channel: 0, // Not used
                clamp: true,      // Clamp fg values
            };
            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            assert_close!(all, &bg_r, &expected_r, ABS_DELTA);
            assert_close!(all, &bg_g, &expected_g, ABS_DELTA);
            assert_close!(all, &bg_b, &expected_b, ABS_DELTA);
        }

        #[test]
        fn test_ec_blend_data_with_separate_alpha_premultiplied() {
            // Color: Replace FG over BG (to keep it simple)
            // EC0: Data channel
            // EC1: Alpha channel for EC0
            let mut bg_r = [0.1];
            let mut bg_g = [0.1];
            let mut bg_b = [0.1];
            let mut bg_ec0 = [0.2];
            let mut bg_ec1_alpha = [0.9]; // EC1 is alpha for EC0

            let fg_r = [0.5];
            let fg_g = [0.5];
            let fg_b = [0.5];
            let fg_ec0 = [0.6];
            let fg_ec1_alpha = [0.4];

            // EC1 (Alpha channel for EC0) blending: BlendAbove, uses itself as alpha.
            // fg_alpha = fg_ec1_alpha[0], bg_alpha = bg_ec1_alpha[0]
            let expected_out_ec1_alpha = expected_alpha_blend(fg_ec1_alpha[0], bg_ec1_alpha[0]);

            // EC0 (Data channel) blending: BlendAbove, uses EC1 as alpha.
            // fg_alpha_for_ec0 = fg_ec1_alpha[0] (not clamped as ec_blending[0].clamp is false)
            // is_premultiplied = extra_channel_info[ec_blending[0].alpha_channel (is 1)].alpha_associated = true.
            // Formula: out = fg_data + bg_data * (1.f - fg_alpha_of_data)
            let expected_out_ec0 =
                expected_color_blend_premultiplied(fg_ec0[0], bg_ec0[0], fg_ec1_alpha[0]);

            let mut bg_channels: [&mut [f32]; 5] = [
                &mut bg_r,
                &mut bg_g,
                &mut bg_b,
                &mut bg_ec0,
                &mut bg_ec1_alpha,
            ];
            let fg_channels: [&[f32]; 5] = [&fg_r, &fg_g, &fg_b, &fg_ec0, &fg_ec1_alpha];

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

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            // Expected Color (Replace)
            assert_close!(all, &bg_r, &fg_r, ABS_DELTA);
            assert_close!(all, &bg_g, &fg_g, ABS_DELTA);
            assert_close!(all, &bg_b, &fg_b, ABS_DELTA);
            assert_close!(all, &bg_ec1_alpha, &[expected_out_ec1_alpha], ABS_DELTA);
            assert_close!(all, &bg_ec0, &[expected_out_ec0], ABS_DELTA);
        }

        #[test]
        fn test_no_alpha_channel_blend_above_falls_back_to_copy_fg() {
            let mut bg_r = [0.1];
            let mut bg_g = [0.2];
            let mut bg_b = [0.3];
            let fg_r = [0.7];
            let fg_g = [0.8];
            let fg_b = [0.9];

            let mut bg_channels: [&mut [f32]; 3] = [&mut bg_r, &mut bg_g, &mut bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::BlendAbove,
                alpha_channel: 0, // Irrelevant as no alpha EIs
                clamp: false,
            };

            let ec_blending: [PatchBlending; 0] = [];
            // No ExtraChannelInfo means has_alpha will be false.
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            // Expected: output color is fg color due to fallback
            assert_close!(all, &bg_r, &fg_r, ABS_DELTA);
            assert_close!(all, &bg_g, &fg_g, ABS_DELTA);
            assert_close!(all, &bg_b, &fg_b, ABS_DELTA);
        }

        #[test]
        fn test_empty_pixels() {
            let mut bg_r: [f32; 0] = [];
            let mut bg_g: [f32; 0] = [];
            let mut bg_b: [f32; 0] = [];
            let fg_r: [f32; 0] = [];
            let fg_g: [f32; 0] = [];
            let fg_b: [f32; 0] = [];

            let mut bg_channels: [&mut [f32]; 3] = [&mut bg_r, &mut bg_g, &mut bg_b];
            let fg_channels: [&[f32]; 3] = [&fg_r, &fg_g, &fg_b];

            let color_blending = PatchBlending {
                mode: PatchBlendMode::Replace,
                alpha_channel: 0,
                clamp: false,
            };
            let ec_blending: [PatchBlending; 0] = [];
            let extra_channel_info: [ExtraChannelInfo; 0] = [];

            blend(
                &mut bg_channels,
                &fg_channels,
                &color_blending,
                &ec_blending,
                &extra_channel_info,
            );

            // Expect output slices to also be empty and unchanged.
            assert_eq!(bg_r.len(), 0);
            assert_eq!(bg_g.len(), 0);
            assert_eq!(bg_b.len(), 0);
        }
    }
}
