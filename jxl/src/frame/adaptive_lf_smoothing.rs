// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, SimdDescriptor, simd_function};
use num_traits::abs;

use crate::api::JxlParallelRunner;
use crate::error::Result;
use crate::headers::frame_header::FrameHeader;
use crate::image::Image;
use crate::render::buffer_splitter::OutputChannelSplitter;
use crate::util::f16;

#[allow(clippy::excessive_precision)]
const W_SIDE: f32 = 0.20345139757231578;
#[allow(clippy::excessive_precision)]
const W_CORNER: f32 = 0.0334829185968739;
const W_CENTER: f32 = 1.0 - 4.0 * (W_SIDE + W_CORNER);

#[inline(always)]
fn compute_pixel_channel(
    inv_dc_factor: f32,
    gap: f32,
    x: usize,
    row_top: &[f16],
    row: &[f16],
    row_bottom: &[f16],
) -> (f32, f32, f32) {
    let tl = row_top[x - 1].to_f32();
    let tc = row_top[x].to_f32();
    let tr = row_top[x + 1].to_f32();
    let ml = row[x - 1].to_f32();
    let mc = row[x].to_f32();
    let mr = row[x + 1].to_f32();
    let bl = row_bottom[x - 1].to_f32();
    let bc = row_bottom[x].to_f32();
    let br = row_bottom[x + 1].to_f32();
    let corner = tl + tr + bl + br;
    let side = ml + mr + tc + bc;
    let sm = corner * W_CORNER + side * W_SIDE + mc * W_CENTER;
    (mc, sm, gap.max(abs((mc - sm) * inv_dc_factor)))
}

#[inline(always)]
fn compute_chunk_channel<D: SimdDescriptor>(
    d: D,
    inv_dc_factor: D::F32Vec,
    gap: D::F32Vec,
    gx: usize,
    row_top: &[u16],
    row: &[u16],
    row_bottom: &[u16],
) -> (D::F32Vec, D::F32Vec, D::F32Vec) {
    let tl = D::F32Vec::load_f16_bits(d, &row_top[gx - 1..]);
    let tc = D::F32Vec::load_f16_bits(d, &row_top[gx..]);
    let tr = D::F32Vec::load_f16_bits(d, &row_top[gx + 1..]);
    let ml = D::F32Vec::load_f16_bits(d, &row[gx - 1..]);
    let mc = D::F32Vec::load_f16_bits(d, &row[gx..]);
    let mr = D::F32Vec::load_f16_bits(d, &row[gx + 1..]);
    let bl = D::F32Vec::load_f16_bits(d, &row_bottom[gx - 1..]);
    let bc = D::F32Vec::load_f16_bits(d, &row_bottom[gx..]);
    let br = D::F32Vec::load_f16_bits(d, &row_bottom[gx + 1..]);
    let corner = tl + tr + bl + br;
    let side = ml + mr + tc + bc;
    let sm = corner.mul_add(
        D::F32Vec::splat(d, W_CORNER),
        side.mul_add(
            D::F32Vec::splat(d, W_SIDE),
            mc * D::F32Vec::splat(d, W_CENTER),
        ),
    );
    (mc, sm, gap.max(((mc - sm) * inv_dc_factor).abs()))
}

simd_function!(
    smooth_row_dispatch,
    d: D,
    #[allow(clippy::too_many_arguments)]
    fn smooth_row(
        inv_lf_factors: [f32; 3],
        xsize: usize,
        origin_x: usize,
        row_len: usize,
        in0: [&[f16]; 3],
        in1: [&[f16]; 3],
        in2: [&[f16]; 3],
        row_0: &mut [f16],
        row_1: &mut [f16],
        row_2: &mut [f16],
    ) {
        if row_len == 0 {
            return;
        }
        let mut lx_start = 0;
        let mut lx_end = row_len;
        if origin_x == 0 {
            row_0[0] = in0[1][0];
            row_1[0] = in1[1][0];
            row_2[0] = in2[1][0];
            lx_start = 1;
        }
        if origin_x + row_len == xsize {
            let last = row_len - 1;
            let gx = origin_x + last;
            row_0[last] = in0[1][gx];
            row_1[last] = in1[1][gx];
            row_2[last] = in2[1][gx];
            lx_end = last;
        }
        if lx_start >= lx_end {
            return;
        }

        let len = D::F32Vec::LEN;
        let inv_fac0 = D::F32Vec::splat(d, inv_lf_factors[0]);
        let inv_fac1 = D::F32Vec::splat(d, inv_lf_factors[1]);
        let inv_fac2 = D::F32Vec::splat(d, inv_lf_factors[2]);
        let c_half = D::F32Vec::splat(d, 0.5);
        let c_three = D::F32Vec::splat(d, 3.0);
        let c_neg_four = D::F32Vec::splat(d, -4.0);
        let c_zero = D::F32Vec::zero(d);

        let in0_bits = in0.map(f16::slice_to_bits);
        let in1_bits = in1.map(f16::slice_to_bits);
        let in2_bits = in2.map(f16::slice_to_bits);

        let mut lx = lx_start;
        {
            let row_0_bits = f16::slice_to_bits_mut(row_0);
            let row_1_bits = f16::slice_to_bits_mut(row_1);
            let row_2_bits = f16::slice_to_bits_mut(row_2);
            while lx + len <= lx_end {
                let gx = origin_x + lx;
                let (mc_x, sm_x, gap) = compute_chunk_channel(
                    d,
                    inv_fac0,
                    c_half,
                    gx,
                    in0_bits[0],
                    in0_bits[1],
                    in0_bits[2],
                );
                let (mc_y, sm_y, gap) = compute_chunk_channel(
                    d,
                    inv_fac1,
                    gap,
                    gx,
                    in1_bits[0],
                    in1_bits[1],
                    in1_bits[2],
                );
                let (mc_b, sm_b, gap) = compute_chunk_channel(
                    d,
                    inv_fac2,
                    gap,
                    gx,
                    in2_bits[0],
                    in2_bits[1],
                    in2_bits[2],
                );
                let factor = c_neg_four.mul_add(gap, c_three).max(c_zero);
                (sm_x - mc_x)
                    .mul_add(factor, mc_x)
                    .store_f16_bits(&mut row_0_bits[lx..]);
                (sm_y - mc_y)
                    .mul_add(factor, mc_y)
                    .store_f16_bits(&mut row_1_bits[lx..]);
                (sm_b - mc_b)
                    .mul_add(factor, mc_b)
                    .store_f16_bits(&mut row_2_bits[lx..]);
                lx += len;
            }
        }

        for lx in lx..lx_end {
            let gx = origin_x + lx;
            let gap = 0.5;
            let (mc_x, sm_x, gap) =
                compute_pixel_channel(inv_lf_factors[0], gap, gx, in0[0], in0[1], in0[2]);
            let (mc_y, sm_y, gap) =
                compute_pixel_channel(inv_lf_factors[1], gap, gx, in1[0], in1[1], in1[2]);
            let (mc_b, sm_b, gap) =
                compute_pixel_channel(inv_lf_factors[2], gap, gx, in2[0], in2[1], in2[2]);
            let factor = (3.0 - 4.0 * gap).max(0.0);
            row_0[lx] = f16::from_f32((sm_x - mc_x) * factor + mc_x);
            row_1[lx] = f16::from_f32((sm_y - mc_y) * factor + mc_y);
            row_2[lx] = f16::from_f32((sm_b - mc_b) * factor + mc_b);
        }
    }
);

pub fn adaptive_lf_smoothing(
    lf_factors: [f32; 3],
    frame_header: &FrameHeader,
    lf_image: &mut [Image<f16>; 3],
    parallel_runner: &mut dyn JxlParallelRunner,
) -> Result<()> {
    let (xsize, ysize) = lf_image[0].size();
    if ysize <= 2 || xsize <= 2 {
        return Ok(());
    }
    let mut smoothed0 = Image::<f16>::new((xsize, ysize))?;
    let mut smoothed1 = Image::<f16>::new((xsize, ysize))?;
    let mut smoothed2 = Image::<f16>::new((xsize, ysize))?;

    let splitter0 = OutputChannelSplitter::from_image(&mut smoothed0);
    let splitter1 = OutputChannelSplitter::from_image(&mut smoothed1);
    let splitter2 = OutputChannelSplitter::from_image(&mut smoothed2);

    let num_lf_groups = frame_header.num_lf_groups();
    let inv_lf_factors = [
        1.0 / lf_factors[0],
        1.0 / lf_factors[1],
        1.0 / lf_factors[2],
    ];

    parallel_runner.run_ordered(num_lf_groups, None, &|g| {
        let r = frame_header.lf_group_rect(g);
        let mut out_ref_0 = splitter0.borrow_typed_rect::<f16>(r);
        let mut out_ref_1 = splitter1.borrow_typed_rect::<f16>(r);
        let mut out_ref_2 = splitter2.borrow_typed_rect::<f16>(r);

        for ly in 0..r.size.1 {
            let gy = r.origin.1 + ly;
            let row_0 = out_ref_0.typed_row_mut::<f16>(ly);
            let row_1 = out_ref_1.typed_row_mut::<f16>(ly);
            let row_2 = out_ref_2.typed_row_mut::<f16>(ly);

            if gy == 0 || gy == ysize - 1 {
                let x_range = r.origin.0..r.origin.0 + r.size.0;
                row_0[..r.size.0].copy_from_slice(&lf_image[0].row(gy)[x_range.clone()]);
                row_1[..r.size.0].copy_from_slice(&lf_image[1].row(gy)[x_range.clone()]);
                row_2[..r.size.0].copy_from_slice(&lf_image[2].row(gy)[x_range]);
                continue;
            }

            smooth_row_dispatch(
                inv_lf_factors,
                xsize,
                r.origin.0,
                r.size.0,
                [
                    lf_image[0].row(gy - 1),
                    lf_image[0].row(gy),
                    lf_image[0].row(gy + 1),
                ],
                [
                    lf_image[1].row(gy - 1),
                    lf_image[1].row(gy),
                    lf_image[1].row(gy + 1),
                ],
                [
                    lf_image[2].row(gy - 1),
                    lf_image[2].row(gy),
                    lf_image[2].row(gy + 1),
                ],
                &mut row_0[..r.size.0],
                &mut row_1[..r.size.0],
                &mut row_2[..r.size.0],
            );
        }
        Ok(())
    })?;

    drop(splitter0);
    drop(splitter1);
    drop(splitter2);
    *lf_image = [smoothed0, smoothed1, smoothed2];
    Ok(())
}
