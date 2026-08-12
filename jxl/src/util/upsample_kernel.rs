// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use jxl_simd::F32SimdVec;

/// Precomputes flattened 5x5 kernels for 2x upsampling from CustomTransformData weights (15 values).
/// Output layout: kernel[oy * 2 + ox] -> [f32; 25] for oy, ox in 0..2
pub fn compute_5x5_kernels_2x(weights: &[f32; 15]) -> [[f32; 25]; 4] {
    let mut kernel = [[[[0.0f32; 5]; 5]; 2]; 2];
    let n = 1isize;
    for i in 0..5 {
        for j in 0..5 {
            let y = (i as isize).min(j as isize);
            let x = (i as isize).max(j as isize);
            let index = (5 * n * y - y * (y - 1) / 2 + x - y) as usize;
            // Filling in the top left corner from the weights
            kernel[j / 5][i / 5][j % 5][i % 5] = weights[index];
            // Mirroring to get the rest of the kernel.
            kernel[1 - j / 5][i / 5][4 - (j % 5)][i % 5] = weights[index];
            kernel[j / 5][1 - i / 5][j % 5][4 - (i % 5)] = weights[index];
            kernel[1 - j / 5][1 - i / 5][4 - (j % 5)][4 - (i % 5)] = weights[index];
        }
    }

    let mut flat_kernels = [[0.0f32; 25]; 4];
    for di in 0..2 {
        for dj in 0..2 {
            let idx = di * 2 + dj;
            for i in 0..5 {
                for j in 0..5 {
                    flat_kernels[idx][i * 5 + j] = kernel[di][dj][i][j];
                }
            }
        }
    }
    flat_kernels
}

/// Derives 1D horizontal 2x kernels from 2D 2x kernels by vertical averaging.
/// k_h[0] = 0.5 * (k2d[0] + k2d[2]) (even/left)
/// k_h[1] = 0.5 * (k2d[1] + k2d[3]) (odd/right)
pub fn compute_1d_h_kernels_2x(k2d: &[[f32; 25]; 4]) -> [[f32; 25]; 2] {
    let mut k_h = [[0.0f32; 25]; 2];
    for i in 0..25 {
        k_h[0][i] = 0.5 * (k2d[0][i] + k2d[2][i]);
        k_h[1][i] = 0.5 * (k2d[1][i] + k2d[3][i]);
    }
    k_h
}

/// Derives 1D vertical 2x kernels from 2D 2x kernels by horizontal averaging.
/// k_v[0] = 0.5 * (k2d[0] + k2d[1]) (top)
/// k_v[1] = 0.5 * (k2d[2] + k2d[3]) (bottom)
pub fn compute_1d_v_kernels_2x(k2d: &[[f32; 25]; 4]) -> [[f32; 25]; 2] {
    let mut k_v = [[0.0f32; 25]; 2];
    for i in 0..25 {
        k_v[0][i] = 0.5 * (k2d[0][i] + k2d[1][i]);
        k_v[1][i] = 0.5 * (k2d[2][i] + k2d[3][i]);
    }
    k_v
}

/// Shared min/max computation - generic helper function called from within dispatched functions
#[inline(always)]
pub fn compute_minmax<D: jxl_simd::SimdDescriptor>(
    d: D,
    input: &[&[f32]],
    xsize: usize,
    col_min: &mut [f32],
    col_max: &mut [f32],
    mins: &mut [f32],
    maxs: &mut [f32],
) {
    let r0 = input[0];
    let r1 = input[1];
    let r2 = input[2];
    let r3 = input[3];
    let r4 = input[4];

    // Step 1: Compute column-wise min/max (vertical reduction across 5 rows)
    // Use div_ceil to process all elements (may over-read but buffers are padded)
    let col_fill_len = xsize + 4;
    let num_vecs = col_fill_len.div_ceil(D::F32Vec::LEN);
    for i in 0..num_vecs {
        let offset = i * D::F32Vec::LEN;
        let v0 = D::F32Vec::load(d, &r0[offset..]);
        let v1 = D::F32Vec::load(d, &r1[offset..]);
        let v2 = D::F32Vec::load(d, &r2[offset..]);
        let v3 = D::F32Vec::load(d, &r3[offset..]);
        let v4 = D::F32Vec::load(d, &r4[offset..]);

        let col_min_v = v0.min(v1).min(v2).min(v3).min(v4);
        let col_max_v = v0.max(v1).max(v2).max(v3).max(v4);

        col_min_v.store(&mut col_min[offset..]);
        col_max_v.store(&mut col_max[offset..]);
    }

    // Step 2: Compute row-wise min/max from column temps (horizontal 5-wide window)
    let num_output_vecs = xsize.div_ceil(D::F32Vec::LEN);
    for i in 0..num_output_vecs {
        let offset = i * D::F32Vec::LEN;
        let m0 = D::F32Vec::load(d, &col_min[offset..]);
        let m1 = D::F32Vec::load(d, &col_min[offset + 1..]);
        let m2 = D::F32Vec::load(d, &col_min[offset + 2..]);
        let m3 = D::F32Vec::load(d, &col_min[offset + 3..]);
        let m4 = D::F32Vec::load(d, &col_min[offset + 4..]);
        let min_v = m0.min(m1).min(m2).min(m3).min(m4);
        min_v.store(&mut mins[offset..]);

        let m0 = D::F32Vec::load(d, &col_max[offset..]);
        let m1 = D::F32Vec::load(d, &col_max[offset + 1..]);
        let m2 = D::F32Vec::load(d, &col_max[offset + 2..]);
        let m3 = D::F32Vec::load(d, &col_max[offset + 3..]);
        let m4 = D::F32Vec::load(d, &col_max[offset + 4..]);
        let max_v = m0.max(m1).max(m2).max(m3).max(m4);
        max_v.store(&mut maxs[offset..]);
    }
}

/// Macro to generate the 5x5 kernel convolution code (shared across upsample and squeeze)
#[macro_export]
macro_rules! kernel_conv {
    ($d:expr, $kv:expr, $r0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $x:expr) => {{
        // Compute 5x5 kernel using FMA with 3-way ILP
        // Row 0
        let mut acc0 = <D::F32Vec>::load($d, &$r0[$x..]) * $kv[0];
        let mut acc1 = <D::F32Vec>::load($d, &$r0[$x + 1..]) * $kv[1];
        let mut acc2 = <D::F32Vec>::load($d, &$r0[$x + 2..]) * $kv[2];
        acc0 = <D::F32Vec>::load($d, &$r0[$x + 3..]).mul_add($kv[3], acc0);
        acc1 = <D::F32Vec>::load($d, &$r0[$x + 4..]).mul_add($kv[4], acc1);
        // Row 1
        acc2 = <D::F32Vec>::load($d, &$r1[$x..]).mul_add($kv[5], acc2);
        acc0 = <D::F32Vec>::load($d, &$r1[$x + 1..]).mul_add($kv[6], acc0);
        acc1 = <D::F32Vec>::load($d, &$r1[$x + 2..]).mul_add($kv[7], acc1);
        acc2 = <D::F32Vec>::load($d, &$r1[$x + 3..]).mul_add($kv[8], acc2);
        acc0 = <D::F32Vec>::load($d, &$r1[$x + 4..]).mul_add($kv[9], acc0);
        // Row 2
        acc1 = <D::F32Vec>::load($d, &$r2[$x..]).mul_add($kv[10], acc1);
        acc2 = <D::F32Vec>::load($d, &$r2[$x + 1..]).mul_add($kv[11], acc2);
        acc0 = <D::F32Vec>::load($d, &$r2[$x + 2..]).mul_add($kv[12], acc0);
        acc1 = <D::F32Vec>::load($d, &$r2[$x + 3..]).mul_add($kv[13], acc1);
        acc2 = <D::F32Vec>::load($d, &$r2[$x + 4..]).mul_add($kv[14], acc2);
        // Row 3
        acc0 = <D::F32Vec>::load($d, &$r3[$x..]).mul_add($kv[15], acc0);
        acc1 = <D::F32Vec>::load($d, &$r3[$x + 1..]).mul_add($kv[16], acc1);
        acc2 = <D::F32Vec>::load($d, &$r3[$x + 2..]).mul_add($kv[17], acc2);
        acc0 = <D::F32Vec>::load($d, &$r3[$x + 3..]).mul_add($kv[18], acc0);
        acc1 = <D::F32Vec>::load($d, &$r3[$x + 4..]).mul_add($kv[19], acc1);
        // Row 4
        acc2 = <D::F32Vec>::load($d, &$r4[$x..]).mul_add($kv[20], acc2);
        acc0 = <D::F32Vec>::load($d, &$r4[$x + 1..]).mul_add($kv[21], acc0);
        acc1 = <D::F32Vec>::load($d, &$r4[$x + 2..]).mul_add($kv[22], acc1);
        acc2 = <D::F32Vec>::load($d, &$r4[$x + 3..]).mul_add($kv[23], acc2);
        acc0 = <D::F32Vec>::load($d, &$r4[$x + 4..]).mul_add($kv[24], acc0);

        acc0 + acc1 + acc2
    }};
}
