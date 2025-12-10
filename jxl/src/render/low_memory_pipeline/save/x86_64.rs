// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::{
    arch::x86_64::{
        __m256i, __m512i, _mm256_add_epi32, _mm256_blend_epi32, _mm256_loadu_si256,
        _mm256_permute2f128_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
        _mm256_storeu_si256, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
        _mm256_unpacklo_epi64, _mm512_loadu_si512, _mm512_mask_permutevar_epi32,
        _mm512_permutex2var_epi32, _mm512_storeu_si512,
    },
    mem::MaybeUninit,
};

#[target_feature(enable = "avx2")]
fn load_avx2_u32(data: &[u32; 8]) -> __m256i {
    // SAFETY: `data` has the correct size.
    unsafe { _mm256_loadu_si256(data.as_ptr() as *const _) }
}

#[target_feature(enable = "avx2")]
fn load_avx2(data: &[u8; 32]) -> __m256i {
    // SAFETY: `data` has the correct size.
    unsafe { _mm256_loadu_si256(data.as_ptr() as *const _) }
}

#[target_feature(enable = "avx2")]
fn store_avx2(data: __m256i, out: &mut [MaybeUninit<u8>; 32]) {
    // SAFETY: `data` has the correct size.
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut _, data) }
}

#[target_feature(enable = "avx512f")]
fn load_avx512_u32(data: &[u32; 16]) -> __m512i {
    // SAFETY: `data` has the correct size.
    unsafe { _mm512_loadu_si512(data.as_ptr() as *const _) }
}

#[target_feature(enable = "avx512f")]
fn load_avx512(data: &[u8; 64]) -> __m512i {
    // SAFETY: `data` has the correct size.
    unsafe { _mm512_loadu_si512(data.as_ptr() as *const _) }
}

#[target_feature(enable = "avx512f")]
fn store_avx512(data: __m512i, out: &mut [MaybeUninit<u8>; 64]) {
    // SAFETY: `data` has the correct size.
    unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut _, data) }
}

#[target_feature(enable = "avx2")]
fn interleave3_32b_avx2(inp: &[&[u8]; 3], out: &mut [MaybeUninit<u8>]) -> usize {
    let [a, b, c] = inp;

    let idx_a0 = load_avx2_u32(&[0, 0, 0, 1, 0, 0, 2, 0]);
    // c1 = idx_a0 + 2
    // b2 = idx_a0 + 5

    let idx_b0 = load_avx2_u32(&[0, 0, 0, 0, 1, 0, 0, 2]);
    // a1 = idx_b0 + 3
    // c2 = idx_b0 + 5

    let idx_c0 = load_avx2_u32(&[0, 0, 0, 0, 0, 1, 0, 0]);
    // b1 = idx_c0 + 3
    // a2 = idx_c0 + 6

    let two = _mm256_set1_epi32(2);
    let three = _mm256_set1_epi32(3);
    let five = _mm256_set1_epi32(5);
    let six = _mm256_set1_epi32(6);

    const LEN: usize = 32;
    let mut processed = 0;
    for (((a, b), c), out) in a
        .chunks_exact(LEN)
        .zip(b.chunks_exact(LEN))
        .zip(c.chunks_exact(LEN))
        .zip(out.chunks_exact_mut(LEN * 3))
    {
        let a = load_avx2(a.try_into().unwrap());
        let b = load_avx2(b.try_into().unwrap());
        let c = load_avx2(c.try_into().unwrap());

        let a0 = _mm256_permutevar8x32_epi32(a, idx_a0);
        let b0 = _mm256_permutevar8x32_epi32(b, idx_b0);
        let c0 = _mm256_permutevar8x32_epi32(c, idx_c0);
        let out0 = _mm256_blend_epi32::<0b10010010>(a0, b0);
        let out0 = _mm256_blend_epi32::<0b00100100>(out0, c0);

        let a1 = _mm256_permutevar8x32_epi32(a, _mm256_add_epi32(idx_b0, three));
        let b1 = _mm256_permutevar8x32_epi32(b, _mm256_add_epi32(idx_c0, three));
        let c1 = _mm256_permutevar8x32_epi32(c, _mm256_add_epi32(idx_a0, two));
        let out1 = _mm256_blend_epi32::<0b00100100>(a1, b1);
        let out1 = _mm256_blend_epi32::<0b01001001>(out1, c1);

        let a2 = _mm256_permutevar8x32_epi32(a, _mm256_add_epi32(idx_c0, six));
        let b2 = _mm256_permutevar8x32_epi32(b, _mm256_add_epi32(idx_a0, five));
        let c2 = _mm256_permutevar8x32_epi32(c, _mm256_add_epi32(idx_b0, five));
        let out2 = _mm256_blend_epi32::<0b01001001>(a2, b2);
        let out2 = _mm256_blend_epi32::<0b10010010>(out2, c2);

        store_avx2(out0, (&mut out[0..LEN]).try_into().unwrap());
        store_avx2(out1, (&mut out[LEN..2 * LEN]).try_into().unwrap());
        store_avx2(out2, (&mut out[2 * LEN..3 * LEN]).try_into().unwrap());
        processed += LEN / 4;
    }

    processed
}

#[inline(never)]
#[target_feature(enable = "avx512f")]
fn interleave3_32b_avx512(inp: &[&[u8]; 3], out: &mut [MaybeUninit<u8>]) -> usize {
    let [a, b, c] = inp;

    let idx_ab0 = load_avx512_u32(&[0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5]);
    let idx_c0 = load_avx512_u32(&[0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0]);

    let idx_ab1 = load_avx512_u32(&[21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26]);
    let idx_c1 = load_avx512_u32(&[0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0]);

    let idx_ab2 = load_avx512_u32(&[0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0]);
    let idx_c2 = load_avx512_u32(&[10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15]);

    const LEN: usize = 64;
    let mut processed = 0;
    for (((a, b), c), out) in a
        .chunks_exact(LEN)
        .zip(b.chunks_exact(LEN))
        .zip(c.chunks_exact(LEN))
        .zip(out.chunks_exact_mut(LEN * 3))
    {
        let a = load_avx512(a.try_into().unwrap());
        let b = load_avx512(b.try_into().unwrap());
        let c = load_avx512(c.try_into().unwrap());

        let out0 = _mm512_permutex2var_epi32(a, idx_ab0, b);
        let out0 = _mm512_mask_permutevar_epi32(out0, 0b0100100100100100, idx_c0, c);

        let out1 = _mm512_permutex2var_epi32(a, idx_ab1, b);
        let out1 = _mm512_mask_permutevar_epi32(out1, 0b0010010010010010, idx_c1, c);

        let out2 = _mm512_permutex2var_epi32(a, idx_ab2, b);
        let out2 = _mm512_mask_permutevar_epi32(out2, 0b1001001001001001, idx_c2, c);

        store_avx512(out0, (&mut out[0..LEN]).try_into().unwrap());
        store_avx512(out1, (&mut out[LEN..2 * LEN]).try_into().unwrap());
        store_avx512(out2, (&mut out[2 * LEN..3 * LEN]).try_into().unwrap());
        processed += LEN / 4;
    }

    processed
}

/// Safety note: does not write uninit data in `out`.
pub(super) fn interleave3_32b(inp: &[&[u8]; 3], out: &mut [MaybeUninit<u8>]) -> usize {
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: we just checked for avx512f.
        unsafe { interleave3_32b_avx512(inp, out) }
    } else if is_x86_feature_detected!("avx2") {
        // SAFETY: we just checked for avx2.
        unsafe { interleave3_32b_avx2(inp, out) }
    } else {
        0
    }
}

// ============================================================================
// 4-channel 32-bit interleaving (RGBA f32)
// ============================================================================

#[target_feature(enable = "avx2")]
fn interleave4_32b_avx2(inp: &[&[u8]; 4], out: &mut [MaybeUninit<u8>]) -> usize {
    let [a, b, c, d] = inp;

    const LEN: usize = 32; // 8 floats per vector
    let mut processed = 0;

    for ((((a, b), c), d), out) in a
        .chunks_exact(LEN)
        .zip(b.chunks_exact(LEN))
        .zip(c.chunks_exact(LEN))
        .zip(d.chunks_exact(LEN))
        .zip(out.chunks_exact_mut(LEN * 4))
    {
        let a = load_avx2(a.try_into().unwrap());
        let b = load_avx2(b.try_into().unwrap());
        let c = load_avx2(c.try_into().unwrap());
        let d = load_avx2(d.try_into().unwrap());

        // Interleave: a0,b0,c0,d0, a1,b1,c1,d1, ...
        // unpacklo/hi work on 32-bit elements within 128-bit lanes
        let ab_lo = _mm256_unpacklo_epi32(a, b); // a0,b0,a1,b1, a4,b4,a5,b5
        let ab_hi = _mm256_unpackhi_epi32(a, b); // a2,b2,a3,b3, a6,b6,a7,b7
        let cd_lo = _mm256_unpacklo_epi32(c, d); // c0,d0,c1,d1, c4,d4,c5,d5
        let cd_hi = _mm256_unpackhi_epi32(c, d); // c2,d2,c3,d3, c6,d6,c7,d7

        let abcd_0 = _mm256_unpacklo_epi64(ab_lo, cd_lo); // a0,b0,c0,d0, a4,b4,c4,d4
        let abcd_1 = _mm256_unpackhi_epi64(ab_lo, cd_lo); // a1,b1,c1,d1, a5,b5,c5,d5
        let abcd_2 = _mm256_unpacklo_epi64(ab_hi, cd_hi); // a2,b2,c2,d2, a6,b6,c6,d6
        let abcd_3 = _mm256_unpackhi_epi64(ab_hi, cd_hi); // a3,b3,c3,d3, a7,b7,c7,d7

        // Now we need to rearrange to get contiguous output
        // Use permute2f128 to swap the 128-bit halves
        let out0 = _mm256_permute2f128_si256::<0x20>(abcd_0, abcd_1); // a0,b0,c0,d0, a1,b1,c1,d1
        let out1 = _mm256_permute2f128_si256::<0x20>(abcd_2, abcd_3); // a2,b2,c2,d2, a3,b3,c3,d3
        let out2 = _mm256_permute2f128_si256::<0x31>(abcd_0, abcd_1); // a4,b4,c4,d4, a5,b5,c5,d5
        let out3 = _mm256_permute2f128_si256::<0x31>(abcd_2, abcd_3); // a6,b6,c6,d6, a7,b7,c7,d7

        store_avx2(out0, (&mut out[0..LEN]).try_into().unwrap());
        store_avx2(out1, (&mut out[LEN..2 * LEN]).try_into().unwrap());
        store_avx2(out2, (&mut out[2 * LEN..3 * LEN]).try_into().unwrap());
        store_avx2(out3, (&mut out[3 * LEN..4 * LEN]).try_into().unwrap());
        processed += LEN / 4; // 8 pixels processed
    }

    processed
}

/// Safety note: does not write uninit data in `out`.
pub(super) fn interleave4_32b(inp: &[&[u8]; 4], out: &mut [MaybeUninit<u8>]) -> usize {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: we just checked for avx2.
        unsafe { interleave4_32b_avx2(inp, out) }
    } else {
        0
    }
}

// ============================================================================
// 3-channel 8-bit interleaving (RGB u8)
// ============================================================================

#[target_feature(enable = "ssse3")]
fn interleave3_8b_ssse3(inp: &[&[u8]; 3], out: &mut [MaybeUninit<u8>]) -> usize {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_or_si128, _mm_shuffle_epi8, _mm_storeu_si128,
    };

    let [r, g, b] = inp;

    // Process 16 pixels at a time -> 48 bytes output
    const LEN: usize = 16;
    let mut processed = 0;

    // Shuffle masks to interleave RGB
    // Input: r0-r15, g0-g15, b0-b15
    // Output: r0,g0,b0, r1,g1,b1, ... r15,g15,b15 (48 bytes)

    // For first 16 output bytes (pixels 0-5, partial 6):
    // r0,g0,b0, r1,g1,b1, r2,g2,b2, r3,g3,b3, r4,g4,b4, r5 (16 bytes)
    let shuffle_r0: [u8; 16] = [0, 0x80, 0x80, 1, 0x80, 0x80, 2, 0x80, 0x80, 3, 0x80, 0x80, 4, 0x80, 0x80, 5];
    let shuffle_g0: [u8; 16] = [0x80, 0, 0x80, 0x80, 1, 0x80, 0x80, 2, 0x80, 0x80, 3, 0x80, 0x80, 4, 0x80, 0x80];
    let shuffle_b0: [u8; 16] = [0x80, 0x80, 0, 0x80, 0x80, 1, 0x80, 0x80, 2, 0x80, 0x80, 3, 0x80, 0x80, 4, 0x80];

    // For second 16 output bytes (partial 5, pixels 6-10, partial 11):
    let shuffle_r1: [u8; 16] = [0x80, 0x80, 6, 0x80, 0x80, 7, 0x80, 0x80, 8, 0x80, 0x80, 9, 0x80, 0x80, 10, 0x80];
    let shuffle_g1: [u8; 16] = [5, 0x80, 0x80, 6, 0x80, 0x80, 7, 0x80, 0x80, 8, 0x80, 0x80, 9, 0x80, 0x80, 10];
    let shuffle_b1: [u8; 16] = [0x80, 5, 0x80, 0x80, 6, 0x80, 0x80, 7, 0x80, 0x80, 8, 0x80, 0x80, 9, 0x80, 0x80];

    // For third 16 output bytes (partial 10, pixels 11-15):
    let shuffle_r2: [u8; 16] = [0x80, 11, 0x80, 0x80, 12, 0x80, 0x80, 13, 0x80, 0x80, 14, 0x80, 0x80, 15, 0x80, 0x80];
    let shuffle_g2: [u8; 16] = [0x80, 0x80, 11, 0x80, 0x80, 12, 0x80, 0x80, 13, 0x80, 0x80, 14, 0x80, 0x80, 15, 0x80];
    let shuffle_b2: [u8; 16] = [10, 0x80, 0x80, 11, 0x80, 0x80, 12, 0x80, 0x80, 13, 0x80, 0x80, 14, 0x80, 0x80, 15];

    fn load_shuffle(data: &[u8; 16]) -> __m128i {
        unsafe { _mm_loadu_si128(data.as_ptr() as *const _) }
    }

    let shuf_r0 = load_shuffle(&shuffle_r0);
    let shuf_g0 = load_shuffle(&shuffle_g0);
    let shuf_b0 = load_shuffle(&shuffle_b0);
    let shuf_r1 = load_shuffle(&shuffle_r1);
    let shuf_g1 = load_shuffle(&shuffle_g1);
    let shuf_b1 = load_shuffle(&shuffle_b1);
    let shuf_r2 = load_shuffle(&shuffle_r2);
    let shuf_g2 = load_shuffle(&shuffle_g2);
    let shuf_b2 = load_shuffle(&shuffle_b2);

    for (((r, g), b), out) in r
        .chunks_exact(LEN)
        .zip(g.chunks_exact(LEN))
        .zip(b.chunks_exact(LEN))
        .zip(out.chunks_exact_mut(LEN * 3))
    {
        // SAFETY: We have target_feature enabled for ssse3
        unsafe {
            let rv = load_shuffle(r.try_into().unwrap());
            let gv = load_shuffle(g.try_into().unwrap());
            let bv = load_shuffle(b.try_into().unwrap());

            let out0 = _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(rv, shuf_r0),
                    _mm_shuffle_epi8(gv, shuf_g0),
                ),
                _mm_shuffle_epi8(bv, shuf_b0),
            );

            let out1 = _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(rv, shuf_r1),
                    _mm_shuffle_epi8(gv, shuf_g1),
                ),
                _mm_shuffle_epi8(bv, shuf_b1),
            );

            let out2 = _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(rv, shuf_r2),
                    _mm_shuffle_epi8(gv, shuf_g2),
                ),
                _mm_shuffle_epi8(bv, shuf_b2),
            );

            _mm_storeu_si128(out[0..16].as_mut_ptr() as *mut _, out0);
            _mm_storeu_si128(out[16..32].as_mut_ptr() as *mut _, out1);
            _mm_storeu_si128(out[32..48].as_mut_ptr() as *mut _, out2);
        }
        processed += LEN;
    }

    processed
}

/// Safety note: does not write uninit data in `out`.
pub(super) fn interleave3_8b(inp: &[&[u8]; 3], out: &mut [MaybeUninit<u8>]) -> usize {
    if is_x86_feature_detected!("ssse3") {
        // SAFETY: we just checked for ssse3.
        unsafe { interleave3_8b_ssse3(inp, out) }
    } else {
        0
    }
}

// ============================================================================
// 4-channel 8-bit interleaving (RGBA u8)
// ============================================================================

#[target_feature(enable = "ssse3")]
fn interleave4_8b_ssse3(inp: &[&[u8]; 4], out: &mut [MaybeUninit<u8>]) -> usize {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
        _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    };

    let [r, g, b, a] = inp;

    // Process 16 pixels at a time -> 64 bytes output
    const LEN: usize = 16;
    let mut processed = 0;

    for ((((r, g), b), a), out) in r
        .chunks_exact(LEN)
        .zip(g.chunks_exact(LEN))
        .zip(b.chunks_exact(LEN))
        .zip(a.chunks_exact(LEN))
        .zip(out.chunks_exact_mut(LEN * 4))
    {
        // SAFETY: We have target_feature enabled for ssse3
        unsafe {
            let rv: __m128i = _mm_loadu_si128(r.as_ptr() as *const _);
            let gv: __m128i = _mm_loadu_si128(g.as_ptr() as *const _);
            let bv: __m128i = _mm_loadu_si128(b.as_ptr() as *const _);
            let av: __m128i = _mm_loadu_si128(a.as_ptr() as *const _);

            // Interleave r,g and b,a
            let rg_lo = _mm_unpacklo_epi8(rv, gv); // r0,g0,r1,g1,...,r7,g7
            let rg_hi = _mm_unpackhi_epi8(rv, gv); // r8,g8,...,r15,g15
            let ba_lo = _mm_unpacklo_epi8(bv, av); // b0,a0,b1,a1,...,b7,a7
            let ba_hi = _mm_unpackhi_epi8(bv, av); // b8,a8,...,b15,a15

            // Interleave rg and ba
            let rgba_0 = _mm_unpacklo_epi16(rg_lo, ba_lo); // r0,g0,b0,a0,r1,g1,b1,a1,...
            let rgba_1 = _mm_unpackhi_epi16(rg_lo, ba_lo); // r4,g4,b4,a4,...,r7,g7,b7,a7
            let rgba_2 = _mm_unpacklo_epi16(rg_hi, ba_hi); // r8,g8,b8,a8,...
            let rgba_3 = _mm_unpackhi_epi16(rg_hi, ba_hi); // r12,g12,b12,a12,...

            _mm_storeu_si128(out[0..16].as_mut_ptr() as *mut _, rgba_0);
            _mm_storeu_si128(out[16..32].as_mut_ptr() as *mut _, rgba_1);
            _mm_storeu_si128(out[32..48].as_mut_ptr() as *mut _, rgba_2);
            _mm_storeu_si128(out[48..64].as_mut_ptr() as *mut _, rgba_3);
        }
        processed += LEN;
    }

    processed
}

#[target_feature(enable = "avx2")]
fn interleave4_8b_avx2(inp: &[&[u8]; 4], out: &mut [MaybeUninit<u8>]) -> usize {
    use std::arch::x86_64::{
        _mm256_unpackhi_epi8, _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
    };

    let [r, g, b, a] = inp;

    // Process 32 pixels at a time -> 128 bytes output
    const LEN: usize = 32;
    let mut processed = 0;

    for ((((r, g), b), a), out) in r
        .chunks_exact(LEN)
        .zip(g.chunks_exact(LEN))
        .zip(b.chunks_exact(LEN))
        .zip(a.chunks_exact(LEN))
        .zip(out.chunks_exact_mut(LEN * 4))
    {
        let rv = load_avx2(r.try_into().unwrap());
        let gv = load_avx2(g.try_into().unwrap());
        let bv = load_avx2(b.try_into().unwrap());
        let av = load_avx2(a.try_into().unwrap());

        // Interleave r,g and b,a (works within 128-bit lanes)
        let rg_lo = _mm256_unpacklo_epi8(rv, gv);
        let rg_hi = _mm256_unpackhi_epi8(rv, gv);
        let ba_lo = _mm256_unpacklo_epi8(bv, av);
        let ba_hi = _mm256_unpackhi_epi8(bv, av);

        // Interleave rg and ba
        let rgba_0 = _mm256_unpacklo_epi16(rg_lo, ba_lo);
        let rgba_1 = _mm256_unpackhi_epi16(rg_lo, ba_lo);
        let rgba_2 = _mm256_unpacklo_epi16(rg_hi, ba_hi);
        let rgba_3 = _mm256_unpackhi_epi16(rg_hi, ba_hi);

        // AVX2 unpack works within 128-bit lanes, so we need to permute
        // to get contiguous output. Each result has data from both lanes mixed.
        let out0 = _mm256_permute2f128_si256::<0x20>(rgba_0, rgba_1);
        let out1 = _mm256_permute2f128_si256::<0x20>(rgba_2, rgba_3);
        let out2 = _mm256_permute2f128_si256::<0x31>(rgba_0, rgba_1);
        let out3 = _mm256_permute2f128_si256::<0x31>(rgba_2, rgba_3);

        store_avx2(out0, (&mut out[0..32]).try_into().unwrap());
        store_avx2(out1, (&mut out[32..64]).try_into().unwrap());
        store_avx2(out2, (&mut out[64..96]).try_into().unwrap());
        store_avx2(out3, (&mut out[96..128]).try_into().unwrap());
        processed += LEN;
    }

    processed
}

/// Safety note: does not write uninit data in `out`.
pub(super) fn interleave4_8b(inp: &[&[u8]; 4], out: &mut [MaybeUninit<u8>]) -> usize {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: we just checked for avx2.
        unsafe { interleave4_8b_avx2(inp, out) }
    } else if is_x86_feature_detected!("ssse3") {
        // SAFETY: we just checked for ssse3.
        unsafe { interleave4_8b_ssse3(inp, out) }
    } else {
        0
    }
}
