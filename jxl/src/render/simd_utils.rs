// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! SIMD utilities for interleaving and deinterleaving channel data.

use jxl_simd::{F32SimdVec, simd_function};

simd_function!(
    interleave_2_dispatch,
    d: D,
    /// Interleave 2 planar channels into packed format.
    pub fn interleave_2(a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
        let len = D::F32Vec::LEN;
        let pixels = a.len().min(b.len());
        let mut processed = 0;

        for ((chunk_a, chunk_b), chunk_out) in a
            .chunks_exact(len)
            .zip(b.chunks_exact(len))
            .zip(out.chunks_exact_mut(len * 2))
        {
            let va = D::F32Vec::load(d, chunk_a);
            let vb = D::F32Vec::load(d, chunk_b);
            D::F32Vec::store_interleaved_2(va, vb, chunk_out);
            processed += len;
        }

        // Scalar fallback for remainder
        for i in processed..pixels {
            out[i * 2] = a[i];
            out[i * 2 + 1] = b[i];
        }

        pixels
    }
);

simd_function!(
    deinterleave_2_dispatch,
    d: D,
    /// Deinterleave packed format into 2 planar channels.
    pub fn deinterleave_2(input: &[f32], a: &mut [f32], b: &mut [f32]) -> usize {
        let len = D::F32Vec::LEN;
        let pixels = (input.len() / 2).min(a.len()).min(b.len());
        let mut processed = 0;

        for ((chunk_a, chunk_b), chunk_in) in a
            .chunks_exact_mut(len)
            .zip(b.chunks_exact_mut(len))
            .zip(input.chunks_exact(len * 2))
        {
            let (va, vb) = D::F32Vec::load_deinterleaved_2(d, chunk_in);
            va.store(chunk_a);
            vb.store(chunk_b);
            processed += len;
        }

        // Scalar fallback for remainder
        for i in processed..pixels {
            a[i] = input[i * 2];
            b[i] = input[i * 2 + 1];
        }

        pixels
    }
);

simd_function!(
    interleave_3_dispatch,
    d: D,
    /// Interleave 3 planar channels into packed RGB format.
    ///
    /// Takes 3 slices of planar channel data and interleaves them into packed
    /// RGB format: [R0, G0, B0, R1, G1, B1, ...].
    ///
    /// # Arguments
    /// * `a`, `b`, `c` - Input planar channel slices (R, G, B)
    /// * `out` - Output buffer, must have length >= min(a.len(), b.len(), c.len()) * 3
    ///
    /// # Returns
    /// Number of pixels processed (length of each input channel used)
    pub fn interleave_3(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) -> usize {
        let len = D::F32Vec::LEN;
        let pixels = a.len().min(b.len()).min(c.len());
        let mut processed = 0;

        for (((chunk_a, chunk_b), chunk_c), chunk_out) in a
            .chunks_exact(len)
            .zip(b.chunks_exact(len))
            .zip(c.chunks_exact(len))
            .zip(out.chunks_exact_mut(len * 3))
        {
            let va = D::F32Vec::load(d, chunk_a);
            let vb = D::F32Vec::load(d, chunk_b);
            let vc = D::F32Vec::load(d, chunk_c);
            D::F32Vec::store_interleaved_3(va, vb, vc, chunk_out);
            processed += len;
        }

        // Scalar fallback for remainder
        for i in processed..pixels {
            out[i * 3] = a[i];
            out[i * 3 + 1] = b[i];
            out[i * 3 + 2] = c[i];
        }

        pixels
    }
);

simd_function!(
    deinterleave_3_dispatch,
    d: D,
    /// Deinterleave packed RGB format into 3 planar channels.
    ///
    /// Takes packed RGB data [R0, G0, B0, R1, G1, B1, ...] and splits it into
    /// separate planar channels.
    ///
    /// # Arguments
    /// * `input` - Packed RGB input, length must be divisible by 3
    /// * `a`, `b`, `c` - Output planar channel slices (R, G, B)
    ///
    /// # Returns
    /// Number of pixels processed
    pub fn deinterleave_3(input: &[f32], a: &mut [f32], b: &mut [f32], c: &mut [f32]) -> usize {
        let len = D::F32Vec::LEN;
        let pixels = (input.len() / 3).min(a.len()).min(b.len()).min(c.len());
        let mut processed = 0;

        for (((chunk_a, chunk_b), chunk_c), chunk_in) in a
            .chunks_exact_mut(len)
            .zip(b.chunks_exact_mut(len))
            .zip(c.chunks_exact_mut(len))
            .zip(input.chunks_exact(len * 3))
        {
            let (va, vb, vc) = D::F32Vec::load_deinterleaved_3(d, chunk_in);
            va.store(chunk_a);
            vb.store(chunk_b);
            vc.store(chunk_c);
            processed += len;
        }

        // Scalar fallback for remainder
        for i in processed..pixels {
            a[i] = input[i * 3];
            b[i] = input[i * 3 + 1];
            c[i] = input[i * 3 + 2];
        }

        pixels
    }
);

simd_function!(
    interleave_4_dispatch,
    d: D,
    /// Interleave 4 planar channels into packed RGBA format.
    pub fn interleave_4(a: &[f32], b: &[f32], c: &[f32], e: &[f32], out: &mut [f32]) -> usize {
        let len = D::F32Vec::LEN;
        let pixels = a.len().min(b.len()).min(c.len()).min(e.len());
        let mut processed = 0;

        for ((((chunk_a, chunk_b), chunk_c), chunk_d), chunk_out) in a
            .chunks_exact(len)
            .zip(b.chunks_exact(len))
            .zip(c.chunks_exact(len))
            .zip(e.chunks_exact(len))
            .zip(out.chunks_exact_mut(len * 4))
        {
            let va = D::F32Vec::load(d, chunk_a);
            let vb = D::F32Vec::load(d, chunk_b);
            let vc = D::F32Vec::load(d, chunk_c);
            let vd = D::F32Vec::load(d, chunk_d);
            D::F32Vec::store_interleaved_4(va, vb, vc, vd, chunk_out);
            processed += len;
        }

        // Scalar fallback for remainder
        for i in processed..pixels {
            out[i * 4] = a[i];
            out[i * 4 + 1] = b[i];
            out[i * 4 + 2] = c[i];
            out[i * 4 + 3] = e[i];
        }

        pixels
    }
);

simd_function!(
    deinterleave_4_dispatch,
    d: D,
    /// Deinterleave packed RGBA format into 4 planar channels.
    pub fn deinterleave_4(
        input: &[f32],
        a: &mut [f32],
        b: &mut [f32],
        c: &mut [f32],
        e: &mut [f32],
    ) -> usize {
        let len = D::F32Vec::LEN;
        let pixels = (input.len() / 4).min(a.len()).min(b.len()).min(c.len()).min(e.len());
        let mut processed = 0;

        for ((((chunk_a, chunk_b), chunk_c), chunk_d), chunk_in) in a
            .chunks_exact_mut(len)
            .zip(b.chunks_exact_mut(len))
            .zip(c.chunks_exact_mut(len))
            .zip(e.chunks_exact_mut(len))
            .zip(input.chunks_exact(len * 4))
        {
            let (va, vb, vc, vd) = D::F32Vec::load_deinterleaved_4(d, chunk_in);
            va.store(chunk_a);
            vb.store(chunk_b);
            vc.store(chunk_c);
            vd.store(chunk_d);
            processed += len;
        }

        // Scalar fallback for remainder
        for i in processed..pixels {
            a[i] = input[i * 4];
            b[i] = input[i * 4 + 1];
            c[i] = input[i * 4 + 2];
            e[i] = input[i * 4 + 3];
        }

        pixels
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleave_deinterleave_2_roundtrip() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let mut packed = vec![0.0; 16];
        let processed = interleave_2_dispatch(&a, &b, &mut packed);
        assert_eq!(processed, 8);

        // Check interleaved format
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 10.0);
        assert_eq!(packed[2], 2.0);
        assert_eq!(packed[3], 20.0);

        // Deinterleave back
        let mut a_out = vec![0.0; 8];
        let mut b_out = vec![0.0; 8];
        let processed = deinterleave_2_dispatch(&packed, &mut a_out, &mut b_out);
        assert_eq!(processed, 8);

        assert_eq!(a_out, a);
        assert_eq!(b_out, b);
    }

    #[test]
    fn test_interleave_deinterleave_3_roundtrip() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let c = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];

        let mut packed = vec![0.0; 24];
        let processed = interleave_3_dispatch(&a, &b, &c, &mut packed);
        assert_eq!(processed, 8);

        // Check interleaved format
        assert_eq!(packed[0], 1.0); // R0
        assert_eq!(packed[1], 10.0); // G0
        assert_eq!(packed[2], 100.0); // B0
        assert_eq!(packed[3], 2.0); // R1
        assert_eq!(packed[4], 20.0); // G1
        assert_eq!(packed[5], 200.0); // B1

        // Deinterleave back
        let mut a_out = vec![0.0; 8];
        let mut b_out = vec![0.0; 8];
        let mut c_out = vec![0.0; 8];
        let processed = deinterleave_3_dispatch(&packed, &mut a_out, &mut b_out, &mut c_out);
        assert_eq!(processed, 8);

        assert_eq!(a_out, a);
        assert_eq!(b_out, b);
        assert_eq!(c_out, c);
    }

    #[test]
    fn test_interleave_deinterleave_4_roundtrip() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let c = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
        let d = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0];

        let mut packed = vec![0.0; 32];
        let processed = interleave_4_dispatch(&a, &b, &c, &d, &mut packed);
        assert_eq!(processed, 8);

        // Check interleaved format
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 10.0);
        assert_eq!(packed[2], 100.0);
        assert_eq!(packed[3], 1000.0);
        assert_eq!(packed[4], 2.0);
        assert_eq!(packed[5], 20.0);

        // Deinterleave back
        let mut a_out = vec![0.0; 8];
        let mut b_out = vec![0.0; 8];
        let mut c_out = vec![0.0; 8];
        let mut d_out = vec![0.0; 8];
        let processed =
            deinterleave_4_dispatch(&packed, &mut a_out, &mut b_out, &mut c_out, &mut d_out);
        assert_eq!(processed, 8);

        assert_eq!(a_out, a);
        assert_eq!(b_out, b);
        assert_eq!(c_out, c);
        assert_eq!(d_out, d);
    }
}
