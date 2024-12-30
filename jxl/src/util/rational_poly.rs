// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Computes `(p0 + p1 x + p2 x^2 + ...) / (q0 + q1 x + q2 x^2 + ...)`.
///
/// # Panics
/// Panics if either `P` or `Q` is zero.
#[inline]
pub fn eval_rational_poly<const P: usize, const Q: usize>(x: f32, p: [f32; P], q: [f32; Q]) -> f32 {
    let yp = p.into_iter().rev().reduce(|yp, p| yp * x + p).unwrap();
    let yq = q.into_iter().rev().reduce(|yq, q| yq * x + q).unwrap();
    yp / yq
}
