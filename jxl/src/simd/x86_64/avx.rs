// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::super::{F32SimdVec, ScalarDescriptor, SimdDescriptor};
use std::{
    arch::x86_64::{
        __m256, __m256i, _mm_loadu_ps, _mm_storeu_ps, _mm_unpackhi_ps, _mm_unpacklo_ps,
        _mm256_add_ps, _mm256_andnot_si256, _mm256_castps_si256, _mm256_castsi256_ps,
        _mm256_div_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
        _mm256_maskload_ps, _mm256_maskstore_ps, _mm256_max_ps, _mm256_mul_ps,
        _mm256_permute2f128_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_shuffle_ps,
        _mm256_storeu_ps, _mm256_sub_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps, _mm256_xor_si256,
    },
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

// Safety invariant: this type is only ever constructed if avx2 and fma are available.
#[derive(Clone, Copy, Debug)]
pub struct AvxDescriptor(());

impl AvxDescriptor {
    /// Safety:
    /// The caller must guarantee that the "avx2" and "fma" target features are available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl AvxDescriptor {
    #[target_feature(enable = "avx")]
    #[inline]
    fn transpose4x4f32(
        self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
    ) {
        assert!(input_stride >= 4);
        assert!(input.len() >= input_stride.checked_mul(3).unwrap().checked_add(4).unwrap());
        assert!(
            output.len()
                >= output_stride
                    .checked_mul(3)
                    .unwrap()
                    .checked_add(4)
                    .unwrap()
        );

        // SAFETY: input is verified to be large enough for this pointer arithmetic.
        let (p0, p1, p2, p3) = unsafe {
            (
                _mm_loadu_ps(input.as_ptr()),
                _mm_loadu_ps(input.as_ptr().add(input_stride)),
                _mm_loadu_ps(input.as_ptr().add(2 * input_stride)),
                _mm_loadu_ps(input.as_ptr().add(3 * input_stride)),
            )
        };

        let q0 = _mm_unpacklo_ps(p0, p2);
        let q1 = _mm_unpacklo_ps(p1, p3);
        let q2 = _mm_unpackhi_ps(p0, p2);
        let q3 = _mm_unpackhi_ps(p1, p3);

        let r0 = _mm_unpacklo_ps(q0, q1);
        let r1 = _mm_unpackhi_ps(q0, q1);
        let r2 = _mm_unpacklo_ps(q2, q3);
        let r3 = _mm_unpackhi_ps(q2, q3);

        // SAFETY: output is verified to be large enough for this pointer arithmetic.
        unsafe {
            _mm_storeu_ps(output.as_mut_ptr(), r0);
            _mm_storeu_ps(output.as_mut_ptr().add(output_stride), r1);
            _mm_storeu_ps(output.as_mut_ptr().add(2 * output_stride), r2);
            _mm_storeu_ps(output.as_mut_ptr().add(3 * output_stride), r3);
        }
    }
}

impl SimdDescriptor for AvxDescriptor {
    type F32Vec = F32VecAvx;
    fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: we just checked avx2 and fma.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn transpose<const ROWS: usize, const COLS: usize>(self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), ROWS * COLS);
        assert_eq!(output.len(), ROWS * COLS);

        if ROWS.is_multiple_of(8) && COLS.is_multiple_of(8) {
            for r in (0..ROWS).step_by(8) {
                let input_row = &input[r * COLS..];
                for c in (0..COLS).step_by(8) {
                    let output_row = &mut output[c * ROWS..];
                    // SAFETY: We know avx is available from the safety invariant on `self`.
                    unsafe {
                        Self::transpose8x8f32(&input_row[c..], COLS, &mut output_row[r..], ROWS);
                    }
                }
            }
        } else if ROWS.is_multiple_of(4) && COLS.is_multiple_of(4) {
            for r in (0..ROWS).step_by(4) {
                let input_row = &input[r * COLS..];
                for c in (0..COLS).step_by(4) {
                    let output_row = &mut output[c * ROWS..];
                    // SAFETY: We know avx is available from the safety invariant on `self`.
                    unsafe {
                        self.transpose4x4f32(&input_row[c..], COLS, &mut output_row[r..], ROWS);
                    }
                }
            }
        } else {
            let scalar = ScalarDescriptor {};
            scalar.transpose::<ROWS, COLS>(input, output);
        }
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn inner<R>(d: AvxDescriptor, f: impl FnOnce(AvxDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees avx2 and fma.
        unsafe { inner(self, f) }
    }
}

impl AvxDescriptor {
    #[target_feature(enable = "avx2")]
    #[inline]
    fn transpose8x8f32(
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
    ) {
        assert!(input_stride >= 8);
        assert!(input.len() >= input_stride.checked_mul(7).unwrap().checked_add(8).unwrap());
        assert!(output_stride >= 8);
        assert!(
            output.len()
                >= output_stride
                    .checked_mul(7)
                    .unwrap()
                    .checked_add(8)
                    .unwrap()
        );

        let (r0, r1, r2, r3, r4, r5, r6, r7);
        // SAFETY: The asserts at the top of the function guarantee that the input slice is large
        // enough for these memory operations and that the offset calculations don't overflow.
        unsafe {
            r0 = _mm256_loadu_ps(input.as_ptr());
            r1 = _mm256_loadu_ps(input.as_ptr().add(input_stride));
            r2 = _mm256_loadu_ps(input.as_ptr().add(input_stride * 2));
            r3 = _mm256_loadu_ps(input.as_ptr().add(input_stride * 3));
            r4 = _mm256_loadu_ps(input.as_ptr().add(input_stride * 4));
            r5 = _mm256_loadu_ps(input.as_ptr().add(input_stride * 5));
            r6 = _mm256_loadu_ps(input.as_ptr().add(input_stride * 6));
            r7 = _mm256_loadu_ps(input.as_ptr().add(input_stride * 7));
        }

        // Stage 1: Unpack low/high pairs
        let t0 = _mm256_unpacklo_ps(r0, r1);
        let t1 = _mm256_unpackhi_ps(r0, r1);
        let t2 = _mm256_unpacklo_ps(r2, r3);
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        // Stage 2: Shuffle to group 32-bit elements
        let s0 = _mm256_shuffle_ps(t0, t2, 0b01_00_01_00); // _MM_SHUFFLE(1,0,1,0)
        let s1 = _mm256_shuffle_ps(t0, t2, 0b11_10_11_10); // _MM_SHUFFLE(3,2,3,2)
        let s2 = _mm256_shuffle_ps(t1, t3, 0b01_00_01_00);
        let s3 = _mm256_shuffle_ps(t1, t3, 0b11_10_11_10);
        let s4 = _mm256_shuffle_ps(t4, t6, 0b01_00_01_00);
        let s5 = _mm256_shuffle_ps(t4, t6, 0b11_10_11_10);
        let s6 = _mm256_shuffle_ps(t5, t7, 0b01_00_01_00);
        let s7 = _mm256_shuffle_ps(t5, t7, 0b11_10_11_10);

        // Stage 3: 128-bit permute to finalize transpose
        let c0 = _mm256_permute2f128_ps(s0, s4, 0x20);
        let c1 = _mm256_permute2f128_ps(s1, s5, 0x20);
        let c2 = _mm256_permute2f128_ps(s2, s6, 0x20);
        let c3 = _mm256_permute2f128_ps(s3, s7, 0x20);
        let c4 = _mm256_permute2f128_ps(s0, s4, 0x31);
        let c5 = _mm256_permute2f128_ps(s1, s5, 0x31);
        let c6 = _mm256_permute2f128_ps(s2, s6, 0x31);
        let c7 = _mm256_permute2f128_ps(s3, s7, 0x31);

        // SAFETY: The asserts at the top of the function guarantee that the output slice is large
        // enough for these memory operations and that the offset calculations don't overflow.
        unsafe {
            _mm256_storeu_ps(output.as_mut_ptr(), c0);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride), c1);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride * 2), c2);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride * 3), c3);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride * 4), c4);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride * 5), c5);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride * 6), c6);
            _mm256_storeu_ps(output.as_mut_ptr().add(output_stride * 7), c7);
        }
    }
}

// TODO(veluca): retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_avx {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "fma,avx2")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff avx2 and fma are available.
            unsafe { inner(self, $($arg),*) }
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecAvx(__m256, AvxDescriptor);

#[target_feature(enable = "avx2")]
#[inline]
fn get_mask(size: usize) -> __m256i {
    const MASKS: [i32; 15] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0];
    // SAFETY: the pointer arithmetic is safe because:
    // `(size - 1) & 7` is between 0 and 7 (inclusive);
    // `7 - ((size - 1) & 7)` is therefore also between 0 and 7 (inclusive);
    // all starting indices between 0 and 7 into a 15-element array leave at least 8 elements to load.
    unsafe { _mm256_loadu_si256(MASKS.as_ptr().add(7 - ((size - 1) & 7)) as *const _) }
}

impl F32SimdVec for F32VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx is available
        // from the safety invariant on `d`.
        Self(unsafe { _mm256_loadu_ps(mem.as_ptr()) }, d)
    }

    #[inline(always)]
    fn load_partial(d: Self::Descriptor, size: usize, mem: &[f32]) -> Self {
        debug_assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        // Fast path: avoid mask setup overhead when loading full vectors
        // This optimization skips the expensive mask creation and masked load when size == LEN
        if size == Self::LEN {
            return Self::load(d, mem);
        }
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx is available
        // from the safety invariant on `d`.
        Self(
            unsafe { _mm256_maskload_ps(mem.as_ptr(), get_mask(size)) },
            d,
        )
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx is available
        // from the safety invariant on `self.1`.
        unsafe { _mm256_storeu_ps(mem.as_mut_ptr(), self.0) }
    }
    #[inline(always)]
    fn store_partial(&self, size: usize, mem: &mut [f32]) {
        assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        if size == Self::LEN {
            return self.store(mem);
        }
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx is available
        // from the safety invariant on `d`.
        unsafe { _mm256_maskstore_ps(mem.as_mut_ptr(), get_mask(size), self.0) }
    }

    fn_avx!(this: F32VecAvx, fn mul_add(mul: F32VecAvx, add: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_fmadd_ps(this.0, mul.0, add.0), this.1)
    });

    fn_avx!(this: F32VecAvx, fn neg_mul_add(mul: F32VecAvx, add: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_fnmadd_ps(this.0, mul.0, add.0), this.1)
    });

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know avx is available from the safety invariant on `d`.
        unsafe { Self(_mm256_set1_ps(v), d) }
    }

    fn_avx!(this: F32VecAvx, fn abs() -> F32VecAvx {
        F32VecAvx(
            _mm256_castsi256_ps(_mm256_andnot_si256(
                _mm256_set1_epi32(i32::MIN),
                _mm256_castps_si256(this.0),
            )),
            this.1)
    });

    fn_avx!(this: F32VecAvx, fn neg() -> F32VecAvx {
        F32VecAvx(
            _mm256_castsi256_ps(_mm256_xor_si256(
                _mm256_set1_epi32(i32::MIN),
                _mm256_castps_si256(this.0),
            )),
            this.1)
    });

    fn_avx!(this: F32VecAvx, fn max(other: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_max_ps(this.0, other.0), this.1)
    });
}

impl Add<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn add(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn sub(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn mul(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn div(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn add_assign(rhs: F32VecAvx) {
        this.0 = _mm256_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn sub_assign(rhs: F32VecAvx) {
        this.0 = _mm256_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn mul_assign(rhs: F32VecAvx) {
        this.0 = _mm256_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn div_assign(rhs: F32VecAvx) {
        this.0 = _mm256_div_ps(this.0, rhs.0)
    });
}
