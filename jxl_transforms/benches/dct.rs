// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use jxl_simd::{bench_all_instruction_sets, SimdDescriptor};
use jxl_transforms::dct::{compute_scaled_dct, DCT1DImpl, DCT1D};
use jxl_transforms::idct2d::*;
use jxl_transforms::transform_map::MAX_COEFF_AREA;

fn bench_idct2d<D: SimdDescriptor>(d: D, c: &mut BenchmarkGroup<'_, impl Measurement>, name: &str) {
    let mut data = vec![1.0; MAX_COEFF_AREA];

    macro_rules! run {
        ($fun: ident, $name: literal, $sz: expr) => {
            let id = BenchmarkId::new(name, format_args!("{}", $name));
            c.bench_function(id, |b| {
                b.iter(|| {
                    d.call(
                        #[inline(always)]
                        |d| $fun(d, &mut data[..$sz]),
                    );
                })
            });
        };
    }

    run!(idct2d_2_2, "2x2", 2 * 2);
    run!(idct2d_4_4, "4x4", 4 * 4);
    run!(idct2d_4_8, "4x8", 4 * 8);
    run!(idct2d_8_4, "8x4", 4 * 8);
    run!(idct2d_8_8, "8x8", 8 * 8);
    run!(idct2d_16_8, "16x8", 16 * 8);
    run!(idct2d_8_16, "8x16", 8 * 16);
    run!(idct2d_16_16, "16x16", 16 * 16);
    run!(idct2d_32_8, "32x8", 32 * 8);
    run!(idct2d_8_32, "8x32", 8 * 32);
    run!(idct2d_32_16, "32x16", 32 * 16);
    run!(idct2d_16_32, "16x32", 16 * 32);
    run!(idct2d_32_32, "32x32", 32 * 32);
    run!(idct2d_64_32, "64x32", 64 * 32);
    run!(idct2d_32_64, "32x64", 32 * 64);
    run!(idct2d_64_64, "64x64", 64 * 64);
    run!(idct2d_128_64, "128x64", 128 * 64);
    run!(idct2d_64_128, "64x128", 64 * 128);
    run!(idct2d_128_128, "128x128", 128 * 128);
    run!(idct2d_256_128, "256x128", 256 * 128);
    run!(idct2d_128_256, "128x256", 128 * 256);
    run!(idct2d_256_256, "256x256", 256 * 256);
}

fn bench_compute_scaled_dct<D: SimdDescriptor>(
    d: D,
    c: &mut BenchmarkGroup<'_, impl Measurement>,
    name: &str,
) {
    fn run_size<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
        c: &mut BenchmarkGroup<'_, impl Measurement>,
        d: D,
        name: &str,
    ) where
        DCT1DImpl<ROWS>: DCT1D,
        DCT1DImpl<COLS>: DCT1D,
    {
        let id = BenchmarkId::new(name, format_args!("{ROWS}x{COLS}"));

        let input_vec = vec![vec![1.0; COLS]; ROWS];
        let mut input = [[0.0; COLS]; ROWS];
        for (i, row) in input_vec.iter().enumerate() {
            input[i].copy_from_slice(row);
        }
        let mut input = [1.0; MAX_COEFF_AREA / 64];
        let mut output = vec![0.0; ROWS * COLS];
        c.bench_function(id, |b| {
            b.iter(|| {
                d.call(|d| compute_scaled_dct::<_, ROWS, COLS>(d, &mut input, &mut output));
            })
        });
    }

    run_size::<_, 2, 2>(c, d, name);
    run_size::<_, 4, 4>(c, d, name);
    run_size::<_, 8, 4>(c, d, name);
    run_size::<_, 4, 8>(c, d, name);
    run_size::<_, 8, 8>(c, d, name);
    run_size::<_, 8, 16>(c, d, name);
    run_size::<_, 16, 8>(c, d, name);
    run_size::<_, 16, 16>(c, d, name);
    run_size::<_, 32, 16>(c, d, name);
    run_size::<_, 16, 32>(c, d, name);
    run_size::<_, 32, 32>(c, d, name);
}

fn idct_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("idct2d");
    let g = &mut group;

    bench_all_instruction_sets!(bench_idct2d, g);

    group.finish();
}

fn compute_scaled_dct_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_scaled_dct");
    let g = &mut group;

    bench_all_instruction_sets!(bench_compute_scaled_dct, g);

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = idct_benches, compute_scaled_dct_benches
);
criterion_main!(benches);
