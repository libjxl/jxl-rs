// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use criterion::measurement::Measurement;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use jxl::var_dct::dct::{DCT1D, DCT1DImpl, IDCT1D, IDCT1DImpl, compute_scaled_dct, idct2d};
use jxl_simd::{SimdDescriptor, bench_all_instruction_sets};

fn bench_idct2d<D: SimdDescriptor>(d: D, c: &mut BenchmarkGroup<'_, impl Measurement>, name: &str) {
    fn run_size<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
        c: &mut BenchmarkGroup<'_, impl Measurement>,
        d: D,
        name: &str,
    ) where
        IDCT1DImpl<ROWS>: IDCT1D,
        IDCT1DImpl<COLS>: IDCT1D,
    {
        let id = BenchmarkId::new(name, format_args!("{ROWS}x{COLS}"));

        let mut data = vec![1.0; ROWS * COLS];
        let mut scratch = vec![0.0; ROWS * COLS];
        c.bench_function(id, |b| {
            b.iter(|| {
                d.call(|d| idct2d::<_, ROWS, COLS>(d, &mut data, &mut scratch));
            })
        });
    }

    run_size::<_, 2, 2>(c, d, name);
    run_size::<_, 4, 4>(c, d, name);
    run_size::<_, 4, 8>(c, d, name);
    run_size::<_, 8, 4>(c, d, name);
    run_size::<_, 8, 8>(c, d, name);
    run_size::<_, 16, 8>(c, d, name);
    run_size::<_, 8, 16>(c, d, name);
    run_size::<_, 16, 16>(c, d, name);
    run_size::<_, 32, 8>(c, d, name);
    run_size::<_, 8, 32>(c, d, name);
    run_size::<_, 32, 16>(c, d, name);
    run_size::<_, 16, 32>(c, d, name);
    run_size::<_, 32, 32>(c, d, name);
    run_size::<_, 64, 32>(c, d, name);
    run_size::<_, 32, 64>(c, d, name);
    run_size::<_, 64, 64>(c, d, name);
    run_size::<_, 128, 64>(c, d, name);
    run_size::<_, 64, 128>(c, d, name);
    run_size::<_, 128, 128>(c, d, name);
    run_size::<_, 256, 128>(c, d, name);
    run_size::<_, 128, 256>(c, d, name);
    run_size::<_, 256, 256>(c, d, name);
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
        let mut output = vec![0.0; ROWS * COLS];
        c.bench_function(id, |b| {
            b.iter(|| {
                d.call(|d| compute_scaled_dct::<_, ROWS, COLS>(d, input, &mut output));
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
