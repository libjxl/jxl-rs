// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs;
use std::path::{Path, PathBuf};

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use jxl::api::{Event, JxlDecoder, JxlDecoderOptions};
use jxl_cli::dec::{OutputDataType, decode_frames};

fn decode_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.sampling_mode(SamplingMode::Flat);

    let paths: Vec<PathBuf> = std::env::var("JXL_FILES").map_or_else(
        |_| {
            let root_test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("jxl")
                .join("resources")
                .join("test");
            [
                &root_test_dir,
                &root_test_dir.join("conformance_test_images"),
            ]
            .iter()
            .flat_map(|path| {
                fs::read_dir(path)
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "jxl"))
                    .collect::<Vec<PathBuf>>()
            })
            .collect()
        },
        |csv| csv.split(',').map(PathBuf::from).collect(),
    );

    for path in paths {
        let bytes = fs::read(&path).unwrap();
        let mut header_input = bytes.as_slice();
        let mut header_decoder = JxlDecoder::new(JxlDecoderOptions::default());
        assert_eq!(
            header_decoder
                .process(&mut header_input, None, None)
                .unwrap(),
            Event::BasicInfo,
        );
        let basic_info = header_decoder.basic_info().unwrap();
        let pixel_count = basic_info.size.0 * basic_info.size.1;
        group.throughput(criterion::Throughput::Elements(pixel_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(path.to_string_lossy()),
            &bytes,
            |b, bytes| {
                let mut last_output = None;
                b.iter(|| {
                    let mut input = bytes.as_slice();
                    let (output, _) = decode_frames(
                        &mut input,
                        JxlDecoderOptions::default(),
                        None,
                        None,
                        &[
                            OutputDataType::U8,
                            OutputDataType::U16,
                            OutputDataType::F16,
                            OutputDataType::F32,
                        ],
                        true,
                        true,
                        false,
                        None,
                        false,
                        last_output.take(),
                    )
                    .unwrap();
                    last_output = Some(output);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = decode;
    config = Criterion::default().sample_size(50);
    targets = decode_benches
);
criterion_main!(decode);
