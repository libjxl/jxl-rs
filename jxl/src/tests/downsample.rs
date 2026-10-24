// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::decode::{DecodeParams, decode};

fn compare(data: &[u8]) {
    for factor in 1..=3 {
        let expected = decode::<f32>(
            data,
            DecodeParams {
                downsample_factor: factor,
                use_simple_pipeline: true,
                ..Default::default()
            },
        )
        .unwrap();
        for chunk in [usize::MAX, 4096] {
            #[cfg(not(any(target_family = "wasm", target_arch = "wasm32")))]
            let mut runner = super::parallel_runner::TestParallelRunner { max_threads: 4 };
            let actual = decode::<f32>(
                data,
                DecodeParams {
                    downsample_factor: factor,
                    chunk_size: chunk,
                    do_flush: chunk != usize::MAX,
                    #[cfg(not(any(target_family = "wasm", target_arch = "wasm32")))]
                    parallel_runner: if chunk == usize::MAX {
                        None
                    } else {
                        Some(&mut runner)
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.len(), expected.len());
                for (a, e) in actual.iter().zip(expected) {
                    assert_eq!(a.size(), e.size());
                    for y in 0..a.size().1 {
                        for (&a, &e) in a.row(y).iter().zip(e.row(y)) {
                            assert!((a - e).abs() < 1e-5 * e.abs().max(1.0), "{a} != {e}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn downsample_patches() {
    compare(&std::fs::read("resources/test/conformance_test_images/patches.jxl").unwrap());
}

#[test]
fn downsample_signalled_upsampling() {
    compare(&std::fs::read("resources/test/conformance_test_images/upsampling.jxl").unwrap());
}

#[test]
fn downsample_vardct_input_grids() {
    for name in [
        "red_420",
        "red_422",
        "red_440",
        "narrow_edge_group",
        "oddsize_ups",
    ] {
        compare(&std::fs::read(format!("resources/test/{name}.jxl")).unwrap());
    }
}

#[test]
fn downsample_modular_input_grids() {
    for name in [
        "squeeze_alpha",
        "ec_dim_shift_no_upsampling",
        "ec_upsampling8_multi_group",
    ] {
        compare(&std::fs::read(format!("resources/test/{name}.jxl")).unwrap());
    }
}

#[test]
fn downsample_animation() {
    compare(&std::fs::read("resources/test/animation_seek_sprite_sheets.jxl").unwrap());
}

#[test]
#[ignore = "requires a local screenshot; set JXL_DOWNSAMPLE_SCREENSHOT"]
fn downsample_blending_screenshot() {
    compare(&std::fs::read(std::env::var("JXL_DOWNSAMPLE_SCREENSHOT").unwrap()).unwrap());
}
