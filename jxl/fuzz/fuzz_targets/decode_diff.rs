// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl_fuzz::{FuzzConfig, fuzz_decode};
use libfuzzer_sys::fuzz_target;

fn fuzz_decode_diff(data: &[u8]) {
    let Ok(seq_frames) = fuzz_decode(
        data,
        FuzzConfig {
            progressive: false,
            parallel: false,
            ..Default::default()
        },
    ) else {
        return;
    };

    let Ok(par_frames) = fuzz_decode(
        data,
        FuzzConfig {
            progressive: false,
            parallel: true,
            num_threads: 2,
            ..Default::default()
        },
    ) else {
        panic!("Parallel decoding failed on a bitstream that succeeded sequentially!");
    };

    assert_eq!(
        seq_frames.len(),
        par_frames.len(),
        "Frame count mismatch between sequential and parallel decode!"
    );

    for (f_idx, (s_frame, p_frame)) in seq_frames.iter().zip(par_frames.iter()).enumerate() {
        assert_eq!(
            s_frame.len(),
            p_frame.len(),
            "Frame {f_idx}: Channel count mismatch between sequential and parallel decode!"
        );
        for (c_idx, (s_img, p_img)) in s_frame.iter().zip(p_frame.iter()).enumerate() {
            assert_eq!(
                s_img.size(),
                p_img.size(),
                "Frame {f_idx} Channel {c_idx}: Size mismatch {:?} vs {:?}",
                s_img.size(),
                p_img.size()
            );
            for y in 0..s_img.size().1 {
                let s_row = s_img.row(y);
                let p_row = p_img.row(y);
                for (x, (&s_val, &p_val)) in s_row.iter().zip(p_row.iter()).enumerate() {
                    assert!(
                        s_val == p_val || (s_val.is_nan() && p_val.is_nan()),
                        "Frame {f_idx} Channel {c_idx} ({x}, {y}): Exact pixel mismatch between sequential ({s_val}) and parallel ({p_val})!"
                    );
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    fuzz_decode_diff(data);
});
