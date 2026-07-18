// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

macro_rules! declare_test_file {
    ($ident:ident, $path:expr) => {
        declare_test_file!($ident, $path, checkpoints: &[]);
    };
    ($ident:ident, $path:expr, checkpoints: $checkpoints:expr) => {
        paste::paste! {
            #[test]
            fn [<test_decode_test_file_ $ident>]() {
                let path = std::path::Path::new("resources/test/").join($path);
                let file = std::fs::read(&path).unwrap();
                crate::tests::decode::decode(&file).unwrap();
            }

            #[test]
            fn [<test_decode_test_file_chunks_ $ident>]() {
                let path = std::path::Path::new("resources/test/").join($path);
                let file = std::fs::read(&path).unwrap();
                crate::tests::decode::decode_internal(&file, 1, false, false, None, None).unwrap();
            }

            #[test]
            fn [<test_scan_test_file_ $ident>]() {
                let path = std::path::Path::new("resources/test/").join($path);
                let file = std::fs::read(&path).unwrap();
                crate::tests::decode::scan_frames_with_decoder(&file, usize::MAX);
            }

            #[test]
            fn [<test_scan_test_file_chunks_ $ident>]() {
                let path = std::path::Path::new("resources/test/").join($path);
                let file = std::fs::read(&path).unwrap();
                crate::tests::decode::scan_frames_with_decoder(&file, 1);
            }

            #[test]
            fn [<test_compare_pipelines_ $ident>]() {
                let path = std::path::Path::new("resources/test/").join($path);
                let file = std::fs::read(&path).unwrap();
                let simple_frames = crate::tests::decode::decode_internal(&file, usize::MAX, true, false, None, None).unwrap().1;
                let frames = crate::tests::decode::decode(&file).unwrap().1;
                assert_eq!(frames.len(), simple_frames.len());
                for (fc, (f, sf)) in frames.into_iter().zip(simple_frames).enumerate() {
                    crate::tests::decode::compare_frames(&path, fc, &f, &sf);
                }
            }

            #[test]
            fn [<test_compare_incremental_ $ident>]() {
                let path = std::path::Path::new("resources/test/").join($path);
                crate::tests::compare_incremental::run(&path, $checkpoints);
            }
        }
    };
}
