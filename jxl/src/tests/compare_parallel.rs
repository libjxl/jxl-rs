// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlParallelRunner;
use crate::error::Error;
use crate::image::Image;
use crate::tests::decode::{compare_frames, decode_internal};
use std::path::Path;

pub struct TestParallelRunner {
    pub max_threads: usize,
}

impl JxlParallelRunner for TestParallelRunner {
    fn run(&mut self, num: usize, fun: &crate::api::JxlParallelRunnerFun<'_>) -> Result<(), Error> {
        if num <= 1 || self.max_threads <= 1 {
            for i in 0..num {
                fun(i)?;
            }
            return Ok(());
        }
        let num_threads = self.max_threads.min(num);
        let next_task = std::sync::atomic::AtomicUsize::new(0);
        let error = std::sync::Mutex::new(None);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for _ in 0..num_threads {
                handles.push(s.spawn(|| {
                    loop {
                        if error.lock().unwrap().is_some() {
                            break;
                        }
                        let task = next_task.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if task >= num {
                            break;
                        }
                        if let Err(e) = fun(task) {
                            let mut err = error.lock().unwrap();
                            if err.is_none() {
                                *err = Some(e);
                            }
                            break;
                        }
                    }
                }));
            }
            for handle in handles {
                if let Err(e) = handle.join() {
                    std::panic::resume_unwind(e);
                }
            }
        });

        if let Some(err) = error.into_inner().unwrap() {
            Err(err)
        } else {
            Ok(())
        }
    }
}

fn clone_images(imgs: &[Image<f32>]) -> Vec<Image<f32>> {
    imgs.iter()
        .map(|img| {
            let mut copy = Image::new(img.size()).unwrap();
            for y in 0..img.size().1 {
                copy.row_mut(y).copy_from_slice(img.row(y));
            }
            copy
        })
        .collect()
}

pub fn run_oneshot(path: &Path) {
    let file = std::fs::read(path).unwrap();

    // Oneshot sequential decode
    let (_, seq_frames) =
        decode_internal(&file, usize::MAX, false, false, None, None, None).unwrap();

    if seq_frames.is_empty() {
        return;
    }

    // Oneshot parallel decode
    let mut runner = TestParallelRunner { max_threads: 4 };
    let (_, par_frames) = decode_internal(
        &file,
        usize::MAX,
        false,
        false,
        None,
        None,
        Some(&mut runner),
    )
    .unwrap();

    assert_eq!(
        seq_frames.len(),
        par_frames.len(),
        "Parallel and sequential frame counts differ for {:?}",
        path
    );

    for (fc, (seq_f, par_f)) in seq_frames.into_iter().zip(par_frames).enumerate() {
        compare_frames(path, fc, &par_f, &seq_f);
    }
}

pub fn run_progressive(path: &Path) {
    let file = std::fs::read(path).unwrap();

    let chunk_size = (file.len() / 8).max(1024);

    let mut seq_flushes: Vec<(usize, usize, Vec<Image<f32>>)> = Vec::new();
    let mut seq_callback =
        |consumed_bytes: usize, f_idx: usize, buffers: &[Image<f32>]| -> Result<(), Error> {
            seq_flushes.push((consumed_bytes, f_idx, clone_images(buffers)));
            Ok(())
        };

    // Sequential progressive decode
    let _ = decode_internal(
        &file,
        chunk_size,
        false,
        true,
        None,
        Some(&mut seq_callback),
        None,
    );

    let mut par_flushes: Vec<(usize, usize, Vec<Image<f32>>)> = Vec::new();
    let mut par_callback =
        |consumed_bytes: usize, f_idx: usize, buffers: &[Image<f32>]| -> Result<(), Error> {
            par_flushes.push((consumed_bytes, f_idx, clone_images(buffers)));
            Ok(())
        };

    // Parallel progressive decode
    let mut runner = TestParallelRunner { max_threads: 4 };
    let _ = decode_internal(
        &file,
        chunk_size,
        false,
        true,
        None,
        Some(&mut par_callback),
        Some(&mut runner),
    );

    assert_eq!(
        seq_flushes.len(),
        par_flushes.len(),
        "Parallel and sequential flush counts differ for {:?}",
        path
    );

    for (idx, ((seq_bytes, seq_f_idx, seq_bufs), (par_bytes, par_f_idx, par_bufs))) in
        seq_flushes.into_iter().zip(par_flushes).enumerate()
    {
        assert_eq!(
            seq_bytes, par_bytes,
            "Flush {} consumed bytes mismatch for {:?}",
            idx, path
        );
        assert_eq!(
            seq_f_idx, par_f_idx,
            "Flush {} frame index mismatch for {:?}",
            idx, path
        );
        compare_frames(path, seq_f_idx, &par_bufs, &seq_bufs);
    }
}
