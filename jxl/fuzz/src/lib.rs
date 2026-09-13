// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{
    JxlColorType, JxlDecoder, JxlDecoderOptions, JxlParallelRunner, JxlParallelRunnerFun,
    ProcessingResult, states,
};
use jxl::image::{Image, JxlOutputBuffer, Rect};

pub struct SimpleParallelRunner {
    max_threads: usize,
}

impl SimpleParallelRunner {
    pub fn new(max_threads: usize) -> Self {
        Self { max_threads }
    }
}

impl JxlParallelRunner for SimpleParallelRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> Result<(), jxl::error::Error> {
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

    fn num_threads(&self) -> usize {
        self.max_threads
    }
}

pub fn reborrow<'a>(
    runner: &'a mut Option<&mut dyn JxlParallelRunner>,
) -> Option<&'a mut dyn JxlParallelRunner> {
    match runner {
        Some(r) => Some(&mut **r),
        None => None,
    }
}

pub struct SimpleRng(u64);

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdeadbeef } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        min + (self.next_u64() as usize % (max - min + 1))
    }
}

#[allow(clippy::result_unit_err)]
pub fn as_complete<T, U, E>(result: Result<ProcessingResult<T, U>, E>) -> Result<T, ()> {
    match result {
        Ok(ProcessingResult::Complete { result }) => Ok(result),
        _ => Err(()),
    }
}

#[derive(Clone, Debug)]
pub struct FuzzConfig {
    pub progressive: bool,
    pub parallel: bool,
    pub num_threads: usize,
    pub sample_limit: Option<usize>,
    pub flush_intermediate: bool,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            progressive: false,
            parallel: false,
            num_threads: 2,
            sample_limit: Some(1 << 22),
            flush_intermediate: false,
        }
    }
}

fn create_frame_buffers(
    frame_size: (usize, usize),
    samples_per_pixel: usize,
    extra_channels: usize,
) -> Result<Vec<Image<f32>>, ()> {
    let mut outputs = match Image::<f32>::new((frame_size.0 * samples_per_pixel, frame_size.1)) {
        Ok(img) => vec![img],
        Err(_) => return Err(()),
    };
    for _ in 0..extra_channels {
        match Image::<f32>::new(frame_size) {
            Ok(img) => outputs.push(img),
            Err(_) => return Err(()),
        }
    }
    Ok(outputs)
}

fn make_output_bufs(outputs: &mut [Image<f32>]) -> Vec<JxlOutputBuffer<'_>> {
    outputs
        .iter_mut()
        .map(|x| {
            let rect = Rect {
                size: x.size(),
                origin: (0, 0),
            };
            JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect).into_raw())
        })
        .collect()
}

#[allow(clippy::result_unit_err)]
pub fn fuzz_decode(data: &[u8], config: FuzzConfig) -> Result<Vec<Vec<Image<f32>>>, ()> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut runner = if config.parallel {
        Some(SimpleParallelRunner::new(config.num_threads))
    } else {
        None
    };
    let mut runner_opt: Option<&mut dyn JxlParallelRunner> = runner
        .as_mut()
        .map(|r| r as &mut dyn JxlParallelRunner);

    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.sample_limit = config.sample_limit;

    if !config.progressive {
        let mut input = data;
        let initialized_decoder = JxlDecoder::<states::Initialized>::new(decoder_options);
        let mut decoder_with_image_info =
            as_complete(initialized_decoder.process(&mut input, reborrow(&mut runner_opt)))?;

        let info = decoder_with_image_info.basic_info();
        let extra_channels = info.extra_channels.len();
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let color_type = pixel_format.color_type;
        let samples_per_pixel = if color_type == JxlColorType::Grayscale {
            1
        } else {
            3
        };

        let mut all_frames = Vec::new();

        loop {
            let decoder_with_frame_info =
                as_complete(decoder_with_image_info.process(&mut input, reborrow(&mut runner_opt)))?;
            let frame_header = decoder_with_frame_info.frame_header();
            let frame_size = frame_header.size;

            let mut outputs =
                create_frame_buffers(frame_size, samples_per_pixel, extra_channels)?;
            let mut output_bufs = make_output_bufs(&mut outputs);

            decoder_with_image_info = as_complete(decoder_with_frame_info.process(
                &mut input,
                &mut output_bufs,
                reborrow(&mut runner_opt),
            ))?;

            all_frames.push(outputs);

            if !decoder_with_image_info.has_more_frames() {
                break;
            }
        }

        Ok(all_frames)
    } else {
        let seed = data
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut rng = SimpleRng::new(seed);

        let mut decoder = JxlDecoder::<states::Initialized>::new(decoder_options);
        let mut chunk_input = &data[0..0];
        let mut remaining_input = data;

        let mut decoder_with_image_info = loop {
            let chunk_size = match rng.gen_range(0, 4) {
                0 => 1,
                1 => rng.gen_range(2, 32),
                2 => rng.gen_range(32, 512),
                _ => rng.gen_range(512, 4096),
            };
            let all_provided = chunk_input.len().saturating_add(chunk_size) >= remaining_input.len();
            let next_len =
                (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
            chunk_input = &remaining_input[..next_len];
            let available_before = chunk_input.len();
            let res = match decoder.process(&mut chunk_input, reborrow(&mut runner_opt)) {
                Ok(r) => r,
                Err(_) => return Err(()),
            };
            let consumed = available_before - chunk_input.len();
            remaining_input = &remaining_input[consumed..];
            match res {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if remaining_input.is_empty() || (all_provided && consumed == 0) {
                        return Ok(Vec::new());
                    }
                    decoder = fallback;
                }
            }
        };

        let basic_info = decoder_with_image_info.basic_info().clone();
        let (width, height) = basic_info.size;
        let color_type = decoder_with_image_info.current_pixel_format().color_type;
        let samples_per_pixel = if color_type == JxlColorType::Grayscale {
            1
        } else {
            3
        };
        let extra_channels = basic_info.extra_channels.len();

        let mut intermediate_outputs =
            match create_frame_buffers((width, height), samples_per_pixel, extra_channels) {
                Ok(img) => img,
                Err(_) => return Err(()),
            };

        let mut all_frames = Vec::new();

        loop {
            let mut decoder_with_frame_info = loop {
                let chunk_size = rng.gen_range(1, 512);
                let all_provided =
                    chunk_input.len().saturating_add(chunk_size) >= remaining_input.len();
                let next_len =
                    (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
                chunk_input = &remaining_input[..next_len];
                let available_before = chunk_input.len();
                let res = match decoder_with_image_info
                    .process(&mut chunk_input, reborrow(&mut runner_opt))
                {
                    Ok(r) => r,
                    Err(_) => return Err(()),
                };
                let consumed = available_before - chunk_input.len();
                remaining_input = &remaining_input[consumed..];
                match res {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                        if config.flush_intermediate {
                            let mut output_bufs = make_output_bufs(&mut intermediate_outputs);
                            let _ =
                                fallback.flush_pixels(&mut output_bufs, reborrow(&mut runner_opt));
                            if rng.gen_range(0, 7) == 0 {
                                let _ = fallback
                                    .flush_pixels(&mut output_bufs, reborrow(&mut runner_opt));
                            }
                        }
                        if remaining_input.is_empty() || (all_provided && consumed == 0) {
                            return Ok(all_frames);
                        }
                        decoder_with_image_info = fallback;
                    }
                }
            };

            let frame_header = decoder_with_frame_info.frame_header();
            let frame_size = frame_header.size;
            let mut outputs =
                create_frame_buffers(frame_size, samples_per_pixel, extra_channels)?;

            decoder_with_image_info = loop {
                let chunk_size = match rng.gen_range(0, 5) {
                    0 => 1,
                    1 => rng.gen_range(2, 64),
                    2 => rng.gen_range(64, 512),
                    3 => rng.gen_range(512, 2048),
                    _ => remaining_input.len().max(1),
                };
                let all_provided =
                    chunk_input.len().saturating_add(chunk_size) >= remaining_input.len();
                let next_len =
                    (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
                chunk_input = &remaining_input[..next_len];
                let available_before = chunk_input.len();
                let mut output_bufs = make_output_bufs(&mut outputs);
                let res = match decoder_with_frame_info.process(
                    &mut chunk_input,
                    &mut output_bufs,
                    reborrow(&mut runner_opt),
                ) {
                    Ok(r) => r,
                    Err(_) => return Err(()),
                };
                let consumed = available_before - chunk_input.len();
                remaining_input = &remaining_input[consumed..];
                match res {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                        if config.flush_intermediate {
                            let _ =
                                fallback.flush_pixels(&mut output_bufs, reborrow(&mut runner_opt));
                            if rng.gen_range(0, 7) == 0 {
                                let _ = fallback
                                    .flush_pixels(&mut output_bufs, reborrow(&mut runner_opt));
                            }
                        }
                        if remaining_input.is_empty() || (all_provided && consumed == 0) {
                            return Ok(all_frames);
                        }
                        decoder_with_frame_info = fallback;
                    }
                }
            };


            all_frames.push(outputs);

            if !decoder_with_image_info.has_more_frames() {
                break;
            }
        }

        Ok(all_frames)
    }
}

#[allow(clippy::result_unit_err)]
pub fn fuzz_decode_diff(data: &[u8]) -> Result<(), ()> {
    let Ok(seq_frames) = fuzz_decode(
        data,
        FuzzConfig {
            progressive: false,
            parallel: false,
            ..Default::default()
        },
    ) else {
        return Ok(());
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

    Ok(())
}
