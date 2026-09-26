// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{
    JxlColorType, JxlDecoder, JxlDecoderOptions, JxlParallelRunner, JxlParallelRunnerFun,
    ProcessingResult, states,
};
use jxl::image::{Image, JxlOutputBuffer, Rect};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

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
    let mut runner_opt: Option<&mut dyn JxlParallelRunner> =
        runner.as_mut().map(|r| r as &mut dyn JxlParallelRunner);

    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.sample_limit = config.sample_limit;

    let mut rng = if config.progressive {
        let seed = data
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
        Some(XorShiftRng::seed_from_u64(seed))
    } else {
        None
    };

    let mut chunk_input = &data[0..0];
    let mut remaining_input = data;

    macro_rules! advance_decoder {
        ($decoder:ident, $chunk_size:expr, $process:expr $(, flush: $fallback:ident => $flush:expr)?) => {{
            loop {
                let chunk_size = match rng.as_mut() {
                    Some(rng) => $chunk_size(rng),
                    None => remaining_input.len().max(1),
                };
                let all_provided = chunk_input.len().saturating_add(chunk_size) >= remaining_input.len();
                let next_len = (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
                chunk_input = &remaining_input[..next_len];
                let available_before = chunk_input.len();

                let res = match $process {
                    Ok(r) => r,
                    Err(_) => return Err(()),
                };

                let consumed = available_before - chunk_input.len();
                remaining_input = &remaining_input[consumed..];

                match res {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        #[allow(unused_mut)]
                        let mut fallback = fallback;
                        $(
                            if config.flush_intermediate {
                                let $fallback = &mut fallback;
                                $flush;
                                if let Some(ref mut rng) = rng {
                                    if rng.random_range(0..7) == 0 {
                                        $flush;
                                    }
                                }
                            }
                        )?
                        if remaining_input.is_empty() || (all_provided && consumed == 0) {
                            return Err(());
                        }
                        $decoder = fallback;
                    }
                }
            }
        }};
    }

    let mut decoder = JxlDecoder::<states::Initialized>::new(decoder_options);
    let mut decoder_with_image_info = advance_decoder!(
        decoder,
        |rng: &mut XorShiftRng| match rng.random_range(0..4) {
            0 => 1,
            1 => rng.random_range(2..32),
            2 => rng.random_range(32..512),
            _ => rng.random_range(512..4096),
        },
        decoder.process(&mut chunk_input, reborrow(&mut runner_opt))
    );

    let basic_info = decoder_with_image_info.basic_info().clone();
    let (width, height) = basic_info.size;
    let color_type = decoder_with_image_info.current_pixel_format().color_type;
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };
    let extra_channels = basic_info.extra_channels.len();

    let mut intermediate_outputs = if config.flush_intermediate {
        Some(create_frame_buffers(
            (width, height),
            samples_per_pixel,
            extra_channels,
        )?)
    } else {
        None
    };

    let mut all_frames = Vec::new();

    loop {
        let mut decoder_with_frame_info = advance_decoder!(
            decoder_with_image_info,
            |rng: &mut XorShiftRng| rng.random_range(1..512),
            decoder_with_image_info.process(&mut chunk_input, reborrow(&mut runner_opt)),
            flush: fallback => {
                if let Some(ref mut intermediate) = intermediate_outputs {
                    let mut bufs = make_output_bufs(intermediate);
                    let _ = fallback.flush_pixels(&mut bufs, reborrow(&mut runner_opt));
                }
            }
        );

        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = frame_header.size;
        let mut outputs = create_frame_buffers(frame_size, samples_per_pixel, extra_channels)?;

        decoder_with_image_info = advance_decoder!(
            decoder_with_frame_info,
            |rng: &mut XorShiftRng| match rng.random_range(0..5) {
                0 => 1,
                1 => rng.random_range(2..64),
                2 => rng.random_range(64..512),
                3 => rng.random_range(512..2048),
                _ => remaining_input.len().max(1),
            },
            {
                let mut bufs = make_output_bufs(&mut outputs);
                decoder_with_frame_info.process(&mut chunk_input, &mut bufs, reborrow(&mut runner_opt))
            },
            flush: fallback => {
                let mut bufs = make_output_bufs(&mut outputs);
                let _ = fallback.flush_pixels(&mut bufs, reborrow(&mut runner_opt));
            }
        );

        all_frames.push(outputs);

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }
    Ok(all_frames)
}
