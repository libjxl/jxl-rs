// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{
    JxlColorType, JxlDecoder, JxlDecoderOptions, JxlParallelRunner, ProcessingResult, states,
};
use jxl::image::{Image, JxlOutputBuffer, Rect};
use libfuzzer_sys::fuzz_target;

fn as_complete<T, U, E>(result: Result<ProcessingResult<T, U>, E>) -> Result<T, ()> {
    match result {
        Ok(ProcessingResult::Complete { result }) => Ok(result),
        _ => Err(()),
    }
}

struct SimpleParallelRunner {
    max_threads: usize,
}

impl JxlParallelRunner for SimpleParallelRunner {
    fn run(
        &mut self,
        num: usize,
        fun: &jxl::api::JxlParallelRunnerFun<'_>,
    ) -> Result<(), jxl::error::Error> {
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

fn reborrow<'a>(
    runner: &'a mut Option<&mut dyn JxlParallelRunner>,
) -> Option<&'a mut dyn JxlParallelRunner> {
    match runner {
        Some(r) => Some(&mut **r),
        None => None,
    }
}

fn fuzz_decode_parallel(mut data: &[u8]) -> Result<(), ()> {
    let mut runner = SimpleParallelRunner { max_threads: 2 };
    let mut runner_opt: Option<&mut dyn JxlParallelRunner> = Some(&mut runner);

    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.sample_limit = Some(1 << 22);
    let initialized_decoder = JxlDecoder::<states::Initialized>::new(decoder_options);
    let mut decoder_with_image_info =
        as_complete(initialized_decoder.process(&mut data, reborrow(&mut runner_opt)))?;

    let info = decoder_with_image_info.basic_info();

    let extra_channels = info.extra_channels.len();
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };

    loop {
        let decoder_with_frame_info =
            as_complete(decoder_with_image_info.process(&mut data, reborrow(&mut runner_opt)))?;
        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = frame_header.size;

        let mut outputs =
            vec![Image::<f32>::new((frame_size.0 * samples_per_pixel, frame_size.1)).unwrap()];

        for _ in 0..extra_channels {
            outputs.push(Image::<f32>::new(frame_size).unwrap());
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
            .iter_mut()
            .map(|x| {
                let rect = Rect {
                    size: x.size(),
                    origin: (0, 0),
                };
                JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect).into_raw())
            })
            .collect();

        decoder_with_image_info = as_complete(decoder_with_frame_info.process(
            &mut data,
            &mut output_bufs,
            reborrow(&mut runner_opt),
        ))?;

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_decode_parallel(data);
});
