// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{JxlColorType, JxlDecoder, JxlDecoderOptions, ProcessingResult, states};
use jxl::image::{Image, JxlOutputBuffer, Rect};
use libfuzzer_sys::fuzz_target;

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdeadbeef } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        min + (self.next_u64() as usize % (max - min + 1))
    }
}

fn fuzz_decode_progressive(data: &[u8]) -> Result<(), ()> {
    if data.is_empty() {
        return Ok(());
    }

    let seed = data
        .iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng = SimpleRng::new(seed);

    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.sample_limit = Some(1 << 22);
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
        let next_len = (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
        chunk_input = &remaining_input[..next_len];
        let available_before = chunk_input.len();
        let res = match decoder.process(&mut chunk_input, None) {
            Ok(r) => r,
            Err(_) => return Err(()),
        };
        remaining_input = &remaining_input[(available_before - chunk_input.len())..];
        match res {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if remaining_input.is_empty() {
                    return Ok(());
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

    let mut outputs = match Image::<f32>::new((width * samples_per_pixel, height)) {
        Ok(img) => vec![img],
        Err(_) => return Err(()),
    };
    for _ in 0..extra_channels {
        match Image::<f32>::new((width, height)) {
            Ok(img) => outputs.push(img),
            Err(_) => return Err(()),
        }
    }

    loop {
        let mut decoder_with_frame_info = loop {
            let chunk_size = rng.gen_range(1, 512);
            let next_len =
                (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
            chunk_input = &remaining_input[..next_len];
            let available_before = chunk_input.len();
            let res = match decoder_with_image_info.process(&mut chunk_input, None) {
                Ok(r) => r,
                Err(_) => return Err(()),
            };
            remaining_input = &remaining_input[(available_before - chunk_input.len())..];
            match res {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
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
                    let _ = fallback.flush_pixels(&mut output_bufs, None);
                    if rng.gen_range(0, 7) == 0 {
                        let _ = fallback.flush_pixels(&mut output_bufs, None);
                    }
                    if remaining_input.is_empty() {
                        return Ok(());
                    }
                    decoder_with_image_info = fallback;
                }
            }
        };

        decoder_with_image_info = loop {
            let chunk_size = match rng.gen_range(0, 5) {
                0 => 1,
                1 => rng.gen_range(2, 64),
                2 => rng.gen_range(64, 512),
                3 => rng.gen_range(512, 2048),
                _ => remaining_input.len().max(1),
            };
            let next_len =
                (chunk_input.len().saturating_add(chunk_size)).min(remaining_input.len());
            chunk_input = &remaining_input[..next_len];
            let available_before = chunk_input.len();
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
            let res =
                match decoder_with_frame_info.process(&mut chunk_input, &mut output_bufs, None) {
                    Ok(r) => r,
                    Err(_) => return Err(()),
                };
            remaining_input = &remaining_input[(available_before - chunk_input.len())..];
            match res {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                    let _ = fallback.flush_pixels(&mut output_bufs, None);
                    if rng.gen_range(0, 7) == 0 {
                        let _ = fallback.flush_pixels(&mut output_bufs, None);
                    }
                    if remaining_input.is_empty() {
                        return Ok(());
                    }
                    decoder_with_frame_info = fallback;
                }
            }
        };

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_decode_progressive(data);
});
