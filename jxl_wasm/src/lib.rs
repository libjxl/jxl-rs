// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::JxlOutputBuffer;
use jxl::api::{JxlBitDepth, ProcessingResult};
use jxl::api::{
    JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDecoderOptions, JxlPrimaries,
    JxlTransferFunction, JxlWhitePoint,
};
use jxl::image::{Image, Rect};
use std::mem;
use wasm_bindgen::prelude::*;

#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Information about a decoded JXL image
#[wasm_bindgen]
pub struct ImageInfo {
    width: u32,
    height: u32,
    num_frames: u32,
    has_alpha: bool,
}

#[wasm_bindgen]
impl ImageInfo {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> u32 {
        self.num_frames
    }

    #[wasm_bindgen(getter)]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn calculate_apng_delay(duration_ms: f64) -> Result<(u16, u16), JsValue> {
    if duration_ms < 0.0 {
        return Err(JsValue::from_str(&format!(
            "Negative frame duration: {}",
            duration_ms
        )));
    }
    if duration_ms == 0.0 {
        return Ok((0, 1));
    }

    let mut num = duration_ms.round() as u64;
    let mut den = 1000u64;

    let common = gcd(num, den);
    num /= common;
    den /= common;

    if num > u16::MAX as u64 || den > u16::MAX as u64 {
        Err(JsValue::from_str(&format!(
            "APNG frame delay overflow after GCD: {}/{}",
            num, den
        )))
    } else {
        Ok((num as u16, den as u16))
    }
}

fn png_color(num_channels: usize) -> Result<png::ColorType, JsValue> {
    match num_channels {
        1 => Ok(png::ColorType::Grayscale),
        2 => Ok(png::ColorType::GrayscaleAlpha),
        3 => Ok(png::ColorType::Rgb),
        4 => Ok(png::ColorType::Rgba),
        _ => Err(JsValue::from_str(&format!(
            "Invalid number of channels for PNG output: {}",
            num_channels
        ))),
    }
}

fn make_cicp(encoding: &JxlColorEncoding) -> Option<png::CodingIndependentCodePoints> {
    let JxlColorEncoding::RgbColorSpace {
        white_point,
        primaries,
        transfer_function,
        ..
    } = encoding
    else {
        return None;
    };

    Some(png::CodingIndependentCodePoints {
        color_primaries: match white_point {
            JxlWhitePoint::DCI => {
                if *primaries == JxlPrimaries::P3 {
                    11
                } else {
                    return None;
                }
            }
            JxlWhitePoint::D65 => match primaries {
                JxlPrimaries::SRGB => 1,
                JxlPrimaries::BT2100 => 9,
                JxlPrimaries::P3 => 12,
                JxlPrimaries::Chromaticities { .. } => return None,
            },
            _ => return None,
        },
        transfer_function: match transfer_function {
            JxlTransferFunction::BT709 => 1,
            JxlTransferFunction::Linear => 8,
            JxlTransferFunction::SRGB => 13,
            JxlTransferFunction::PQ => 16,
            JxlTransferFunction::DCI => 17,
            JxlTransferFunction::HLG => 18,
            JxlTransferFunction::Gamma(_) => return None,
        },
        matrix_coefficients: 0,
        is_video_full_range_image: true,
    })
}

// Extract RGB channels from interleaved RGB buffer
fn planes_from_interleaved(interleaved: &Image<f32>) -> Result<Vec<Image<f32>>, JsValue> {
    let size = interleaved.size();
    let size = (size.0 / 3, size.1);
    let mut r_image = Image::<f32>::new(size)
        .map_err(|e| JsValue::from_str(&format!("Failed to create image: {}", e)))?;
    let mut g_image = Image::<f32>::new(size)
        .map_err(|e| JsValue::from_str(&format!("Failed to create image: {}", e)))?;
    let mut b_image = Image::<f32>::new(size)
        .map_err(|e| JsValue::from_str(&format!("Failed to create image: {}", e)))?;

    for y in 0..size.1 {
        let r_row = r_image.row_mut(y);
        let g_row = g_image.row_mut(y);
        let b_row = b_image.row_mut(y);
        let src_row = interleaved.row(y);
        for x in 0..size.0 {
            r_row[x] = src_row[3 * x];
            g_row[x] = src_row[3 * x + 1];
            b_row[x] = src_row[3 * x + 2];
        }
    }
    Ok(vec![r_image, g_image, b_image])
}

struct ImageFrame {
    channels: Vec<Image<f32>>,
    duration: f64,
    color_type: JxlColorType,
}

struct DecodeOutput {
    size: (usize, usize),
    frames: Vec<ImageFrame>,
    original_bit_depth: JxlBitDepth,
    output_profile: JxlColorProfile,
    num_loops: u32,
}

fn decode_jxl_internal(jxl_data: &[u8]) -> Result<DecodeOutput, JsValue> {
    let mut input = jxl_data;

    let decoder_options = JxlDecoderOptions::default();
    let initialized_decoder =
        jxl::api::JxlDecoder::<jxl::api::states::Initialized>::new(decoder_options);

    let mut decoder_with_image_info = match initialized_decoder
        .process(&mut input)
        .map_err(|e| JsValue::from_str(&format!("Failed to decode header: {}", e)))?
    {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(JsValue::from_str("Source file truncated"));
        }
    };

    let info = decoder_with_image_info.basic_info();
    let image_size = info.size;
    let output_profile = decoder_with_image_info.output_color_profile().clone();
    let num_loops = info.animation.as_ref().map(|a| a.num_loops).unwrap_or(0);
    let extra_channels = info.extra_channels.len();
    let original_bit_depth = info.bit_depth.clone();

    let mut image_data = DecodeOutput {
        size: image_size,
        frames: Vec::new(),
        original_bit_depth,
        output_profile,
        num_loops,
    };

    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };

    loop {
        let decoder_with_frame_info = match decoder_with_image_info
            .process(&mut input)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode frame info: {}", e)))?
        {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => {
                return Err(JsValue::from_str("Source file truncated"));
            }
        };

        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = image_size;

        let mut outputs = vec![
            Image::<f32>::new((frame_size.0 * samples_per_pixel, frame_size.1))
                .map_err(|e| JsValue::from_str(&format!("Failed to create output image: {}", e)))?,
        ];

        for _ in 0..extra_channels {
            outputs.push(Image::<f32>::new(frame_size).map_err(|e| {
                JsValue::from_str(&format!("Failed to create extra channel: {}", e))
            })?);
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

        decoder_with_image_info = match decoder_with_frame_info
            .process(&mut input, &mut output_bufs)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode frame data: {}", e)))?
        {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => {
                return Err(JsValue::from_str("Source file truncated"));
            }
        };

        image_data.frames.push(ImageFrame {
            duration: frame_header.duration.unwrap_or(0.0),
            channels: outputs,
            color_type,
        });

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok(image_data)
}

fn encode_to_png(mut image_data: DecodeOutput) -> Result<Vec<u8>, JsValue> {
    if image_data.frames.is_empty()
        || image_data.frames[0].channels.is_empty()
        || image_data.size.0 == 0
        || image_data.size.1 == 0
    {
        return Err(JsValue::from_str("Invalid JXL image: no frames"));
    }

    let (width, height) = image_data.size;

    // Convert interleaved RGB to planar
    for frame in image_data.frames.iter_mut() {
        if frame.color_type != JxlColorType::Grayscale {
            let mut new_channels = planes_from_interleaved(&frame.channels[0])?;
            new_channels.extend(mem::take(&mut frame.channels).into_iter().skip(1));
            frame.channels = new_channels;
        }
    }

    // Get num_channels AFTER conversion from interleaved to planar
    let num_channels = image_data.frames[0].channels.len();

    let mut info = png::Info::with_size(width as u32, height as u32);
    match &image_data.output_profile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent,
        }) => {
            use jxl::headers::color_encoding::RenderingIntent;
            info.srgb = Some(match rendering_intent {
                RenderingIntent::Absolute => png::SrgbRenderingIntent::AbsoluteColorimetric,
                RenderingIntent::Relative => png::SrgbRenderingIntent::RelativeColorimetric,
                RenderingIntent::Perceptual => png::SrgbRenderingIntent::Perceptual,
                RenderingIntent::Saturation => png::SrgbRenderingIntent::Saturation,
            });
            info.source_gamma = Some(png::ScaledFloat::from_scaled(45455));
            info.source_chromaticities = Some(png::SourceChromaticities {
                white: (
                    png::ScaledFloat::from_scaled(31270),
                    png::ScaledFloat::from_scaled(32900),
                ),
                red: (
                    png::ScaledFloat::from_scaled(64000),
                    png::ScaledFloat::from_scaled(33000),
                ),
                green: (
                    png::ScaledFloat::from_scaled(30000),
                    png::ScaledFloat::from_scaled(60000),
                ),
                blue: (
                    png::ScaledFloat::from_scaled(15000),
                    png::ScaledFloat::from_scaled(6000),
                ),
            });
        }
        JxlColorProfile::Simple(encoding) => {
            info.coding_independent_code_points = make_cicp(encoding);
            let icc_bytes = encoding
                .maybe_create_profile()
                .map_err(|e| JsValue::from_str(&format!("Failed to create ICC profile: {}", e)))?
                .unwrap();
            info.icc_profile = Some(std::borrow::Cow::from(icc_bytes));
        }
        JxlColorProfile::Icc(icc_bytes) => {
            info.icc_profile = Some(std::borrow::Cow::Borrowed(icc_bytes));
        }
    }

    let mut buf = Vec::new();
    let mut encoder = png::Encoder::with_info(&mut buf, info)
        .map_err(|e| JsValue::from_str(&format!("Failed to create PNG encoder: {}", e)))?;

    encoder.set_color(png_color(num_channels)?);

    // Use 8-bit for web compatibility
    let bit_depth = image_data.original_bit_depth.bits_per_sample().min(8);
    encoder.set_depth(if bit_depth <= 8 {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    });

    if image_data.frames.len() > 1 {
        encoder
            .set_animated(image_data.frames.len() as u32, image_data.num_loops)
            .map_err(|e| JsValue::from_str(&format!("Failed to set animation: {}", e)))?;
    }

    let mut writer = encoder
        .write_header()
        .map_err(|e| JsValue::from_str(&format!("Failed to write PNG header: {}", e)))?;

    let num_pixels = height * width * num_channels;
    let mut data: Vec<u8> = vec![0; num_pixels];

    for (index, frame) in image_data.frames.iter().enumerate() {
        for y in 0..height {
            for x in 0..width {
                for c in 0..num_channels {
                    data[(y * width + x) * num_channels + c] =
                        ((frame.channels[c].row(y)[x] * 255.0).clamp(0.0, 255.0) + 0.5) as u8;
                }
            }
        }
        writer
            .write_image_data(&data)
            .map_err(|e| JsValue::from_str(&format!("Failed to write image data: {}", e)))?;

        if index + 1 < image_data.frames.len() && image_data.frames.len() > 1 {
            let (delay_num, delay_den) = calculate_apng_delay(frame.duration)?;
            writer
                .set_frame_delay(delay_num, delay_den)
                .map_err(|e| JsValue::from_str(&format!("Failed to set frame delay: {}", e)))?;
        }
    }

    drop(writer);
    Ok(buf)
}

/// Decode a JXL image and return PNG bytes
///
/// # Arguments
/// * `jxl_data` - The JXL image data as a byte array
///
/// # Returns
/// PNG image data that can be used directly in a browser
#[wasm_bindgen]
pub fn decode_jxl_to_png(jxl_data: &[u8]) -> Result<Vec<u8>, JsValue> {
    let decoded = decode_jxl_internal(jxl_data)?;
    encode_to_png(decoded)
}

/// Get information about a JXL image without fully decoding it
#[wasm_bindgen]
pub fn get_jxl_info(jxl_data: &[u8]) -> Result<ImageInfo, JsValue> {
    let mut input = jxl_data;

    let decoder_options = JxlDecoderOptions::default();
    let initialized_decoder =
        jxl::api::JxlDecoder::<jxl::api::states::Initialized>::new(decoder_options);

    let decoder_with_image_info = match initialized_decoder
        .process(&mut input)
        .map_err(|e| JsValue::from_str(&format!("Failed to decode header: {}", e)))?
    {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(JsValue::from_str("Source file truncated"));
        }
    };

    let info = decoder_with_image_info.basic_info();
    let num_frames = if info.animation.is_some() {
        // For animated images, we'd need to decode all frames to know the count
        // For now, just indicate it's animated
        2 // Minimum for animation
    } else {
        1
    };

    Ok(ImageInfo {
        width: info.size.0 as u32,
        height: info.size.1 as u32,
        num_frames,
        has_alpha: !info.extra_channels.is_empty(),
    })
}
