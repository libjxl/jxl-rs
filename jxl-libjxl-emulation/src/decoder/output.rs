// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;

use jxl::api::{
    Endianness, JxlBitDepth as RsBitDepth, JxlColorType, JxlDataFormat, JxlOutputBuffer,
    JxlPixelFormat as RsPixelFormat,
};

use crate::types::{
    JXL_BIG_ENDIAN, JXL_BIT_DEPTH_CUSTOM, JXL_BIT_DEPTH_FROM_CODESTREAM,
    JXL_BIT_DEPTH_FROM_PIXEL_FORMAT, JXL_LITTLE_ENDIAN, JXL_NATIVE_ENDIAN, JXL_TYPE_FLOAT,
    JXL_TYPE_FLOAT16, JXL_TYPE_UINT8, JXL_TYPE_UINT16, JxlBitDepth, JxlDataType, JxlEndianness,
    JxlImageOutCallback, JxlPixelFormat,
};

pub struct FrameOutput {
    pub image_out_target: Option<ImageOutTarget>,
    pub extra_channel_targets: Vec<Option<ImageOutTarget>>,
    pub output_bit_depth: JxlBitDepth,
}

impl Default for FrameOutput {
    fn default() -> Self {
        Self {
            image_out_target: None,
            extra_channel_targets: Vec::new(),
            output_bit_depth: JxlBitDepth {
                type_: JXL_BIT_DEPTH_FROM_PIXEL_FORMAT,
                bits_per_sample: 0,
                exponent_bits_per_sample: 0,
            },
        }
    }
}

impl FrameOutput {
    pub fn has_targets(&self) -> bool {
        self.image_out_target.is_some() || self.extra_channel_targets.iter().any(|t| t.is_some())
    }

    pub fn clear_targets(&mut self) {
        self.image_out_target = None;
        for ec_target in &mut self.extra_channel_targets {
            *ec_target = None;
        }
    }

    pub fn init_extra_channels(&mut self, num_ec: usize) {
        self.extra_channel_targets = (0..num_ec).map(|_| None).collect();
    }

    pub fn desired_rs_pixel_format(
        &self,
        codestream_bd: &RsBitDepth,
        is_gray: bool,
    ) -> Option<RsPixelFormat> {
        let color_fmt = *self.image_out_target.as_ref()?.format();
        let ec_fmts: Vec<Option<JxlPixelFormat>> = self
            .extra_channel_targets
            .iter()
            .map(|t| t.as_ref().map(|target| *target.format()))
            .collect();
        build_rs_pixel_format(
            Some(&color_fmt),
            &ec_fmts,
            &self.output_bit_depth,
            codestream_bd,
            is_gray,
        )
    }

    pub fn make_output_buffers(
        &mut self,
        width: usize,
        height: usize,
    ) -> Option<Vec<JxlOutputBuffer<'_>>> {
        let mut output_buffers = Vec::new();
        if let Some(color_target) = self.image_out_target.as_mut() {
            output_buffers.push(color_target.make_output_buffer(width, height)?);
        }
        for ec_target in self.extra_channel_targets.iter_mut().flatten() {
            output_buffers.push(ec_target.make_output_buffer(width, height)?);
        }
        Some(output_buffers)
    }

    pub fn finalize_all(&self, width: usize) -> bool {
        if let Some(color_target) = self.image_out_target.as_ref()
            && !color_target.finalize_output(width)
        {
            return false;
        }
        for ec_target in self.extra_channel_targets.iter().flatten() {
            if !ec_target.finalize_output(width) {
                return false;
            }
        }
        true
    }
}

pub fn bytes_per_sample(data_type: JxlDataType) -> Option<usize> {
    match data_type {
        JXL_TYPE_UINT8 => Some(1),
        JXL_TYPE_UINT16 | JXL_TYPE_FLOAT16 => Some(2),
        JXL_TYPE_FLOAT => Some(4),
        _ => None,
    }
}

pub fn compute_row_layout(width: usize, format: &JxlPixelFormat) -> Option<(usize, usize)> {
    if !(1..=4).contains(&format.num_channels) {
        return None;
    }
    let _ = to_rs_endianness(format.endianness)?;
    let bps = bytes_per_sample(format.data_type)?;
    let ch = format.num_channels as usize;
    let bytes_per_row = width.checked_mul(ch)?.checked_mul(bps)?;
    let stride = if format.align > 1 {
        bytes_per_row.checked_next_multiple_of(format.align)?
    } else {
        bytes_per_row
    };
    Some((bytes_per_row, stride))
}

pub fn compute_buffer_size(width: usize, height: usize, format: &JxlPixelFormat) -> Option<usize> {
    let (bytes_per_row, stride) = compute_row_layout(width, format)?;
    if width == 0 || height == 0 {
        return Some(0);
    }
    let total = stride.checked_mul(height - 1)?.checked_add(bytes_per_row)?;
    if total > isize::MAX as usize {
        return None;
    }
    Some(total)
}

pub fn to_rs_endianness(endianness: JxlEndianness) -> Option<Endianness> {
    match endianness {
        JXL_NATIVE_ENDIAN => Some(Endianness::native()),
        JXL_LITTLE_ENDIAN => Some(Endianness::LittleEndian),
        JXL_BIG_ENDIAN => Some(Endianness::BigEndian),
        _ => None,
    }
}

pub fn to_rs_data_format(
    format: &JxlPixelFormat,
    bit_depth: &JxlBitDepth,
    codestream_bit_depth: &RsBitDepth,
) -> Option<JxlDataFormat> {
    let endianness = to_rs_endianness(format.endianness)?;
    match format.data_type {
        JXL_TYPE_UINT8 => {
            let bd = match bit_depth.type_ {
                JXL_BIT_DEPTH_FROM_PIXEL_FORMAT => 8,
                JXL_BIT_DEPTH_FROM_CODESTREAM => {
                    codestream_bit_depth.bits_per_sample().clamp(1, 8) as u8
                }
                JXL_BIT_DEPTH_CUSTOM => {
                    if !(1..=8).contains(&bit_depth.bits_per_sample) {
                        return None;
                    }
                    bit_depth.bits_per_sample as u8
                }
                _ => return None,
            };
            Some(JxlDataFormat::U8 { bit_depth: bd })
        }
        JXL_TYPE_UINT16 => {
            let bd = match bit_depth.type_ {
                JXL_BIT_DEPTH_FROM_PIXEL_FORMAT => 16,
                JXL_BIT_DEPTH_FROM_CODESTREAM => {
                    codestream_bit_depth.bits_per_sample().clamp(1, 16) as u8
                }
                JXL_BIT_DEPTH_CUSTOM => {
                    if !(1..=16).contains(&bit_depth.bits_per_sample) {
                        return None;
                    }
                    bit_depth.bits_per_sample as u8
                }
                _ => return None,
            };
            Some(JxlDataFormat::U16 {
                endianness,
                bit_depth: bd,
            })
        }
        JXL_TYPE_FLOAT16 => Some(JxlDataFormat::F16 { endianness }),
        JXL_TYPE_FLOAT => Some(JxlDataFormat::F32 { endianness }),
        _ => None,
    }
}

pub fn to_rs_color_type(num_channels: u32, is_gray: bool) -> Option<JxlColorType> {
    if !is_gray && num_channels < 3 {
        return None;
    }
    match num_channels {
        1 => Some(JxlColorType::Grayscale),
        2 => Some(JxlColorType::GrayscaleAlpha),
        3 => Some(JxlColorType::Rgb),
        4 => Some(JxlColorType::Rgba),
        _ => None,
    }
}

pub fn build_rs_pixel_format(
    color_format: Option<&JxlPixelFormat>,
    extra_channel_formats: &[Option<JxlPixelFormat>],
    bit_depth: &JxlBitDepth,
    codestream_bit_depth: &RsBitDepth,
    is_gray: bool,
) -> Option<RsPixelFormat> {
    let (color_type, color_data_format) = if let Some(cf) = color_format {
        let ct = to_rs_color_type(cf.num_channels, is_gray)?;
        let df = to_rs_data_format(cf, bit_depth, codestream_bit_depth)?;
        (ct, Some(df))
    } else {
        let ct = if is_gray {
            JxlColorType::Grayscale
        } else {
            JxlColorType::Rgb
        };
        (ct, None)
    };

    let mut extra_channel_format = Vec::with_capacity(extra_channel_formats.len());
    for ecf in extra_channel_formats {
        if let Some(f) = ecf {
            if f.num_channels != 1 {
                return None;
            }
            extra_channel_format.push(Some(to_rs_data_format(f, bit_depth, codestream_bit_depth)?));
        } else {
            extra_channel_format.push(None);
        }
    }

    Some(RsPixelFormat {
        color_type,
        color_data_format,
        extra_channel_format,
    })
}

pub enum ImageOutTarget {
    Buffer {
        ptr: *mut u8,
        size: usize,
        format: JxlPixelFormat,
    },
    Callback {
        callback: unsafe extern "C" fn(
            opaque: *mut c_void,
            x: usize,
            y: usize,
            num_pixels: usize,
            pixels: *const c_void,
        ),
        opaque: *mut c_void,
        format: JxlPixelFormat,
        buf: Vec<u8>,
    },
}

impl ImageOutTarget {
    pub fn format(&self) -> &JxlPixelFormat {
        match self {
            Self::Buffer { format, .. } | Self::Callback { format, .. } => format,
        }
    }

    pub fn new_callback(
        format: JxlPixelFormat,
        callback: JxlImageOutCallback,
        opaque: *mut c_void,
    ) -> Option<Self> {
        Some(Self::Callback {
            callback: callback?,
            opaque,
            format,
            buf: Vec::new(),
        })
    }

    pub fn make_output_buffer(
        &mut self,
        width: usize,
        height: usize,
    ) -> Option<JxlOutputBuffer<'_>> {
        let format = *self.format();
        let (bytes_per_row, stride) = compute_row_layout(width, &format)?;

        match self {
            Self::Buffer { ptr, size, .. } => {
                let needed = compute_buffer_size(width, height, &format)?;
                if *size < needed {
                    return None;
                }
                // SAFETY: Caller of `JxlDecoderSetImageOutBuffer` / `JxlDecoderSetExtraChannelBuffer`
                // provided a valid buffer of at least `needed` bytes.
                Some(unsafe { JxlOutputBuffer::new_from_ptr(*ptr, height, bytes_per_row, stride) })
            }
            Self::Callback { buf, .. } => {
                buf.resize(height.checked_mul(bytes_per_row)?, 0);
                Some(JxlOutputBuffer::new(buf, height, bytes_per_row))
            }
        }
    }

    pub fn finalize_output(&self, width: usize) -> bool {
        let Self::Callback {
            callback,
            opaque,
            format,
            buf,
        } = self
        else {
            return true;
        };
        let Some((bytes_per_row, _)) = compute_row_layout(width, format) else {
            return false;
        };
        for (y, row) in buf.chunks_exact(bytes_per_row).enumerate() {
            // SAFETY: Invoking caller-supplied C row callback per `JxlDecoderSetImageOutCallback` contract.
            unsafe {
                callback(*opaque, 0, y, width, row.as_ptr().cast::<c_void>());
            }
        }
        true
    }
}
