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
    JxlImageOutCallback, JxlImageOutDestroyCallback, JxlImageOutInitCallback,
    JxlImageOutRunCallback, JxlPixelFormat,
};

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

pub fn to_rs_color_type(num_channels: u32, bgr: bool, is_gray: bool) -> Option<JxlColorType> {
    if !is_gray && num_channels < 3 {
        return None;
    }
    match (num_channels, bgr) {
        (1, _) => Some(JxlColorType::Grayscale),
        (2, _) => Some(JxlColorType::GrayscaleAlpha),
        (3, false) => Some(JxlColorType::Rgb),
        (3, true) => Some(JxlColorType::Bgr),
        (4, false) => Some(JxlColorType::Rgba),
        (4, true) => Some(JxlColorType::Bgra),
        _ => None,
    }
}

pub fn build_rs_pixel_format(
    color_format: Option<&JxlPixelFormat>,
    extra_channel_formats: &[(Option<JxlPixelFormat>, RsBitDepth)],
    bit_depth: &JxlBitDepth,
    codestream_bit_depth: &RsBitDepth,
    is_gray: bool,
) -> Option<RsPixelFormat> {
    let (color_type, color_data_format) = if let Some(cf) = color_format {
        let ct = to_rs_color_type(cf.num_channels, false, is_gray)?;
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
    for (ecf, ec_bit_depth) in extra_channel_formats {
        if let Some(f) = ecf {
            if f.num_channels != 1 {
                return None;
            }
            extra_channel_format.push(Some(to_rs_data_format(f, bit_depth, ec_bit_depth)?));
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

/// 8-byte aligned scratch buffer for decoding via row callbacks.
#[derive(Debug, Clone)]
pub struct AlignedScratch {
    words: Vec<u64>,
    pub bytes_per_row: usize,
    pub num_rows: usize,
}

impl AlignedScratch {
    pub fn new(num_rows: usize, bytes_per_row: usize) -> Self {
        let total_bytes = num_rows.saturating_mul(bytes_per_row);
        let num_words = total_bytes.div_ceil(8);
        Self {
            words: vec![0u64; num_words],
            bytes_per_row,
            num_rows,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.words.as_mut_ptr().cast::<u8>()
    }

    pub fn row_bytes(&self, y: usize) -> &[u8] {
        let start = y * self.bytes_per_row;
        let all_bytes =
            // SAFETY: `self.words` is valid for `self.words.len() * 8` bytes, which is at least
            // `self.num_rows * self.bytes_per_row`.
            unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.words.len() * 8) };
        &all_bytes[start..start + self.bytes_per_row]
    }
}

pub enum ImageOutTarget {
    Buffer {
        ptr: *mut u8,
        size: usize,
        format: JxlPixelFormat,
    },
    SimpleCallback {
        callback: unsafe extern "C" fn(
            opaque: *mut c_void,
            x: usize,
            y: usize,
            num_pixels: usize,
            pixels: *const c_void,
        ),
        opaque: *mut c_void,
        format: JxlPixelFormat,
        scratch: Option<AlignedScratch>,
    },
    MultithreadedCallback {
        init_callback: unsafe extern "C" fn(
            init_opaque: *mut c_void,
            num_threads: usize,
            num_pixels_per_thread: usize,
        ) -> *mut c_void,
        run_callback: unsafe extern "C" fn(
            run_opaque: *mut c_void,
            thread_id: usize,
            x: usize,
            y: usize,
            num_pixels: usize,
            pixels: *const c_void,
        ),
        destroy_callback: Option<unsafe extern "C" fn(run_opaque: *mut c_void)>,
        init_opaque: *mut c_void,
        format: JxlPixelFormat,
        scratch: Option<AlignedScratch>,
    },
}

impl ImageOutTarget {
    pub fn format(&self) -> &JxlPixelFormat {
        match self {
            Self::Buffer { format, .. }
            | Self::SimpleCallback { format, .. }
            | Self::MultithreadedCallback { format, .. } => format,
        }
    }

    pub fn new_simple_callback(
        format: JxlPixelFormat,
        callback: JxlImageOutCallback,
        opaque: *mut c_void,
    ) -> Option<Self> {
        Some(Self::SimpleCallback {
            callback: callback?,
            opaque,
            format,
            scratch: None,
        })
    }

    pub fn new_multithreaded_callback(
        format: JxlPixelFormat,
        init_callback: JxlImageOutInitCallback,
        run_callback: JxlImageOutRunCallback,
        destroy_callback: JxlImageOutDestroyCallback,
        init_opaque: *mut c_void,
    ) -> Option<Self> {
        Some(Self::MultithreadedCallback {
            init_callback: init_callback?,
            run_callback: run_callback?,
            destroy_callback,
            init_opaque,
            format,
            scratch: None,
        })
    }

    /// Prepares a `JxlOutputBuffer` for `jxl-rs` to write into directly (for user buffers)
    /// or into an aligned scratch buffer (for row callbacks).
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
                // SAFETY: The caller of `JxlDecoderSetImageOutBuffer` / `JxlDecoderSetPreviewOutBuffer`
                // / `JxlDecoderSetExtraChannelBuffer` provided a valid buffer of at least `needed`
                // bytes, which satisfies the row stride and size preconditions of `new_from_ptr`.
                Some(unsafe { JxlOutputBuffer::new_from_ptr(*ptr, height, bytes_per_row, stride) })
            }
            Self::SimpleCallback { scratch, .. } | Self::MultithreadedCallback { scratch, .. } => {
                let s = scratch.get_or_insert_with(|| AlignedScratch::new(height, bytes_per_row));
                if s.num_rows != height || s.bytes_per_row != bytes_per_row {
                    *s = AlignedScratch::new(height, bytes_per_row);
                }
                // SAFETY: `AlignedScratch` owns a contiguous 8-byte aligned buffer of
                // `height * bytes_per_row` initialized bytes.
                Some(unsafe {
                    JxlOutputBuffer::new_from_ptr(
                        s.as_mut_ptr(),
                        height,
                        bytes_per_row,
                        bytes_per_row,
                    )
                })
            }
        }
    }

    /// Invokes row callbacks once frame decoding completes.
    /// Returns `false` if a multithreaded callback initialization failed.
    pub fn finalize_output(&self, width: usize, height: usize) -> bool {
        match self {
            Self::Buffer { .. } => true,
            Self::SimpleCallback {
                callback,
                opaque,
                scratch: Some(scratch),
                ..
            } => {
                for y in 0..height {
                    let row = scratch.row_bytes(y);
                    // SAFETY: Invoking caller-supplied C callback with valid row data for row `y`.
                    unsafe {
                        callback(*opaque, 0, y, width, row.as_ptr().cast::<c_void>());
                    }
                }
                true
            }
            Self::MultithreadedCallback {
                init_callback,
                run_callback,
                destroy_callback,
                init_opaque,
                scratch: Some(scratch),
                ..
            } => {
                // SAFETY: Invoking caller-supplied C init/run/destroy callbacks per libjxl contract.
                unsafe {
                    let run_opaque = init_callback(*init_opaque, 1, width);
                    if run_opaque.is_null() {
                        return false;
                    }
                    for y in 0..height {
                        let row = scratch.row_bytes(y);
                        run_callback(run_opaque, 0, 0, y, width, row.as_ptr().cast::<c_void>());
                    }
                    if let Some(destroy) = destroy_callback {
                        destroy(run_opaque);
                    }
                }
                true
            }
            _ => true,
        }
    }
}
