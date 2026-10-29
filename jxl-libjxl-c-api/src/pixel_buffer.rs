// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use jxl::api::JxlOutputBuffer;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::conversions::to_rs_endianness;
use crate::types::{
    JXL_TYPE_FLOAT, JXL_TYPE_FLOAT16, JXL_TYPE_UINT8, JXL_TYPE_UINT16, JxlDataType,
    JxlImageOutDestroyCallback, JxlPixelFormat,
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

/// A borrowed C input byte buffer registered via `JxlDecoderSetInput`.
pub(crate) struct InputBuffer {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped), either
    // `buf.len() == 0` or `buf` points to `buf.len()` initialized bytes (within a single
    // allocation of at most `isize::MAX` bytes) that are valid for reads and not modified
    // concurrently.
    buf: NonNull<[u8]>,
}

impl InputBuffer {
    /// Wraps a raw C input pointer and byte length. Returns `None` if `data` is null and
    /// `size > 0`.
    ///
    /// # Safety
    /// If `data` is non-null and `size > 0`, the caller must ensure that during calls to functions
    /// in this crate until this `InputBuffer` is dropped, `data` is valid for reads of `size`
    /// initialized bytes (within a single allocation of at most `isize::MAX` bytes) and the
    /// pointed-to bytes are not modified concurrently.
    pub(crate) unsafe fn new(data: *const u8, size: usize) -> Option<Self> {
        let non_null_ptr = if size == 0 {
            NonNull::<u8>::dangling()
        } else {
            NonNull::new(data.cast_mut())?
        };
        Some(Self {
            buf: NonNull::slice_from_raw_parts(non_null_ptr, size),
        })
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: We are inside a function in this crate, and `&self` cannot outlive this call or
        // coexist with `&mut self` methods that release the buffer, so by `InputBuffer`'s safety
        // invariant `self.buf` is valid for reads and not modified for the lifetime of `&self`.
        unsafe { self.buf.as_ref() }
    }

    pub(crate) fn consume(&mut self, amount: usize) {
        self.buf = NonNull::from(&self.as_slice()[amount..]);
    }
}

/// A borrowed C output byte buffer registered via `JxlDecoderSetBoxBuffer`.
pub(crate) struct BoxOutBuffer {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped), either
    // `buf.len() == 0` or `buf` points to `buf.len()` writable bytes (within a single allocation
    // of at most `isize::MAX` bytes) that remain exclusively accessible.
    buf: NonNull<[u8]>,
}

impl BoxOutBuffer {
    /// Wraps a raw C output pointer and byte length. Returns `None` if `data` is null and
    /// `size > 0`.
    ///
    /// # Safety
    /// If `data` is non-null and `size > 0`, the caller must ensure that during calls to functions
    /// in this crate until this `BoxOutBuffer` is dropped, `data` is valid for writes of `size`
    /// bytes (within a single allocation of at most `isize::MAX` bytes) and remains exclusively
    /// accessible.
    pub(crate) unsafe fn new(data: *mut u8, size: usize) -> Option<Self> {
        let non_null_ptr = if size == 0 {
            NonNull::<u8>::dangling()
        } else {
            NonNull::new(data)?
        };
        Some(Self {
            buf: NonNull::slice_from_raw_parts(non_null_ptr, size),
        })
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn write(&mut self, src: &[u8]) -> usize {
        let n = self.buf.len().min(src.len());
        let ptr = self.buf.as_ptr().cast::<u8>();
        // SAFETY: Either `n == 0` (with non-null `ptr` and `ptr.add(0) == ptr`), or we are inside
        // a function in this crate so by `BoxOutBuffer`'s safety invariant `ptr` is valid for
        // writes of `n <= self.buf.len()` exclusively accessible bytes within a single
        // `<= isize::MAX` allocation (discharging `copy_nonoverlapping`, `ptr.add(n)`, and
        // `NonNull::new_unchecked`).
        let next_ptr = unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, n);
            NonNull::new_unchecked(ptr.add(n))
        };
        self.buf = NonNull::slice_from_raw_parts(next_ptr, self.buf.len() - n);
        n
    }
}

/// A validated user-provided output pixel buffer and its format.
#[derive(Debug)]
pub(crate) struct ImageOutBuffer {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped), `ptr`
    // is valid for writes of `size` bytes (within a single allocation) that remain exclusively
    // accessible.
    ptr: NonNull<u8>,
    size: usize,
    format: JxlPixelFormat,
}

impl ImageOutBuffer {
    /// Creates a new `ImageOutBuffer` after verifying `buffer` is non-null and `size >= min_size`.
    ///
    /// # Safety
    /// If `buffer` is non-null and `size >= min_size`, the caller must ensure that during calls to
    /// functions in this crate until this `ImageOutBuffer` is dropped, `buffer` is valid for
    /// writes of `size` bytes (within a single allocation) and remains exclusively accessible.
    pub(crate) unsafe fn new(
        buffer: *mut u8,
        size: usize,
        min_size: usize,
        format: JxlPixelFormat,
    ) -> Option<Self> {
        if size < min_size {
            return None;
        }
        let ptr = NonNull::new(buffer)?;
        Some(Self { ptr, size, format })
    }

    pub(crate) fn format(&self) -> JxlPixelFormat {
        self.format
    }

    /// Prepares a `JxlOutputBuffer` for `jxl-rs` to write into directly.
    pub(crate) fn make_output_buffer(
        &mut self,
        width: usize,
        height: usize,
    ) -> Option<JxlOutputBuffer<'_>> {
        let (bytes_per_row, stride) = compute_row_layout(width, &self.format)?;
        let needed = compute_buffer_size(width, height, &self.format)?;
        if self.size < needed {
            return None;
        }
        // SAFETY: We are inside a function in this crate, so by `ImageOutBuffer`'s safety invariant
        // `self.ptr` points to at least `self.size >= needed` writable, exclusively accessible
        // bytes for the lifetime of `&mut self`, satisfying the preconditions of `new_from_ptr`.
        Some(unsafe {
            JxlOutputBuffer::new_from_ptr(self.ptr.as_ptr(), height, bytes_per_row, stride)
        })
    }
}

pub(crate) type SimpleImageOutFn =
    unsafe extern "C" fn(*mut c_void, usize, usize, usize, *const c_void);
pub(crate) type MultithreadedInitFn =
    unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void;
pub(crate) type MultithreadedRunFn =
    unsafe extern "C" fn(*mut c_void, usize, usize, usize, usize, *const c_void);

#[derive(Clone, Copy)]
struct SendOpaque(*mut c_void);

impl SendOpaque {
    fn get(self) -> *mut c_void {
        self.0
    }
}

// SAFETY: `SendOpaque` is only passed to `dispatch_rows`: when `parallel` is `false`
// (`num_threads == 1`), `dispatch_rows` runs synchronously on the calling thread without sending or
// sharing `SendOpaque`; when `parallel` is `true` (`num_threads > 1`), `ImageOutCallback`'s safety
// invariant guarantees that `opaque` / `run_opaque` may be passed concurrently from up to
// `num_threads` worker threads.
unsafe impl Send for SendOpaque {}
// SAFETY: Same as `Send` above (`SendOpaque` is shared across parallel worker closures only when
// `parallel` is `true`).
unsafe impl Sync for SendOpaque {}

fn packed_frame_layout(
    width: usize,
    height: usize,
    format: &JxlPixelFormat,
) -> Option<(usize, usize)> {
    if width == 0 || height == 0 {
        return None;
    }
    let (bytes_per_row, _) = compute_row_layout(width, format)?;
    let total_bytes = bytes_per_row.checked_mul(height)?;
    (total_bytes <= isize::MAX as usize).then_some((bytes_per_row, total_bytes))
}

fn dispatch_rows<F>(height: usize, bytes_per_row: usize, bytes: &[u8], num_threads: usize, f: F)
where
    F: Fn(usize, usize, *const c_void) + Sync,
{
    if num_threads <= 1 {
        for y in 0..height {
            let row_ptr = bytes[y * bytes_per_row..].as_ptr().cast::<c_void>();
            f(0, y, row_ptr);
        }
    } else {
        let next_y = AtomicUsize::new(0);
        (0..num_threads).into_par_iter().for_each(|thread_id| {
            loop {
                let y = next_y.fetch_add(1, Ordering::Relaxed);
                if y >= height {
                    break;
                }
                let row_ptr = bytes[y * bytes_per_row..].as_ptr().cast::<c_void>();
                f(thread_id, y, row_ptr);
            }
        });
    }
}

#[derive(Debug)]
enum CallbackKind {
    Simple {
        callback: SimpleImageOutFn,
        opaque: *mut c_void,
    },
    Multithreaded {
        init: MultithreadedInitFn,
        run: MultithreadedRunFn,
        destroy: JxlImageOutDestroyCallback,
        init_opaque: *mut c_void,
    },
}

/// A validated user-provided output pixel callback and its internal aligned frame buffer.
#[derive(Debug)]
pub(crate) struct ImageOutCallback {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped):
    // - For `CallbackKind::Simple`, `callback(opaque, x, y, num_pixels, pixels)` is safe to invoke
    //   for any valid scanline stripe `(x, y, num_pixels)` in `format` (including concurrently on
    //   disjoint rows when `parallel` is true), does not unwind across FFI, and does not access the
    //   owning `JxlDecoder`.
    // - For `CallbackKind::Multithreaded`, `init(init_opaque, num_threads, num_pixels_per_thread)`
    //   is safe to invoke and, when it returns a non-null `run_opaque`,
    //   `run(run_opaque, thread_id, x, y, num_pixels, pixels)` is safe to invoke concurrently for
    //   distinct `thread_id < num_threads` and `num_pixels <= num_pixels_per_thread`, followed by
    //   `destroy(run_opaque)` (if `Some`) after all `run` calls complete, without unwinding across
    //   FFI or accessing the owning `JxlDecoder`.
    kind: CallbackKind,
    format: JxlPixelFormat,
    // 4-byte-aligned backing storage so every tightly packed row (`y * width * channels * bps`,
    // with `bps` in `{1, 2, 4}`) is naturally aligned for `u8`, `u16`/`f16`, and `f32` samples.
    scratch: Vec<u32>,
}

impl ImageOutCallback {
    /// Creates a new single-callback `ImageOutCallback`.
    ///
    /// # Safety
    /// During calls to functions in this crate until this `ImageOutCallback` is dropped, the caller
    /// must ensure `callback(opaque, x, y, num_pixels, pixels)` is safe to invoke for any valid
    /// scanline stripe in `format` (including concurrently across disjoint rows when a parallel
    /// runner is set), does not unwind across FFI, and does not access the owning `JxlDecoder`.
    pub(crate) unsafe fn new_simple(
        format: JxlPixelFormat,
        callback: SimpleImageOutFn,
        opaque: *mut c_void,
    ) -> Self {
        Self {
            kind: CallbackKind::Simple { callback, opaque },
            format,
            scratch: Vec::new(),
        }
    }

    /// Creates a new multithreaded `ImageOutCallback`.
    ///
    /// # Safety
    /// During calls to functions in this crate until this `ImageOutCallback` is dropped, the caller
    /// must ensure `init(init_opaque, num_threads, num_pixels_per_thread)` is safe to invoke and,
    /// whenever it returns a non-null `run_opaque`,
    /// `run(run_opaque, thread_id, x, y, num_pixels, pixels)` is safe to invoke concurrently for
    /// distinct `thread_id < num_threads` and `num_pixels <= num_pixels_per_thread`, followed by
    /// `destroy(run_opaque)` (when `destroy` is `Some`) after all `run` calls complete, without
    /// unwinding across FFI or accessing the owning `JxlDecoder`.
    pub(crate) unsafe fn new_multithreaded(
        format: JxlPixelFormat,
        init: MultithreadedInitFn,
        run: MultithreadedRunFn,
        destroy: JxlImageOutDestroyCallback,
        init_opaque: *mut c_void,
    ) -> Self {
        Self {
            kind: CallbackKind::Multithreaded {
                init,
                run,
                destroy,
                init_opaque,
            },
            format,
            scratch: Vec::new(),
        }
    }

    pub(crate) fn format(&self) -> JxlPixelFormat {
        self.format
    }

    pub(crate) fn make_output_buffer(
        &mut self,
        width: usize,
        height: usize,
    ) -> Option<JxlOutputBuffer<'_>> {
        let (bytes_per_row, total_bytes) = packed_frame_layout(width, height, &self.format)?;
        let words = total_bytes.div_ceil(size_of::<u32>());
        if self.scratch.len() < words {
            self.scratch
                .try_reserve_exact(words - self.scratch.len())
                .ok()?;
            self.scratch.resize(words, 0);
        }
        // SAFETY: `self.scratch` holds at least `words * 4 >= total_bytes <= isize::MAX`
        // initialized, exclusively borrowed bytes (`u32` has no padding or invalid bit patterns)
        // for the lifetime of `&mut self`.
        let byte_slice = unsafe {
            std::slice::from_raw_parts_mut(self.scratch.as_mut_ptr().cast::<u8>(), total_bytes)
        };
        Some(JxlOutputBuffer::new(byte_slice, height, bytes_per_row))
    }

    pub(crate) fn emit(&self, width: usize, height: usize, parallel: bool) -> bool {
        let Some((bytes_per_row, total_bytes)) = packed_frame_layout(width, height, &self.format)
        else {
            return false;
        };
        if self.scratch.len().saturating_mul(size_of::<u32>()) < total_bytes {
            return false;
        }
        // SAFETY: `self.scratch` holds at least `total_bytes` initialized, immutable bytes for the
        // lifetime of `&self`.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.scratch.as_ptr().cast::<u8>(), total_bytes) };
        let num_threads = if parallel {
            rayon::current_num_threads().min(height).max(1)
        } else {
            1
        };

        match self.kind {
            CallbackKind::Simple { callback, opaque } => {
                let opaque = SendOpaque(opaque);
                dispatch_rows(
                    height,
                    bytes_per_row,
                    bytes,
                    num_threads,
                    |_, y, row_ptr| {
                        // SAFETY: Each row `y < height` is visited exclusively with `row_ptr` pointing
                        // to `bytes_per_row` initialized, sample-aligned bytes valid for the callback
                        // duration, and `ImageOutCallback`'s safety invariant guarantees `callback` is
                        // safe to invoke (including concurrently across disjoint rows when `parallel`
                        // is true).
                        unsafe {
                            callback(opaque.get(), 0, y, width, row_ptr);
                        }
                    },
                );
                true
            }
            CallbackKind::Multithreaded {
                init,
                run,
                destroy,
                init_opaque,
            } => {
                // SAFETY: Guaranteed by `ImageOutCallback`'s safety invariant (`num_threads >= 1`
                // and `num_pixels_per_thread == width`).
                let run_opaque = unsafe { init(init_opaque, num_threads, width) };
                if run_opaque.is_null() {
                    return false;
                }
                let run_opaque_send = SendOpaque(run_opaque);
                dispatch_rows(
                    height,
                    bytes_per_row,
                    bytes,
                    num_threads,
                    |thread_id, y, row_ptr| {
                        // SAFETY: `run_opaque` is the non-null handle returned by `init`; each
                        // `thread_id < num_threads` executes sequentially inside its own worker so
                        // no two concurrent `run` calls share a `thread_id`; `width` equals
                        // `num_pixels_per_thread`; and `row_ptr` points to `bytes_per_row`
                        // initialized, sample-aligned bytes.
                        unsafe {
                            run(run_opaque_send.get(), thread_id, 0, y, width, row_ptr);
                        }
                    },
                );
                if let Some(destroy_fn) = destroy {
                    // SAFETY: All `run` invocations for `run_opaque` have completed.
                    unsafe {
                        destroy_fn(run_opaque);
                    }
                }
                true
            }
        }
    }
}

/// Main-frame pixel output destination (either a user buffer or a scanline callback).
#[derive(Debug)]
pub(crate) enum ImageOutTarget {
    Buffer(ImageOutBuffer),
    Callback(ImageOutCallback),
}

impl ImageOutTarget {
    pub(crate) fn format(&self) -> JxlPixelFormat {
        match self {
            Self::Buffer(buf) => buf.format(),
            Self::Callback(cb) => cb.format(),
        }
    }

    pub(crate) fn make_output_buffer(
        &mut self,
        width: usize,
        height: usize,
    ) -> Option<JxlOutputBuffer<'_>> {
        match self {
            Self::Buffer(buf) => buf.make_output_buffer(width, height),
            Self::Callback(cb) => cb.make_output_buffer(width, height),
        }
    }

    pub(crate) fn emit_callbacks(&self, width: usize, height: usize, parallel: bool) -> bool {
        match self {
            Self::Buffer(_) => true,
            Self::Callback(cb) => cb.emit(width, height, parallel),
        }
    }
}
