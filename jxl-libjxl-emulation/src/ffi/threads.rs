// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;

use crate::parallel_runner::{ThreadRunnerState, default_num_threads, run_with_threads};
use crate::types::{
    JxlMemoryManager, JxlParallelRetCode, JxlParallelRunFunction, JxlParallelRunInit,
};

/// Returns the default number of worker threads for `JxlThreadParallelRunner`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunnerDefaultNumWorkerThreads() -> usize {
    default_num_threads()
}

/// Creates a thread parallel runner state.
#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunnerCreate(
    memory_manager: Option<&JxlMemoryManager>,
    num_worker_threads: usize,
) -> *mut c_void {
    if let Some(mm) = memory_manager
        && (mm.alloc.is_none() || mm.free.is_none())
    {
        return std::ptr::null_mut();
    }
    let num_threads = if num_worker_threads == 0 {
        default_num_threads()
    } else {
        num_worker_threads
    };
    Box::into_raw(Box::new(ThreadRunnerState { num_threads })).cast::<c_void>()
}

/// Destroys a thread parallel runner created by `JxlThreadParallelRunnerCreate`.
///
/// # Safety
/// `runner_opaque` must be null or a pointer returned by `JxlThreadParallelRunnerCreate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlThreadParallelRunnerDestroy(runner_opaque: *mut c_void) {
    if !runner_opaque.is_null() {
        // SAFETY: `runner_opaque` was allocated with `Box::into_raw` in `JxlThreadParallelRunnerCreate`.
        unsafe {
            drop(Box::from_raw(runner_opaque.cast::<ThreadRunnerState>()));
        }
    }
}

/// Parallel runner implementation using standard threads.
///
/// # Safety
/// `runner_opaque` must be a valid pointer returned by `JxlThreadParallelRunnerCreate` (or null
/// to use default thread count), and `init`/`func`/`jpegxl_opaque` must satisfy the
/// `JxlParallelRunner` C-ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlThreadParallelRunner(
    runner_opaque: *mut c_void,
    jpegxl_opaque: *mut c_void,
    init: JxlParallelRunInit,
    func: JxlParallelRunFunction,
    start_range: u32,
    end_range: u32,
) -> JxlParallelRetCode {
    let num_threads = if runner_opaque.is_null() {
        default_num_threads()
    } else {
        // SAFETY: `runner_opaque` is a valid `ThreadRunnerState` pointer.
        unsafe { (*runner_opaque.cast::<ThreadRunnerState>()).num_threads }
    };
    // SAFETY: Forwarding validated runner arguments.
    unsafe {
        run_with_threads(
            num_threads,
            jpegxl_opaque,
            init,
            func,
            start_range,
            end_range,
        )
    }
}

/// Suggests a number of threads for a given image size.
#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunnerSuggestThreads(xsize: u64, ysize: u64) -> u32 {
    let num_groups = xsize.div_ceil(256).saturating_mul(ysize.div_ceil(256));
    let max_threads = default_num_threads() as u64;
    num_groups.clamp(1, max_threads) as u32
}

/// Creates a resizable parallel runner state.
#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunnerCreate(
    memory_manager: Option<&JxlMemoryManager>,
) -> *mut c_void {
    JxlThreadParallelRunnerCreate(memory_manager, 1)
}

/// Updates the number of threads for a resizable parallel runner.
///
/// # Safety
/// `runner_opaque` must be null or a valid pointer returned by `JxlResizableParallelRunnerCreate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlResizableParallelRunnerSetThreads(
    runner_opaque: *mut c_void,
    num_threads: usize,
) {
    if !runner_opaque.is_null() {
        // SAFETY: `runner_opaque` is a valid `ThreadRunnerState` pointer.
        unsafe {
            (*runner_opaque.cast::<ThreadRunnerState>()).num_threads = num_threads.max(1);
        }
    }
}

/// Destroys a resizable parallel runner created by `JxlResizableParallelRunnerCreate`.
///
/// # Safety
/// `runner_opaque` must be null or a pointer returned by `JxlResizableParallelRunnerCreate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlResizableParallelRunnerDestroy(runner_opaque: *mut c_void) {
    // SAFETY: `JxlResizableParallelRunnerCreate` uses the same representation as `JxlThreadParallelRunnerCreate`.
    unsafe {
        JxlThreadParallelRunnerDestroy(runner_opaque);
    }
}

/// Resizable parallel runner callback.
///
/// # Safety
/// `runner_opaque` must be null or a valid pointer returned by `JxlResizableParallelRunnerCreate`,
/// and `init`/`func`/`jpegxl_opaque` must satisfy the `JxlParallelRunner` C-ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlResizableParallelRunner(
    runner_opaque: *mut c_void,
    jpegxl_opaque: *mut c_void,
    init: JxlParallelRunInit,
    func: JxlParallelRunFunction,
    start_range: u32,
    end_range: u32,
) -> JxlParallelRetCode {
    // SAFETY: Forwarding to `JxlThreadParallelRunner`.
    unsafe {
        JxlThreadParallelRunner(
            runner_opaque,
            jpegxl_opaque,
            init,
            func,
            start_range,
            end_range,
        )
    }
}
