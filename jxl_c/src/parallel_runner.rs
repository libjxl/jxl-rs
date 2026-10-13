// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use jxl::api::{JxlParallelRunner as RsParallelRunner, JxlParallelRunnerFun};
use jxl::error::{Error as JxlError, Result as JxlResult};

use crate::types::{
    JXL_PARALLEL_RET_RUNNER_ERROR, JXL_PARALLEL_RET_SUCCESS, JxlMemoryManager, JxlParallelRetCode,
    JxlParallelRunFunction, JxlParallelRunInit,
};

#[derive(Clone, Copy)]
pub struct CParallelRunner {
    pub runner: unsafe extern "C" fn(
        runner_opaque: *mut c_void,
        jpegxl_opaque: *mut c_void,
        init: JxlParallelRunInit,
        func: JxlParallelRunFunction,
        start_range: u32,
        end_range: u32,
    ) -> JxlParallelRetCode,
    pub runner_opaque: *mut c_void,
}

struct RunnerContext<'a, 'b> {
    fun: &'a JxlParallelRunnerFun<'b>,
    failed: AtomicBool,
}

unsafe extern "C" fn init_trampoline(
    _jpegxl_opaque: *mut c_void,
    _num_threads: usize,
) -> JxlParallelRetCode {
    JXL_PARALLEL_RET_SUCCESS
}

unsafe extern "C" fn func_trampoline(jpegxl_opaque: *mut c_void, value: u32, _thread_id: usize) {
    // SAFETY: `jpegxl_opaque` was created in `CParallelRunner::run` from a valid `&RunnerContext`
    // that outlives all `func_trampoline` invocations during `(self.runner)(...)`.
    let ctx = unsafe { &*(jpegxl_opaque as *const RunnerContext<'_, '_>) };
    if ctx.failed.load(Ordering::Relaxed) {
        return;
    }
    let res = catch_unwind(AssertUnwindSafe(|| (ctx.fun)(value as usize)));
    if !matches!(res, Ok(Ok(()))) {
        ctx.failed.store(true, Ordering::Relaxed);
    }
}

impl RsParallelRunner for CParallelRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> JxlResult<()> {
        let Ok(end_range) = u32::try_from(num) else {
            return Err(JxlError::ApiUsageError("parallel runner range overflow"));
        };
        let mut ctx = RunnerContext {
            fun,
            failed: AtomicBool::new(false),
        };
        let jpegxl_opaque = (&raw mut ctx).cast::<c_void>();
        // SAFETY: Calling the C parallel runner configured via `JxlDecoderSetParallelRunner`
        // with valid function pointers and a context pointer valid for the duration of the call.
        let ret = unsafe {
            (self.runner)(
                self.runner_opaque,
                jpegxl_opaque,
                Some(init_trampoline),
                Some(func_trampoline),
                0,
                end_range,
            )
        };
        if ret != JXL_PARALLEL_RET_SUCCESS || ctx.failed.load(Ordering::Relaxed) {
            return Err(JxlError::ApiUsageError("parallel runner failed"));
        }
        Ok(())
    }

    fn num_threads(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

struct ThreadRunnerState {
    num_threads: usize,
}

fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

unsafe fn run_with_threads(
    num_threads: usize,
    jpegxl_opaque: *mut c_void,
    init: JxlParallelRunInit,
    func: JxlParallelRunFunction,
    start_range: u32,
    end_range: u32,
) -> JxlParallelRetCode {
    let Some(init_fn) = init else {
        return JXL_PARALLEL_RET_RUNNER_ERROR;
    };
    let Some(func_fn) = func else {
        return JXL_PARALLEL_RET_RUNNER_ERROR;
    };
    let num_threads = num_threads.max(1);
    // SAFETY: Caller guarantees `init_fn` and `jpegxl_opaque` are valid per `JxlParallelRunner` ABI.
    let init_ret = unsafe { init_fn(jpegxl_opaque, num_threads) };
    if init_ret != JXL_PARALLEL_RET_SUCCESS {
        return init_ret;
    }
    if start_range >= end_range {
        return JXL_PARALLEL_RET_SUCCESS;
    }

    if num_threads == 1 || end_range - start_range == 1 {
        for value in start_range..end_range {
            // SAFETY: Caller guarantees `func_fn` and `jpegxl_opaque` are valid for `0..num_threads`.
            unsafe {
                func_fn(jpegxl_opaque, value, 0);
            }
        }
        return JXL_PARALLEL_RET_SUCCESS;
    }

    let next_val = AtomicU64::new(start_range as u64);
    let end_val = end_range as u64;
    let opaque_addr = jpegxl_opaque as usize;
    let worker_count = num_threads.min((end_range - start_range) as usize);

    std::thread::scope(|s| {
        for thread_id in 1..worker_count {
            let next_ref = &next_val;
            s.spawn(move || {
                loop {
                    let val = next_ref.fetch_add(1, Ordering::Relaxed);
                    if val >= end_val {
                        break;
                    }
                    // SAFETY: `func_fn` and `jpegxl_opaque` are thread-safe across `0..num_threads`
                    // as guaranteed by the `JxlParallelRunner` contract.
                    unsafe {
                        func_fn(opaque_addr as *mut c_void, val as u32, thread_id);
                    }
                }
            });
        }
        loop {
            let val = next_val.fetch_add(1, Ordering::Relaxed);
            if val >= end_val {
                break;
            }
            // SAFETY: Thread 0 executes in the calling thread within `0..num_threads`.
            unsafe {
                func_fn(jpegxl_opaque, val as u32, 0);
            }
        }
    });

    JXL_PARALLEL_RET_SUCCESS
}

/// Returns the default number of worker threads for `JxlThreadParallelRunner`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunnerDefaultNumWorkerThreads() -> usize {
    default_num_threads()
}

/// Creates a thread parallel runner state.
///
/// # Safety
/// `memory_manager` must be null or point to a valid `JxlMemoryManager`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlThreadParallelRunnerCreate(
    memory_manager: *const JxlMemoryManager,
    num_worker_threads: usize,
) -> *mut c_void {
    if !memory_manager.is_null() {
        // SAFETY: `memory_manager` is non-null and valid per caller contract.
        let mm = unsafe { &*memory_manager };
        if mm.alloc.is_some() != mm.free.is_some() {
            return std::ptr::null_mut();
        }
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
///
/// # Safety
/// `memory_manager` must be null or point to a valid `JxlMemoryManager`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlResizableParallelRunnerCreate(
    memory_manager: *const JxlMemoryManager,
) -> *mut c_void {
    // SAFETY: Forwarding to `JxlThreadParallelRunnerCreate` with 1 initial thread.
    unsafe { JxlThreadParallelRunnerCreate(memory_manager, 1) }
}

/// Updates the number of threads for a resizable parallel runner.
///
/// # Safety
/// `runner_opaque` must be a valid pointer returned by `JxlResizableParallelRunnerCreate`.
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
/// `runner_opaque` must be a valid pointer returned by `JxlResizableParallelRunnerCreate`, and
/// `init`/`func`/`jpegxl_opaque` must satisfy the `JxlParallelRunner` C-ABI contract.
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
