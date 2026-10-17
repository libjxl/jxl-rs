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
    JXL_PARALLEL_RET_RUNNER_ERROR, JXL_PARALLEL_RET_SUCCESS, JxlParallelRetCode,
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
        default_num_threads()
    }

    fn run_ordered(
        &mut self,
        num: usize,
        _max_threads: Option<usize>,
        fun: &JxlParallelRunnerFun<'_>,
    ) -> JxlResult<()> {
        if num == 0 {
            return Ok(());
        }
        self.run(num, fun)
    }
}

pub(crate) struct ThreadRunnerState {
    pub(crate) num_threads: usize,
}

pub(crate) fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

pub(crate) unsafe fn run_with_threads(
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
    let num_threads = if cfg!(any(target_family = "wasm", target_arch = "wasm32")) {
        1
    } else {
        num_threads.max(1)
    };
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
