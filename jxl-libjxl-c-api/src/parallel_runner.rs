// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;

use jxl::api::{JxlParallelRunner as RsParallelRunner, JxlParallelRunnerFun};
use jxl::error::Result;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::types::{
    JXL_PARALLEL_RET_SUCCESS, JxlMemoryManager, JxlParallelRetCode, JxlParallelRunFunction,
    JxlParallelRunInit,
};

pub(crate) struct RayonParallelRunner;

impl RsParallelRunner for RayonParallelRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> Result<()> {
        (0..num).into_par_iter().try_for_each(fun)
    }

    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

// Unsupported feature: custom C parallel runners (`JxlParallelRunner`).
// `jxl-libjxl-c-api` always uses its own internal Rayon parallel runner during decoding.
// `JxlThreadParallelRunner*` and `JxlResizableParallelRunner*` are provided as minimal stubs so
// existing OSS callers that create/pass/destroy them continue to work without changes. The memory
// manager is ignored, and callers must not assume that `alloc` was used to allocate the returned
// handle.

#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunnerDefaultNumWorkerThreads() -> usize {
    rayon::current_num_threads()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunnerCreate(
    _memory_manager: Option<&JxlMemoryManager>,
    _num_worker_threads: usize,
) -> *mut c_void {
    std::ptr::dangling_mut::<u8>().cast::<c_void>()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunnerDestroy(_runner_opaque: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlThreadParallelRunner(
    _runner_opaque: *mut c_void,
    _jpegxl_opaque: *mut c_void,
    _init: JxlParallelRunInit,
    _func: JxlParallelRunFunction,
    _start_range: u32,
    _end_range: u32,
) -> JxlParallelRetCode {
    JXL_PARALLEL_RET_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunnerSuggestThreads(_xsize: u64, _ysize: u64) -> u32 {
    rayon::current_num_threads() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunnerCreate(
    memory_manager: Option<&JxlMemoryManager>,
) -> *mut c_void {
    JxlThreadParallelRunnerCreate(memory_manager, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunnerSetThreads(
    _runner_opaque: *mut c_void,
    _num_threads: usize,
) {
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunnerDestroy(_runner_opaque: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlResizableParallelRunner(
    _runner_opaque: *mut c_void,
    _jpegxl_opaque: *mut c_void,
    _init: JxlParallelRunInit,
    _func: JxlParallelRunFunction,
    _start_range: u32,
    _end_range: u32,
) -> JxlParallelRetCode {
    JXL_PARALLEL_RET_SUCCESS
}
