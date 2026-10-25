// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{JxlParallelRunner as RsParallelRunner, JxlParallelRunnerFun};
use jxl::error::Result;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub(crate) struct RayonParallelRunner;

impl RsParallelRunner for RayonParallelRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun<'_>) -> Result<()> {
        (0..num).into_par_iter().try_for_each(fun)
    }

    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}
