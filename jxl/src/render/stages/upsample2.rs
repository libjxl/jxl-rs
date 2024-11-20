// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{RenderPipelineInOutStage, RenderPipelineStage};

pub struct Upsample2x {
    channel: usize,
}

impl std::fmt::Display for Upsample2x {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "2x2 upsampling of channel {}",
            self.channel
        )
    }
}

impl RenderPipelineStage for Upsample2x {
    type Type = RenderPipelineInOutStage<u8,u8,0,0,1,1>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(&mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[u8]], &mut [&mut [u8]])],
    ) {
        todo!();
    }

}