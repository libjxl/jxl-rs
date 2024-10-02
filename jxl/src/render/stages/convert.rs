// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{RenderPipelineInOutStage, RenderPipelineStage};

pub struct ConvertU8F32Stage {
    channel: usize,
}

impl ConvertU8F32Stage {
    pub fn new(channel: usize) -> ConvertU8F32Stage {
        ConvertU8F32Stage { channel }
    }
}

impl RenderPipelineStage for ConvertU8F32Stage {
    type Type = RenderPipelineInOutStage<u8, f32, 0, 0, 0, 0>;

    fn name(&self) -> String {
        format!("convert U8 data to F32 in channel {}", self.channel)
    }

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[u8]], &mut [&mut [f32]])],
    ) {
        let (input, output) = &mut row[0];
        for i in 0..xsize {
            output[0][i] = input[0][i] as f32 * (1.0 / 255.0);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::error::Result;
    use test_log::test;

    #[test]
    fn u8_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, u8, f32>(
            ConvertU8F32Stage::new(0),
            (500, 500),
            1,
        )
    }
}
