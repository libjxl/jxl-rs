// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::image::{DataTypeTag, Image, Rect, box_average};
use crate::render::stages::DownsampleStage;
use crate::util::ShiftRightCeil;

impl DownsampleStage {
    pub(super) fn run_simple(
        &self,
        input_buffers: &[Image<f64>],
        output_buffers: &mut [Image<f64>],
    ) -> Result<()> {
        let channels = self.from_channel..self.from_channel + self.alpha.len();
        let input_buffers = &input_buffers[channels.clone()];
        let output_buffers = &mut output_buffers[channels];
        let Some(first) = input_buffers.first() else {
            return Ok(());
        };
        let size = first.size();
        let shift = self.shift() as usize;
        for (input, output) in input_buffers.iter().zip(output_buffers.iter()) {
            assert_eq!(input.size(), size);
            assert_eq!(output.size(), (size.0.shrc(shift), size.1.shrc(shift)));
        }

        let rect = Rect {
            origin: (0, 0),
            size,
        };
        for ((input, output), alpha) in input_buffers.iter().zip(output_buffers).zip(&self.alpha) {
            let input = input.get_rect(rect).into_raw();
            let alpha_image =
                alpha.map(|(channel, _)| input_buffers[channel].get_rect(rect).into_raw());
            let mut output = output
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (size.0.shrc(shift), size.1.shrc(shift)),
                })
                .into_raw();
            for y in 0..size.1.shrc(shift) {
                let rows = (y << shift)..((y + 1) << shift).min(size.1);
                let source: Vec<_> = rows.clone().map(|y| input.row(y)).collect();
                let alpha_rows = alpha_image
                    .as_ref()
                    .map(|image| rows.map(|y| image.row(y)).collect::<Vec<_>>());
                box_average(
                    DataTypeTag::F64,
                    &source,
                    shift,
                    alpha_rows.as_deref(),
                    alpha.is_some_and(|(_, clamp)| clamp),
                    output.row(y),
                );
            }
        }
        Ok(())
    }
}
