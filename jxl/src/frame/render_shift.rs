// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::render::pipeline;
use crate::error::Result;
use crate::headers::frame_header::{FrameHeader, FrameType};
use crate::image::{Image, ImageDataType, Rect};
use crate::render::RenderPipeline;
use crate::util::ShiftRightCeil;

#[derive(Clone, Copy)]
pub(super) struct RenderShiftInfo {
    pub shift: usize,
    pub upsampling_shift: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum RenderShiftInfoMode {
    FirstPass,
    ReferenceOutput,
    FinalOutput,
}

impl RenderShiftInfo {
    pub fn new(header: &FrameHeader, requested_shift: usize, mode: RenderShiftInfoMode) -> Self {
        let is_lf = header.frame_type == FrameType::LFFrame;
        let is_patch_ref = header.can_be_referenced && header.save_before_ct;
        let needs_full_res = match mode {
            RenderShiftInfoMode::FirstPass => {
                is_lf || is_patch_ref || header.needs_full_resolution_blending()
            }
            RenderShiftInfoMode::ReferenceOutput => is_lf || is_patch_ref,
            RenderShiftInfoMode::FinalOutput => is_lf,
        };

        Self {
            shift: if needs_full_res { 0 } else { requested_shift },
            upsampling_shift: header.upsampling.ilog2() as usize,
        }
    }

    pub fn downsampled_size(self, size: (usize, usize)) -> (usize, usize) {
        (size.0.shrc(self.shift), size.1.shrc(self.shift))
    }

    /// Downsampling factor in effect when rendering frame features.
    pub fn feature_rendering_shift(self) -> usize {
        self.shift.saturating_sub(self.upsampling_shift)
    }

    fn signalled_shift(self, header: &FrameHeader, channel: usize) -> (usize, usize) {
        if channel < 3 {
            let shift = self.upsampling_shift;
            (
                shift + header.hshift(channel),
                shift + header.vshift(channel),
            )
        } else {
            let shift = header.ec_upsampling[channel - 3].ilog2() as usize;
            (shift, shift)
        }
    }

    /// Required additional downsampling after producing raw samples from Modular and VarDCT data.
    pub fn output_shift(self, header: &FrameHeader, channel: usize) -> usize {
        let us = if channel < 3 {
            self.upsampling_shift
        } else {
            header.ec_upsampling[channel - 3].ilog2() as usize
        };
        self.shift.saturating_sub(us)
    }
}

/// Nearest-grid alignment, with ties towards positive infinity. Use the same
/// rule for frame origins and patch positions, including negative frame crops.
pub(crate) fn align_origin(origin: isize, shift: usize) -> isize {
    if shift == 0 {
        origin
    } else {
        (origin + (1isize << (shift - 1))) >> shift
    }
}

/// Computes a box average, optionally accounting for an alpha channel and whether it's
/// premultiplied.
pub(crate) fn box_average<T: ImageDataType, Dest>(
    source: &Image<T>,
    rect: Rect,
    shift: usize,
    alpha: Option<(&Image<T>, bool)>,
    destination: &mut Dest,
    get_destination_row: impl Fn(&mut Dest, usize) -> &mut [T],
) {
    let valid_size = rect.size;
    let output_size = (valid_size.0.shrc(shift), valid_size.1.shrc(shift));
    assert!(source.size().0 >= rect.end().0 && source.size().1 >= rect.end().1);
    if let Some((alpha, _)) = alpha {
        assert!(alpha.size().0 >= rect.end().0 && alpha.size().1 >= rect.end().1);
    }

    let source = source.get_rect(rect);
    let alpha = alpha
        .as_ref()
        .map(|(image, premul)| (image.get_rect(rect), *premul));

    for y in 0..output_size.1 {
        let output = get_destination_row(destination, y);
        let y0 = y << shift;
        let y1 = ((y + 1) << shift).min(valid_size.1);
        for (x, value) in output[..output_size.0].iter_mut().enumerate() {
            let x0 = x << shift;
            let x1 = ((x + 1) << shift).min(source.size().0);

            let mut sum = 0.0;
            let mut weight = 0.0;
            for sy in y0..y1 {
                let src_row = &source.row(sy)[x0..x1];
                if let Some((alpha, premul)) = alpha {
                    let alpha_row = &alpha.row(sy)[x0..x1];
                    for (sample, alpha) in src_row.iter().zip(alpha_row) {
                        let a = alpha.to_f64();
                        let mult = if premul { 1.0 } else { a };
                        sum += sample.to_f64() * mult;
                        weight += a;
                    }
                } else {
                    for sample in src_row {
                        sum += sample.to_f64();
                        weight += 1.0;
                    }
                }
            }

            *value = T::from_f64(if weight > 0.0 { sum / weight } else { 0.0 });
        }
    }
}

impl super::Frame {
    pub(super) fn render_shift_info(&self, mode: RenderShiftInfoMode) -> RenderShiftInfo {
        RenderShiftInfo::new(
            &self.header,
            self.decoder_state.output_downsampling_shift,
            mode,
        )
    }

    /// Perform box average downsampling on the given source image.
    pub(super) fn downsample_for_pipeline<T: ImageDataType>(
        &self,
        channel: usize,
        group: usize,
        source: Image<T>,
    ) -> Result<Image<T>> {
        let mut destination = pipeline!(self, p, p.get_buffer(channel))?;

        let shift_info = self.render_shift_info(RenderShiftInfoMode::FinalOutput);
        let reduction = shift_info.output_shift(&self.header, channel);
        let signalled_shift = shift_info.signalled_shift(&self.header, channel);

        // Clip in the final frame grid before downsampling.
        let upsampled_size = self.header.size_upsampled();
        let (gx, gy) = (
            group % self.header.size_groups().0,
            group / self.header.size_groups().0,
        );
        let group_shift = self.header.log_group_dim() + shift_info.upsampling_shift;
        let size = (
            (upsampled_size.0 - (gx << group_shift)).min(1 << group_shift),
            (upsampled_size.1 - (gy << group_shift)).min(1 << group_shift),
        );
        let valid_size = (
            size.0.shrc(signalled_shift.0),
            size.1.shrc(signalled_shift.1),
        );
        box_average(
            &source,
            Rect {
                origin: (0, 0),
                size: valid_size,
            },
            reduction,
            None,
            &mut destination,
            |dest, y| dest.row_mut(y),
        );

        self.buffer_recycler.recycle_buffer(source);
        Ok(destination)
    }
}
