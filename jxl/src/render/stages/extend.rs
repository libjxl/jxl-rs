// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    frame::ReferenceFrame,
    headers::{FileHeader, extra_channels::ExtraChannelInfo, frame_header::*},
    render::{RenderPipelineExtendStage, RenderPipelineStage},
};

#[allow(dead_code)]
pub struct ExtendToImageDimensionsStage {
    pub frame_origin: (isize, isize),
    pub image_size: (isize, isize),
    pub blending_info: BlendingInfo,
    pub ec_blending_info: Vec<BlendingInfo>,
    pub extra_channels: Vec<ExtraChannelInfo>,
    pub reference_frames: Vec<Option<ReferenceFrame>>,
    pub zeros: Vec<f32>,
}

impl ExtendToImageDimensionsStage {
    pub fn new(
        frame_header: &FrameHeader,
        file_header: &FileHeader,
        reference_frames: &[Option<ReferenceFrame>],
    ) -> Result<ExtendToImageDimensionsStage> {
        Ok(ExtendToImageDimensionsStage {
            frame_origin: (frame_header.x0 as isize, frame_header.y0 as isize),
            image_size: (
                file_header.size.xsize() as isize,
                file_header.size.ysize() as isize,
            ),
            blending_info: frame_header.blending_info.clone(),
            ec_blending_info: frame_header.ec_blending_info.clone(),
            extra_channels: file_header.image_metadata.extra_channel_info.clone(),
            reference_frames: reference_frames.to_owned(),
            zeros: vec![0f32; file_header.size.xsize() as usize],
        })
    }
}

impl std::fmt::Display for ExtendToImageDimensionsStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "extend-to-image-dims")
    }
}

impl RenderPipelineStage for ExtendToImageDimensionsStage {
    type Type = RenderPipelineExtendStage<f32>;

    fn uses_channel(&self, _c: usize) -> bool {
        true
    }

    fn new_size(&self, _current_size: (usize, usize)) -> (usize, usize) {
        (self.image_size.0 as usize, self.image_size.1 as usize)
    }

    fn original_data_origin(&self) -> (isize, isize) {
        self.frame_origin
    }

    fn process_row_chunk(
        &self,
        position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let num_ec = self.extra_channels.len();
        let num_c = 3 + num_ec;
        let x0 = position.0;
        let x1 = x0 + xsize;
        let y0 = position.1;

        let mut bg = vec![self.zeros.as_slice(); num_c];
        for (c, bg_ptr) in bg.iter_mut().enumerate().take(3) {
            if self.reference_frames[self.blending_info.source as usize].is_some() {
                *bg_ptr = self.reference_frames[self.blending_info.source as usize]
                    .as_ref()
                    .unwrap()
                    .frame[c]
                    .as_rect()
                    .row(y0);
            }
        }
        for i in 0..num_ec {
            if self.reference_frames[self.ec_blending_info[i].source as usize].is_some() {
                bg[3 + i] = self.reference_frames[self.ec_blending_info[i].source as usize]
                    .as_ref()
                    .unwrap()
                    .frame[3 + i]
                    .as_rect()
                    .row(y0);
            }
        }

        for c in 0..num_c {
            row[c][0..xsize].copy_from_slice(&bg[c][x0..x1]);
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::util::test::read_headers_and_toc;

    #[test]
    fn extend_consistency() -> Result<()> {
        let (file_header, frame_header, _) =
            read_headers_and_toc(include_bytes!("../../../resources/test/basic.jxl")).unwrap();
        let reference_frames: Vec<Option<ReferenceFrame>> = vec![None, None, None, None];
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ExtendToImageDimensionsStage::new(&frame_header, &file_header, &reference_frames)?,
            (500, 500),
            4,
        )
    }
}
