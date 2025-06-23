// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::image::{DataTypeTag, ImageDataType};

use super::{
    RenderPipelineExtendStage, RenderPipelineInOutStage, RenderPipelineInPlaceStage,
    RenderPipelineInspectStage,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderPipelineStageType {
    Inspect,
    InPlace,
    InOut,
    Extend,
}

pub trait RenderPipelineStageInfo: super::simple_pipeline::RenderPipelineRunStage {
    const TYPE: RenderPipelineStageType;
    const BORDER: (u8, u8);
    const SHIFT: (u8, u8);
    const INPUT_TYPE: DataTypeTag;
    const OUTPUT_TYPE: Option<DataTypeTag>;
    type RowType<'a>;
}

impl<InputT: ImageDataType> RenderPipelineStageInfo for RenderPipelineInspectStage<InputT> {
    const TYPE: RenderPipelineStageType = RenderPipelineStageType::Inspect;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (u8, u8) = (0, 0);
    const INPUT_TYPE: DataTypeTag = InputT::DATA_TYPE_ID;
    const OUTPUT_TYPE: Option<DataTypeTag> = None;
    type RowType<'a> = &'a [InputT];
}

impl<T: ImageDataType> RenderPipelineStageInfo for RenderPipelineInPlaceStage<T> {
    const TYPE: RenderPipelineStageType = RenderPipelineStageType::InPlace;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (u8, u8) = (0, 0);
    const INPUT_TYPE: DataTypeTag = T::DATA_TYPE_ID;
    const OUTPUT_TYPE: Option<DataTypeTag> = None;
    type RowType<'a> = &'a mut [T];
}

impl<
    InputT: ImageDataType,
    OutputT: ImageDataType,
    const BORDER_X: u8,
    const BORDER_Y: u8,
    const SHIFT_X: u8,
    const SHIFT_Y: u8,
> RenderPipelineStageInfo
    for RenderPipelineInOutStage<InputT, OutputT, BORDER_X, BORDER_Y, SHIFT_X, SHIFT_Y>
{
    const TYPE: RenderPipelineStageType = RenderPipelineStageType::InOut;
    const BORDER: (u8, u8) = (BORDER_X, BORDER_Y);
    const SHIFT: (u8, u8) = (SHIFT_X, SHIFT_Y);
    const INPUT_TYPE: DataTypeTag = InputT::DATA_TYPE_ID;
    const OUTPUT_TYPE: Option<DataTypeTag> = Some(OutputT::DATA_TYPE_ID);
    type RowType<'a> = (&'a [&'a [InputT]], &'a mut [&'a mut [OutputT]]);
}

impl<T: ImageDataType> RenderPipelineStageInfo for RenderPipelineExtendStage<T> {
    const TYPE: RenderPipelineStageType = RenderPipelineStageType::Extend;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (u8, u8) = (0, 0);
    const INPUT_TYPE: DataTypeTag = T::DATA_TYPE_ID;
    const OUTPUT_TYPE: Option<DataTypeTag> = None;
    type RowType<'a> = &'a mut [T];
}
