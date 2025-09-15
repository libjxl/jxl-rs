// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Display;
use std::ops::ControlFlow;

use crate::image::{DataTypeTag, ImageDataType};
use crate::util::{ShiftRightCeil, tracing_wrappers::*};

use super::save::SaveStage;
use super::{
    RenderPipelineExtendStage, RenderPipelineInOutStage, RenderPipelineInPlaceStage,
    RenderPipelineStage,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderPipelineStageType {
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

impl<T: ImageDataType> RenderPipelineStageInfo for RenderPipelineInPlaceStage<T> {
    const TYPE: RenderPipelineStageType = RenderPipelineStageType::InPlace;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (u8, u8) = (0, 0);
    const INPUT_TYPE: DataTypeTag = T::DATA_TYPE_ID;
    const OUTPUT_TYPE: Option<DataTypeTag> = None;
    type RowType<'a> = &'a mut [T];
}

pub type InOutChannel<'a, InputT, OutputT> = (&'a [&'a [InputT]], &'a mut [&'a mut [OutputT]]);

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
    type RowType<'a> = InOutChannel<'a, InputT, OutputT>;
}

impl<T: ImageDataType> RenderPipelineStageInfo for RenderPipelineExtendStage<T> {
    const TYPE: RenderPipelineStageType = RenderPipelineStageType::Extend;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (u8, u8) = (0, 0);
    const INPUT_TYPE: DataTypeTag = T::DATA_TYPE_ID;
    const OUTPUT_TYPE: Option<DataTypeTag> = None;
    type RowType<'a> = &'a mut [T];
}

pub trait BoxedStage: Display {
    fn new<S: RenderPipelineStage>(stage: S) -> Self;
    fn uses_channel(&self, channel: usize) -> bool;
    fn input_type(&self) -> DataTypeTag;
    fn output_type(&self) -> DataTypeTag;
}

pub enum Stage<BoxedStage> {
    Process(BoxedStage),
    Save(SaveStage),
}

impl<BoxedStage: Display> Display for Stage<BoxedStage> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stage::Process(s) => write!(f, "{}", s),
            Stage::Save(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChannelInfo {
    pub ty: Option<DataTypeTag>,
    pub downsample: (u8, u8),
}

pub struct RenderPipelineShared<BoxedStage> {
    pub channel_info: Vec<Vec<ChannelInfo>>,
    pub input_size: (usize, usize),
    pub log_group_size: usize,
    pub xgroups: usize,
    pub group_chan_ready_passes: Vec<Vec<usize>>,
    pub completed_passes: usize,
    pub num_passes: usize,
    pub chunk_size: usize,
    pub stages: Vec<Stage<BoxedStage>>,
}

impl<BoxedStage> RenderPipelineShared<BoxedStage> {
    pub fn group_position(&self, group_id: usize) -> (usize, usize) {
        (group_id % self.xgroups, group_id / self.xgroups)
    }

    pub fn group_offset(&self, group_id: usize) -> (usize, usize) {
        let group = self.group_position(group_id);
        (
            group.0 << self.log_group_size,
            group.1 << self.log_group_size,
        )
    }

    pub fn group_size(&self, group_id: usize) -> (usize, usize) {
        let goffset = self.group_offset(group_id);
        (
            self.input_size
                .0
                .min(goffset.0 + (1 << self.log_group_size))
                - goffset.0,
            self.input_size
                .1
                .min(goffset.1 + (1 << self.log_group_size))
                - goffset.1,
        )
    }

    pub fn group_size_for_channel(
        &self,
        channel: usize,
        group_id: usize,
        requested_data_type: DataTypeTag,
    ) -> (usize, usize) {
        let goffset = self.group_offset(group_id);
        let ChannelInfo { downsample, ty } = self.channel_info[0][channel];
        if ty.unwrap() != requested_data_type {
            panic!(
                "Invalid pipeline usage: incorrect channel type, requested {:?}, but pipeline wants {ty:?}",
                requested_data_type
            );
        }
        assert_eq!(goffset.0 % (1 << downsample.0), 0);
        assert_eq!(goffset.1 % (1 << downsample.1), 0);
        let group_size = self.group_size(group_id);
        (
            group_size.0.shrc(downsample.0),
            group_size.1.shrc(downsample.1),
        )
    }

    pub fn num_groups(&self) -> usize {
        self.xgroups * self.input_size.1.shrc(self.log_group_size)
    }
}
