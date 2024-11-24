// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

use crate::headers::{
    self,
    modular::{SqueezeParams, TransformId},
};
use crate::util::tracing_wrappers::*;

use super::ChannelInfo;

#[derive(Debug, FromPrimitive, PartialEq)]
pub enum RctPermutation {
    Rgb = 0,
    Gbr = 1,
    Brg = 2,
    Rbg = 3,
    Grb = 4,
    Bgr = 5,
}

#[derive(Debug, FromPrimitive, PartialEq)]
pub enum RctOp {
    Noop = 0,
    AddFirstToThird = 1,
    AddFirstToSecond = 2,
    AddFirstToSecondAndThird = 3,
    AddAvgToSecond = 4,
    AddFirstToThirdAndAvgToSecond = 5,
    YCoCg = 6,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum Transform {
    Rct {
        begin_channel: usize,
        op: RctOp,
        perm: RctPermutation,
    },
    Palette {
        begin_channel: usize,
        num_channels: usize,
        num_colors: usize,
        num_deltas: usize,
    },
    Squeeze(Vec<SqueezeParams>),
}

fn default_squeeze(
    num_meta_channels: usize,
    data_channel_info: &[ChannelInfo],
) -> Vec<SqueezeParams> {
    let mut w = data_channel_info[0].size.0;
    let mut h = data_channel_info[0].size.1;
    let nc = data_channel_info.len();

    let mut params = vec![];

    if nc > 2 && data_channel_info[1].size == (w, h) {
        // 420 previews
        let sp = SqueezeParams {
            horizontal: true,
            in_place: false,
            begin_channel: num_meta_channels as u32 + 1,
            num_channels: 2,
        };
        params.push(sp);
        params.push(SqueezeParams {
            horizontal: false,
            ..sp
        });
    }

    const MAX_FIRST_PREVIEW_SIZE: usize = 8;

    let sp = SqueezeParams {
        begin_channel: num_meta_channels as u32,
        num_channels: nc as u32,
        in_place: true,
        horizontal: false,
    };

    // vertical first on tall images
    if w <= h && h > MAX_FIRST_PREVIEW_SIZE {
        params.push(SqueezeParams {
            horizontal: false,
            ..sp
        });
        h = (h + 1) / 2;
    }
    while w > MAX_FIRST_PREVIEW_SIZE || h > MAX_FIRST_PREVIEW_SIZE {
        if w > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: true,
                ..sp
            });
            w = (w + 1) / 2;
        }
        if h > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: false,
                ..sp
            });
            h = (h + 1) / 2;
        }
    }

    params
}

impl Transform {
    #[instrument(level = "trace", ret)]
    pub fn from_bitstream(
        t: &headers::modular::Transform,
        num_meta_channels: usize,
        data_channel_info: &[ChannelInfo],
    ) -> Transform {
        match t.id {
            TransformId::Rct => Transform::Rct {
                begin_channel: t.begin_channel as usize,
                op: RctOp::from_u32(t.rct_type % 7).unwrap(),
                perm: RctPermutation::from_u32(t.rct_type / 7)
                    .expect("header decoding should ensure rct_type < 42"),
            },
            TransformId::Palette => Transform::Palette {
                begin_channel: t.begin_channel as usize,
                num_channels: t.num_channels as usize,
                num_colors: t.num_colors as usize,
                num_deltas: t.num_deltas as usize,
            },
            TransformId::Squeeze => {
                if t.squeezes.is_empty() {
                    Transform::Squeeze(default_squeeze(num_meta_channels, data_channel_info))
                } else {
                    Transform::Squeeze(t.squeezes.clone())
                }
            }
            TransformId::Invalid => {
                unreachable!("header decoding for invalid transforms should fail")
            }
        }
    }

    #[instrument(level = "trace", ret)]
    pub fn is_noop(&self) -> bool {
        match self {
            Self::Rct {
                begin_channel: _,
                op,
                perm,
            } => *op == RctOp::Noop && *perm == RctPermutation::Rgb,
            Self::Squeeze(x) if x.is_empty() => true,
            _ => false,
        }
    }
}
