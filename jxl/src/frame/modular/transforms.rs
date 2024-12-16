// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(veluca): remove this.
#![allow(dead_code)]

use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

use crate::error::Error;
use crate::util::tracing_wrappers::*;
use crate::{
    error::Result,
    headers::{
        self,
        modular::{SqueezeParams, TransformId},
    },
};

use super::{ChannelInfo, ModularBufferInfo, Predictor};

#[derive(Debug, FromPrimitive, PartialEq, Clone, Copy)]
pub enum RctPermutation {
    Rgb = 0,
    Gbr = 1,
    Brg = 2,
    Rbg = 3,
    Grb = 4,
    Bgr = 5,
}

#[derive(Debug, FromPrimitive, PartialEq, Clone, Copy)]
pub enum RctOp {
    Noop = 0,
    AddFirstToThird = 1,
    AddFirstToSecond = 2,
    AddFirstToSecondAndThird = 3,
    AddAvgToSecond = 4,
    AddFirstToThirdAndAvgToSecond = 5,
    YCoCg = 6,
}

#[derive(Debug)]
pub enum TransformStep {
    Rct {
        buf_in: [usize; 3],
        buf_out: [usize; 3],
        op: RctOp,
        perm: RctPermutation,
    },
    Palette {
        buf_in: usize,
        buf_pal: usize,
        buf_out: usize,
        num_colors: usize,
        num_deltas: usize,
        chan_index: usize,
        predictor: Predictor,
    },
    HSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
    },
    VSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
    },
}

fn default_squeeze(data_channel_info: &[(usize, ChannelInfo)]) -> Vec<SqueezeParams> {
    let mut w = data_channel_info[0].1.size.0;
    let mut h = data_channel_info[0].1.size.1;
    let nc = data_channel_info.len();

    let mut params = vec![];

    let num_meta_channels = data_channel_info
        .iter()
        .take_while(|x| x.1.is_meta())
        .count();

    if nc > 2 && data_channel_info[1].1.size == (w, h) {
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

#[instrument(level = "trace", err)]
fn check_equal_channels(
    channels: &[(usize, ChannelInfo)],
    first_chan: usize,
    num: usize,
) -> Result<()> {
    if first_chan + num > channels.len() {
        return Err(Error::InvalidChannelRange(
            first_chan,
            first_chan + num,
            channels.len(),
        ));
    }
    for inc in 1..num {
        if channels[first_chan].1 != channels[first_chan + inc].1 {
            return Err(Error::MixingDifferentChannels);
        }
    }
    Ok(())
}

#[instrument(level = "trace", err)]
fn check_squeeze_params(channels: &[(usize, ChannelInfo)], params: &SqueezeParams) -> Result<()> {
    let end_channel = (params.begin_channel + params.num_channels) as usize;
    if end_channel > channels.len() {
        return Err(Error::InvalidChannelRange(
            params.begin_channel as usize,
            params.num_channels as usize,
            channels.len(),
        ));
    }
    if channels[params.begin_channel as usize].1.is_meta() != channels[end_channel - 1].1.is_meta()
    {
        return Err(Error::MixingDifferentChannels);
    }
    if channels[params.begin_channel as usize].1.is_meta() && !params.in_place {
        return Err(Error::MetaSqueezeRequiresInPlace);
    }
    Ok(())
}

#[instrument(level = "trace", ret)]
pub fn meta_apply_transforms(
    channels: &[ChannelInfo],
    transforms: &[headers::modular::Transform],
) -> Result<(Vec<ModularBufferInfo>, Vec<TransformStep>)> {
    let mut buffer_info = vec![];
    let mut transform_steps = vec![];
    // (buffer id, channel info)
    let mut channels: Vec<_> = channels.iter().cloned().enumerate().collect();

    // First, add all the pre-transform channels to the buffer list.
    for chan in channels.iter() {
        buffer_info.push(ModularBufferInfo {
            info: chan.1.clone(),
            channel_id: chan.0,
            is_output: true,
            is_coded: false,
            description: format!(
                "Input channel {}, size {}x{}",
                chan.0, chan.1.size.0, chan.1.size.1
            ),
        });
    }

    let add_transform_buffer = |buffer_info: &mut Vec<ModularBufferInfo>, info, description| {
        buffer_info.push(ModularBufferInfo {
            info,
            channel_id: 0,
            is_output: false,
            is_coded: false,
            description,
        });
        buffer_info.len() - 1
    };

    // Apply transforms to the channel list.
    for transform in transforms {
        match transform.id {
            TransformId::Rct => {
                let begin_channel = transform.begin_channel as usize;
                let op = RctOp::from_u32(transform.rct_type % 7).unwrap();
                let perm = RctPermutation::from_u32(transform.rct_type / 7)
                    .expect("header decoding should ensure rct_type < 42");
                check_equal_channels(&channels, begin_channel, 3)?;
                let mut buf_in = [0; 3];
                let buf_out = [
                    channels[begin_channel].0,
                    channels[begin_channel + 1].0,
                    channels[begin_channel + 2].0,
                ];
                for i in 0..3 {
                    let c = &mut channels[begin_channel + i];
                    c.0 = add_transform_buffer(
                        &mut buffer_info,
                        c.1.clone(),
                        format!(
                            "RCT (op {:?} perm {:?}) starting at channel {}, input {}",
                            op, perm, begin_channel, i
                        ),
                    );
                    buf_in[i] = c.0;
                }
                transform_steps.push(TransformStep::Rct {
                    buf_out,
                    buf_in,
                    op,
                    perm,
                });
                trace!("applied RCT: {channels:?}");
            }
            TransformId::Squeeze => {
                let steps = if transform.squeezes.is_empty() {
                    default_squeeze(&channels)
                } else {
                    transform.squeezes.clone()
                };
                for step in steps {
                    check_squeeze_params(&channels, &step)?;
                    let in_place = step.in_place;
                    let horizontal = step.horizontal;
                    let begin_channel = step.begin_channel as usize;
                    let num_channels = step.num_channels as usize;
                    let end_channel = begin_channel + num_channels;
                    let new_chan_offset = if in_place {
                        end_channel
                    } else {
                        channels.len()
                    };
                    for ic in 0..num_channels {
                        let chan = &channels[begin_channel + ic].1;
                        let new_shift = if let Some(shift) = chan.shift {
                            if shift.0 > 30 || shift.1 > 30 {
                                return Err(Error::TooManySqueezes);
                            }
                            if horizontal {
                                Some((shift.0 + 1, shift.1))
                            } else {
                                Some((shift.0, shift.1 + 1))
                            }
                        } else {
                            None
                        };
                        let w = chan.size.0;
                        let h = chan.size.1;
                        let (new_size_0, new_size_1) = if horizontal {
                            ((w.div_ceil(2), h), (w - w.div_ceil(2), h))
                        } else {
                            ((w, h.div_ceil(2)), (w, h - h.div_ceil(2)))
                        };
                        let new_0 = ChannelInfo {
                            shift: new_shift,
                            size: new_size_0,
                        };
                        let buf_0 = add_transform_buffer(
                            &mut buffer_info,
                            new_0.clone(),
                            format!("Squeezed channel, original channel {}", begin_channel + ic),
                        );
                        let new_1 = ChannelInfo {
                            shift: new_shift,
                            size: new_size_1,
                        };
                        let buf_1 = add_transform_buffer(
                            &mut buffer_info,
                            new_1.clone(),
                            format!("Squeeze residual, original channel {}", begin_channel + ic),
                        );
                        if horizontal {
                            transform_steps.push(TransformStep::HSqueeze {
                                buf_in: [buf_0, buf_1],
                                buf_out: channels[begin_channel + ic].0,
                            });
                        } else {
                            transform_steps.push(TransformStep::VSqueeze {
                                buf_in: [buf_0, buf_1],
                                buf_out: channels[begin_channel + ic].0,
                            });
                        }
                        channels[begin_channel + ic] = (buf_0, new_0);
                        channels.insert(new_chan_offset + ic, (buf_1, new_1));
                        trace!("applied squeeze: {channels:?}");
                    }
                }
            }
            TransformId::Palette => {
                let begin_channel = transform.begin_channel as usize;
                let num_channels = transform.num_channels as usize;
                let num_colors = transform.num_colors as usize;
                let num_deltas = transform.num_deltas as usize;
                let pred = Predictor::from_u32(transform.predictor_id)
                    .expect("header decoding should ensure a valid predictor");
                check_equal_channels(&channels, begin_channel, num_channels)?;
                let pchan_info = ChannelInfo {
                    shift: None,
                    size: (num_colors + num_deltas, num_channels),
                };
                let pchan = add_transform_buffer(
                    &mut buffer_info,
                    pchan_info.clone(),
                    format!(
                        "Palette for palette transform starting at channel {} with {} channels",
                        begin_channel, num_channels
                    ),
                );
                let inchan = add_transform_buffer(
                    &mut buffer_info,
                    channels[begin_channel].1.clone(),
                    format!(
                        "Pixel data for palette transform starting at channel {} with {} channels",
                        begin_channel, num_channels
                    ),
                );
                for i in 0..num_channels {
                    transform_steps.push(TransformStep::Palette {
                        buf_in: inchan,
                        buf_pal: pchan,
                        buf_out: channels[begin_channel + i].0,
                        num_colors,
                        num_deltas,
                        chan_index: i,
                        predictor: pred,
                    });
                }
                channels.drain(begin_channel + 1..begin_channel + num_channels);
                channels[begin_channel].0 = inchan;
                channels.insert(0, (pchan, pchan_info));
                trace!("applied palette: {channels:?}");
            }
            TransformId::Invalid => {
                unreachable!("header decoding for invalid transforms should fail");
            }
        }
    }

    // All the channels left over at the end of applying transforms are the channels that are
    // actually coded.
    for (chid, chan) in channels.iter().enumerate() {
        buffer_info[chan.0].is_coded = true;
        buffer_info[chan.0].channel_id = chid;
    }

    Ok((buffer_info, transform_steps))
}
