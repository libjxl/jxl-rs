// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub struct DownsampleStage {
    pub(crate) factor: u8,
    pub(crate) from_channel: usize,
    // (alpha_channel, clamp)
    pub(crate) alpha: Vec<Option<(usize, bool)>>,
}

impl DownsampleStage {
    pub fn new(factor: u8, alpha: Vec<Option<(usize, bool)>>) -> Self {
        assert!((1..=3).contains(&factor));

        let from_channel = 0;
        let num_channels = alpha.len();
        for (idx, a) in alpha.iter().enumerate() {
            let Some((a_idx, _)) = *a else {
                continue;
            };
            assert!(
                a_idx < num_channels,
                "alpha channel index of channel #{idx} is out of bounds: \
                the index is {a_idx} but num_channels is {num_channels}",
            );
        }

        Self {
            factor,
            from_channel,
            alpha,
        }
    }

    pub fn single(factor: u8, from_channel: usize) -> Self {
        Self {
            from_channel,
            ..Self::new(factor, vec![None])
        }
    }

    pub fn shift(&self) -> u8 {
        self.factor
    }

    pub fn uses_channel(&self, c: usize) -> bool {
        let from = self.from_channel;
        let num_channels = self.alpha.len();
        (from..from + num_channels).contains(&c)
    }
}

impl std::fmt::Display for DownsampleStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from = self.from_channel;
        let num_channels = self.alpha.len();
        let to = from + num_channels;
        write!(
            f,
            "downsample channel {from}..{to} by {}x",
            1 << self.factor
        )
    }
}
