// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Originally written for jxl-oxide.

use crate::bit_reader::BitReader;
use crate::error::{Error, Result};
use crate::util::tracing::*;

use super::decode::ReaderInner;
use super::hybrid_uint::HybridUint;

pub struct Lz77ReaderInner<'a> {
    min_symbol: u32,
    min_length: u32,
    length_config: &'a HybridUint,
    dist_multiplier: u32,
    context_map: &'a [u8],
    reader: ReaderInner<'a>,
    window: Vec<u32>,
    num_to_copy: u32,
    copy_pos: u32,
    num_decoded: u32,
}

impl std::fmt::Debug for Lz77ReaderInner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lz77ReaderInner")
            .field("min_symbol", &self.min_symbol)
            .field("min_length", &self.min_length)
            .field("length_config", &self.length_config)
            .field("dist_multiplier", &self.dist_multiplier)
            .field("context_map", &self.context_map)
            .field("reader", &self.reader)
            .field("window", &"(...)")
            .field("num_to_copy", &self.num_to_copy)
            .field("copy_pos", &self.copy_pos)
            .field("num_decoded", &self.num_decoded)
            .finish()
    }
}

impl<'a> Lz77ReaderInner<'a> {
    const LOG_WINDOW_SIZE: u32 = 20;
    const WINDOW_MASK: u32 = (1 << Self::LOG_WINDOW_SIZE) - 1;

    #[rustfmt::skip]
    const SPECIAL_DISTANCES: [(i8, u8); 120] = [
        ( 0, 1), ( 1, 0), ( 1, 1), (-1, 1), ( 0, 2), ( 2, 0), ( 1, 2), (-1, 2), ( 2, 1), (-2, 1),
        ( 2, 2), (-2, 2), ( 0, 3), ( 3, 0), ( 1, 3), (-1, 3), ( 3, 1), (-3, 1), ( 2, 3), (-2, 3),
        ( 3, 2), (-3, 2), ( 0, 4), ( 4, 0), ( 1, 4), (-1, 4), ( 4, 1), (-4, 1), ( 3, 3), (-3, 3),
        ( 2, 4), (-2, 4), ( 4, 2), (-4, 2), ( 0, 5), ( 3, 4), (-3, 4), ( 4, 3), (-4, 3), ( 5, 0),
        ( 1, 5), (-1, 5), ( 5, 1), (-5, 1), ( 2, 5), (-2, 5), ( 5, 2), (-5, 2), ( 4, 4), (-4, 4),
        ( 3, 5), (-3, 5), ( 5, 3), (-5, 3), ( 0, 6), ( 6, 0), ( 1, 6), (-1, 6), ( 6, 1), (-6, 1),
        ( 2, 6), (-2, 6), ( 6, 2), (-6, 2), ( 4, 5), (-4, 5), ( 5, 4), (-5, 4), ( 3, 6), (-3, 6),
        ( 6, 3), (-6, 3), ( 0, 7), ( 7, 0), ( 1, 7), (-1, 7), ( 5, 5), (-5, 5), ( 7, 1), (-7, 1),
        ( 4, 6), (-4, 6), ( 6, 4), (-6, 4), ( 2, 7), (-2, 7), ( 7, 2), (-7, 2), ( 3, 7), (-3, 7),
        ( 7, 3), (-7, 3), ( 5, 6), (-5, 6), ( 6, 5), (-6, 5), ( 8, 0), ( 4, 7), (-4, 7), ( 7, 4),
        (-7, 4), ( 8, 1), ( 8, 2), ( 6, 6), (-6, 6), ( 8, 3), ( 5, 7), (-5, 7), ( 7, 5), (-7, 5),
        ( 8, 4), ( 6, 7), (-6, 7), ( 7, 6), (-7, 6), ( 8, 5), ( 7, 7), (-7, 7), ( 8, 6), ( 8, 7),
    ];

    pub(super) fn new(
        min_symbol: u32,
        min_length: u32,
        length_config: &'a HybridUint,
        dist_multiplier: u32,
        context_map: &'a [u8],
        reader: ReaderInner<'a>,
    ) -> Self {
        Self {
            min_symbol,
            min_length,
            length_config,
            dist_multiplier,
            context_map,
            reader,
            window: Vec::new(),
            num_to_copy: 0,
            copy_pos: 0,
            num_decoded: 0,
        }
    }

    fn push_decoded_symbol(&mut self, token: u32) {
        let offset = (self.num_decoded & Self::WINDOW_MASK) as usize;
        if let Some(slot) = self.window.get_mut(offset) {
            *slot = token;
        } else {
            debug_assert_eq!(self.window.len(), offset);
            self.window.push(token);
        }
        self.num_decoded += 1;
    }

    fn pull_symbol(&mut self) -> Option<u32> {
        if let Some(next_num_to_copy) = self.num_to_copy.checked_sub(1) {
            let sym = self.window[(self.copy_pos & Self::WINDOW_MASK) as usize];
            self.copy_pos += 1;
            self.num_to_copy = next_num_to_copy;
            Some(sym)
        } else {
            None
        }
    }

    #[instrument(err, skip(self, br))]
    pub fn read_clustered(&mut self, br: &mut BitReader, cluster: usize) -> Result<u32> {
        if let Some(sym) = self.pull_symbol() {
            self.push_decoded_symbol(sym);
            return Ok(sym);
        }

        let Self {
            min_symbol,
            min_length,
            length_config,
            dist_multiplier,
            context_map,
            num_decoded,
            ..
        } = *self;
        let reader = &mut self.reader;

        let token = reader.read_token_clustered(br, cluster)?;
        let Some(lz77_token) = token.checked_sub(min_symbol) else {
            let sym = reader.read_uint_clustered(token, br, cluster)?;
            self.push_decoded_symbol(sym);
            return Ok(sym);
        };

        if num_decoded == 0 {
            return Err(Error::UnexpectedLz77Repeat);
        }

        let lz_dist_cluster = *context_map.last().unwrap() as usize;

        let num_to_copy = length_config.read(lz77_token, br)?;
        let Some(num_to_copy) = num_to_copy.checked_add(min_length) else {
            warn!(num_to_copy, min_length, "LZ77 num_to_copy overflow");
            return Err(Error::ArithmeticOverflow);
        };
        self.num_to_copy = num_to_copy;

        let distance_sym = reader.read_clustered(br, lz_dist_cluster)?;
        let distance_sub_1 = if dist_multiplier == 0 {
            distance_sym
        } else if let Some(distance) = distance_sym.checked_sub(120) {
            distance
        } else {
            let (offset, dist) = Self::SPECIAL_DISTANCES[distance_sym as usize];
            let dist = (dist_multiplier * dist as u32).checked_add_signed(offset as i32 - 1);
            dist.unwrap_or(0)
        };

        let distance = (((1 << 20) - 1).min(distance_sub_1) + 1).min(num_decoded);
        self.copy_pos = num_decoded - distance;

        let sym = self.pull_symbol().unwrap();
        self.push_decoded_symbol(sym);
        Ok(sym)
    }

    pub fn check_final_state(self) -> Result<()> {
        self.reader.check_final_state()
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::super::decode::Histograms;
    use super::*;

    #[test]
    fn copy_repeat() {
        let histogram = Histograms::reverse_octet(1);
        let reader = histogram.as_reader_inner();
        let length_config = HybridUint::new(4, 0, 0);
        let mut lz77_reader = Lz77ReaderInner::new(240, 4, &length_config, 0, &[0], reader);

        let bytes = [0u8, 1, 2, 3, 240, 1, 244, 7];
        let bytes_reversed = bytes.map(|v| v.reverse_bits());
        let mut br = BitReader::new(&bytes_reversed);

        let expected_arr = [0u32, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 2, 3, 2, 3];
        for expected in expected_arr {
            let actual = lz77_reader.read_clustered(&mut br, 0).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn special_distances() {
        let histogram = Histograms::reverse_octet(1);
        let reader = histogram.as_reader_inner();
        let length_config = HybridUint::new(4, 0, 0);
        let mut lz77_reader = Lz77ReaderInner::new(240, 4, &length_config, 4, &[0], reader);

        let bytes = [0u8, 1, 2, 3, 240, 9, 244, 4];
        let bytes_reversed = bytes.map(|v| v.reverse_bits());
        let mut br = BitReader::new(&bytes_reversed);

        let expected_arr = [0u32, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 2, 3, 2, 3];
        for expected in expected_arr {
            let actual = lz77_reader.read_clustered(&mut br, 0).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
