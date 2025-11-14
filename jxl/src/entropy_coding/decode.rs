// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_macros::UnconditionalCoder;

use crate::bit_reader::BitReader;
use crate::entropy_coding::ans::*;
use crate::entropy_coding::context_map::*;
use crate::entropy_coding::huffman::*;
use crate::entropy_coding::hybrid_uint::*;
use crate::error::{Error, Result};
use crate::headers::encodings::*;
use crate::util::tracing_wrappers::*;

pub fn decode_varint16(br: &mut BitReader) -> Result<u16> {
    if br.read(1)? != 0 {
        let nbits = br.read(4)? as usize;
        if nbits == 0 {
            Ok(1)
        } else {
            Ok((1 << nbits) + br.read(nbits)? as u16)
        }
    } else {
        Ok(0)
    }
}

pub fn unpack_signed(unsigned: u32) -> i32 {
    ((unsigned >> 1) ^ ((!unsigned) & 1).wrapping_sub(1)) as i32
}

#[derive(UnconditionalCoder, Debug)]
struct Lz77Params {
    pub enabled: bool,
    #[condition(enabled)]
    #[coder(u2S(224, 512, 4096, Bits(15) + 8))]
    pub min_symbol: Option<u32>,
    #[condition(enabled)]
    #[coder(u2S(3, 4, Bits(2) + 5, Bits(8) + 9))]
    pub min_length: Option<u32>,
}

#[derive(Debug)]
enum Codes {
    Huffman(HuffmanCodes),
    Ans(AnsCodes),
}

impl Codes {
    fn single_symbol(&self, ctx: usize) -> Option<u32> {
        match self {
            Self::Huffman(hc) => hc.single_symbol(ctx),
            Self::Ans(ans) => ans.single_symbol(ctx),
        }
    }
}

#[derive(Debug)]
pub struct Histograms {
    lz77_params: Lz77Params,
    lz77_length_uint: Option<HybridUint>,
    context_map: Vec<u8>,
    // TODO(veluca): figure out why this is unused.
    #[allow(dead_code)]
    log_alpha_size: usize,
    uint_configs: Vec<HybridUint>,
    codes: Codes,
}

#[derive(Debug)]
pub struct Lz77State {
    min_symbol: u32,
    min_length: u32,
    dist_multiplier: u32,
    window: Vec<u32>,
    num_to_copy: u32,
    copy_pos: u32,
    num_decoded: u32,
}

impl Lz77State {
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
}

#[derive(Debug)]
pub struct SymbolReader {
    pub lz77_state: Option<Lz77State>,
    pub ans_reader: AnsReader,
}

impl SymbolReader {
    pub fn new(
        histograms: &Histograms,
        br: &mut BitReader,
        image_width: Option<usize>,
    ) -> Result<SymbolReader> {
        let ans_reader = if matches!(histograms.codes, Codes::Ans(_)) {
            AnsReader::init(br)?
        } else {
            AnsReader::new_unused()
        };

        let lz77_state = if histograms.lz77_params.enabled {
            Some(Lz77State {
                min_symbol: histograms.lz77_params.min_symbol.unwrap(),
                min_length: histograms.lz77_params.min_length.unwrap(),
                dist_multiplier: image_width.unwrap_or(0) as u32,
                window: Vec::new(),
                num_to_copy: 0,
                copy_pos: 0,
                num_decoded: 0,
            })
        } else {
            None
        };

        Ok(SymbolReader {
            lz77_state,
            ans_reader,
        })
    }

    pub fn try_into_rle(self, histograms: &Histograms) -> Result<RleSymbolReader, Self> {
        let &Some(Lz77State {
            min_symbol,
            min_length,
            dist_multiplier: 1..,
            num_decoded: 0,
            ..
        }) = &self.lz77_state
        else {
            return Err(self);
        };

        let lz_dist_cluster = *histograms.context_map.last().unwrap() as usize;
        let lz_conf = &histograms.uint_configs[lz_dist_cluster];
        let Some(sym) = histograms.codes.single_symbol(lz_dist_cluster) else {
            return Err(self);
        };

        if sym == 1 && lz_conf.is_split_exponent_zero() {
            Ok(RleSymbolReader {
                min_symbol,
                min_length,
                ans_reader: self.ans_reader,
                last_sym: None,
                repeat_count: 0,
            })
        } else {
            Err(self)
        }
    }

    #[inline]
    pub fn read_unsigned(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        context: usize,
    ) -> Result<u32> {
        let cluster = histograms.map_context_to_cluster(context);
        self.read_unsigned_clustered(histograms, br, cluster)
    }

    #[inline(always)]
    pub fn read_signed(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        context: usize,
    ) -> Result<i32> {
        let unsigned = self.read_unsigned(histograms, br, context)?;
        Ok(unpack_signed(unsigned))
    }

    #[inline]
    pub fn read_unsigned_clustered(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<u32> {
        if histograms.lz77_params.enabled {
            let lz77_state = self.lz77_state.as_mut().unwrap();
            if let Some(sym) = lz77_state.pull_symbol() {
                lz77_state.push_decoded_symbol(sym);
                return Ok(sym);
            }
            let token = match &histograms.codes {
                Codes::Huffman(hc) => hc.read(br, cluster)?,
                Codes::Ans(ans) => self.ans_reader.read(ans, br, cluster)?,
            };
            let Some(lz77_token) = token.checked_sub(lz77_state.min_symbol) else {
                let sym = histograms.uint_configs[cluster].read(token, br)?;
                lz77_state.push_decoded_symbol(sym);
                return Ok(sym);
            };
            if lz77_state.num_decoded == 0 {
                return Err(Error::UnexpectedLz77Repeat);
            }

            let num_to_copy = histograms
                .lz77_length_uint
                .as_ref()
                .unwrap()
                .read(lz77_token, br)?;
            let Some(num_to_copy) = num_to_copy.checked_add(lz77_state.min_length) else {
                warn!(
                    num_to_copy,
                    lz77_state.min_length, "LZ77 num_to_copy overflow"
                );
                return Err(Error::ArithmeticOverflow);
            };
            let lz_dist_cluster = *histograms.context_map.last().unwrap() as usize;

            let distance_sym = match &histograms.codes {
                Codes::Huffman(hc) => hc.read(br, lz_dist_cluster)?,
                Codes::Ans(ans) => self.ans_reader.read(ans, br, lz_dist_cluster)?,
            };
            let distance_sym = histograms.uint_configs[lz_dist_cluster].read(distance_sym, br)?;

            let distance_sub_1 = if lz77_state.dist_multiplier == 0 {
                distance_sym
            } else if let Some(distance) = distance_sym.checked_sub(120) {
                distance
            } else {
                let (offset, dist) = Lz77State::SPECIAL_DISTANCES[distance_sym as usize];
                let dist = (lz77_state.dist_multiplier * dist as u32)
                    .checked_add_signed(offset as i32 - 1);
                dist.unwrap_or(0)
            };

            let distance = (((1 << 20) - 1).min(distance_sub_1) + 1).min(lz77_state.num_decoded);
            lz77_state.copy_pos = lz77_state.num_decoded - distance;

            lz77_state.num_to_copy = num_to_copy;
            let sym = lz77_state.pull_symbol().unwrap();
            lz77_state.push_decoded_symbol(sym);
            Ok(sym)
        } else {
            let token = match &histograms.codes {
                Codes::Huffman(hc) => hc.read(br, cluster)?,
                Codes::Ans(ans) => self.ans_reader.read(ans, br, cluster)?,
            };
            histograms.uint_configs[cluster].read(token, br)
        }
    }

    #[inline(always)]
    pub fn read_signed_clustered(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<i32> {
        let unsigned = self.read_unsigned_clustered(histograms, br, cluster)?;
        Ok(unpack_signed(unsigned))
    }

    pub fn check_final_state(self, histograms: &Histograms) -> Result<()> {
        match &histograms.codes {
            Codes::Huffman(_) => Ok(()),
            Codes::Ans(_) => self.ans_reader.check_final_state(),
        }
    }
}

#[derive(Debug)]
pub struct RleSymbolReader {
    min_symbol: u32,
    min_length: u32,
    ans_reader: AnsReader,
    last_sym: Option<u32>,
    repeat_count: u32,
}

#[derive(Debug, Copy, Clone)]
enum RleToken {
    Value(u32),
    Repeat(u32),
}

impl RleSymbolReader {
    #[inline]
    fn read_token_clustered(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<RleToken> {
        let token = match &histograms.codes {
            Codes::Huffman(hc) => hc.read(br, cluster)?,
            Codes::Ans(ans) => self.ans_reader.read(ans, br, cluster)?,
        };

        Ok(if let Some(token) = token.checked_sub(self.min_symbol) {
            let lz_length_conf = histograms.lz77_length_uint.as_ref().unwrap();
            let count = lz_length_conf.read(token, br)?;
            RleToken::Repeat(count + self.min_length)
        } else {
            let sym = histograms.uint_configs[cluster].read(token, br)?;
            RleToken::Value(sym)
        })
    }

    #[inline]
    pub fn read_unsigned(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        context: usize,
    ) -> Result<u32> {
        let cluster = histograms.map_context_to_cluster(context);
        self.read_unsigned_clustered(histograms, br, cluster)
    }

    #[inline(always)]
    pub fn read_signed(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        context: usize,
    ) -> Result<i32> {
        let unsigned = self.read_unsigned(histograms, br, context)?;
        Ok(unpack_signed(unsigned))
    }

    #[inline]
    pub fn read_unsigned_clustered(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<u32> {
        if self.repeat_count == 0 {
            let token = self.read_token_clustered(histograms, br, cluster)?;
            match token {
                RleToken::Value(sym) => {
                    self.last_sym = Some(sym);
                    self.repeat_count = 1;
                }
                RleToken::Repeat(count) => {
                    self.repeat_count = count;
                }
            }
        }
        self.repeat_count -= 1;
        self.last_sym.ok_or(Error::UnexpectedLz77Repeat)
    }

    #[inline(always)]
    pub fn read_signed_clustered(
        &mut self,
        histograms: &Histograms,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<i32> {
        let unsigned = self.read_unsigned_clustered(histograms, br, cluster)?;
        Ok(unpack_signed(unsigned))
    }

    pub fn check_final_state(self, histograms: &Histograms) -> Result<()> {
        match &histograms.codes {
            Codes::Huffman(_) => Ok(()),
            Codes::Ans(_) => self.ans_reader.check_final_state(),
        }
    }
}

impl Histograms {
    pub fn decode(num_contexts: usize, br: &mut BitReader, allow_lz77: bool) -> Result<Histograms> {
        let lz77_params = Lz77Params::read_unconditional(&(), br, &Empty {})?;
        if !allow_lz77 && lz77_params.enabled {
            return Err(Error::Lz77Disallowed);
        }
        let (num_contexts, lz77_length_uint) = if lz77_params.enabled {
            (
                num_contexts + 1,
                Some(HybridUint::decode(/*log_alpha_size=*/ 8, br)?),
            )
        } else {
            (num_contexts, None)
        };

        let context_map = if num_contexts > 1 {
            decode_context_map(num_contexts, br)?
        } else {
            vec![0]
        };
        assert_eq!(context_map.len(), num_contexts);

        let use_prefix_code = br.read(1)? != 0;
        let log_alpha_size = if use_prefix_code {
            HUFFMAN_MAX_BITS
        } else {
            br.read(2)? as usize + 5
        };
        let num_histograms = *context_map.iter().max().unwrap() + 1;
        let uint_configs = ((0..num_histograms).map(|_| HybridUint::decode(log_alpha_size, br)))
            .collect::<Result<_>>()?;

        let codes = if use_prefix_code {
            Codes::Huffman(HuffmanCodes::decode(num_histograms as usize, br)?)
        } else {
            Codes::Ans(AnsCodes::decode(
                num_histograms as usize,
                log_alpha_size,
                br,
            )?)
        };

        Ok(Histograms {
            lz77_params,
            lz77_length_uint,
            context_map,
            log_alpha_size,
            uint_configs,
            codes,
        })
    }

    pub fn map_context_to_cluster(&self, context: usize) -> usize {
        self.context_map[context] as usize
    }

    pub fn num_histograms(&self) -> usize {
        *self.context_map.iter().max().unwrap() as usize + 1
    }
}

#[cfg(test)]
impl Histograms {
    /// Builds a decoder that reads an octet at a time and emits its bit-reversed value.
    pub fn reverse_octet(num_contexts: usize) -> Self {
        let d = HuffmanCodes::byte_histogram();
        let codes = Codes::Huffman(d);
        let uint_configs = vec![HybridUint::new(8, 0, 0)];
        Self {
            lz77_params: Lz77Params {
                enabled: false,
                min_symbol: None,
                min_length: None,
            },
            lz77_length_uint: None,
            uint_configs,
            log_alpha_size: 15,
            context_map: vec![0u8; num_contexts],
            codes,
        }
    }

    pub fn rle(num_contexts: usize, min_symbol: u32, min_length: u32) -> Self {
        let d = HuffmanCodes::byte_histogram_rle();
        let codes = Codes::Huffman(d);
        let uint_configs = vec![HybridUint::new(8, 0, 0), HybridUint::new(0, 0, 0)];
        let mut context_map = vec![0u8; num_contexts + 1];
        *context_map.last_mut().unwrap() = 1;
        Self {
            lz77_params: Lz77Params {
                enabled: true,
                min_symbol: Some(min_symbol),
                min_length: Some(min_length),
            },
            lz77_length_uint: Some(HybridUint::new(8, 0, 0)),
            uint_configs,
            log_alpha_size: 15,
            context_map,
            codes,
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::ControlFlow;

    use test_log::test;

    use super::*;

    #[test]
    fn rle_arb() {
        let histograms = Histograms::rle(1, 240, 3);

        arbtest::arbtest(|u| {
            let width = u.int_in_range(1usize..=256)?;

            let mut bytes = Vec::new();
            let mut is_prev_repeat = true;
            let mut expected_num_bytes = 0usize;
            u.arbitrary_loop(None, None, |u| {
                let do_repeat = !is_prev_repeat && u.ratio(1, 4)?;
                let range = if do_repeat { 240u8..=255 } else { 0u8..=239 };
                let byte = u.int_in_range(range)?;
                bytes.push(byte);

                is_prev_repeat = do_repeat;
                expected_num_bytes += if do_repeat { byte as usize - 237 } else { 1 };
                Ok(if expected_num_bytes >= 256 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                })
            })?;
            for b in &mut bytes {
                *b = b.reverse_bits();
            }

            // Read normally
            let mut normal_br = BitReader::new(&bytes);
            let mut normal_reader =
                SymbolReader::new(&histograms, &mut normal_br, Some(width)).unwrap();

            // Read RLE
            let mut rle_br = BitReader::new(&bytes);
            let rle_reader = SymbolReader::new(&histograms, &mut rle_br, Some(width)).unwrap();
            let mut rle_reader = rle_reader.try_into_rle(&histograms).unwrap();

            for _ in 0..expected_num_bytes {
                let expected = normal_reader
                    .read_unsigned_clustered(&histograms, &mut normal_br, 0)
                    .unwrap();
                let actual = rle_reader
                    .read_unsigned_clustered(&histograms, &mut rle_br, 0)
                    .unwrap();
                assert_eq!(actual, expected);
            }
            assert_eq!(rle_reader.repeat_count, 0);
            assert_eq!(normal_br.total_bits_available(), 0);
            assert_eq!(rle_br.total_bits_available(), 0);

            Ok(())
        });
    }
}
