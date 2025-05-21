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

use super::lz77::Lz77ReaderInner;

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

#[derive(Debug)]
pub struct Histograms {
    lz77_params: Lz77Params,
    lz77_length_uint: Option<HybridUint>,
    context_map: Vec<u8>,
    // TODO(firsching): remove once we use this!
    #[allow(dead_code)]
    log_alpha_size: usize,
    uint_configs: Vec<HybridUint>,
    codes: Codes,
}

#[derive(Debug)]
pub struct Reader<'a> {
    histograms: &'a Histograms,
    lz77_config: ReaderLz77Config<'a>,
}

#[derive(Debug)]
enum ReaderLz77Config<'a> {
    Disabled(ReaderInner<'a>),
    Enabled(Lz77ReaderInner<'a>),
}

impl Reader<'_> {
    pub fn read(&mut self, br: &mut BitReader, context: usize) -> Result<u32> {
        let cluster = self.histograms.map_context_to_cluster(context);
        match &mut self.lz77_config {
            ReaderLz77Config::Disabled(inner) => inner.read_clustered(br, cluster),
            ReaderLz77Config::Enabled(inner) => inner.read_clustered(br, cluster),
        }
    }

    pub fn read_signed(&mut self, br: &mut BitReader, cluster: usize) -> Result<i32> {
        let unsigned = self.read(br, cluster)?;
        Ok(unpack_signed(unsigned))
    }

    pub fn check_final_state(self) -> Result<()> {
        match self.lz77_config {
            ReaderLz77Config::Disabled(inner) => inner.check_final_state(),
            ReaderLz77Config::Enabled(inner) => inner.check_final_state(),
        }
    }
}

#[derive(Debug)]
pub(super) struct ReaderInner<'a> {
    codes: &'a Codes,
    uint_configs: &'a [HybridUint],
    ans_reader: AnsReader,
}

impl ReaderInner<'_> {
    pub(super) fn read_token_clustered(
        &mut self,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<u32> {
        match &self.codes {
            Codes::Huffman(hc) => hc.read(br, cluster),
            Codes::Ans(ans) => self.ans_reader.read(ans, br, cluster),
        }
    }

    pub(super) fn read_uint_clustered(
        &self,
        token: u32,
        br: &mut BitReader,
        cluster: usize,
    ) -> Result<u32> {
        self.uint_configs[cluster].read(token, br)
    }

    pub(super) fn read_clustered(&mut self, br: &mut BitReader, cluster: usize) -> Result<u32> {
        let symbol = self.read_token_clustered(br, cluster)?;
        self.read_uint_clustered(symbol, br, cluster)
    }

    pub(super) fn check_final_state(self) -> Result<()> {
        match &self.codes {
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

    fn make_reader_impl(&self, br: &mut BitReader, image_width: Option<usize>) -> Result<Reader> {
        let ans_reader = if matches!(self.codes, Codes::Ans(_)) {
            AnsReader::init(br)?
        } else {
            AnsReader::new_unused()
        };

        let reader = ReaderInner {
            codes: &self.codes,
            uint_configs: &self.uint_configs,
            ans_reader,
        };

        let lz77_config = if self.lz77_params.enabled {
            let dist_multiplier = image_width.unwrap_or(0) as u32;
            let min_symbol = self.lz77_params.min_symbol.unwrap();
            let min_length = self.lz77_params.min_length.unwrap();
            let length_config = self.lz77_length_uint.as_ref().unwrap();
            let reader = Lz77ReaderInner::new(
                min_symbol,
                min_length,
                length_config,
                dist_multiplier,
                &self.context_map,
                reader,
            );
            ReaderLz77Config::Enabled(reader)
        } else {
            ReaderLz77Config::Disabled(reader)
        };

        Ok(Reader {
            histograms: self,
            lz77_config,
        })
    }

    pub fn make_reader(&self, br: &mut BitReader) -> Result<Reader> {
        self.make_reader_impl(br, None)
    }

    pub fn make_reader_with_width(&self, br: &mut BitReader, image_width: usize) -> Result<Reader> {
        self.make_reader_impl(br, Some(image_width))
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

    pub(super) fn as_reader_inner(&self) -> ReaderInner<'_> {
        ReaderInner {
            codes: &self.codes,
            uint_configs: &self.uint_configs,
            ans_reader: AnsReader::new_unused(),
        }
    }
}
