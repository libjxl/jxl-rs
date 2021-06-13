// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate jxl_headers_derive;

use jxl_headers_derive::UnconditionalCoder;

use crate::bit_reader::BitReader;
use crate::entropy_coding::ans::*;
use crate::entropy_coding::context_map::*;
use crate::entropy_coding::huffman::*;
use crate::entropy_coding::hybrid_uint::*;
use crate::error::Error;
use crate::headers::encodings::*;

pub fn decode_varint16(br: &mut BitReader) -> Result<u16, Error> {
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

#[derive(UnconditionalCoder, Debug)]
struct LZ77Params {
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
}

#[derive(Debug)]
pub struct Histograms {
    lz77_params: LZ77Params,
    lz77_length_uint: Option<HybridUint>,
    context_map: Vec<u8>,
    log_alpha_size: usize,
    uint_configs: Vec<HybridUint>,
    codes: Codes,
}

#[derive(Debug)]
pub struct Reader<'a> {
    histograms: &'a Histograms,
}

impl<'a> Reader<'a> {
    fn read_internal(
        &self,
        br: &mut BitReader,
        uint_config: &HybridUint,
        cluster: usize,
    ) -> Result<u32, Error> {
        let symbol = match &self.histograms.codes {
            Codes::Huffman(hc) => hc.read(br, cluster)?,
        };
        uint_config.read(symbol, br)
    }

    pub fn read(&self, br: &mut BitReader, context: usize) -> Result<u32, Error> {
        assert!(!self.histograms.lz77_params.enabled);
        let cluster = self.histograms.context_map[context] as usize;
        self.read_internal(br, &self.histograms.uint_configs[cluster], cluster)
    }

    pub fn check_final_state(self) -> Result<(), Error> {
        match &self.histograms.codes {
            Codes::Huffman(_) => Ok(()),
        }
    }
}

impl Histograms {
    pub fn decode(
        num_contexts: usize,
        br: &mut BitReader,
        allow_lz77: bool,
    ) -> Result<Histograms, Error> {
        use std::iter::FromIterator;
        let lz77_params = LZ77Params::read_unconditional(&(), br, &Empty {})?;
        if !allow_lz77 && lz77_params.enabled {
            return Err(Error::LZ77Disallowed);
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
        // TODO(veluca93): debug print.
        println!(
            "nc: {} {:?} {:?} {:?}",
            num_contexts, lz77_params, context_map, lz77_length_uint
        );
        assert_eq!(context_map.len(), num_contexts);

        let use_prefix_code = br.read(1)? != 0;
        let log_alpha_size = if use_prefix_code {
            HUFFMAN_MAX_BITS
        } else {
            br.read(2)? as usize + 5
        };
        let num_histograms = *context_map.iter().max().unwrap() + 1;
        let uint_configs =
            Result::from_iter((0..num_histograms).map(|_| HybridUint::decode(log_alpha_size, br)))?;

        let codes = if use_prefix_code {
            Codes::Huffman(HuffmanCodes::decode(num_histograms as usize, br)?)
        } else {
            unimplemented!();
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
    fn make_reader_impl(
        &self,
        _br: &mut BitReader,
        _image_width: Option<usize>,
    ) -> Result<Reader, Error> {
        if self.lz77_params.enabled {
            unimplemented!()
        }
        Ok(Reader { histograms: self })
    }

    pub fn make_reader(&self, br: &mut BitReader) -> Result<Reader, Error> {
        self.make_reader_impl(br, None)
    }

    pub fn make_reader_with_width(
        &self,
        br: &mut BitReader,
        image_width: usize,
    ) -> Result<Reader, Error> {
        self.make_reader_impl(br, Some(image_width))
    }
}
