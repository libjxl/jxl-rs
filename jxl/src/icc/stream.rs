// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::{Read, Write};

use byteorder::{ReadBytesExt, WriteBytesExt};

use crate::bit_reader::*;
use crate::entropy_coding::decode::{Histograms, Reader};
use crate::error::{Error, Result};
use crate::util::tracing_wrappers::{instrument, warn};
use crate::util::try_with_capacity;

fn read_varint(mut read_one: impl FnMut() -> Result<u8>) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    while shift < 63 {
        let b = read_one()?;
        value |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(value)
}

pub(super) fn read_varint_from_reader(stream: &mut impl Read) -> Result<u64> {
    read_varint(|| stream.read_u8().map_err(|_| Error::IccEndOfStream))
}

pub(super) struct IccStream<'br, 'buf, 'hist> {
    br: &'br mut BitReader<'buf>,
    reader: Reader<'hist>,
    len: u64,
    bytes_read: u64,
    // [prev, prev_prev]
    prev_bytes: [u8; 2],
}

impl<'br, 'buf, 'hist> IccStream<'br, 'buf, 'hist> {
    pub(super) fn new(
        br: &'br mut BitReader<'buf>,
        histograms: &'hist Histograms,
        len: u64,
    ) -> Result<Self> {
        let reader = histograms.make_reader(br)?;
        Ok(Self {
            br,
            reader,
            len,
            bytes_read: 0,
            prev_bytes: [0, 0],
        })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub fn remaining_bytes(&self) -> u64 {
        self.len - self.bytes_read
    }

    fn get_icc_ctx(&self) -> u32 {
        if self.bytes_read <= super::ICC_HEADER_SIZE {
            return 0;
        }

        let [b1, b2] = self.prev_bytes;

        let p1 = match b1 {
            b'a'..=b'z' | b'A'..=b'Z' => 0,
            b'0'..=b'9' | b'.' | b',' => 1,
            0..=1 => 2 + b1 as u32,
            2..=15 => 4,
            241..=254 => 5,
            255 => 6,
            _ => 7,
        };
        let p2 = match b2 {
            b'a'..=b'z' | b'A'..=b'Z' => 0,
            b'0'..=b'9' | b'.' | b',' => 1,
            0..=15 => 2,
            241..=255 => 3,
            _ => 4,
        };

        1 + p1 + 8 * p2
    }

    fn read_one(&mut self) -> Result<u8> {
        if self.remaining_bytes() == 0 {
            return Err(Error::IccEndOfStream);
        }

        let ctx = self.get_icc_ctx() as usize;
        let sym = self.reader.read(&mut *self.br, ctx)?;
        if sym >= 256 {
            warn!(sym, "Invalid symbol in ICC stream");
            return Err(Error::InvalidIccStream);
        }
        let b = sym as u8;

        self.bytes_read += 1;
        self.prev_bytes = [b, self.prev_bytes[0]];
        Ok(b)
    }

    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        if buf.len() > self.remaining_bytes() as usize {
            return Err(Error::IccEndOfStream);
        }

        for b in buf {
            *b = self.read_one()?;
        }

        Ok(())
    }

    pub fn read_to_vec_exact(&mut self, len: usize) -> Result<Vec<u8>> {
        if len > self.remaining_bytes() as usize {
            return Err(Error::IccEndOfStream);
        }

        let mut out = try_with_capacity(len)?;

        for _ in 0..len {
            out.push(self.read_one()?);
        }

        Ok(out)
    }

    pub fn read_varint(&mut self) -> Result<u64> {
        read_varint(|| self.read_one())
    }

    pub fn copy_bytes(&mut self, writer: &mut impl Write, len: usize) -> Result<()> {
        if len > self.remaining_bytes() as usize {
            return Err(Error::IccEndOfStream);
        }

        for _ in 0..len {
            let b = self.read_one()?;
            writer.write_u8(b).map_err(|_| Error::InvalidIccStream)?;
        }

        Ok(())
    }

    #[instrument(skip_all, err)]
    pub fn finalize(self) -> Result<()> {
        // Test entropy decoder checksum
        self.reader.check_final_state()?;

        // Check if all bytes are read
        if self.bytes_read == self.len {
            Ok(())
        } else {
            warn!("ICC stream is not fully consumed");
            Err(Error::InvalidIccStream)
        }
    }
}
