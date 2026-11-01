// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::Cursor;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::bit_reader::*;
use crate::entropy_coding::decode::{Histograms, SymbolReader};
use crate::error::{Error, Result};
use crate::headers::encodings::*;
use crate::util::NewWithCapacity;
use crate::util::tracing_wrappers::warn;

mod header;
mod stream;
mod tag;

use header::read_header;
use stream::IccStream;
pub use stream::read_varint_from_reader;
use tag::{read_single_command, read_tag_list};

const ICC_CONTEXTS: usize = 41;
const ICC_HEADER_SIZE: u64 = 128;
const PREAMBLE_SIZE: usize = 20;

/// Level 5 allows ICC profiles of up to 2^22 bytes, Level 10 up to 2^28.
fn max_icc_size(level5_limits: bool) -> u64 {
    if level5_limits { 1 << 22 } else { 1 << 28 }
}

fn check_preamble(
    output_size: u64,
    commands_size: u64,
    bytes_read: u64,
    total_len: u64,
    level5_limits: bool,
) -> Result<()> {
    if bytes_read.saturating_add(commands_size) > total_len {
        return Err(Error::InvalidIccStream);
    }
    if output_size > max_icc_size(level5_limits) || output_size.saturating_add(65536) < total_len {
        return Err(Error::IccTooLarge);
    }
    Ok(())
}

fn read_icc_inner(stream: &mut IccStream, level5_limits: bool) -> Result<Vec<u8>, Error> {
    let output_size = stream.read_varint()?;
    let commands_size = stream.read_varint()?;
    check_preamble(
        output_size,
        commands_size,
        stream.bytes_read(),
        stream.len(),
        level5_limits,
    )?;

    // Extract command stream first.
    let commands = stream.read_to_vec_exact(commands_size as usize)?;
    let mut commands_stream = Cursor::new(commands);
    // `stream` contains data stream from here.
    let data_stream = stream;

    // Decode ICC profile header.
    let mut decoded_profile = read_header(data_stream, output_size)?;
    if output_size <= ICC_HEADER_SIZE {
        return Ok(decoded_profile);
    }

    // Convert to slice writer to prevent buffer from growing.
    // `read_header` above returns buffer with capacity of `output_size`, so this doesn't realloc.
    debug_assert_eq!(decoded_profile.capacity(), output_size as usize);
    decoded_profile.resize(output_size as usize, 0);
    let mut decoded_profile_writer = Cursor::new(&mut *decoded_profile);
    decoded_profile_writer.set_position(ICC_HEADER_SIZE);

    // Decode tag list.
    let v = read_varint_from_reader(&mut commands_stream)?;
    if let Some(num_tags) = v.checked_sub(1) {
        if (output_size - ICC_HEADER_SIZE) / 12 < num_tags {
            warn!(output_size, num_tags, "num_tags too large");
            return Err(Error::InvalidIccStream);
        }

        let num_tags = num_tags as u32;
        decoded_profile_writer
            .write_u32::<BigEndian>(num_tags)
            .map_err(|_| Error::InvalidIccStream)?;

        read_tag_list(
            data_stream,
            &mut commands_stream,
            &mut decoded_profile_writer,
            num_tags,
            output_size,
        )?;
    }

    // Decode tag data.
    // Will not enter the loop if end of stream was reached while decoding tag list.
    while let Ok(command) = commands_stream.read_u8() {
        read_single_command(
            data_stream,
            &mut commands_stream,
            &mut decoded_profile_writer,
            command,
        )?;
    }

    // Validate output size.
    let actual_len = decoded_profile_writer.position();
    if actual_len != output_size {
        warn!(output_size, actual_len, "ICC profile size mismatch");
        return Err(Error::InvalidIccStream);
    }

    Ok(decoded_profile)
}

/// Struct to incrementally decode an ICC profile.
pub struct IncrementalIccReader {
    histograms: Histograms,
    reader: SymbolReader,
    out_buf: Vec<u8>,
    len: usize,
    // [prev, prev_prev]
    prev_bytes: [u8; 2],
    level5_limits: bool,
}

impl IncrementalIccReader {
    pub fn new(br: &mut BitReader, level5_limits: bool) -> Result<Self> {
        let len = u64::read_unconditional(&(), br, &Empty {})?;
        if len > 1u64 << 24 {
            return Err(Error::IccTooLarge);
        }

        let len = len as usize;

        let mut lz77_scratch = Vec::new();
        let histograms = Histograms::decode(ICC_CONTEXTS, br, true, &mut lz77_scratch)?;
        let reader = SymbolReader::new(&histograms, br, None, &mut lz77_scratch)?;
        let initial_alloc = len.min(64 * 1024);
        let out_buf = Vec::new_with_capacity(initial_alloc)?;
        Ok(Self {
            histograms,
            reader,
            len,
            out_buf,
            prev_bytes: [0, 0],
            level5_limits,
        })
    }

    fn get_icc_ctx(&self) -> u32 {
        if self.out_buf.len() <= ICC_HEADER_SIZE as usize {
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

    pub fn num_coded_bytes(&self) -> usize {
        self.len
    }

    pub fn remaining(&self) -> usize {
        assert!(self.num_coded_bytes() >= self.out_buf.len());
        self.num_coded_bytes() - self.out_buf.len()
    }

    fn check_preamble(&self) -> Result<()> {
        let mut cursor = &self.out_buf[..];
        let output_size =
            read_varint_from_reader(&mut cursor).map_err(|_| Error::InvalidIccStream)?;
        let commands_size =
            read_varint_from_reader(&mut cursor).map_err(|_| Error::InvalidIccStream)?;
        let bytes_read = (self.out_buf.len() - cursor.len()) as u64;
        check_preamble(
            output_size,
            commands_size,
            bytes_read,
            self.len as u64,
            self.level5_limits,
        )
    }

    pub fn read_one(&mut self, br: &mut BitReader) -> Result<()> {
        let ctx = self.get_icc_ctx() as usize;
        let checkpoint = self.reader.checkpoint::<1>();
        let sym = self.reader.read_unsigned(&self.histograms, br, ctx);

        if let Err(err) = br.check_for_error() {
            self.reader.restore(checkpoint);
            return Err(err);
        }
        self.reader.check_for_error()?;
        if sym >= 256 {
            warn!(sym, "Invalid symbol in ICC stream");
            return Err(Error::InvalidIccStream);
        }

        let b = sym as u8;
        if let Err(err) = self.out_buf.try_reserve(1) {
            self.reader.restore(checkpoint);
            return Err(err.into());
        }
        self.out_buf.push(b);
        self.prev_bytes = [b, self.prev_bytes[0]];
        if self.len > PREAMBLE_SIZE && self.out_buf.len() == PREAMBLE_SIZE {
            self.check_preamble()?;
        }
        Ok(())
    }

    pub fn finalize(self, br: &mut BitReader) -> Result<Vec<u8>> {
        assert_eq!(self.num_coded_bytes(), self.out_buf.len());
        self.reader.check_final_state(&self.histograms, br)?;
        let mut stream = IccStream::new(self.out_buf);
        let profile = read_icc_inner(&mut stream, self.level5_limits)?;
        stream.finalize()?;
        Ok(profile)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn level5_limits_the_icc_size() {
        let size = |output_size: u64, level5_limits: bool| {
            check_preamble(output_size, 0, 2, output_size, level5_limits)
        };
        assert!(size(1 << 22, true).is_ok());
        assert!(matches!(size((1 << 22) + 1, true), Err(Error::IccTooLarge)));
        assert!(size((1 << 22) + 1, false).is_ok());
        assert!(size(1 << 28, false).is_ok());
        assert!(matches!(
            size((1 << 28) + 1, false),
            Err(Error::IccTooLarge)
        ));
    }
}
