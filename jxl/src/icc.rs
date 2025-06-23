// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::Cursor;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::bit_reader::*;
use crate::entropy_coding::decode::Histograms;
use crate::error::{Error, Result};
use crate::headers::encodings::*;
use crate::util::tracing_wrappers::warn;

mod header;
mod stream;
mod tag;

use header::read_header;
use stream::{IccStream, read_varint_from_reader};
use tag::{read_single_command, read_tag_list};

const ICC_CONTEXTS: usize = 41;
const ICC_HEADER_SIZE: u64 = 128;

fn read_icc_inner(stream: &mut IccStream) -> Result<Vec<u8>, Error> {
    let output_size = stream.read_varint()?;
    let commands_size = stream.read_varint()?;
    if stream.bytes_read().saturating_add(commands_size) > stream.len() {
        return Err(Error::InvalidIccStream);
    }

    // Simple check to avoid allocating too large buffer.
    if output_size > (1 << 28) {
        return Err(Error::IccTooLarge);
    }

    if output_size + 65536 < stream.len() {
        return Err(Error::IccTooLarge);
    }

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

/// Read and decode ICC profile from `BitReader`.
// TODO(tirr-c): Make this resumable.
pub fn read_icc(br: &mut BitReader) -> Result<Vec<u8>, Error> {
    let len = u64::read_unconditional(&(), br, &Empty {})?;
    if len > 1u64 << 20 {
        return Err(Error::IccTooLarge);
    }

    let histograms = Histograms::decode(ICC_CONTEXTS, br, /*allow_lz77=*/ true)?;
    let mut stream = IccStream::new(br, &histograms, len)?;
    let profile = read_icc_inner(&mut stream)?;
    stream.finalize()?;
    Ok(profile)
}
