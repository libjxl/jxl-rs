// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Parser for the JPEG XL Frame Index box (`jxli`), as specified in
//! the JPEG XL container specification.
//!
//! The frame index box provides a seek table for animated JXL files,
//! listing keyframe byte offsets in the codestream, timestamps, and
//! frame counts.

use crate::error::{Error, Result};

/// A single entry in the frame index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameIndexEntry {
    /// Absolute byte offset of this keyframe in the codestream.
    /// (Accumulated from the delta-coded OFFi values.)
    pub codestream_offset: u64,
    /// Duration in ticks from this indexed frame to the next indexed frame
    /// (or end of stream for the last entry). A tick lasts TNUM/TDEN seconds.
    pub duration_ticks: u64,
    /// Number of displayed frames from this indexed frame to the next indexed
    /// frame (or end of stream for the last entry).
    pub frame_count: u64,
}

/// Parsed contents of a Frame Index box (`jxli`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameIndexBox {
    /// Tick numerator. A tick lasts `tnum / tden` seconds.
    pub tnum: u32,
    /// Tick denominator.
    pub tden: u32,
    /// Indexed frame entries.
    pub entries: Vec<FrameIndexEntry>,
}

impl FrameIndexBox {
    /// Returns the number of indexed frames.
    pub fn num_frames(&self) -> usize {
        self.entries.len()
    }

    /// Returns the duration of one tick in seconds.
    pub fn tick_duration_secs(&self) -> f64 {
        self.tnum as f64 / self.tden as f64
    }

    /// Finds the index entry for the keyframe at or before the given
    /// codestream byte offset.
    pub fn entry_for_offset(&self, offset: u64) -> Option<&FrameIndexEntry> {
        // Entries are sorted by codestream_offset (monotonically increasing).
        match self
            .entries
            .binary_search_by_key(&offset, |e| e.codestream_offset)
        {
            Ok(i) => Some(&self.entries[i]),
            Err(0) => None,
            Err(i) => Some(&self.entries[i - 1]),
        }
    }

    /// Parse a frame index box from its raw content bytes (after the box header).
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        let nf = cursor.read_varint()?;
        if nf > u32::MAX as u64 {
            return Err(Error::InvalidBox);
        }
        let nf = nf as usize;

        let tnum = cursor.read_u32_be()?;
        let tden = cursor.read_u32_be()?;

        if tden == 0 {
            return Err(Error::InvalidBox);
        }

        let mut entries = Vec::with_capacity(nf);
        let mut absolute_offset: u64 = 0;

        for _ in 0..nf {
            let off_delta = cursor.read_varint()?;
            let duration_ticks = cursor.read_varint()?;
            let frame_count = cursor.read_varint()?;

            absolute_offset = absolute_offset
                .checked_add(off_delta)
                .ok_or(Error::InvalidBox)?;

            entries.push(FrameIndexEntry {
                codestream_offset: absolute_offset,
                duration_ticks,
                frame_count,
            });
        }

        Ok(FrameIndexBox {
            tnum,
            tden,
            entries,
        })
    }
}

/// Simple cursor over a byte slice for reading varints and fixed-width integers.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Error::OutOfBounds(1));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u32_be(&mut self) -> Result<u32> {
        if self.pos + 4 > self.data.len() {
            return Err(Error::OutOfBounds(4));
        }
        let val = u32::from_be_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(val)
    }

    /// Read a Varint:
    /// LEB128, 7 bits per byte, high bit means "more", up to 63 bits total.
    fn read_varint(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if shift > 56 {
                return Err(Error::InvalidBox);
            }
            let b = self.read_byte()?;
            value += ((b & 0x7f) as u64) << shift;
            if b <= 0x7f {
                break;
            }
            shift += 7;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to encode a varint (LEB128).
    fn encode_varint(mut value: u64) -> Vec<u8> {
        let mut result = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value > 0 {
                byte |= 0x80;
            }
            result.push(byte);
            if value == 0 {
                break;
            }
        }
        result
    }

    /// Helper to build a jxli box content buffer.
    fn build_frame_index(tnum: u32, tden: u32, entries: &[(u64, u64, u64)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend(encode_varint(entries.len() as u64));
        buf.extend(tnum.to_be_bytes());
        buf.extend(tden.to_be_bytes());
        for &(off, ti, fi) in entries {
            buf.extend(encode_varint(off));
            buf.extend(encode_varint(ti));
            buf.extend(encode_varint(fi));
        }
        buf
    }

    #[test]
    fn test_parse_empty_index() {
        let data = build_frame_index(1, 1000, &[]);
        let index = FrameIndexBox::parse(&data).unwrap();
        assert_eq!(index.num_frames(), 0);
        assert_eq!(index.tnum, 1);
        assert_eq!(index.tden, 1000);
    }

    #[test]
    fn test_parse_single_entry() {
        // One frame at offset 0, duration 100 ticks, 1 frame
        let data = build_frame_index(1, 1000, &[(0, 100, 1)]);
        let index = FrameIndexBox::parse(&data).unwrap();
        assert_eq!(index.num_frames(), 1);
        assert_eq!(
            index.entries[0],
            FrameIndexEntry {
                codestream_offset: 0,
                duration_ticks: 100,
                frame_count: 1,
            }
        );
    }

    #[test]
    fn test_parse_multiple_entries_delta_coding() {
        // Three frames with delta-coded offsets:
        //   OFF0=100 (absolute: 100), T0=50, F0=2
        //   OFF1=200 (absolute: 300), T1=50, F1=2
        //   OFF2=150 (absolute: 450), T2=30, F2=1
        let data = build_frame_index(1, 1000, &[(100, 50, 2), (200, 50, 2), (150, 30, 1)]);
        let index = FrameIndexBox::parse(&data).unwrap();
        assert_eq!(index.num_frames(), 3);
        assert_eq!(index.entries[0].codestream_offset, 100);
        assert_eq!(index.entries[1].codestream_offset, 300);
        assert_eq!(index.entries[2].codestream_offset, 450);
        assert_eq!(index.entries[0].duration_ticks, 50);
        assert_eq!(index.entries[1].duration_ticks, 50);
        assert_eq!(index.entries[2].duration_ticks, 30);
    }

    #[test]
    fn test_parse_large_varint() {
        // Test with a value that requires multiple varint bytes
        let mut data = Vec::new();
        data.extend(encode_varint(1)); // NF = 1
        data.extend(1u32.to_be_bytes()); // TNUM
        data.extend(1000u32.to_be_bytes()); // TDEN
        data.extend(encode_varint(0x1234_5678_9ABC)); // large offset
        data.extend(encode_varint(42));
        data.extend(encode_varint(1));
        let index = FrameIndexBox::parse(&data).unwrap();
        assert_eq!(index.entries[0].codestream_offset, 0x1234_5678_9ABC);
    }

    #[test]
    fn test_entry_for_offset() {
        let data = build_frame_index(1, 1000, &[(100, 50, 2), (200, 50, 2), (150, 30, 1)]);
        let index = FrameIndexBox::parse(&data).unwrap();
        // Absolute offsets: 100, 300, 450

        // Before first entry
        assert!(index.entry_for_offset(50).is_none());
        // Exact match
        assert_eq!(index.entry_for_offset(100).unwrap().codestream_offset, 100);
        // Between entries
        assert_eq!(index.entry_for_offset(200).unwrap().codestream_offset, 100);
        assert_eq!(index.entry_for_offset(350).unwrap().codestream_offset, 300);
        // Exact match on last
        assert_eq!(index.entry_for_offset(450).unwrap().codestream_offset, 450);
        // Past last
        assert_eq!(index.entry_for_offset(999).unwrap().codestream_offset, 450);
    }

    #[test]
    fn test_zero_tden_rejected() {
        let data = build_frame_index(1, 0, &[]);
        assert!(FrameIndexBox::parse(&data).is_err());
    }

    #[test]
    fn test_truncated_data() {
        // Just NF=1, no TNUM/TDEN
        let data = encode_varint(1);
        assert!(FrameIndexBox::parse(&data).is_err());
    }

    #[test]
    fn test_tick_duration() {
        let data = build_frame_index(1, 1000, &[]);
        let index = FrameIndexBox::parse(&data).unwrap();
        assert!((index.tick_duration_secs() - 0.001).abs() < 1e-9);
    }
}
