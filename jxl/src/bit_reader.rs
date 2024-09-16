// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Error;
use byteorder::{ByteOrder, LittleEndian};

/// Reads bits from a sequence of bytes.
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_buf: u64,
    bits_in_buf: usize,
    total_bits_read: usize,
}

pub const MAX_BITS_PER_CALL: usize = 56;

impl<'a> BitReader<'a> {
    /// Constructs a BitReader for a given range of data.
    pub fn new(data: &[u8]) -> BitReader {
        BitReader {
            data,
            bit_buf: 0,
            bits_in_buf: 0,
            total_bits_read: 0,
        }
    }

    /// Reads `num` bits from the buffer without consuming them.
    pub fn peek(&mut self, num: usize) -> u64 {
        debug_assert!(num <= MAX_BITS_PER_CALL);
        self.refill();
        self.bit_buf & ((1u64 << num) - 1)
    }

    /// Advances by `num` bits. Similar to `skip_bits`, but bits must be in the buffer.
    pub fn consume(&mut self, num: usize) -> Result<(), Error> {
        if self.bits_in_buf < num {
            return Err(Error::OutOfBounds);
        }
        self.bit_buf >>= num;
        self.bits_in_buf -= num;
        self.total_bits_read += num;
        Ok(())
    }

    /// Reads `num` bits from the buffer.
    /// ```
    /// # use jxl::bit_reader::BitReader;
    /// let mut br = BitReader::new(&[0, 1]);
    /// assert_eq!(br.read(8)?, 0);
    /// assert_eq!(br.read(4)?, 1);
    /// assert_eq!(br.read(4)?, 0);
    /// assert_eq!(br.total_bits_read(), 16);
    /// assert!(br.read(1).is_err());
    /// # Ok::<(), jxl::error::Error>(())
    /// ```
    pub fn read(&mut self, num: usize) -> Result<u64, Error> {
        let ret = self.peek(num);
        self.consume(num)?;
        Ok(ret)
    }

    /// Returns the total number of bits that have been read or skipped.
    pub fn total_bits_read(&self) -> usize {
        self.total_bits_read
    }

    /// Skips `num` bits.
    /// ```
    /// # use jxl::bit_reader::BitReader;
    /// let mut br = BitReader::new(&[0, 1]);
    /// assert_eq!(br.read(8)?, 0);
    /// br.skip_bits(4)?;
    /// assert_eq!(br.total_bits_read(), 12);
    /// # Ok::<(), jxl::error::Error>(())
    /// ```
    #[inline(never)]
    pub fn skip_bits(&mut self, mut num: usize) -> Result<(), Error> {
        self.total_bits_read += num;
        if num <= self.bits_in_buf {
            self.bits_in_buf -= num;
            self.bit_buf >>= num;
            return Ok(());
        }
        num -= self.bits_in_buf;
        self.bits_in_buf = 0;
        if num > self.data.len() * 8 {
            return Err(Error::OutOfBounds);
        }
        self.data = &self.data[num / 8..];
        self.refill();
        if num > self.bits_in_buf {
            return Err(Error::OutOfBounds);
        }
        self.bits_in_buf -= num;
        self.bit_buf >>= num;
        Ok(())
    }

    /// Jumps to the next byte boundary. The skipped bytes have to be 0.
    /// ```
    /// # use jxl::bit_reader::BitReader;
    /// let mut br = BitReader::new(&[0, 1]);
    /// assert_eq!(br.read(8)?, 0);
    /// br.skip_bits(4)?;
    /// br.jump_to_byte_boundary()?;
    /// assert_eq!(br.total_bits_read(), 16);
    /// # Ok::<(), jxl::error::Error>(())
    /// ```
    #[inline(never)]
    pub fn jump_to_byte_boundary(&mut self) -> Result<(), Error> {
        let byte_boundary = (self.total_bits_read + 7) / 8 * 8;
        if self.read(byte_boundary - self.total_bits_read)? != 0 {
            return Err(Error::NonZeroPadding);
        }
        Ok(())
    }

    fn refill(&mut self) {
        // See Refill() in C++ code.
        if self.data.len() >= 8 {
            let bits = LittleEndian::read_u64(self.data);
            self.bit_buf |= bits << self.bits_in_buf;
            let read_bytes = (63 - self.bits_in_buf) >> 3;
            self.bits_in_buf |= 56;
            self.data = &self.data[read_bytes..];
            debug_assert!(56 <= self.bits_in_buf && self.bits_in_buf < 64);
        } else {
            self.refill_slow()
        }
    }

    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.bits_in_buf < 56 {
            if self.data.is_empty() {
                return;
            }
            self.bit_buf |= (self.data[0] as u64) << self.bits_in_buf;
            self.bits_in_buf += 8;
            self.data = &self.data[1..];
        }
    }
}
