// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use crate::{error::Error, util::tracing_wrappers::*};
use byteorder::{ByteOrder, LittleEndian};

/// Reads bits from a sequence of bytes.
#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_buf: u64,
    bits_in_buf: usize,
    total_bits_read: usize,
}

impl Debug for BitReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BitReader{{ data: [{} bytes], bit_buf: {:0width$b}, total_bits_read: {} }}",
            self.data.len(),
            self.bit_buf,
            self.total_bits_read,
            width = self.bits_in_buf
        )
    }
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

    /// Un-reads `num` bits back into the buffer. `num` must be such that `self.bits_in_buf + num`
    /// is at most 64.
    pub fn unconsume(&mut self, num: usize, bits: u64) {
        assert!(self.bits_in_buf + num <= 64);
        assert!(bits >> num == 0);
        self.bit_buf = (self.bit_buf << num) | bits;
        self.bits_in_buf += num;
        self.total_bits_read = self.total_bits_read.wrapping_sub(num);
    }

    /// Advances by `num` bits. Similar to `skip_bits`, but bits must be in the buffer.
    pub fn consume(&mut self, num: usize) -> Result<(), Error> {
        if self.bits_in_buf < num {
            return Err(Error::OutOfBounds((num - self.bits_in_buf).div_ceil(8)));
        }
        self.bit_buf >>= num;
        self.bits_in_buf -= num;
        self.total_bits_read = self.total_bits_read.wrapping_add(num);
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
    /// Note that this might be negative if there have been calls to `unread()`.
    pub fn total_bits_read(&self) -> isize {
        self.total_bits_read as isize
    }

    /// Returns the total number of bits that can still be read or skipped.
    pub fn total_bits_available(&self) -> usize {
        self.data.len() * 8 + self.bits_in_buf
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
    pub fn skip_bits(&mut self, mut n: usize) -> Result<(), Error> {
        // Check if we can skip within the current buffer
        if let Some(next_remaining_bits) = self.bits_in_buf.checked_sub(n) {
            self.total_bits_read += n;
            self.bits_in_buf = next_remaining_bits;
            self.bit_buf >>= n;
            return Ok(());
        }

        // Adjust the number of bits to skip and reset the buffer
        n -= self.bits_in_buf;
        self.total_bits_read += self.bits_in_buf;
        self.bit_buf = 0;
        self.bits_in_buf = 0;

        // Check if the remaining bits to skip exceed the total bits in `data`
        let bits_available = self.data.len() * 8;
        if n > bits_available {
            self.total_bits_read += bits_available;
            return Err(Error::OutOfBounds(n - bits_available));
        }

        // Skip bytes directly in `data`, then handle leftover bits
        self.total_bits_read += n / 8 * 8;
        self.data = &self.data[n / 8..];
        n %= 8;

        // Refill the buffer and adjust for any remaining bits
        self.refill();
        let to_consume = self.bits_in_buf.min(n);
        self.total_bits_read -= to_consume;
        n -= to_consume;
        self.bit_buf >>= to_consume;
        if n > 0 {
            Err(Error::OutOfBounds(n))
        } else {
            Ok(())
        }
    }

    /// Return the number of bits
    pub fn bits_to_next_byte(&self) -> usize {
        let byte_boundary = self.total_bits_read.div_ceil(8) * 8;
        byte_boundary - self.total_bits_read
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
        if self.read(self.bits_to_next_byte())? != 0 {
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

    /// Splits off a separate BitReader to handle the next `n` *full* bytes.
    /// If `self` is not aligned to a byte boundary, it skips to the next byte boundary.
    /// `self` is automatically advanced by `n` bytes.
    pub fn split_at(&mut self, n: usize) -> Result<BitReader<'a>, Error> {
        self.jump_to_byte_boundary()?;
        let mut ret = Self { ..*self };
        self.skip_bits(n * 8)?;
        let bytes_in_buf = ret.bits_in_buf / 8;
        if n > bytes_in_buf {
            // Prevent the returned bitreader from over-reading.
            ret.data = &ret.data[..n - bytes_in_buf];
        } else {
            ret.bits_in_buf = n * 8;
            ret.bit_buf &= (1u64 << (n * 8)) - 1;
            ret.data = &[];
        }
        debug!(?n, ret=?ret);
        Ok(ret)
    }
}
