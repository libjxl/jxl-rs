// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Error;
use byteorder::{BigEndian, ByteOrder};

pub struct JxlCodestream {
    data: Vec<u8>,
    codestream_start: usize,
    codestream_end: usize,
}

#[derive(PartialEq, Debug)]
enum State {
    Empty,
    Jxlp(usize), // box index
}

impl JxlCodestream {
    pub fn get(&self) -> &[u8] {
        &self.data[self.codestream_start..self.codestream_end]
    }
    pub fn new(data: Vec<u8>) -> Result<JxlCodestream, Error> {
        // Box-based file format.
        if data.starts_with(&[
            0x00, 0x00, 0x00, 0x0C, 'J' as u8, 'X' as u8, 'L' as u8, ' ' as u8, 0x0D, 0x0A, 0x87,
            0x0A,
        ]) {
            let mut state = State::Empty;
            let mut assembled_codestream = vec![];
            let mut pos = 0usize;
            loop {
                if let State::Jxlp(_) = state {
                    if pos >= data.len() {
                        return Err(Error::FileTruncated);
                    }
                }
                if pos + 8 >= data.len() {
                    return Err(Error::FileTruncated);
                }
                let box_start = pos;
                let mut box_size: usize = BigEndian::read_u32(&data[pos..]) as usize;
                pos += 4;
                let ty = &data[pos..pos + 4];
                pos += 4;
                if pos + 8 >= data.len() {
                    return Err(Error::FileTruncated);
                }
                if box_size == 1 {
                    let sz = BigEndian::read_u64(&data[pos..]) as usize;
                    if sz >= usize::MAX {
                        return Err(Error::InvalidBox);
                    }
                    box_size = sz;
                    pos += 8;
                }
                if box_start + box_size > data.len() {
                    return Err(Error::FileTruncated);
                }
                let eof_box = box_size == 0;
                if box_size == 0 {
                    box_size = data.len() - box_start;
                }
                let mut handle_jxlp =
                    |jxlp_idx: Option<usize>| -> Result<Option<JxlCodestream>, Error> {
                        if box_size <= 4 {
                            return Err(Error::InvalidBox);
                        }
                        let jxlp_count_and_last = BigEndian::read_u32(&data[pos..]);
                        pos += 4;
                        let jxlp_count = jxlp_count_and_last & (((1 as u32) << 31) - 1);
                        if jxlp_count as usize != jxlp_idx.map_or_else(|| 0, |x| x + 1) {
                            return Err(Error::InvalidBox);
                        }
                        let jxlp_is_last = jxlp_count_and_last >= ((1 as u32) << 31);
                        if eof_box && !jxlp_is_last {
                            return Err(Error::InvalidBox);
                        }
                        assembled_codestream.extend_from_slice(&data[pos..box_size + box_start]);
                        if !jxlp_is_last {
                            Ok(None)
                        } else {
                            let mut cs = vec![];
                            std::mem::swap(&mut cs, &mut assembled_codestream);
                            let len = cs.len();
                            Ok(Some(JxlCodestream {
                                data: cs,
                                codestream_start: 0,
                                codestream_end: len,
                            }))
                        }
                    };
                println!("{:?}", std::str::from_utf8(ty));
                match (ty, &state) {
                    (b"jxlc", State::Empty) => {
                        break Ok(JxlCodestream {
                            data,
                            codestream_start: pos,
                            codestream_end: box_size + box_start,
                        });
                    }
                    (b"jxlc", State::Jxlp(_)) => {
                        // Can't mix jxlp and jxlc.
                        return Err(Error::InvalidBox);
                    }
                    (b"jxlp", State::Empty) => {
                        if let Some(cs) = handle_jxlp(None)? {
                            break Ok(cs);
                        }
                        state = State::Jxlp(0);
                    }
                    (b"jxlp", &State::Jxlp(idx)) => {
                        if let Some(cs) = handle_jxlp(Some(idx))? {
                            break Ok(cs);
                        }
                        state = State::Jxlp(idx + 1);
                    }
                    _ => {}
                }
                pos = box_size + box_start;
            }
        } else if data.starts_with(&[0xff, 0x0A]) {
            let codestream_end = data.len();
            Ok(JxlCodestream {
                data,
                codestream_start: 0usize,
                codestream_end,
            })
        } else {
            Err(Error::InvalidSignature(data[0], data[1]))
        }
    }
}
