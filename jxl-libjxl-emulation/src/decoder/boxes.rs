// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::VecDeque;
use std::ffi::c_char;

use jxl::api::{JxlAuxBox, JxlAuxBoxType, JxlDecoder as RsDecoder};

use crate::types::{
    JXL_DEC_BOX, JXL_DEC_BOX_COMPLETE, JXL_DEC_BOX_NEED_MORE_OUTPUT, JxlBoxType, JxlDecoderStatus,
};

pub const AUX_BOX_TYPES: [JxlAuxBoxType; 3] = [
    JxlAuxBoxType::EXIF,
    JxlAuxBoxType::XMP,
    JxlAuxBoxType::JUMBF,
];

struct PendingBox {
    raw_type: [u8; 4],
    inner_type: [u8; 4],
    raw_size: u64,
    contents_size: u64,
    data: Vec<u8>,
}

pub struct BoxStreamer {
    seen_aux_counts: [usize; 3],
    pending_boxes: VecDeque<PendingBox>,
    current_box: Option<PendingBox>,
    current_box_offset: usize,
    box_out_ptr: *mut u8,
    box_out_cap: usize,
    box_out_written: usize,
    pub trailing_parsed: bool,
    pub trailing_box_extra: Vec<u8>,
}

impl Default for BoxStreamer {
    fn default() -> Self {
        Self {
            seen_aux_counts: [0; 3],
            pending_boxes: VecDeque::new(),
            current_box: None,
            current_box_offset: 0,
            box_out_ptr: std::ptr::null_mut(),
            box_out_cap: 0,
            box_out_written: 0,
            trailing_parsed: false,
            trailing_box_extra: Vec::new(),
        }
    }
}

impl BoxStreamer {
    pub fn collect_from(&mut self, dec: &RsDecoder, decompress_boxes: bool) -> bool {
        for (i, &ty) in AUX_BOX_TYPES.iter().enumerate() {
            let boxes = dec.aux_boxes(ty);
            for b in &boxes[self.seen_aux_counts[i]..] {
                let Some(pb) = make_pending_box(b, &[], decompress_boxes, false) else {
                    return false;
                };
                self.pending_boxes.push_back(pb);
            }
            self.seen_aux_counts[i] = boxes.len();
        }
        true
    }

    pub fn enqueue_trailing_box(&mut self, tb: &JxlAuxBox, decompress_boxes: bool) -> bool {
        let Some(pb) = make_pending_box(tb, &self.trailing_box_extra, decompress_boxes, true)
        else {
            return false;
        };
        self.pending_boxes.push_back(pb);
        self.trailing_box_extra.clear();
        true
    }

    pub fn step(&mut self, events_wanted: i32) -> Option<JxlDecoderStatus> {
        if let Some(cb) = &self.current_box {
            let had_buffer = !self.box_out_ptr.is_null();
            if had_buffer {
                let avail_out = self.box_out_cap - self.box_out_written;
                let rem_in_box = cb.data.len() - self.current_box_offset;
                let to_copy = avail_out.min(rem_in_box);
                if to_copy > 0 {
                    // SAFETY: `self.box_out_ptr` has `self.box_out_cap` writable bytes from
                    // `Self::set_buffer`, and `self.box_out_written + to_copy <= self.box_out_cap`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            cb.data.as_ptr().add(self.current_box_offset),
                            self.box_out_ptr.add(self.box_out_written),
                            to_copy,
                        );
                    }
                    self.box_out_written += to_copy;
                    self.current_box_offset += to_copy;
                }
                if self.current_box_offset < cb.data.len() {
                    return Some(JXL_DEC_BOX_NEED_MORE_OUTPUT);
                }
            }
            self.current_box = None;
            self.current_box_offset = 0;
            if had_buffer && (events_wanted & (JXL_DEC_BOX_COMPLETE.0 as i32)) != 0 {
                return Some(JXL_DEC_BOX_COMPLETE);
            }
        }
        if let Some(next_box) = self.pending_boxes.pop_front() {
            self.current_box = Some(next_box);
            self.current_box_offset = 0;
            return Some(JXL_DEC_BOX);
        }
        None
    }

    /// Sets the output buffer for the current box.
    ///
    /// # Safety
    /// `data` must point to at least `size` writable bytes that remain valid until
    /// `Self::release_buffer` is called.
    pub unsafe fn set_buffer(&mut self, data: *mut u8, size: usize) -> bool {
        if data.is_null() || !self.box_out_ptr.is_null() || self.current_box.is_none() {
            return false;
        }
        self.box_out_ptr = data;
        self.box_out_cap = size;
        self.box_out_written = 0;
        true
    }

    pub fn release_buffer(&mut self) -> usize {
        if self.box_out_ptr.is_null() {
            return 0;
        }
        let remaining = self.box_out_cap - self.box_out_written;
        self.box_out_ptr = std::ptr::null_mut();
        self.box_out_cap = 0;
        self.box_out_written = 0;
        remaining
    }

    pub fn box_type(&self, decompress_boxes: bool, decompressed: bool) -> Option<JxlBoxType> {
        let cb = self.current_box.as_ref()?;
        let ty = if decompressed && decompress_boxes {
            cb.inner_type
        } else {
            cb.raw_type
        };
        Some([
            ty[0] as c_char,
            ty[1] as c_char,
            ty[2] as c_char,
            ty[3] as c_char,
        ])
    }

    pub fn raw_size(&self) -> Option<u64> {
        self.current_box.as_ref().map(|cb| cb.raw_size)
    }

    pub fn contents_size(&self) -> Option<u64> {
        self.current_box.as_ref().map(|cb| cb.contents_size)
    }
}

fn make_pending_box(
    aux_box: &JxlAuxBox,
    trailing_extra: &[u8],
    decompress: bool,
    is_infinite: bool,
) -> Option<PendingBox> {
    let inner_type = aux_box.box_type().0;
    let raw_type = if aux_box.is_compressed() {
        *b"brob"
    } else {
        inner_type
    };
    let contents_size = (aux_box.raw_data().len() + trailing_extra.len()) as u64;
    let raw_size = if is_infinite {
        0
    } else {
        let header_size = if aux_box.is_compressed() { 12 } else { 8 };
        header_size + contents_size
    };

    let data = if decompress && aux_box.is_compressed() {
        #[cfg(feature = "brotli")]
        {
            aux_box.data(trailing_extra).ok()?.into_owned()
        }
        #[cfg(not(feature = "brotli"))]
        {
            return None;
        }
    } else if trailing_extra.is_empty() {
        aux_box.raw_data().to_vec()
    } else {
        let mut v = Vec::with_capacity(aux_box.raw_data().len() + trailing_extra.len());
        v.extend_from_slice(aux_box.raw_data());
        v.extend_from_slice(trailing_extra);
        v
    };

    Some(PendingBox {
        raw_type,
        inner_type,
        raw_size,
        contents_size,
        data,
    })
}
