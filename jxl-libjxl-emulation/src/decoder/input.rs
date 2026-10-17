// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{Event, JxlSignature as RsSignature, check_signature};

pub struct InputCursor {
    next_in: *const u8,
    avail_in: usize,
    caller_pos: u64,
    caller_size: usize,
    total_fed: u64,
    parsed_pos: u64,
    input_is_set: bool,
    input_closed: bool,
    have_container: Option<bool>,
}

impl Default for InputCursor {
    fn default() -> Self {
        Self {
            next_in: std::ptr::null(),
            avail_in: 0,
            caller_pos: 0,
            caller_size: 0,
            total_fed: 0,
            parsed_pos: 0,
            input_is_set: false,
            input_closed: false,
            have_container: None,
        }
    }
}

impl InputCursor {
    /// Sets the active caller input slice.
    ///
    /// # Safety
    /// If `size > 0`, `data` must point to at least `size` readable bytes that remain valid until
    /// `Self::release_input` is called.
    pub unsafe fn set_input(&mut self, data: *const u8, size: usize) -> bool {
        if (data.is_null() && size > 0) || self.input_is_set || self.input_closed {
            return false;
        }
        if self.have_container.is_none() && self.caller_pos == 0 && size >= 2 && !data.is_null() {
            // SAFETY: `data` is non-null and has at least `size >= 2` readable bytes.
            let slice = unsafe { std::slice::from_raw_parts(data, size.min(16)) };
            match check_signature(slice) {
                RsSignature::Container => self.have_container = Some(true),
                RsSignature::Codestream => self.have_container = Some(false),
                _ => {}
            }
        }
        let already_fed = (self.total_fed.saturating_sub(self.caller_pos) as usize).min(size);
        self.caller_size = size;
        self.next_in = if size > 0 {
            // SAFETY: `already_fed <= size`, keeping `data.add(already_fed)` within `data..=data + size`.
            unsafe { data.add(already_fed) }
        } else {
            data
        };
        self.avail_in = size - already_fed;
        self.input_is_set = true;
        true
    }

    pub fn release_input(&mut self) -> usize {
        if !self.input_is_set {
            return 0;
        }
        let caller_end = self.caller_pos + self.caller_size as u64;
        let effective_pos = self.parsed_pos.clamp(self.caller_pos, caller_end);
        let rem = (caller_end - effective_pos) as usize;
        self.caller_pos = effective_pos;
        self.caller_size = 0;
        self.next_in = std::ptr::null();
        self.avail_in = 0;
        self.input_is_set = false;
        rem
    }

    pub fn close(&mut self) {
        self.input_closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.input_closed
    }

    pub fn have_container(&self) -> Option<bool> {
        self.have_container
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.avail_in == 0 || self.next_in.is_null() {
            &[]
        } else {
            // SAFETY: `self.next_in` was validated by `Self::set_input` to have at least
            // `self.avail_in` readable bytes until `Self::release_input` is called.
            unsafe { std::slice::from_raw_parts(self.next_in, self.avail_in) }
        }
    }

    pub fn record_rs_process(
        &mut self,
        consumed: usize,
        res: &Result<Event, jxl::error::Error>,
        parsed_bytes: Option<u64>,
    ) {
        if consumed > 0 {
            // SAFETY: `consumed <= self.avail_in`, keeping `self.next_in` in bounds.
            self.next_in = unsafe { self.next_in.add(consumed) };
            self.avail_in -= consumed;
            self.total_fed += consumed as u64;
        }
        match res {
            Ok(Event::NeedMoreInput { .. }) | Err(_) => {
                self.parsed_pos = self.total_fed;
            }
            Ok(
                Event::BasicInfo
                | Event::FrameHeader
                | Event::FrameComplete { .. }
                | Event::Complete,
            ) => {
                if let Some(p) = parsed_bytes {
                    self.parsed_pos = self.parsed_pos.max(p.min(self.total_fed));
                } else {
                    self.parsed_pos = self.total_fed;
                }
            }
        }
    }

    pub fn drain_into(&mut self, out: &mut Vec<u8>) {
        if self.avail_in > 0 && !self.next_in.is_null() {
            // SAFETY: `self.next_in` has `self.avail_in` readable bytes.
            let rem = unsafe { std::slice::from_raw_parts(self.next_in, self.avail_in) };
            out.extend_from_slice(rem);
            // SAFETY: Advancing to the end of `self.avail_in`.
            self.next_in = unsafe { self.next_in.add(self.avail_in) };
            self.total_fed += self.avail_in as u64;
            self.parsed_pos = self.total_fed;
            self.avail_in = 0;
        }
    }
}
