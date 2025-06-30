// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::rc::Rc;

use crate::{bit_reader::BitReader, error::Result};

use super::decode::Histograms;

/// Equivalent to a `crate::entropy_coding::Reader`, but owns the histograms.
// As far as I (veluca) can tell, a self-referential struct is the easiest way to create a
// variant of Reader that owns its histograms. Potential alternatives:
//  - Pass a `Rc<Histograms>` all the way through `Reader`. This is quite cumbersome as
//    one cannot obtain a `Rc` to fields of an object in a `Rc`.
//  - Make `Reader` generic over a `AsRef<Histograms>` type. Again this is cumbersome, and
//    might impact negatively code size.
//  - Don't require owned histograms at all: this could be done if users of this type could be
//    converted into generators, but those are far from being stable.
pub struct Reader {
    // Safety note: 'static is not correct. `inner` actually borrows from `*histograms`.
    // Since `inner` is never exposed outside this struct, we can guarantee that `inner` is always
    // dropped before `histograms`.
    //
    // Drop order note: the order of these fields guarantees that `inner` is dropped before
    // `histograms`.
    inner: super::decode::Reader<'static>,
    // Note: this uses Rc because Box *invalidates its pointee when it is moved*, causing UB if
    // `self` is ever moved.
    _histograms: Rc<Histograms>,
}

impl Reader {
    #[allow(unsafe_code)]
    pub(super) fn new(
        histograms: Histograms,
        br: &mut BitReader,
        width: Option<usize>,
    ) -> Result<Self> {
        let _histograms = Rc::new(histograms);
        // Safety: `Rc::as_ptr` guarantees that the returned pointer is valid while there is
        // at least one `Rc` alive. We guarantee that `borrowed_histograms` is always dropped
        // before `histograms`:
        //  - If a panic or other error occurs when constructing `inner`, `borrowed_histograms` is
        //    dropped before `histograms`.
        //  - `Reader`'s safety invariant guarantees that `inner` is dropped before `histograms`.
        //  - Constructing `self` does not change the reference count.
        let borrowed_histograms = unsafe { &*Rc::as_ptr(&_histograms) };
        let inner = borrowed_histograms.make_reader_impl(br, width)?;
        Ok(Self { inner, _histograms })
    }

    pub fn read(&mut self, br: &mut BitReader, context: usize) -> Result<u32> {
        self.inner.read(br, context)
    }

    pub fn read_signed(&mut self, br: &mut BitReader, cluster: usize) -> Result<i32> {
        self.inner.read_signed(br, cluster)
    }

    pub fn check_final_state(self) -> Result<()> {
        self.inner.check_final_state()
    }
}
