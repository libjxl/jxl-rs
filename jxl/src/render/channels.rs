// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::marker::PhantomData;

use jxl_simd::{F32SimdVec, I32SimdVec, SimdDescriptor};

use crate::render::low_memory_pipeline::row_buffers::RowBuffer;
use crate::util::{SmallVec, StackOnly, mirror};

/// Trait providing vectorized SIMD load operation for an image data type.
pub trait VecLoad<D: SimdDescriptor>: 'static + Copy {
    type Vec: Copy;
    const LEN: usize;

    /// Load a vector from a slice.
    fn load(d: D, slice: &[Self]) -> Self::Vec;
}

/// Trait providing vectorized SIMD store operation for an image data type.
pub trait VecStore<D: SimdDescriptor>: VecLoad<D> {
    /// Store a vector into a slice.
    fn store(vec: Self::Vec, slice: &mut [Self]);
}

impl<D: SimdDescriptor> VecLoad<D> for f32 {
    type Vec = D::F32Vec;
    const LEN: usize = D::F32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::F32Vec::load(d, slice)
    }
}

impl<D: SimdDescriptor> VecStore<D> for f32 {
    #[inline(always)]
    fn store(vec: Self::Vec, slice: &mut [Self]) {
        vec.store(slice);
    }
}

impl<D: SimdDescriptor> VecLoad<D> for i32 {
    type Vec = D::I32Vec;
    const LEN: usize = D::I32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::I32Vec::load(d, slice)
    }
}

impl<D: SimdDescriptor> VecStore<D> for i32 {
    #[inline(always)]
    fn store(vec: Self::Vec, slice: &mut [Self]) {
        vec.store(slice);
    }
}

impl<D: SimdDescriptor> VecLoad<D> for u16 {
    type Vec = D::I32Vec;
    const LEN: usize = D::I32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::I32Vec::load_from_u16(d, slice)
    }
}

impl<D: SimdDescriptor> VecStore<D> for u16 {
    #[inline(always)]
    fn store(vec: Self::Vec, slice: &mut [Self]) {
        vec.store_u16(slice);
    }
}

impl<D: SimdDescriptor> VecLoad<D> for i16 {
    type Vec = D::I32Vec;
    const LEN: usize = D::I32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::I32Vec::load_from_i16(d, slice)
    }
}

/// Trait providing interleaved vector stores (e.g. for horizontal upsampling).
pub trait StoreInterleaved<D: SimdDescriptor, const SCALE: usize>: VecStore<D> {
    /// Store `SCALE` vectors interleaved into `slice`.
    fn store_interleaved(d: D, vals: [Self::Vec; SCALE], slice: &mut [Self]);
}

impl<D: SimdDescriptor> StoreInterleaved<D, 2> for f32 {
    #[inline(always)]
    fn store_interleaved(_d: D, vals: [Self::Vec; 2], slice: &mut [Self]) {
        D::F32Vec::store_interleaved_2(vals[0], vals[1], slice);
    }
}


impl<D: SimdDescriptor> StoreInterleaved<D, 4> for f32 {
    #[inline(always)]
    fn store_interleaved(_d: D, vals: [Self::Vec; 4], slice: &mut [Self]) {
        D::F32Vec::store_interleaved_4(vals[0], vals[1], vals[2], vals[3], slice);
    }
}

impl<D: SimdDescriptor> StoreInterleaved<D, 8> for f32 {
    #[inline(always)]
    fn store_interleaved(_d: D, vals: [Self::Vec; 8], slice: &mut [Self]) {
        D::F32Vec::store_interleaved_8(
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], slice,
        );
    }
}

/// An immutable multi-channel view for vectorized stencil processing.
///
/// # Struct Layout & Addressing
/// Stores `CHANS` base pointers and `ROWS` row offsets.
/// Both `channel_ptrs` and `row_offsets` are fixed-size arrays matching `CHANS` and `ROWS`.
///
/// An element at channel `CHAN`, row `ROW`, column offset `OFFSET` relative to the current position is located at:
/// `*(channel_ptrs[CHAN] + row_offsets[ROW] + OFFSET)`
///
/// On x86-64, this directly leverages the native SIB addressing mode:
/// `disp8(%base_reg, %offset_reg, scale)`
#[repr(C)]
pub struct ChannelsView<'a, D, T, const CHANS: usize, const ROWS: usize, const RADIUS: usize = 0> {
    d: D,
    channel_ptrs: [*const T; CHANS],
    row_offsets: [usize; ROWS],
    step: usize,
    num_vec: usize,
    _marker: PhantomData<&'a T>,
}

impl<
    'a,
    D: SimdDescriptor,
    T: VecLoad<D>,
    const CHANS: usize,
    const ROWS: usize,
    const RADIUS: usize,
> ChannelsView<'a, D, T, CHANS, ROWS, RADIUS>
{
    /// Load a SIMD vector from channel `CHAN`, row `ROW`, at column offset `OFFSET`.
    #[inline(always)]
    pub fn load<const CHAN: usize, const ROW: usize, const OFFSET: isize>(&self) -> T::Vec {
        const { assert!(CHAN < CHANS, "CHAN index out of bounds") };
        const { assert!(ROW < ROWS, "ROW index out of bounds") };
        const {
            assert!(
                OFFSET.unsigned_abs() <= RADIUS,
                "OFFSET out of bounds for declared RADIUS"
            )
        };
        // SAFETY: Const assertions guarantee `CHAN < CHANS`, `ROW < ROWS`, and `|OFFSET| <= RADIUS`.
        // By `ChannelsView` invariants (guaranteed by `create_view`), `ptr` and the range
        // `[ptr, ptr + T::LEN)` reside entirely within the valid, initialized row buffer allocation.
        let slice = unsafe {
            let base = *self.channel_ptrs.get_unchecked(CHAN);
            let row_off = *self.row_offsets.get_unchecked(ROW);
            let ptr = base.add(row_off).offset(OFFSET);
            std::slice::from_raw_parts(ptr, T::LEN)
        };
        T::load(self.d, slice)
    }

    /// Advance all channel base pointers by their precomputed step.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `advance_all` is called at most `self.num_vec` times,
    /// so pointers stay within their allocated objects.
    #[inline(always)]
    unsafe fn advance_all(&mut self) {
        for c in 0..CHANS {
            // SAFETY: Invariants established in `create_view` ensure the buffer has capacity
            // for `num_vec * step` elements, so advancing by `step` remains within the allocated object.
            unsafe {
                self.channel_ptrs[c] = self.channel_ptrs[c].add(self.step);
            }
        }
    }
}

/// A mutable multi-channel view for vectorized stencil processing.
#[repr(C)]
pub struct ChannelsMutView<'a, D, T, const CHANS: usize, const ROWS: usize, const SCALE: usize = 1>
{
    d: D,
    // Safety invariant: each pointer in `channel_ptrs` is non-null, aligned for `T`,
    // and points to an allocation that remains valid and exclusively accessible for lifetime `'a`.
    channel_ptrs: [*mut T; CHANS],
    // Safety invariant: each offset in `row_offsets` selects a valid row in the channel allocation.
    row_offsets: [usize; ROWS],
    step: usize,
    num_vec: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<
    'a,
    D: SimdDescriptor,
    T: VecStore<D>,
    const CHANS: usize,
    const ROWS: usize,
    const SCALE: usize,
> ChannelsMutView<'a, D, T, CHANS, ROWS, SCALE>
{
    /// Store a SIMD vector into channel `CHAN`, row `ROW` at the current chunk position.
    #[inline(always)]
    pub fn store<const CHAN: usize, const ROW: usize>(&mut self, val: T::Vec) {
        const { assert!(CHAN < CHANS, "CHAN index out of bounds") };
        const { assert!(ROW < ROWS, "ROW index out of bounds") };
        // SAFETY: Const assertions guarantee `CHAN < CHANS` and `ROW < ROWS`.
        // By `ChannelsMutView` invariants, `ptr` and the range `[ptr, ptr + T::LEN)` reside
        // within the allocated row buffer with exclusive mutable access for `'a`.
        let slice = unsafe {
            let base = *self.channel_ptrs.get_unchecked(CHAN);
            let row_off = *self.row_offsets.get_unchecked(ROW);
            let ptr = base.add(row_off);
            std::slice::from_raw_parts_mut(ptr, T::LEN)
        };
        T::store(val, slice);
    }

    /// Store interleaved SIMD vectors (e.g. for horizontal upsampling).
    #[inline(always)]
    pub fn store_interleaved<const CHAN: usize, const ROW: usize>(&mut self, vals: [T::Vec; SCALE])
    where
        T: StoreInterleaved<D, SCALE>,
    {
        const { assert!(CHAN < CHANS, "CHAN index out of bounds") };
        const { assert!(ROW < ROWS, "ROW index out of bounds") };
        // SAFETY: Const assertions guarantee `CHAN < CHANS` and `ROW < ROWS`.
        // By `ChannelsMutView` invariants, `ptr` and the range `[ptr, ptr + T::LEN * SCALE)`
        // reside within the allocated row buffer with exclusive mutable access for `'a`.
        let slice = unsafe {
            let base = *self.channel_ptrs.get_unchecked(CHAN);
            let row_off = *self.row_offsets.get_unchecked(ROW);
            let ptr = base.add(row_off);
            std::slice::from_raw_parts_mut(ptr, T::LEN * SCALE)
        };
        T::store_interleaved(self.d, vals, slice);
    }

    /// Advance all channel base pointers by their precomputed step.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// 1. `advance_all` is called at most `self.num_vec` times as part of chunk iteration.
    /// 2. For each channel pointer in `self.channel_ptrs`, the base pointer was derived from an allocated
    ///    object of length at least `self.num_vec * self.step` elements, so advancing by `self.step` up to
    ///    `self.num_vec` times results in a pointer that remains in-bounds or at most one element past
    ///    the end of the allocated object.
    /// 3. The byte offset `self.step * size_of::<T>()` does not exceed `isize::MAX as usize`.
    ///
    /// # Postconditions
    ///
    /// Each channel base pointer in `self.channel_ptrs` is advanced by `self.step` elements, preserving
    /// non-nullness, alignment for `T`, and exclusive write permissions.
    #[inline(always)]
    unsafe fn advance_all(&mut self) {
        for c in 0..CHANS {
            // SAFETY: Invariants established in `create_view` ensure the buffer has capacity
            // for `num_vec * step` elements, so advancing by `step` remains within the allocated object.
            unsafe {
                self.channel_ptrs[c] = self.channel_ptrs[c].add(self.step);
            }
        }
    }
}

/// Multi-row channel accessor for immutable access.
pub struct Channels<'a, T> {
    // Safety invariant: Each pointer in `channel_ptrs` is non-null, aligned for `T`,
    // and points into a live allocation valid and immutable for lifetime `'a`.
    channel_ptrs: SmallVec<*const T, 8, StackOnly>,
    // Safety invariant: For every channel index `c < channel_ptrs.len()` and row index `r < row_offsets.len()`,
    // the range `[channel_ptrs[c] + row_offsets[r] - x_offset, channel_ptrs[c] + row_offsets[r] + row_len)`
    // resides entirely within a single live allocation of initialized valid `T` values.
    row_offsets: SmallVec<usize, 16, StackOnly>,
    // Safety invariant: Distance in elements of valid left border padding available before each row pointer.
    x_offset: usize,
    // Safety invariant: Length in elements of initialized row data available from each row pointer.
    row_len: usize,
    _marker: PhantomData<&'a T>,
}

#[inline(always)]
fn copy_smallvec<T: Copy, const N: usize>(src: &SmallVec<T, N, StackOnly>) -> SmallVec<T, N, StackOnly> {
    let mut out = SmallVec::new();
    for &x in &**src {
        out.push(x);
    }
    out
}

impl<'a, T> Clone for Channels<'a, T> {
    fn clone(&self) -> Self {
        Self {
            channel_ptrs: copy_smallvec(&self.channel_ptrs),
            row_offsets: copy_smallvec(&self.row_offsets),
            x_offset: self.x_offset,
            row_len: self.row_len,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Channels<'a, T> {
    /// Create a new Channels accessor with explicit pointers and offsets.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `channel_ptrs` and `row_offsets` describe valid, initialized,
    /// properly aligned allocations of `T` for lifetime `'a`.
    #[inline(always)]
    pub(crate) unsafe fn new_with_layout(
        channel_ptrs: SmallVec<*const T, 8, StackOnly>,
        row_offsets: SmallVec<usize, 16, StackOnly>,
        x_offset: usize,
        row_len: usize,
    ) -> Self {
        Self {
            channel_ptrs,
            row_offsets,
            x_offset,
            row_len,
            _marker: PhantomData,
        }
    }

    /// Returns an empty `Channels` instance with zero channels and zero rows.
    pub fn empty() -> Self {
        // SAFETY: Vacuously sound with 0 channels and 0 rows.
        unsafe { Self::new_with_layout(SmallVec::new(), SmallVec::new(), 0, 0) }
    }

    /// Get a reference to the slice for channel `channel`, row `row`.
    #[inline(always)]
    pub fn get_row(&self, channel: usize, row: usize) -> &'a [T] {
        assert!(channel < self.channel_ptrs.len(), "channel out of bounds");
        assert!(row < self.row_offsets.len(), "row out of bounds");
        // SAFETY: Bounds are asserted above. By `Channels` invariants, the range
        // `[base + offset, base + offset + self.row_len)` is valid, initialized, and immutable for `'a`.
        unsafe {
            let base = *self.channel_ptrs.get_unchecked(channel);
            let offset = *self.row_offsets.get_unchecked(row);
            std::slice::from_raw_parts(base.add(offset), self.row_len)
        }
    }

    /// Get a reference to a slice for channel `channel`, row `row`, with offset `offset` and length `len`.
    #[inline(always)]
    pub fn get_row_slice(&self, channel: usize, row: usize, offset: isize, len: usize) -> &'a [T] {
        assert!(channel < self.channel_ptrs.len(), "channel out of bounds");
        assert!(row < self.row_offsets.len(), "row out of bounds");
        let x_offset = self.x_offset as isize;
        let row_len = self.row_len as isize;
        assert!(
            offset >= -x_offset,
            "slice start underflows buffer: offset {offset}, x_offset {}",
            self.x_offset
        );
        let end_elem = offset.checked_add_unsigned(len).expect("len overflow");
        assert!(
            end_elem <= row_len,
            "slice end exceeds row buffer: end {end_elem}, row_len {row_len}"
        );
        // SAFETY: Bounds and offsets are verified above, ensuring the slice `[ptr, ptr + len)`
        // lies strictly within the initialized row allocation for lifetime `'a`.
        unsafe {
            let base = *self.channel_ptrs.get_unchecked(channel);
            let row_start = *self.row_offsets.get_unchecked(row);
            let ptr = base.add(row_start).offset(offset);
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Construct a `ChannelsView` with compile-time dimensions `CHANS`, `ROWS`, and stencil `RADIUS`.
    #[inline(always)]
    pub fn create_view<
        const CHANS: usize,
        const ROWS: usize,
        const RADIUS: usize,
        D: SimdDescriptor,
    >(
        &self,
        d: D,
        xsize: usize,
    ) -> ChannelsView<'a, D, T, CHANS, ROWS, RADIUS>
    where
        T: VecLoad<D>,
    {
        assert!(
            self.channel_ptrs.len() >= CHANS,
            "Channels: requested {CHANS} channels, but only have {}",
            self.channel_ptrs.len()
        );
        assert!(
            self.row_offsets.len() >= ROWS,
            "Channels: requested {ROWS} rows, but only have {}",
            self.row_offsets.len()
        );
        assert!(
            self.x_offset >= RADIUS,
            "Channels: left padding {} too small for RADIUS {RADIUS}",
            self.x_offset
        );

        assert!(xsize > 0, "Channels: xsize must be non-zero");
        let vl = T::LEN;
        let num_vec = xsize.div_ceil(vl);
        let required_len = num_vec
            .checked_mul(vl)
            .and_then(|v| v.checked_add(RADIUS))
            .expect("required_len overflow");
        assert!(
            self.row_len >= required_len,
            "Channels: row_len {} too small for RADIUS {RADIUS} with xsize {xsize} (needs {required_len})",
            self.row_len
        );

        let mut channel_ptrs = [std::ptr::null(); CHANS];
        channel_ptrs.copy_from_slice(&self.channel_ptrs[..CHANS]);

        let mut row_offsets = [0; ROWS];
        row_offsets.copy_from_slice(&self.row_offsets[..ROWS]);

        ChannelsView {
            d,
            channel_ptrs,
            row_offsets,
            step: vl,
            num_vec,
            _marker: PhantomData,
        }
    }

    /// Returns a `Channels` accessor for a single row across all channels.
    #[inline(always)]
    pub fn select_row(&self, row: usize) -> Channels<'a, T> {
        assert!(
            row < self.row_offsets.len(),
            "row index {row} out of bounds ({})",
            self.row_offsets.len()
        );
        let mut row_offsets = SmallVec::new();
        row_offsets.push(self.row_offsets[row]);
        // SAFETY: Inherits existing valid pointers and verified row offset for lifetime `'a`.
        unsafe { Self::new_with_layout(copy_smallvec(&self.channel_ptrs), row_offsets, self.x_offset, self.row_len) }
    }

    /// Returns the number of channels.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.channel_ptrs.len()
    }

    /// Returns true if there are no channels.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.channel_ptrs.is_empty()
    }

    /// Returns the number of rows per channel.
    #[inline(always)]
    pub fn rows_per_channel(&self) -> usize {
        self.row_offsets.len()
    }
}

impl<'a, T: crate::image::ImageDataType> Channels<'a, T> {
    /// Create `Channels` directly from a slice of `RowBuffer` references for the given row and vertical border.
    #[inline(always)]
    pub fn from_row_buffers(
        buffers: &[&'a RowBuffer],
        xstart: usize,
        current_row: usize,
        border_y: usize,
        image_height: usize,
    ) -> Self {
        if buffers.is_empty() {
            return Self::empty();
        }
        let first = buffers[0];
        let stride = first.stride_elements::<T>();
        assert!(xstart <= stride, "xstart exceeds stride");
        let num_rows = first.num_rows();
        let row_mask = num_rows - 1;
        let mut channel_ptrs = SmallVec::new();
        for b in buffers.iter() {
            let ptr = b.as_slice::<T>().as_ptr();
            // SAFETY: `xstart <= stride` ensures `ptr.add(xstart)` is in-bounds of the buffer.
            channel_ptrs.push(unsafe { ptr.add(xstart) });
        }
        let by = border_y as isize;
        let mut row_offsets = SmallVec::new();
        if current_row >= border_y && current_row + border_y < image_height {
            for iy in -by..=by {
                let row_idx = (current_row as isize + iy) as usize & row_mask;
                row_offsets.push(row_idx * stride);
            }
        } else {
            for iy in -by..=by {
                let row_idx = mirror(current_row as isize + iy, image_height) & row_mask;
                row_offsets.push(row_idx * stride);
            }
        }
        let row_len = stride - xstart;
        // SAFETY: Each pointer and circular row offset maps to an allocated row in `RowBuffer`.
        unsafe { Self::new_with_layout(channel_ptrs, row_offsets, xstart, row_len) }
    }
}

/// Multi-row channel accessor for mutable access.
pub struct ChannelsMut<'a, T> {
    // Safety invariant: Each pointer in `channel_ptrs` is non-null, aligned for `T`,
    // and points into a live allocation valid and exclusively owned for lifetime `'a`.
    channel_ptrs: SmallVec<*mut T, 8, StackOnly>,
    // Safety invariant: Each offset in `row_offsets` selects a valid row such that no two (c, r) pairs
    // refer to overlapping memory regions, and each row has capacity for at least `row_len` elements of `T`.
    row_offsets: SmallVec<usize, 16, StackOnly>,
    // Safety invariant: Length in elements of exclusively accessible row data starting at each row pointer.
    row_len: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> ChannelsMut<'a, T> {
    /// Create a new ChannelsMut accessor with explicit pointers and offsets.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// 1. Each pointer in `channel_ptrs` is non-null, properly aligned for `T`, and points into an
    ///    allocated object that remains valid for lifetime `'a`.
    /// 2. For every channel index `c < channel_ptrs.len()` and row index `r < row_offsets.len()`,
    ///    the memory range `[channel_ptrs[c] + row_offsets[r], channel_ptrs[c] + row_offsets[r] + row_len)`
    ///    resides entirely within the same allocated object and contains initialized values of `T`.
    /// 3. Exclusivity / No Aliasing: No two `(c, r)` pairs refer to overlapping memory regions,
    ///    and no other live reference or pointer aliases any part of the accessible memory for lifetime `'a`.
    /// 4. No arithmetic overflow occurs in `row_offsets[r] + row_len` when converted to byte offsets:
    ///    `(row_offsets[r] + row_len) * size_of::<T>() <= isize::MAX as usize`.
    ///
    /// # Postconditions
    ///
    /// Returns a `ChannelsMut<'a, T>` accessor providing sound, exclusive, mutable access to the described rows
    /// for lifetime `'a`.
    #[inline(always)]
    pub(crate) unsafe fn new_with_layout(
        channel_ptrs: SmallVec<*mut T, 8, StackOnly>,
        row_offsets: SmallVec<usize, 16, StackOnly>,
        row_len: usize,
    ) -> Self {
        Self {
            channel_ptrs,
            row_offsets,
            row_len,
            _marker: PhantomData,
        }
    }

    /// Returns an empty `ChannelsMut` instance with zero channels and zero rows.
    pub fn empty() -> Self {
        // SAFETY: Vacuously sound with 0 channels and 0 rows.
        unsafe { Self::new_with_layout(SmallVec::new(), SmallVec::new(), 0) }
    }

    /// Get a mutable reference to the slice for channel `channel`, row `row`.
    #[inline(always)]
    pub fn get_row_mut(&mut self, channel: usize, row: usize) -> &mut [T] {
        assert!(channel < self.channel_ptrs.len(), "channel out of bounds");
        assert!(row < self.row_offsets.len(), "row out of bounds");
        // SAFETY: Bounds asserted above. `&mut self` guarantees exclusive mutable access for the borrow,
        // and the range `[base + offset, base + offset + self.row_len)` is within the valid row buffer allocation.
        unsafe {
            let base = *self.channel_ptrs.get_unchecked(channel);
            let offset = *self.row_offsets.get_unchecked(row);
            std::slice::from_raw_parts_mut(base.add(offset), self.row_len)
        }
    }

    /// Construct a `ChannelsMutView` with compile-time dimensions `CHANS`, `ROWS`, and chunk `SCALE`.
    #[inline(always)]
    pub fn create_view<
        'b,
        const CHANS: usize,
        const ROWS: usize,
        const SCALE: usize,
        D: SimdDescriptor,
    >(
        &'b mut self,
        d: D,
        xsize: usize,
    ) -> ChannelsMutView<'b, D, T, CHANS, ROWS, SCALE>
    where
        T: VecStore<D>,
    {
        assert!(
            self.channel_ptrs.len() >= CHANS,
            "ChannelsMut: requested {CHANS} channels, but only have {}",
            self.channel_ptrs.len()
        );
        assert!(
            self.row_offsets.len() >= ROWS,
            "ChannelsMut: requested {ROWS} rows, but only have {}",
            self.row_offsets.len()
        );

        assert!(xsize > 0, "ChannelsMut: xsize must be non-zero");
        let vl = T::LEN;
        let num_vec = xsize.div_ceil(vl);
        let required_len = num_vec
            .checked_mul(vl)
            .and_then(|v| v.checked_mul(SCALE))
            .expect("required_len overflow");
        assert!(
            self.row_len >= required_len,
            "ChannelsMut: row_len {} too small for SCALE {SCALE} with xsize {xsize} (needs {required_len})",
            self.row_len
        );

        let mut channel_ptrs = [std::ptr::null_mut(); CHANS];
        channel_ptrs.copy_from_slice(&self.channel_ptrs[..CHANS]);

        let mut row_offsets = [0; ROWS];
        row_offsets.copy_from_slice(&self.row_offsets[..ROWS]);

        ChannelsMutView {
            d,
            channel_ptrs,
            row_offsets,
            step: vl * SCALE,
            num_vec,
            _marker: PhantomData,
        }
    }

    /// Returns a `ChannelsMut` accessor for a single row across all channels.
    #[inline(always)]
    pub fn select_row(&mut self, row: usize) -> ChannelsMut<'_, T> {
        assert!(
            row < self.row_offsets.len(),
            "row index {row} out of bounds ({})",
            self.row_offsets.len()
        );
        let mut row_offsets = SmallVec::new();
        row_offsets.push(self.row_offsets[row]);
        // SAFETY: Re-borrowing from `&mut self` guarantees exclusive mutable access for `'_`.
        // The pointers and single row offset remain valid within the allocation.
        unsafe { Self::new_with_layout(copy_smallvec(&self.channel_ptrs), row_offsets, self.row_len) }
    }

    /// Returns the number of rows per channel.
    #[inline(always)]
    pub fn rows_per_channel(&self) -> usize {
        self.row_offsets.len()
    }

    /// Returns the number of channels.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.channel_ptrs.len()
    }

    /// Returns true if there are no channels.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.channel_ptrs.is_empty()
    }
}

impl<'a, T: crate::image::ImageDataType> ChannelsMut<'a, T> {
    /// Create `ChannelsMut` directly from mutable `RowBuffer` references, computing circular row offsets in-place.
    #[inline(always)]
    pub fn from_row_buffers(
        buffers: &'a mut [RowBuffer],
        xstart: usize,
        first_row: usize,
        count: usize,
    ) -> Self {
        if buffers.is_empty() || count == 0 {
            return Self::empty();
        }
        let first = &buffers[0];
        let stride = first.stride_elements::<T>();
        assert!(xstart <= stride, "xstart exceeds stride");
        let num_rows = first.num_rows();
        assert!(
            count <= num_rows,
            "count {count} exceeds available rows {num_rows}"
        );
        let row_mask = num_rows - 1;
        let mut channel_ptrs = SmallVec::new();
        for b in buffers.iter_mut() {
            let ptr = b.as_mut_slice::<T>().as_mut_ptr();
            // SAFETY: `xstart <= stride` ensures `ptr.add(xstart)` is in-bounds of the buffer.
            channel_ptrs.push(unsafe { ptr.add(xstart) });
        }
        let mut row_offsets = SmallVec::new();
        for r in 0..count {
            let row_idx = (first_row + r) & row_mask;
            row_offsets.push(row_idx * stride);
        }
        let row_len = stride - xstart;
        // SAFETY: `buffers` provides disjoint `RowBuffer`s. With `count <= num_rows`,
        // circular row indices are distinct, ensuring no two (c, r) pairs alias in memory.
        unsafe { Self::new_with_layout(channel_ptrs, row_offsets, row_len) }
    }
}

/// Helper struct to execute a closure over SIMD chunks of input and output channels.
///
/// Const generic parameters:
/// - `CHANS_IN`: number of input channels
/// - `ROWS_IN`: number of input rows
/// - `RADIUS`: horizontal border radius for stencil loads (offsets in `[-RADIUS, RADIUS]`)
/// - `CHANS_OUT`: number of output channels
/// - `ROWS_OUT`: number of output rows
/// - `SCALE`: chunk scaling / horizontal upsampling factor (defaults to 1)
pub struct ForEachChunk<
    const CHANS_IN: usize,
    const ROWS_IN: usize,
    const RADIUS: usize,
    const CHANS_OUT: usize,
    const ROWS_OUT: usize,
    const SCALE: usize = 1,
>;

impl<
    const CHANS_IN: usize,
    const ROWS_IN: usize,
    const RADIUS: usize,
    const CHANS_OUT: usize,
    const ROWS_OUT: usize,
    const SCALE: usize,
> ForEachChunk<CHANS_IN, ROWS_IN, RADIUS, CHANS_OUT, ROWS_OUT, SCALE>
{
    /// Execute `f` over chunks of width `T::LEN` spanning `xsize`.
    #[inline(always)]
    pub fn run<D: SimdDescriptor, T: VecLoad<D>, U: VecStore<D>, F>(
        d: D,
        xsize: usize,
        input_rows: &Channels<T>,
        output_rows: &mut ChannelsMut<U>,
        mut f: F,
    ) where
        F: FnMut(
            usize,
            &ChannelsView<'_, D, T, CHANS_IN, ROWS_IN, RADIUS>,
            &mut ChannelsMutView<'_, D, U, CHANS_OUT, ROWS_OUT, SCALE>,
        ),
    {
        if xsize == 0 {
            return;
        }
        let mut in_view = input_rows.create_view::<CHANS_IN, ROWS_IN, RADIUS, _>(d, xsize);
        let mut out_view = output_rows.create_view::<CHANS_OUT, ROWS_OUT, SCALE, _>(d, xsize);
        let vl = T::LEN;
        assert_eq!(vl, U::LEN, "Input and Output vector lengths must match");
        let num_vec = in_view.num_vec;
        for i in 0..num_vec {
            let x = i * vl;
            f(x, &in_view, &mut out_view);
            // SAFETY: Invariants established in `create_view` ensure both views have buffer capacity
            // for `num_vec * step` elements, and `i < num_vec`.
            unsafe {
                in_view.advance_all();
                out_view.advance_all();
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::undocumented_unsafe_blocks
)]
mod tests {
    use jxl_simd::{ScalarDescriptor, SimdDescriptor, test_all_instruction_sets};

    use super::*;
    use crate::image::DataTypeTag;
    use crate::render::low_memory_pipeline::row_buffers::RowBuffer;

    #[test]
    fn test_xsize_zero() {
        let d = ScalarDescriptor::new().unwrap();
        let rb_in = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 16).unwrap();
        let mut rb_out = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 16).unwrap();
        let in_x0 = RowBuffer::x0_offset::<f32>();
        let out_x0 = RowBuffer::x0_offset::<f32>();
        let in_channels = Channels::<f32>::from_row_buffers(&[&rb_in], in_x0, 0, 0, 1);
        let mut out_channels =
            ChannelsMut::<f32>::from_row_buffers(std::slice::from_mut(&mut rb_out), out_x0, 0, 1);

        let mut count = 0;
        ForEachChunk::<1, 1, 2, 1, 1>::run(
            d,
            0,
            &in_channels,
            &mut out_channels,
            |_x, _inv, _outv| {
                count += 1;
            },
        );
        assert_eq!(count, 0);
    }

    #[test]
    #[should_panic(expected = "Channels: row_len 8 too small for RADIUS 2 with xsize 8 (needs 10)")]
    fn test_insufficient_buffer_len_panics() {
        let d = ScalarDescriptor::new().unwrap();
        let rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 8).unwrap();
        let stride = rb.stride_elements::<f32>();
        let channels = Channels::<f32>::from_row_buffers(&[&rb], stride - 8, 0, 0, 1);
        // x_offset = stride - 8, row_len = 8.
        // xsize = 8, RADIUS = 2 -> needs 8 + 2 = 10 elements, but only has row_len = 8
        let _ = channels.create_view::<1, 1, 2, _>(d, 8);
    }

    #[test]
    #[should_panic(expected = "Channels: left padding 0 too small for RADIUS 2")]
    fn test_insufficient_left_padding_panics() {
        let d = ScalarDescriptor::new().unwrap();
        let rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 16).unwrap();
        let channels = Channels::<f32>::from_row_buffers(&[&rb], 0, 0, 0, 1);
        // x_offset = 0, but RADIUS = 2
        let _ = channels.create_view::<1, 1, 2, _>(d, 8);
    }

    #[test]
    #[should_panic(expected = "Channels: requested 2 channels, but only have 1")]
    fn test_insufficient_channels_panics() {
        let d = ScalarDescriptor::new().unwrap();
        let rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 16).unwrap();
        let in_x0 = RowBuffer::x0_offset::<f32>();
        let channels = Channels::<f32>::from_row_buffers(&[&rb], in_x0, 0, 0, 1);
        let _ = channels.create_view::<2, 1, 0, _>(d, 8);
    }

    #[test]
    #[should_panic(expected = "Channels: requested 2 rows, but only have 1")]
    fn test_insufficient_rows_panics() {
        let d = ScalarDescriptor::new().unwrap();
        let rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 16).unwrap();
        let in_x0 = RowBuffer::x0_offset::<f32>();
        let channels = Channels::<f32>::from_row_buffers(&[&rb], in_x0, 0, 0, 1);
        let _ = channels.create_view::<1, 2, 0, _>(d, 8);
    }

    fn test_simd_stencil_processing_non_multiple_vl<D: SimdDescriptor>(d: D) {
        let vl = <f32 as VecLoad<D>>::LEN;
        const RADIUS: usize = 2;
        const CHANS_IN: usize = 3;
        const ROWS_IN: usize = 5;
        const CHANS_OUT: usize = 2;
        const ROWS_OUT: usize = 1;

        let test_sizes = [
            1,
            vl.saturating_sub(1).max(1),
            vl,
            vl + 1,
            vl * 2 + 3,
            vl * 3 + 7,
        ];

        for xsize in test_sizes {
            let mut in_rbs = (0..CHANS_IN)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 2, 0, 0, xsize).unwrap())
                .collect::<Vec<_>>();
            let in_x0 = RowBuffer::x0_offset::<f32>();
            for c in 0..CHANS_IN {
                for r in 0..ROWS_IN {
                    let row = in_rbs[c].get_row_mut::<f32>(r);
                    for x in 0..row.len() {
                        row[x] = (c as f32) * 10000.0 + (r as f32) * 100.0 + (x as f32);
                    }
                }
            }
            let in_refs: Vec<&RowBuffer> = in_rbs.iter().collect();
            let in_channels = Channels::<f32>::from_row_buffers(&in_refs, in_x0, 2, 2, 5);

            let mut out_rbs = (0..CHANS_OUT)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize).unwrap())
                .collect::<Vec<_>>();
            let out_x0 = RowBuffer::x0_offset::<f32>();
            let mut out_channels =
                ChannelsMut::<f32>::from_row_buffers(&mut out_rbs, out_x0, 0, ROWS_OUT);

            ForEachChunk::<CHANS_IN, ROWS_IN, RADIUS, CHANS_OUT, ROWS_OUT>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let c0 = in_v.load::<0, 2, 0>();
                    let c1_left = in_v.load::<1, 1, -2>();
                    let c2_right = in_v.load::<2, 3, 2>();
                    out_v.store::<0, 0>(c0 + c1_left);
                    out_v.store::<1, 0>(c1_left + c2_right);
                },
            );

            for i in 0..xsize {
                let exp0 = in_rbs[0].get_row::<f32>(2)[in_x0 + i]
                    + in_rbs[1].get_row::<f32>(1)[in_x0 + i - 2];
                let exp1 = in_rbs[1].get_row::<f32>(1)[in_x0 + i - 2]
                    + in_rbs[2].get_row::<f32>(3)[in_x0 + i + 2];
                assert_eq!(
                    out_rbs[0].get_row::<f32>(0)[out_x0 + i],
                    exp0,
                    "Mismatch channel 0 at index {i} for xsize {xsize}"
                );
                assert_eq!(
                    out_rbs[1].get_row::<f32>(0)[out_x0 + i],
                    exp1,
                    "Mismatch channel 1 at index {i} for xsize {xsize}"
                );
            }
        }
    }

    fn test_simd_interleaved_stores_all_scales<D: SimdDescriptor>(d: D) {
        let vl = <f32 as VecLoad<D>>::LEN;

        // Scale 2
        {
            let xsize = vl * 2;
            let mut in_rbs = (0..2)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize).unwrap())
                .collect::<Vec<_>>();
            let in_x0 = RowBuffer::x0_offset::<f32>();
            for i in 0..xsize {
                in_rbs[0].get_row_mut::<f32>(0)[in_x0 + i] = 100.0 + i as f32;
                in_rbs[1].get_row_mut::<f32>(0)[in_x0 + i] = 200.0 + i as f32;
            }
            let in_refs: Vec<&RowBuffer> = in_rbs.iter().collect();
            let in_channels = Channels::<f32>::from_row_buffers(&in_refs, in_x0, 0, 0, 1);

            let mut out_rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize * 2).unwrap();
            let out_x0 = RowBuffer::x0_offset::<f32>();
            let mut out_channels = ChannelsMut::<f32>::from_row_buffers(
                std::slice::from_mut(&mut out_rb),
                out_x0,
                0,
                1,
            );

            ForEachChunk::<2, 1, 0, 1, 1, 2>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let v0 = in_v.load::<0, 0, 0>();
                    let v1 = in_v.load::<1, 0, 0>();
                    out_v.store_interleaved::<0, 0>([v0, v1]);
                },
            );

            for i in 0..xsize {
                assert_eq!(out_rb.get_row::<f32>(0)[out_x0 + i * 2], 100.0 + i as f32);
                assert_eq!(
                    out_rb.get_row::<f32>(0)[out_x0 + i * 2 + 1],
                    200.0 + i as f32
                );
            }
        }


        // Scale 4
        {
            let xsize = vl * 2;
            let mut in_rbs = (0..4)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize).unwrap())
                .collect::<Vec<_>>();
            let in_x0 = RowBuffer::x0_offset::<f32>();
            for c in 0..4 {
                for i in 0..xsize {
                    in_rbs[c].get_row_mut::<f32>(0)[in_x0 + i] =
                        (c as f32 + 1.0) * 100.0 + i as f32;
                }
            }
            let in_refs: Vec<&RowBuffer> = in_rbs.iter().collect();
            let in_channels = Channels::<f32>::from_row_buffers(&in_refs, in_x0, 0, 0, 1);

            let mut out_rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize * 4).unwrap();
            let out_x0 = RowBuffer::x0_offset::<f32>();
            let mut out_channels = ChannelsMut::<f32>::from_row_buffers(
                std::slice::from_mut(&mut out_rb),
                out_x0,
                0,
                1,
            );

            ForEachChunk::<4, 1, 0, 1, 1, 4>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let v0 = in_v.load::<0, 0, 0>();
                    let v1 = in_v.load::<1, 0, 0>();
                    let v2 = in_v.load::<2, 0, 0>();
                    let v3 = in_v.load::<3, 0, 0>();
                    out_v.store_interleaved::<0, 0>([v0, v1, v2, v3]);
                },
            );

            for i in 0..xsize {
                for c in 0..4 {
                    assert_eq!(
                        out_rb.get_row::<f32>(0)[out_x0 + i * 4 + c],
                        (c as f32 + 1.0) * 100.0 + i as f32
                    );
                }
            }
        }

        // Scale 8
        {
            let xsize = vl * 2;
            let mut in_rbs = (0..8)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize).unwrap())
                .collect::<Vec<_>>();
            let in_x0 = RowBuffer::x0_offset::<f32>();
            for c in 0..8 {
                for i in 0..xsize {
                    in_rbs[c].get_row_mut::<f32>(0)[in_x0 + i] =
                        (c as f32 + 1.0) * 100.0 + i as f32;
                }
            }
            let in_refs: Vec<&RowBuffer> = in_rbs.iter().collect();
            let in_channels = Channels::<f32>::from_row_buffers(&in_refs, in_x0, 0, 0, 1);

            let mut out_rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, xsize * 8).unwrap();
            let out_x0 = RowBuffer::x0_offset::<f32>();
            let mut out_channels = ChannelsMut::<f32>::from_row_buffers(
                std::slice::from_mut(&mut out_rb),
                out_x0,
                0,
                1,
            );

            ForEachChunk::<8, 1, 0, 1, 1, 8>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let vals = [
                        in_v.load::<0, 0, 0>(),
                        in_v.load::<1, 0, 0>(),
                        in_v.load::<2, 0, 0>(),
                        in_v.load::<3, 0, 0>(),
                        in_v.load::<4, 0, 0>(),
                        in_v.load::<5, 0, 0>(),
                        in_v.load::<6, 0, 0>(),
                        in_v.load::<7, 0, 0>(),
                    ];
                    out_v.store_interleaved::<0, 0>(vals);
                },
            );

            for i in 0..xsize {
                for c in 0..8 {
                    assert_eq!(
                        out_rb.get_row::<f32>(0)[out_x0 + i * 8 + c],
                        (c as f32 + 1.0) * 100.0 + i as f32
                    );
                }
            }
        }
    }

    fn test_simd_data_types_all_instruction_sets<D: SimdDescriptor>(d: D) {
        let vl = <i32 as VecLoad<D>>::LEN;
        let xsize = vl * 2;

        // i32 load, stencil, store
        {
            let mut rb_in = RowBuffer::new(DataTypeTag::I32, 0, 0, 0, xsize).unwrap();
            let in_x0 = RowBuffer::x0_offset::<i32>();
            let row_in = rb_in.get_row_mut::<i32>(0);
            for i in 0..row_in.len() {
                row_in[i] = (i as i32) * 10;
            }
            let in_channels = Channels::<i32>::from_row_buffers(&[&rb_in], in_x0, 0, 0, 1);

            let mut rb_out = RowBuffer::new(DataTypeTag::I32, 0, 0, 0, xsize).unwrap();
            let out_x0 = RowBuffer::x0_offset::<i32>();
            let mut out_channels = ChannelsMut::<i32>::from_row_buffers(
                std::slice::from_mut(&mut rb_out),
                out_x0,
                0,
                1,
            );

            ForEachChunk::<1, 1, 2, 1, 1>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let cur = in_v.load::<0, 0, 0>();
                    let left = in_v.load::<0, 0, -2>();
                    let right = in_v.load::<0, 0, 2>();
                    out_v.store::<0, 0>(cur + left + right);
                },
            );

            for i in 0..xsize {
                let exp = rb_in.get_row::<i32>(0)[in_x0 + i]
                    + rb_in.get_row::<i32>(0)[in_x0 + i - 2]
                    + rb_in.get_row::<i32>(0)[in_x0 + i + 2];
                assert_eq!(
                    rb_out.get_row::<i32>(0)[out_x0 + i],
                    exp,
                    "i32 mismatch at {i}"
                );
            }
        }

        // u16 load and store
        {
            let mut rb_in = RowBuffer::new(DataTypeTag::U16, 0, 0, 0, xsize).unwrap();
            let in_x0 = RowBuffer::x0_offset::<u16>();
            let row_in = rb_in.get_row_mut::<u16>(0);
            for i in 0..row_in.len() {
                row_in[i] = (i as u16) * 5 + 1;
            }
            let in_channels = Channels::<u16>::from_row_buffers(&[&rb_in], in_x0, 0, 0, 1);

            let mut rb_out = RowBuffer::new(DataTypeTag::U16, 0, 0, 0, xsize).unwrap();
            let out_x0 = RowBuffer::x0_offset::<u16>();
            let mut out_channels = ChannelsMut::<u16>::from_row_buffers(
                std::slice::from_mut(&mut rb_out),
                out_x0,
                0,
                1,
            );

            ForEachChunk::<1, 1, 1, 1, 1>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let cur = in_v.load::<0, 0, 0>();
                    let left = in_v.load::<0, 0, -1>();
                    out_v.store::<0, 0>(cur + left);
                },
            );

            for i in 0..xsize {
                let exp =
                    rb_in.get_row::<u16>(0)[in_x0 + i] + rb_in.get_row::<u16>(0)[in_x0 + i - 1];
                assert_eq!(
                    rb_out.get_row::<u16>(0)[out_x0 + i],
                    exp,
                    "u16 mismatch at {i}"
                );
            }
        }

        // i16 load
        {
            let mut rb_in = RowBuffer::new(DataTypeTag::I16, 0, 0, 0, xsize).unwrap();
            let in_x0 = RowBuffer::x0_offset::<i16>();
            let row_in = rb_in.get_row_mut::<i16>(0);
            for i in 0..row_in.len() {
                row_in[i] = (i as i16) * 3 - 50;
            }
            let in_channels = Channels::<i16>::from_row_buffers(&[&rb_in], in_x0, 0, 0, 1);

            let mut rb_out = RowBuffer::new(DataTypeTag::I32, 0, 0, 0, xsize).unwrap();
            let out_x0 = RowBuffer::x0_offset::<i32>();
            let mut out_channels = ChannelsMut::<i32>::from_row_buffers(
                std::slice::from_mut(&mut rb_out),
                out_x0,
                0,
                1,
            );

            ForEachChunk::<1, 1, 1, 1, 1>::run(
                d,
                xsize,
                &in_channels,
                &mut out_channels,
                |_x, in_v, out_v| {
                    let cur = in_v.load::<0, 0, 0>();
                    let left = in_v.load::<0, 0, -1>();
                    out_v.store::<0, 0>(cur - left);
                },
            );

            for i in 0..xsize {
                let exp = (rb_in.get_row::<i16>(0)[in_x0 + i]
                    - rb_in.get_row::<i16>(0)[in_x0 + i - 1]) as i32;
                assert_eq!(
                    rb_out.get_row::<i32>(0)[out_x0 + i],
                    exp,
                    "i16 mismatch at {i}"
                );
            }
        }
    }

    #[test]
    fn test_select_row_multi_channel() {
        let mut rb0 = RowBuffer::new(DataTypeTag::F32, 1, 0, 0, 16).unwrap();
        let mut rb1 = RowBuffer::new(DataTypeTag::F32, 1, 0, 0, 16).unwrap();
        let stride = rb0.stride_elements::<f32>();
        for r in 0..3 {
            for x in 0..stride {
                rb0.get_row_mut::<f32>(r)[x] = 1.0;
                rb1.get_row_mut::<f32>(r)[x] = 100.0 + (r * stride + x) as f32;
            }
        }

        let channels = Channels::<f32>::from_row_buffers(&[&rb0, &rb1], 2, 1, 1, 3);
        let mut rb_out = RowBuffer::new(DataTypeTag::F32, 1, 0, 0, 16).unwrap();
        for r in 0..3 {
            for x in 0..stride {
                rb_out.get_row_mut::<f32>(r)[x] = 100.0 + (r * stride + x) as f32;
            }
        }
        let mut channels_mut =
            ChannelsMut::<f32>::from_row_buffers(std::slice::from_mut(&mut rb_out), 2, 0, 3);

        let sel_c = channels.select_row(1);
        assert_eq!(sel_c.len(), 2);
        assert_eq!(sel_c.rows_per_channel(), 1);
        assert_eq!(sel_c.get_row(0, 0)[0], 1.0);
        assert_eq!(sel_c.get_row(1, 0)[0], 100.0 + (stride + 2) as f32);

        let mut sel_m = channels_mut.select_row(1);
        assert_eq!(sel_m.len(), 1);
        assert_eq!(sel_m.rows_per_channel(), 1);
        sel_m.get_row_mut(0, 0)[0] = 777.0;

        assert_eq!(channels_mut.get_row_mut(0, 1)[0], 777.0);
    }

    #[test]
    fn test_channels_empty_and_accessors() {
        let empty_c: Channels<'_, f32> = Channels::empty();
        assert!(empty_c.is_empty());
        assert_eq!(empty_c.len(), 0);
        assert_eq!(empty_c.rows_per_channel(), 0);

        let empty_m: ChannelsMut<'_, f32> = ChannelsMut::empty();
        assert!(empty_m.is_empty());
        assert_eq!(empty_m.len(), 0);
        assert_eq!(empty_m.rows_per_channel(), 0);

        let empty_rb_c = Channels::<f32>::from_row_buffers(&[], 0, 0, 0, 0);
        assert!(empty_rb_c.is_empty());

        let empty_rb_m: ChannelsMut<'_, f32> = ChannelsMut::from_row_buffers(&mut [], 0, 0, 0);
        assert!(empty_rb_m.is_empty());

        let mut rb = RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 16).unwrap();
        let zero_count_m =
            ChannelsMut::<f32>::from_row_buffers(std::slice::from_mut(&mut rb), 0, 0, 0);
        assert!(zero_count_m.is_empty());
    }

    #[test]
    fn test_from_row_buffers_circular_indexing_and_soundness() {
        use crate::image::DataTypeTag;
        use crate::render::low_memory_pipeline::row_buffers::RowBuffer;

        let rb_in = RowBuffer::new(DataTypeTag::F32, 0, 1, 0, 16).unwrap();
        let mut rb_out = RowBuffer::new(DataTypeTag::F32, 0, 1, 0, 16).unwrap();

        let channels_in = Channels::<f32>::from_row_buffers(&[&rb_in], 0, 0, 0, 2);
        assert_eq!(channels_in.len(), 1);
        assert_eq!(channels_in.rows_per_channel(), 1);

        let mut channels_out =
            ChannelsMut::<f32>::from_row_buffers(std::slice::from_mut(&mut rb_out), 0, 0, 2);
        assert_eq!(channels_out.len(), 1);
        assert_eq!(channels_out.rows_per_channel(), 2);

        channels_out.get_row_mut(0, 0)[0] = 42.0;
        channels_out.get_row_mut(0, 1)[0] = 84.0;
        assert_eq!(rb_out.get_row::<f32>(0)[0], 42.0);
        assert_eq!(rb_out.get_row::<f32>(1)[0], 84.0);
    }

    #[test]
    fn test_from_row_buffers_mirroring_and_fast_path() {
        use crate::image::DataTypeTag;
        use crate::render::low_memory_pipeline::row_buffers::RowBuffer;

        // Create a RowBuffer with 8 rows
        let mut rb = RowBuffer::new(DataTypeTag::F32, 2, 0, 0, 16).unwrap();
        let stride = rb.stride_elements::<f32>();
        assert_eq!(rb.num_rows(), 8);

        // Fill each row with distinctive values: row r filled with r as f32
        for r in 0..8 {
            for x in 0..stride {
                rb.get_row_mut::<f32>(r)[x] = (r * 100 + x) as f32;
            }
        }

        let height = 10;

        // 1. BORDER_Y = 0 (single row)
        {
            let c = Channels::<f32>::from_row_buffers(&[&rb], 0, 3, 0, height);
            assert_eq!(c.rows_per_channel(), 1);
            assert_eq!(c.get_row(0, 0)[0], 300.0);
        }

        // 2. BORDER_Y = 1: Fast path (interior row, e.g. row 4)
        {
            let c = Channels::<f32>::from_row_buffers(&[&rb], 0, 4, 1, height);
            assert_eq!(c.rows_per_channel(), 3);
            // Rows should be 3, 4, 5
            assert_eq!(c.get_row(0, 0)[0], 300.0);
            assert_eq!(c.get_row(0, 1)[0], 400.0);
            assert_eq!(c.get_row(0, 2)[0], 500.0);
        }

        // 3. BORDER_Y = 1: Border path at top (current_row = 0)
        {
            let c = Channels::<f32>::from_row_buffers(&[&rb], 0, 0, 1, height);
            assert_eq!(c.rows_per_channel(), 3);
            // mirror(-1, 10) = 0, mirror(0, 10) = 0, mirror(1, 10) = 1
            assert_eq!(c.get_row(0, 0)[0], 0.0);
            assert_eq!(c.get_row(0, 1)[0], 0.0);
            assert_eq!(c.get_row(0, 2)[0], 100.0);
        }

        // 4. BORDER_Y = 2: Border path at bottom (current_row = 9, height = 10)
        {
            let c = Channels::<f32>::from_row_buffers(&[&rb], 0, 9, 2, height);
            assert_eq!(c.rows_per_channel(), 5);
            // mirror(7, 10) = 7 & 7 = 7
            // mirror(8, 10) = 8 & 7 = 0
            // mirror(9, 10) = 9 & 7 = 1
            // mirror(10, 10) = 9 & 7 = 1
            // mirror(11, 10) = 8 & 7 = 0
            assert_eq!(c.get_row(0, 0)[0], 700.0);
            assert_eq!(c.get_row(0, 1)[0], 0.0);
            assert_eq!(c.get_row(0, 2)[0], 100.0);
            assert_eq!(c.get_row(0, 3)[0], 100.0);
            assert_eq!(c.get_row(0, 4)[0], 0.0);
        }

        // 5. ChannelsMut::from_row_buffers circular indexing across wrap
        {
            let mut rb_out = RowBuffer::new(DataTypeTag::F32, 2, 0, 0, 16).unwrap();
            let mut c_mut =
                ChannelsMut::<f32>::from_row_buffers(std::slice::from_mut(&mut rb_out), 0, 6, 4);
            assert_eq!(c_mut.rows_per_channel(), 4);
            // Rows should wrap: 6, 7, 0, 1
            c_mut.get_row_mut(0, 0)[0] = 60.0;
            c_mut.get_row_mut(0, 1)[0] = 70.0;
            c_mut.get_row_mut(0, 2)[0] = 80.0;
            c_mut.get_row_mut(0, 3)[0] = 90.0;

            assert_eq!(rb_out.get_row::<f32>(6)[0], 60.0);
            assert_eq!(rb_out.get_row::<f32>(7)[0], 70.0);
            assert_eq!(rb_out.get_row::<f32>(0)[0], 80.0);
            assert_eq!(rb_out.get_row::<f32>(1)[0], 90.0);
        }
    }

    #[test]
    fn test_channels_arbitrary_miri() {
        use crate::image::DataTypeTag;
        use crate::render::low_memory_pipeline::row_buffers::RowBuffer;

        let d = ScalarDescriptor::new().unwrap();

        arbtest::arbtest(|u| {
            // 1. Multi-channel, multi-row stencil processing with ForEachChunk
            const CHANS_IN: usize = 2;
            const ROWS_IN: usize = 3;
            const RADIUS: usize = 2;
            const CHANS_OUT: usize = 2;
            const ROWS_OUT: usize = 2;

            let max_xsize = if cfg!(miri) { 8 } else { 32 };
            let xsize: usize = u.int_in_range(0..=max_xsize)?;
            let stride_extra: usize = u.int_in_range(2..=8)?;

            let mut in_rbs = (0..CHANS_IN)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 1, 0, 0, xsize + stride_extra).unwrap())
                .collect::<Vec<_>>();
            let in_x0 = RowBuffer::x0_offset::<f32>();
            let stride = in_rbs[0].stride_elements::<f32>();
            for c in 0..CHANS_IN {
                for r in 0..ROWS_IN {
                    let row = in_rbs[c].get_row_mut::<f32>(r);
                    for x in 0..row.len() {
                        row[x] = (c as f32) * 1000.0 + (r as f32) * 100.0 + (x as f32) * 1.5 + 0.25;
                    }
                }
            }

            let in_refs: Vec<&RowBuffer> = in_rbs.iter().collect();
            let in_channels = Channels::<f32>::from_row_buffers(&in_refs, in_x0, 1, 1, 3);

            let mut out_rbs = (0..CHANS_OUT)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 1, 0, xsize + stride_extra).unwrap())
                .collect::<Vec<_>>();
            let out_x0 = RowBuffer::x0_offset::<f32>();
            {
                let mut out_channels =
                    ChannelsMut::<f32>::from_row_buffers(&mut out_rbs, out_x0, 0, ROWS_OUT);

                ForEachChunk::<CHANS_IN, ROWS_IN, RADIUS, CHANS_OUT, ROWS_OUT, 1>::run(
                    d,
                    xsize,
                    &in_channels,
                    &mut out_channels,
                    |_x, in_v, out_v| {
                        // Stencil loads across multiple channels and rows with negative, zero, positive offsets
                        let c0_r0_m2 = in_v.load::<0, 0, -2>();
                        let c0_r1_0 = in_v.load::<0, 1, 0>();
                        let c0_r2_p2 = in_v.load::<0, 2, 2>();

                        let c1_r0_m1 = in_v.load::<1, 0, -1>();
                        let c1_r1_0 = in_v.load::<1, 1, 0>();
                        let c1_r2_p1 = in_v.load::<1, 2, 1>();

                        out_v.store::<0, 0>(c0_r0_m2 + c0_r1_0 + c0_r2_p2);
                        out_v.store::<0, 1>(c1_r0_m1 + c1_r1_0 + c1_r2_p1);
                        out_v.store::<1, 0>(c0_r1_0 * 2.0 + c1_r1_0);
                        out_v.store::<1, 1>(c0_r0_m2 - c1_r2_p1);
                    },
                );
            }

            // Verify ForEachChunk outputs match reference scalar calculations
            for x in 0..xsize {
                let exp_0_0 = in_rbs[0].get_row::<f32>(0)[in_x0 + x - 2]
                    + in_rbs[0].get_row::<f32>(1)[in_x0 + x]
                    + in_rbs[0].get_row::<f32>(2)[in_x0 + x + 2];
                let exp_0_1 = in_rbs[1].get_row::<f32>(0)[in_x0 + x - 1]
                    + in_rbs[1].get_row::<f32>(1)[in_x0 + x]
                    + in_rbs[1].get_row::<f32>(2)[in_x0 + x + 1];
                let exp_1_0 = in_rbs[0].get_row::<f32>(1)[in_x0 + x] * 2.0
                    + in_rbs[1].get_row::<f32>(1)[in_x0 + x];
                let exp_1_1 = in_rbs[0].get_row::<f32>(0)[in_x0 + x - 2]
                    - in_rbs[1].get_row::<f32>(2)[in_x0 + x + 1];

                assert_eq!(out_rbs[0].get_row::<f32>(0)[out_x0 + x], exp_0_0);
                assert_eq!(out_rbs[0].get_row::<f32>(1)[out_x0 + x], exp_0_1);
                assert_eq!(out_rbs[1].get_row::<f32>(0)[out_x0 + x], exp_1_0);
                assert_eq!(out_rbs[1].get_row::<f32>(1)[out_x0 + x], exp_1_1);
            }

            // 2. Exercise get_row on Channels
            for c in 0..CHANS_IN {
                for r in 0..ROWS_IN {
                    let row = in_channels.get_row(c, r);
                    assert_eq!(row.len(), stride - in_x0);
                    assert_eq!(row, &in_rbs[c].get_row::<f32>(r)[in_x0..stride]);
                }
            }

            // 3. Exercise get_row_slice on Channels (positive, negative, and zero offsets)
            let row_len = stride - in_x0;
            let max_neg = in_x0 as isize;
            let offset: isize = u.int_in_range(-max_neg..=(row_len as isize))?;
            let max_possible_len = ((row_len as isize) - offset) as usize;
            let slice_len: usize = u.int_in_range(0..=max_possible_len)?;
            let slice_c: usize = u.int_in_range(0..=(CHANS_IN - 1))?;
            let slice_r: usize = u.int_in_range(0..=(ROWS_IN - 1))?;

            let slice = in_channels.get_row_slice(slice_c, slice_r, offset, slice_len);
            assert_eq!(slice.len(), slice_len);
            let abs_start = (in_x0 as isize + offset) as usize;
            assert_eq!(
                slice,
                &in_rbs[slice_c].get_row::<f32>(slice_r)[abs_start..abs_start + slice_len]
            );

            // 4. Exercise select_row on Channels
            let sel_r: usize = u.int_in_range(0..=(ROWS_IN - 1))?;
            let sel = in_channels.select_row(sel_r);
            assert_eq!(sel.len(), CHANS_IN);
            assert_eq!(sel.rows_per_channel(), 1);
            for c in 0..CHANS_IN {
                assert_eq!(sel.get_row(c, 0), in_channels.get_row(c, sel_r));
            }

            // 5. Exercise get_row_mut and select_row on ChannelsMut
            let mut_r: usize = u.int_in_range(0..=(ROWS_OUT - 1))?;
            let mut_c: usize = u.int_in_range(0..=(CHANS_OUT - 1))?;
            let mut mutated_val = 0.0f32;
            let mut had_elem = false;
            {
                let mut out_channels_recreated =
                    ChannelsMut::<f32>::from_row_buffers(&mut out_rbs, out_x0, 0, ROWS_OUT);
                let r_mut = out_channels_recreated.get_row_mut(mut_c, mut_r);
                if !r_mut.is_empty() {
                    r_mut[0] += 12345.0;
                    mutated_val = r_mut[0];
                    had_elem = true;
                }
            }
            if had_elem {
                assert_eq!(out_rbs[mut_c].get_row::<f32>(mut_r)[out_x0], mutated_val);
            }

            let mut sel_mutated_val = 0.0f32;
            let mut sel_had_elem = false;
            {
                let mut out_channels_recreated =
                    ChannelsMut::<f32>::from_row_buffers(&mut out_rbs, out_x0, 0, ROWS_OUT);
                let mut sel_mut = out_channels_recreated.select_row(mut_r);
                assert_eq!(sel_mut.len(), CHANS_OUT);
                assert_eq!(sel_mut.rows_per_channel(), 1);
                let row_sel = sel_mut.get_row_mut(mut_c, 0);
                if !row_sel.is_empty() {
                    row_sel[0] += 54321.0;
                    sel_mutated_val = row_sel[0];
                    sel_had_elem = true;
                }
            }
            if sel_had_elem {
                assert_eq!(
                    out_rbs[mut_c].get_row::<f32>(mut_r)[out_x0],
                    sel_mutated_val
                );
            }

            // 6. Interleaved stores (SCALE = 2)
            let inter_xsize = if xsize > 8 { 8 } else { xsize };
            let mut inter_in_rbs = (0..2)
                .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 0, 0, inter_xsize + 8).unwrap())
                .collect::<Vec<_>>();
            let in_x0_inter = RowBuffer::x0_offset::<f32>();
            for x in 0..(inter_xsize + 8) {
                inter_in_rbs[0].get_row_mut::<f32>(0)[in_x0_inter + x] = 11.0;
                inter_in_rbs[1].get_row_mut::<f32>(0)[in_x0_inter + x] = 22.0;
            }
            let inter_refs: Vec<&RowBuffer> = inter_in_rbs.iter().collect();
            let inter_channels_in =
                Channels::<f32>::from_row_buffers(&inter_refs, in_x0_inter, 0, 0, 1);

            let mut inter_out_rb =
                RowBuffer::new(DataTypeTag::F32, 0, 0, 0, (inter_xsize + 8) * 2).unwrap();
            let out_x0_inter = RowBuffer::x0_offset::<f32>();
            let mut inter_channels_out = ChannelsMut::<f32>::from_row_buffers(
                std::slice::from_mut(&mut inter_out_rb),
                out_x0_inter,
                0,
                1,
            );

            ForEachChunk::<2, 1, 0, 1, 1, 2>::run(
                d,
                inter_xsize,
                &inter_channels_in,
                &mut inter_channels_out,
                |_x, in_v, out_v| {
                    let v0 = in_v.load::<0, 0, 0>();
                    let v1 = in_v.load::<1, 0, 0>();
                    out_v.store_interleaved::<0, 0>([v0, v1]);
                },
            );

            for x in 0..inter_xsize {
                assert_eq!(inter_out_rb.get_row::<f32>(0)[out_x0_inter + x * 2], 11.0);
                assert_eq!(
                    inter_out_rb.get_row::<f32>(0)[out_x0_inter + x * 2 + 1],
                    22.0
                );
            }

            // 7. Exercise RowBuffer interop
            let rb_in = RowBuffer::new(DataTypeTag::F32, 0, 1, 0, xsize + 16).unwrap();
            let mut rb_out = RowBuffer::new(DataTypeTag::F32, 0, 1, 0, xsize + 16).unwrap();
            let rb_stride = rb_in.stride_elements::<f32>();
            let rb_c_in = Channels::<f32>::from_row_buffers(&[&rb_in], 0, 0, 0, 2);
            let mut rb_c_out =
                ChannelsMut::<f32>::from_row_buffers(std::slice::from_mut(&mut rb_out), 0, 0, 2);

            assert_eq!(rb_c_in.len(), 1);
            assert_eq!(rb_c_in.rows_per_channel(), 1);
            assert_eq!(rb_c_out.len(), 1);
            assert_eq!(rb_c_out.rows_per_channel(), 2);

            let r0 = rb_c_in.get_row(0, 0);
            assert_eq!(r0.len(), rb_stride);
            let r1_mut = rb_c_out.get_row_mut(0, 1);
            r1_mut[0] = 42.42;
            assert_eq!(rb_out.get_row::<f32>(1)[0], 42.42);

            Ok(())
        })
        .budget_ms(if cfg!(miri) { 500 } else { 2000 });
    }

    test_all_instruction_sets!(test_simd_stencil_processing_non_multiple_vl);
    test_all_instruction_sets!(test_simd_interleaved_stores_all_scales);
    test_all_instruction_sets!(test_simd_data_types_all_instruction_sets);
}
