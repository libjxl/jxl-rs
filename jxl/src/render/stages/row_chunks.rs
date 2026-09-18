// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use jxl_simd::{F32SimdVec, I32SimdVec, SimdDescriptor};

pub(super) trait VecFor<D: SimdDescriptor>: 'static + Copy {
    type Vec: Copy;
    const LEN: usize;
    fn load(d: D, slice: &[Self]) -> Self::Vec;
    fn store(vec: Self::Vec, slice: &mut [Self]);
}

impl<D: SimdDescriptor> VecFor<D> for f32 {
    type Vec = D::F32Vec;
    const LEN: usize = D::F32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::F32Vec::load(d, slice)
    }

    #[inline(always)]
    fn store(vec: Self::Vec, slice: &mut [Self]) {
        vec.store(slice);
    }
}

impl<D: SimdDescriptor> VecFor<D> for i32 {
    type Vec = D::I32Vec;
    const LEN: usize = D::I32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::I32Vec::load(d, slice)
    }

    #[inline(always)]
    fn store(vec: Self::Vec, slice: &mut [Self]) {
        vec.store(slice);
    }
}

impl<D: SimdDescriptor> VecFor<D> for u16 {
    type Vec = D::I32Vec;
    const LEN: usize = D::I32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::I32Vec::load_from_u16(d, slice)
    }

    #[inline(always)]
    fn store(vec: Self::Vec, slice: &mut [Self]) {
        vec.store_u16(slice);
    }
}

impl<D: SimdDescriptor> VecFor<D> for i16 {
    type Vec = D::I32Vec;
    const LEN: usize = D::I32Vec::LEN;

    #[inline(always)]
    fn load(d: D, slice: &[Self]) -> Self::Vec {
        D::I32Vec::load_from_i16(d, slice)
    }

    #[inline(always)]
    fn store(_vec: Self::Vec, _slice: &mut [Self]) {
        unimplemented!("storing into i16 is not supported");
    }
}

pub(super) trait ChunkItem<'a> {
    type Item;
}

pub(super) trait GetChunk<D: SimdDescriptor>: for<'a> ChunkItem<'a> {
    fn num_chunks(&self) -> usize;

    /// # Safety
    /// `i < self.num_chunks()` must hold.
    unsafe fn get_chunk_unchecked(&mut self, i: usize) -> <Self as ChunkItem<'_>>::Item;
}

pub(super) trait Chunkable<D: SimdDescriptor> {
    type ChunkProvider: GetChunk<D>;

    fn into_provider(self, d: D, num_chunks: usize) -> Self::ChunkProvider;
}

#[derive(Clone, Copy)]
pub(super) struct ChunkProvider<'a, D: SimdDescriptor, T: VecFor<D> = f32> {
    // Safety invariant: self.data.len() >= self.num_chunks * T::LEN.
    data: &'a [T],
    d: D,
    num_chunks: usize,
}

impl<'a, D: SimdDescriptor, T: VecFor<D>> ChunkItem<'_> for ChunkProvider<'a, D, T> {
    type Item = T::Vec;
}

impl<'a, D: SimdDescriptor, T: VecFor<D>> GetChunk<D> for ChunkProvider<'a, D, T> {
    #[inline(always)]
    fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    #[inline(always)]
    unsafe fn get_chunk_unchecked(&mut self, i: usize) -> <Self as ChunkItem<'_>>::Item {
        let start = i * T::LEN;
        // SAFETY: The caller upholds i < self.num_chunks. By the safety invariant,
        // self.data.len() >= self.num_chunks * T::LEN. Since i < self.num_chunks,
        // start + T::LEN = (i + 1) * T::LEN <= self.data.len().
        let sub = unsafe { self.data.get_unchecked(start..start + T::LEN) };
        T::load(self.d, sub)
    }
}

impl<'a, D: SimdDescriptor, T: VecFor<D>> Chunkable<D> for &'a [T] {
    type ChunkProvider = ChunkProvider<'a, D, T>;

    #[inline(always)]
    fn into_provider(self, d: D, num_chunks: usize) -> Self::ChunkProvider {
        const { assert!(T::LEN == D::F32Vec::LEN, "T::LEN must match F32Vec::LEN") };
        let required = num_chunks
            .checked_mul(T::LEN)
            .expect("slice length calculation overflowed usize");
        assert!(
            self.len() >= required,
            "slice too short: len {} < required {required} (num_chunks={num_chunks})",
            self.len(),
        );
        // Safety note: establishes safety invariant self.data.len() >= self.num_chunks * T::LEN.
        ChunkProvider {
            data: self,
            d,
            num_chunks,
        }
    }
}

pub(super) struct ChunkWriter<'a, D: SimdDescriptor, const SCALE: usize = 1, T = f32> {
    // Safety invariant: self.slice.len() >= SCALE * D::F32Vec::LEN.
    slice: &'a mut [T],
    d: D,
}

impl<'a, D: SimdDescriptor, T: VecFor<D>> ChunkWriter<'a, D, 1, T> {
    #[inline(always)]
    pub(super) fn write(&mut self, vec: T::Vec) {
        T::store(vec, self.slice);
    }

    #[inline(always)]
    pub(super) fn read(&self) -> T::Vec {
        T::load(self.d, self.slice)
    }
}

impl<'a, D: SimdDescriptor> ChunkWriter<'a, D, 1, u8> {
    #[inline(always)]
    pub(super) fn round_store_u8(&mut self, vec: D::F32Vec) {
        vec.round_store_u8(self.slice);
    }

    #[inline(always)]
    pub(super) fn store_u8(&mut self, vec: D::I32Vec) {
        vec.store_u8(self.slice);
    }
}

impl<'a, D: SimdDescriptor> ChunkWriter<'a, D, 1, u16> {
    #[inline(always)]
    pub(super) fn round_store_u16(&mut self, vec: D::F32Vec) {
        vec.round_store_u16(self.slice);
    }
}

pub(super) trait StoreInterleaved<D: SimdDescriptor, const SCALE: usize> {
    fn store_interleaved(&mut self, values: [D::F32Vec; SCALE]);
}

impl<'a, D: SimdDescriptor> StoreInterleaved<D, 2> for ChunkWriter<'a, D, 2> {
    #[inline(always)]
    fn store_interleaved(&mut self, [a, b]: [D::F32Vec; 2]) {
        D::F32Vec::store_interleaved_2(a, b, self.slice);
    }
}

impl<'a, D: SimdDescriptor> StoreInterleaved<D, 4> for ChunkWriter<'a, D, 4> {
    #[inline(always)]
    fn store_interleaved(&mut self, [a, b, c, d]: [D::F32Vec; 4]) {
        D::F32Vec::store_interleaved_4(a, b, c, d, self.slice);
    }
}

impl<'a, D: SimdDescriptor> StoreInterleaved<D, 8> for ChunkWriter<'a, D, 8> {
    #[inline(always)]
    fn store_interleaved(&mut self, [v0, v1, v2, v3, v4, v5, v6, v7]: [D::F32Vec; 8]) {
        D::F32Vec::store_interleaved_8(v0, v1, v2, v3, v4, v5, v6, v7, self.slice);
    }
}

pub(super) struct MutChunkProvider<'a, D: SimdDescriptor, const SCALE: usize = 1, T = f32> {
    // Safety invariant: self.data.len() >= self.num_chunks * SCALE * D::F32Vec::LEN.
    data: &'a mut [T],
    d: D,
    num_chunks: usize,
}

impl<'a, 'b, D: SimdDescriptor, const SCALE: usize, T: 'static> ChunkItem<'b>
    for MutChunkProvider<'a, D, SCALE, T>
{
    type Item = ChunkWriter<'b, D, SCALE, T>;
}

impl<'a, D: SimdDescriptor, const SCALE: usize, T: 'static> GetChunk<D>
    for MutChunkProvider<'a, D, SCALE, T>
{
    #[inline(always)]
    fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    #[inline(always)]
    unsafe fn get_chunk_unchecked(&mut self, i: usize) -> <Self as ChunkItem<'_>>::Item {
        let chunk_len = SCALE * D::F32Vec::LEN;
        let start = i * chunk_len;
        // SAFETY: The caller upholds i < self.num_chunks. By the safety invariant,
        // self.data.len() >= self.num_chunks * SCALE * D::F32Vec::LEN. Since i < self.num_chunks,
        // start + SCALE * D::F32Vec::LEN = (i + 1) * SCALE * D::F32Vec::LEN <= self.data.len().
        let sub = unsafe { self.data.get_unchecked_mut(start..start + chunk_len) };
        ChunkWriter {
            slice: sub,
            d: self.d,
        }
    }
}

pub(super) struct ScaledChunkMut<'a, const SCALE: usize, T = f32>(pub(super) &'a mut [T]);

impl<'a, D: SimdDescriptor, const SCALE: usize, T: 'static> Chunkable<D>
    for ScaledChunkMut<'a, SCALE, T>
{
    type ChunkProvider = MutChunkProvider<'a, D, SCALE, T>;

    #[inline(always)]
    fn into_provider(self, d: D, num_chunks: usize) -> Self::ChunkProvider {
        let chunk_len = SCALE * D::F32Vec::LEN;
        let required = num_chunks
            .checked_mul(chunk_len)
            .expect("ScaledChunkMut buffer calculation overflowed usize");
        assert!(
            self.0.len() >= required,
            "ScaledChunkMut slice too short: len {} < required {required} (num_chunks={num_chunks}, SCALE={SCALE})",
            self.0.len(),
        );
        // Safety note: establishes safety invariant self.data.len() >= self.num_chunks * SCALE * D::F32Vec::LEN.
        MutChunkProvider {
            data: self.0,
            d,
            num_chunks,
        }
    }
}

impl<'a, D: SimdDescriptor, T: 'static> Chunkable<D> for &'a mut [T] {
    type ChunkProvider = MutChunkProvider<'a, D, 1, T>;

    #[inline(always)]
    fn into_provider(self, d: D, num_chunks: usize) -> Self::ChunkProvider {
        ScaledChunkMut::<1, T>(self).into_provider(d, num_chunks)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Window<'a, const EXTRA: usize, T = f32>(pub(super) &'a [T]);

pub(super) struct WindowChunkProvider<'a, D: SimdDescriptor, const EXTRA: usize, T: VecFor<D> = f32>
{
    // Safety invariant: either self.num_chunks == 0, or
    // self.data.len() >= self.num_chunks * T::LEN + EXTRA.
    data: &'a [T],
    d: D,
    num_chunks: usize,
}

impl<'a, 'b, D: SimdDescriptor, const EXTRA: usize, T: VecFor<D>> ChunkItem<'b>
    for WindowChunkProvider<'a, D, EXTRA, T>
{
    type Item = WindowAccessor<'b, D, EXTRA, T>;
}

impl<'a, D: SimdDescriptor, const EXTRA: usize, T: VecFor<D>> GetChunk<D>
    for WindowChunkProvider<'a, D, EXTRA, T>
{
    #[inline(always)]
    fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    #[inline(always)]
    unsafe fn get_chunk_unchecked(&mut self, i: usize) -> <Self as ChunkItem<'_>>::Item {
        let start = i * T::LEN;
        // SAFETY: The caller upholds i < self.num_chunks, which implies self.num_chunks >= 1.
        // By the safety invariant, self.data.len() >= self.num_chunks * T::LEN + EXTRA.
        // Since i <= self.num_chunks - 1, start + T::LEN + EXTRA =
        // (i + 1) * T::LEN + EXTRA <= self.data.len().
        let sub = unsafe { self.data.get_unchecked(start..start + T::LEN + EXTRA) };
        WindowAccessor {
            slice: sub,
            d: self.d,
        }
    }
}

impl<'a, D: SimdDescriptor, const EXTRA: usize, T: VecFor<D>> Chunkable<D>
    for Window<'a, EXTRA, T>
{
    type ChunkProvider = WindowChunkProvider<'a, D, EXTRA, T>;

    #[inline(always)]
    fn into_provider(self, d: D, num_chunks: usize) -> Self::ChunkProvider {
        const { assert!(T::LEN == D::F32Vec::LEN, "T::LEN must match F32Vec::LEN") };
        let required = if num_chunks == 0 {
            0
        } else {
            num_chunks
                .checked_mul(T::LEN)
                .and_then(|v| v.checked_add(EXTRA))
                .expect("Window buffer calculation overflowed usize")
        };
        assert!(
            self.0.len() >= required,
            "Window slice too short: len {} < required {required} (num_chunks={num_chunks}, EXTRA={EXTRA})",
            self.0.len(),
        );
        // Safety note: establishes safety invariant: if num_chunks == 0, true;
        // otherwise self.data.len() >= self.num_chunks * T::LEN + EXTRA.
        WindowChunkProvider {
            data: self.0,
            d,
            num_chunks,
        }
    }
}

pub(super) struct WindowAccessor<'a, D: SimdDescriptor, const EXTRA: usize, T: VecFor<D> = f32> {
    // Safety invariant: self.slice.len() >= T::LEN + EXTRA.
    slice: &'a [T],
    d: D,
}

impl<'a, D: SimdDescriptor, const EXTRA: usize, T: VecFor<D>> WindowAccessor<'a, D, EXTRA, T> {
    #[inline(always)]
    pub(super) fn get<const OFFSET: usize>(&self) -> T::Vec {
        const { assert!(OFFSET <= EXTRA, "Window offset exceeds EXTRA") };
        // SAFETY: OFFSET <= EXTRA is checked by the compile-time assertion.
        // By the safety invariant, self.slice.len() >= T::LEN + EXTRA.
        // Therefore OFFSET + T::LEN <= EXTRA + T::LEN <= self.slice.len().
        let sub = unsafe { self.slice.get_unchecked(OFFSET..OFFSET + T::LEN) };
        T::load(self.d, sub)
    }
}

macro_rules! impl_chunkable_array {
    ($n:literal, $($idx:ident),+) => {
        impl<'b, T: ChunkItem<'b>> ChunkItem<'b> for [T; $n] {
            type Item = [T::Item; $n];
        }

        impl<D: SimdDescriptor, P: GetChunk<D>> GetChunk<D> for [P; $n] {
            #[inline(always)]
            fn num_chunks(&self) -> usize {
                let mut min = self[0].num_chunks();
                let mut idx = 1;
                while idx < $n {
                    min = min.min(self[idx].num_chunks());
                    idx += 1;
                }
                min
            }

            #[inline(always)]
            unsafe fn get_chunk_unchecked(&mut self, i: usize) -> <Self as ChunkItem<'_>>::Item {
                let [$($idx),+] = self;
                // SAFETY: The caller upholds i < self.num_chunks(). By the definition of num_chunks()
                // for arrays, self.num_chunks() <= self[k].num_chunks() for each element k.
                // Therefore i < self[k].num_chunks() holds for each element.
                unsafe {
                    [$($idx.get_chunk_unchecked(i)),+]
                }
            }
        }

        impl<D: SimdDescriptor, T: Chunkable<D>> Chunkable<D> for [T; $n] {
            type ChunkProvider = [T::ChunkProvider; $n];

            #[inline(always)]
            fn into_provider(self, d: D, num_chunks: usize) -> Self::ChunkProvider {
                let [$($idx),+] = self;
                [$($idx.into_provider(d, num_chunks)),+]
            }
        }
    };
}

impl_chunkable_array!(2, x0, x1);
impl_chunkable_array!(3, x0, x1, x2);
impl_chunkable_array!(4, x0, x1, x2, x3);
impl_chunkable_array!(5, x0, x1, x2, x3, x4);
impl_chunkable_array!(7, x0, x1, x2, x3, x4, x5, x6);
impl_chunkable_array!(8, x0, x1, x2, x3, x4, x5, x6, x7);

macro_rules! impl_chunkable_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<'b, $($T: ChunkItem<'b>),+> ChunkItem<'b> for ($($T,)+) {
            type Item = ($($T::Item,)+);
        }

        impl<Desc: SimdDescriptor, $($T: GetChunk<Desc>),+> GetChunk<Desc> for ($($T,)+) {
            #[inline(always)]
            fn num_chunks(&self) -> usize {
                let mut min = self.0.num_chunks();
                $(
                    min = min.min(self.$idx.num_chunks());
                )+
                min
            }

            #[inline(always)]
            unsafe fn get_chunk_unchecked(&mut self, i: usize) -> <Self as ChunkItem<'_>>::Item {
                // SAFETY: The caller upholds i < self.num_chunks(). By the definition of num_chunks()
                // for tuples, self.num_chunks() <= self.$idx.num_chunks() for each element.
                // Therefore i < self.$idx.num_chunks() holds for each element.
                unsafe { ($(self.$idx.get_chunk_unchecked(i),)+) }
            }
        }

        impl<Desc: SimdDescriptor, $($T: Chunkable<Desc>),+> Chunkable<Desc> for ($($T,)+) {
            type ChunkProvider = ($($T::ChunkProvider,)+);

            #[inline(always)]
            fn into_provider(self, d: Desc, num_chunks: usize) -> Self::ChunkProvider {
                ($(self.$idx.into_provider(d, num_chunks),)+)
            }
        }
    };
}

impl_chunkable_tuple!(0: A, 1: B);
impl_chunkable_tuple!(0: A, 1: B, 2: C);
impl_chunkable_tuple!(0: A, 1: B, 2: C, 3: D);
impl_chunkable_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_chunkable_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_chunkable_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);

/// Iterates over row slices in chunks of SIMD vector length for `xsize` pixels,
/// safely eliding bounds checks inside the loop.
///
/// Pre-checks buffer bounds once upfront, and then invokes `f(x, chunks)`
/// for each vector step `0..xsize.div_ceil(D::F32Vec::LEN)`.
/// `x` is the pixel offset `i * D::F32Vec::LEN`.
///
/// # Buffer Padding Requirements
/// All slices passed to `for_each_chunk` must have sufficient length for full vector chunks:
/// - Disjoint slices must have length >= `xsize.div_ceil(D::F32Vec::LEN) * D::F32Vec::LEN`.
/// - `Window<EXTRA>` slices must have length >= `xsize.div_ceil(D::F32Vec::LEN) * D::F32Vec::LEN + EXTRA`.
/// - `ScaledChunkMut<SCALE>` slices must have length >= `xsize.div_ceil(D::F32Vec::LEN) * SCALE * D::F32Vec::LEN`.
#[inline(always)]
pub(super) fn for_each_chunk<D: SimdDescriptor, S: Chunkable<D>, F>(
    d: D,
    xsize: usize,
    slices: S,
    mut f: F,
) where
    F: FnMut(usize, <S::ChunkProvider as ChunkItem<'_>>::Item),
{
    let vector_length = D::F32Vec::LEN;
    assert!(vector_length > 0, "vector_length must be positive");
    let num_vec = xsize.div_ceil(vector_length);
    if num_vec == 0 {
        return;
    }

    let mut provider = slices.into_provider(d, num_vec);
    assert_eq!(
        provider.num_chunks(),
        num_vec,
        "provider chunk count mismatch: expected {num_vec}, got {}",
        provider.num_chunks()
    );
    for i in 0..num_vec {
        // SAFETY: Loop condition ensures i < num_vec. The assertion above established that
        // provider.num_chunks() == num_vec, therefore i < provider.num_chunks().
        let chunk = unsafe { provider.get_chunk_unchecked(i) };
        f(i * vector_length, chunk);
    }
}

#[cfg(test)]
mod test {
    use jxl_simd::ScalarDescriptor;

    use super::*;

    #[test]
    fn test_empty_xsize() {
        let d = ScalarDescriptor::new().unwrap();
        let mut count = 0;
        let mut out = [0.0f32; 8];
        for_each_chunk(d, 0, &mut out[..], |_x, _chunk| {
            count += 1;
        });
        assert_eq!(count, 0);
    }

    #[test]
    fn test_single_slice_exact() {
        let d = ScalarDescriptor::new().unwrap();
        let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut visited_x = Vec::new();
        for_each_chunk(d, 8, &mut data[..], |x, mut chunk| {
            visited_x.push(x);
            let val = chunk.read();
            chunk.write(val * 2.0);
        });
        assert_eq!(visited_x, (0..8).collect::<Vec<_>>());
        assert_eq!(data, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_window_get() {
        let d = ScalarDescriptor::new().unwrap();
        let data = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut result = Vec::new();
        for_each_chunk(d, 3, Window::<2>(&data[..]), |_x, win| {
            result.push((win.get::<0>(), win.get::<1>(), win.get::<2>()));
        });
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], (10.0, 20.0, 30.0));
        assert_eq!(result[1], (20.0, 30.0, 40.0));
        assert_eq!(result[2], (30.0, 40.0, 50.0));
    }

    #[test]
    fn test_scaled_chunk_mut() {
        let d = ScalarDescriptor::new().unwrap();
        let mut out = [0.0f32; 8];
        for_each_chunk(d, 4, ScaledChunkMut::<2>(&mut out[..]), |_x, mut chunk| {
            chunk.store_interleaved([1.0, 2.0]);
        });
        assert_eq!(out, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_array_of_windows() {
        let d = ScalarDescriptor::new().unwrap();
        let r0 = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let r1 = [10.0f32, 11.0, 12.0, 13.0, 14.0];
        let r2 = [20.0f32, 21.0, 22.0, 23.0, 24.0];
        let arr = [
            Window::<2>(&r0[..]),
            Window::<2>(&r1[..]),
            Window::<2>(&r2[..]),
        ];

        let mut collected = Vec::new();
        for_each_chunk(d, 3, arr, |_x, wins| {
            collected.push((wins[0].get::<0>(), wins[1].get::<1>(), wins[2].get::<2>()));
        });
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], (0.0, 11.0, 22.0));
        assert_eq!(collected[1], (1.0, 12.0, 23.0));
        assert_eq!(collected[2], (2.0, 13.0, 24.0));
    }

    #[test]
    fn test_i16_and_u8() {
        let d = ScalarDescriptor::new().unwrap();
        let input = [10i16, 20, 30, 40];
        let mut out_u8 = [0u8; 4];
        let mut out_u16 = [0u16; 4];
        for_each_chunk(
            d,
            4,
            (&input[..], &mut out_u8[..], &mut out_u16[..]),
            |_x, (in_vec, mut out8, mut out16)| {
                out8.store_u8(in_vec);
                out16.round_store_u16(in_vec.as_f32());
            },
        );
        assert_eq!(out_u8, [10, 20, 30, 40]);
        assert_eq!(out_u16, [10, 20, 30, 40]);
    }

    #[test]
    fn test_arb() {
        arbtest::arbtest(|u| {
            let d = ScalarDescriptor::new().unwrap();
            let vector_length = 1; // ScalarDescriptor LEN is 1
            let xsize: usize = u.int_in_range(0..=32)?;
            let num_vec = xsize;

            let win_extra: usize = 2;
            let pad: usize = u.int_in_range(0..=8)?;

            let req_slice = num_vec;
            let req_win = if num_vec == 0 { 0 } else { num_vec + win_extra };
            let req_scaled_rw = num_vec * 2;

            // Buffers for tuple test:
            let buf_slice_ro: Vec<f32> =
                (0..(req_slice + pad)).map(|i| 1000.0 + i as f32).collect();
            let mut buf_slice_rw: Vec<f32> = vec![0.0; req_slice + pad];
            let buf_win: Vec<f32> = (0..(req_win + pad)).map(|i| 2000.0 + i as f32).collect();
            let mut buf_scaled_rw: Vec<f32> = vec![0.0; req_scaled_rw + pad];
            let buf_arr_win = [
                (0..(req_win + pad))
                    .map(|i| 4000.0 + i as f32)
                    .collect::<Vec<f32>>(),
                (0..(req_win + pad))
                    .map(|i| 5000.0 + i as f32)
                    .collect::<Vec<f32>>(),
            ];
            let mut buf_arr_rw_0: Vec<f32> = vec![0.0; req_slice + pad];
            let mut buf_arr_rw_1: Vec<f32> = vec![0.0; req_slice + pad];
            let mut buf_nested_rw_0_0: Vec<f32> = vec![0.0; req_slice + pad];
            let mut buf_nested_rw_0_1: Vec<f32> = vec![0.0; req_slice + pad];
            let mut buf_nested_rw_1_0: Vec<f32> = vec![0.0; req_slice + pad];
            let mut buf_nested_rw_1_1: Vec<f32> = vec![0.0; req_slice + pad];

            let mut count = 0;
            for_each_chunk(
                d,
                xsize,
                (
                    &buf_slice_ro[..],
                    &mut buf_slice_rw[..],
                    Window::<2>(&buf_win[..]),
                    ScaledChunkMut::<2>(&mut buf_scaled_rw[..]),
                    [
                        Window::<2>(&buf_arr_win[0][..]),
                        Window::<2>(&buf_arr_win[1][..]),
                    ],
                    [&mut buf_arr_rw_0[..], &mut buf_arr_rw_1[..]],
                    [
                        [&mut buf_nested_rw_0_0[..], &mut buf_nested_rw_0_1[..]],
                        [&mut buf_nested_rw_1_0[..], &mut buf_nested_rw_1_1[..]],
                    ],
                ),
                #[inline(always)]
                |x, (s_ro, mut s_rw, w, mut sc_rw, arr_w, mut arr_rw, mut nested_rw)| {
                    let step = count;
                    assert_eq!(x, step * vector_length);

                    assert_eq!(s_ro, buf_slice_ro[x]);
                    assert_eq!(w.get::<0>(), buf_win[x]);
                    assert_eq!(w.get::<1>(), buf_win[x + 1]);
                    assert_eq!(w.get::<2>(), buf_win[x + 2]);
                    assert_eq!(arr_w[0].get::<0>(), buf_arr_win[0][x]);
                    assert_eq!(arr_w[1].get::<0>(), buf_arr_win[1][x]);

                    s_rw.write(10000.0 + step as f32);
                    sc_rw.store_interleaved([20000.0 + step as f32, 20001.0 + step as f32]);
                    arr_rw[0].write(30000.0 + step as f32);
                    arr_rw[1].write(40000.0 + step as f32);
                    nested_rw[0][0].write(50000.0 + step as f32);
                    nested_rw[0][1].write(60000.0 + step as f32);
                    nested_rw[1][0].write(70000.0 + step as f32);
                    nested_rw[1][1].write(80000.0 + step as f32);

                    count += 1;
                },
            );

            assert_eq!(count, num_vec);

            for step in 0..num_vec {
                assert_eq!(buf_slice_rw[step], 10000.0 + step as f32);
                assert_eq!(buf_scaled_rw[step * 2], 20000.0 + step as f32);
                assert_eq!(buf_scaled_rw[step * 2 + 1], 20001.0 + step as f32);
                assert_eq!(buf_arr_rw_0[step], 30000.0 + step as f32);
                assert_eq!(buf_arr_rw_1[step], 40000.0 + step as f32);
                assert_eq!(buf_nested_rw_0_0[step], 50000.0 + step as f32);
                assert_eq!(buf_nested_rw_0_1[step], 60000.0 + step as f32);
                assert_eq!(buf_nested_rw_1_0[step], 70000.0 + step as f32);
                assert_eq!(buf_nested_rw_1_1[step], 80000.0 + step as f32);
            }

            Ok(())
        });
    }
}
