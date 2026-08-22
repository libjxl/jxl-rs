// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::common::precompute_references;
use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Histograms, SymbolReader};
use crate::error::Result;
use crate::frame::modular::decode::common::make_pixel;
use crate::frame::modular::decode::specialized_trees::run_on_specialized_tree;
use crate::frame::modular::predict::{PredictionData, WeightedPredictorState};
use crate::frame::modular::tree::{NUM_NONREF_PROPERTIES, PROPERTIES_PER_PREVCHAN, predict};
use crate::frame::modular::{ModularChannel, Tree};
use crate::headers::modular::{GroupHeader, WeightedHeader};
use crate::image::Image;
use crate::util::tracing_wrappers::*;

const SMALL_CHANNEL_THRESHOLD: usize = 64;

pub(super) trait ModularChannelDecoder {
    #[inline(always)]
    fn needs_toptop(&self) -> bool {
        true
    }

    fn init_row(&mut self, _buffers: &mut [&mut ModularChannel], _chan: usize, _y: usize) {}

    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        pos: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn decode_row_slices(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        row_toptop: &[i32],
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
    ) {
        let do_decode_cold = {
            #[inline(never)]
            |decoder: &mut Self,
             row: &mut [i32],
             row_top: &[i32],
             row_toptop: &[i32],
             pos: (usize, usize),
             reader: &mut SymbolReader,
             br: &mut BitReader|
             -> (PredictionData, i32) {
                let prediction_data =
                    PredictionData::get_rows(row, row_top, row_toptop, pos.0, pos.1);
                let val = decoder.decode_one(prediction_data, pos, reader, br, histograms);
                row[pos.0] = val;
                (prediction_data, val)
            }
        };

        let (x0, x1) = if y < 2 { (0, 0) } else { (2, xsize - 2) };

        let mut last = 0;
        let mut prediction_data = PredictionData::default();
        for x in 0..x0 {
            (prediction_data, last) =
                do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
        }
        for (x, r) in row.iter_mut().enumerate().skip(x0).take(x1 - x0) {
            prediction_data = prediction_data.update_for_interior_row(
                row_top,
                row_toptop,
                x,
                last,
                self.needs_toptop(),
            );
            let val = self.decode_one(prediction_data, (x, y), reader, br, histograms);
            *r = val;
            last = val;
        }
        for x in x1..xsize {
            do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn decode_row_slices_i16(
        &mut self,
        row: &mut [i16],
        row_top: &[i16],
        row_toptop: &[i16],
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
    ) {
        let do_decode_cold = {
            #[inline(never)]
            |decoder: &mut Self,
             row: &mut [i16],
             row_top: &[i16],
             row_toptop: &[i16],
             pos: (usize, usize),
             reader: &mut SymbolReader,
             br: &mut BitReader|
             -> (PredictionData, i32) {
                let prediction_data =
                    PredictionData::get_rows_i16(row, row_top, row_toptop, pos.0, pos.1);
                let val = decoder.decode_one(prediction_data, pos, reader, br, histograms);
                row[pos.0] = val as i16;
                (prediction_data, val)
            }
        };

        let (x0, x1) = if y < 2 { (0, 0) } else { (2, xsize - 2) };

        let mut last = 0;
        let mut prediction_data = PredictionData::default();
        for x in 0..x0 {
            (prediction_data, last) =
                do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
        }
        for (x, r) in row.iter_mut().enumerate().skip(x0).take(x1 - x0) {
            prediction_data = prediction_data.update_for_interior_row_i16(
                row_top,
                row_toptop,
                x,
                last,
                self.needs_toptop(),
            );
            let val = self.decode_one(prediction_data, (x, y), reader, br, histograms);
            *r = val as i16;
            last = val;
        }
        for x in x1..xsize {
            do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn decode_row(
        &mut self,
        buffers: &mut [&mut ModularChannel],
        chan: usize,
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
    ) {
        self.init_row(buffers, chan, y);
        let (row, row_top, row_toptop) = match y {
            0 => (buffers[chan].data.as_i32_mut().row_mut(0), &[][..], &[][..]),
            1 => {
                let [row, row_top] = buffers[chan].data.as_i32_mut().distinct_rows_mut([1, 0]);
                (row, &*row_top, &[][..])
            }
            _ => {
                let [row, row_top, row_toptop] =
                    buffers[chan].data.as_i32_mut().distinct_rows_mut([y, y - 1, y - 2]);
                (row, &*row_top, &*row_toptop)
            }
        };
        self.decode_row_slices(row, row_top, row_toptop, histograms, reader, br, y, xsize);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn decode_row_i16(
        &mut self,
        buffers: &mut [&mut ModularChannel],
        chan: usize,
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
    ) {
        self.init_row(buffers, chan, y);
        let (row, row_top, row_toptop) = match y {
            0 => (buffers[chan].data.as_i16_mut().row_mut(0), &[][..], &[][..]),
            1 => {
                let [row, row_top] = buffers[chan].data.as_i16_mut().distinct_rows_mut([1, 0]);
                (row, &*row_top, &[][..])
            }
            _ => {
                let [row, row_top, row_toptop] =
                    buffers[chan].data.as_i16_mut().distinct_rows_mut([y, y - 1, y - 2]);
                (row, &*row_top, &*row_toptop)
            }
        };
        self.decode_row_slices_i16(row, row_top, row_toptop, histograms, reader, br, y, xsize);
    }
}

struct FullTree<'a> {
    tree: &'a Tree,
    references: Image<i32>,
    property_buffer: Box<[i32; 256]>,
    wp_state: WeightedPredictorState,
}

impl<'a> FullTree<'a> {
    fn new(
        tree: &'a Tree,
        wp_header: &WeightedHeader,
        channel: usize,
        stream: usize,
        xsize: usize,
    ) -> Result<Self> {
        let num_ref_props = tree
            .num_properties
            .saturating_sub(NUM_NONREF_PROPERTIES)
            .next_multiple_of(PROPERTIES_PER_PREVCHAN);
        let references = Image::<i32>::new((num_ref_props, xsize))?;
        let mut property_buffer = Box::new([0; 256]);

        property_buffer[0] = channel as i32;
        property_buffer[1] = stream as i32;

        Ok(Self {
            tree,
            references,
            property_buffer,
            wp_state: WeightedPredictorState::new(wp_header, xsize),
        })
    }
}

impl<'a> ModularChannelDecoder for FullTree<'a> {
    fn init_row(&mut self, buffers: &mut [&mut ModularChannel], chan: usize, y: usize) {
        precompute_references(buffers, chan, y, &mut self.references);
        self.property_buffer[9] = 0;
    }

    fn decode_one(
        &mut self,
        _prediction_data: PredictionData,
        _pos: (usize, usize),
        _reader: &mut SymbolReader,
        _br: &mut BitReader,
        _histograms: &Histograms,
    ) -> i32 {
        unreachable!()
    }

    fn decode_row_slices(
        &mut self,
        row: &mut [i32],
        row_top: &[i32],
        row_toptop: &[i32],
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
    ) {
        let _ = (row_top, row_toptop);
        for x in 0..xsize {
            let prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let prediction_result = predict(
                self.tree,
                prediction_data,
                Some(&mut self.wp_state),
                x,
                y,
                &self.references,
                &mut self.property_buffer[..],
            );
            let dec = reader.read_signed(histograms, br, prediction_result.context as usize);
            let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
            self.wp_state.update_errors(val, (x, y));
            row[x] = val;
        }
    }

    fn decode_row_slices_i16(
        &mut self,
        row: &mut [i16],
        row_top: &[i16],
        row_toptop: &[i16],
        histograms: &Histograms,
        reader: &mut SymbolReader,
        br: &mut BitReader,
        y: usize,
        xsize: usize,
    ) {
        for x in 0..xsize {
            let prediction_data = PredictionData::get_rows_i16(row, row_top, row_toptop, x, y);
            let prediction_result = predict(
                self.tree,
                prediction_data,
                Some(&mut self.wp_state),
                x,
                y,
                &self.references,
                &mut self.property_buffer[..],
            );
            let dec = reader.read_signed(histograms, br, prediction_result.context as usize);
            let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
            self.wp_state.update_errors(val, (x, y));
            row[x] = val as i16;
        }
    }
}

#[inline(never)]
fn decode_modular_channel_impl(
    t: &mut dyn ModularChannelDecoder,
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    histo: &Histograms,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    let size = buffers[chan].data.size();
    let xsize = size.0;
    if buffers[chan].data.is_16bit() {
        for y in 0..size.1 {
            t.decode_row_i16(buffers, chan, histo, reader, br, y, xsize);
        }
    } else {
        for y in 0..size.1 {
            t.decode_row(buffers, chan, histo, reader, br, y, xsize);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(buffers, reader, tree))]
pub(super) fn decode_modular_channel(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    debug!("reading channel");
    let size = buffers[chan].data.size();
    if size.0 <= 4 || size.1 <= 2 || size.0 * size.1 <= SMALL_CHANNEL_THRESHOLD {
        let mut decoder = FullTree::new(tree, &header.wp_header, chan, stream_id, size.0)?;
        return decode_modular_channel_impl(
            &mut decoder,
            buffers,
            chan,
            &tree.histograms,
            reader,
            br,
        );
    }

    run_on_specialized_tree(tree, chan, stream_id, size.0, header, {
        |t| decode_modular_channel_impl(t, buffers, chan, &tree.histograms, reader, br)
    })?;
    br.check_for_error()
}
