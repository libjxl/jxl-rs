// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::common::precompute_references;
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::{Histograms, SymbolReader},
    error::Result,
    frame::modular::{
        IMAGE_OFFSET, IMAGE_PADDING, ModularChannel, Tree,
        decode::{common::make_pixel, specialized_trees::run_on_specialized_tree},
        predict::{PredictionData, WeightedPredictorState},
        tree::{NUM_NONREF_PROPERTIES, PROPERTIES_PER_PREVCHAN, predict},
    },
    headers::modular::GroupHeader,
    image::Image,
    util::tracing_wrappers::*,
};

const SMALL_CHANNEL_THRESHOLD: usize = 64;

pub(super) trait ModularChannelDecoder {
    #[inline(always)]
    fn needs_toptop(&self) -> bool {
        true
    }

    fn init_row(&mut self, _buffers: &mut [&mut ModularChannel], _chan: usize, _y: usize) {}

    fn decode_one(
        &mut self,
        _prediction_data: PredictionData,
        _pos: (usize, usize),
        _reader: &mut SymbolReader,
        _br: &mut BitReader,
        _histograms: &Histograms,
    ) -> i32 {
        0
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
        const { assert!(IMAGE_OFFSET.1 == 2) };
        let [row, row_top, row_toptop] =
            buffers[chan].data.distinct_full_rows_mut([y + 2, y + 1, y]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + xsize];
        let row_top = &mut row_top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + xsize];
        let row_toptop = &mut row_toptop[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + xsize];

        let do_decode_cold = {
            #[inline(never)]
            |decoder: &mut Self,
             row: &mut [i32],
             row_top: &mut [i32],
             row_toptop: &mut [i32],
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

        if y < 2 {
            for x in 0..xsize {
                do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
            }
        } else {
            let mut last = 0;
            let mut prediction_data = PredictionData::default();
            for x in 0..2 {
                (prediction_data, last) =
                    do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
            }
            for (x, r) in row.iter_mut().enumerate().skip(2).take(xsize - 4) {
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
            for x in xsize - 2..xsize {
                do_decode_cold(self, row, row_top, row_toptop, (x, y), reader, br);
            }
        }
    }
}

// General case decoder, for small buffers for which it's not worth trying to detect tree special cases.
#[inline(never)]
fn decode_modular_channel_small(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    let size = buffers[chan].data.size();
    let mut wp_state = WeightedPredictorState::new(&header.wp_header, size.0);
    let mut num_ref_props = tree
        .max_property_count()
        .saturating_sub(NUM_NONREF_PROPERTIES);
    // The precompute_references function stores 4 values per reference property (offset + 0,1,2,3)
    num_ref_props = num_ref_props.div_ceil(PROPERTIES_PER_PREVCHAN) * PROPERTIES_PER_PREVCHAN;
    let mut references = Image::<i32>::new((num_ref_props, size.0))?;
    let num_properties = NUM_NONREF_PROPERTIES + num_ref_props;

    const { assert!(IMAGE_OFFSET.1 == 2) };

    for y in 0..size.1 {
        precompute_references(buffers, chan, y, &mut references);
        let mut property_buffer: Vec<i32> = vec![0; num_properties];
        property_buffer[0] = chan as i32;
        property_buffer[1] = stream_id as i32;
        let [row, row_top, row_toptop] =
            buffers[chan].data.distinct_full_rows_mut([y + 2, y + 1, y]);
        let row = &mut row[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let row_top = &mut row_top[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        let row_toptop = &mut row_toptop[IMAGE_OFFSET.0..IMAGE_OFFSET.0 + size.0];
        for x in 0..size.0 {
            let prediction_data = PredictionData::get_rows(row, row_top, row_toptop, x, y);
            let prediction_result = predict(
                &tree.nodes,
                prediction_data,
                Some(&mut wp_state),
                x,
                y,
                &references,
                &mut property_buffer,
            );
            let dec = reader.read_signed(&tree.histograms, br, prediction_result.context as usize);
            let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
            row[x] = val;
            wp_state.update_errors(val, (x, y));
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
    if size.0 <= IMAGE_PADDING.0
        || size.1 <= IMAGE_PADDING.1
        || size.0 * size.1 <= SMALL_CHANNEL_THRESHOLD
    {
        return decode_modular_channel_small(buffers, chan, stream_id, header, tree, reader, br);
    }

    assert_eq!(buffers[chan].data.padding().1, IMAGE_PADDING.1);
    assert!(buffers[chan].data.padding().0 >= IMAGE_PADDING.0);
    assert_eq!(buffers[chan].data.offset(), IMAGE_OFFSET);

    // We now know the channel has size at least IMAGE_PADDING.
    run_on_specialized_tree(tree, chan, stream_id, size.0, header, {
        #[inline(never)]
        |t| {
            let size = buffers[chan].data.size();
            debug_assert!(size.0 >= 4);
            debug_assert!(size.1 >= 2);
            for y in 0..size.1 {
                t.decode_row(buffers, chan, &tree.histograms, reader, br, y, size.0);
            }
            Ok(())
        }
    })?;
    br.check_for_error()
}
