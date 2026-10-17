// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests for the public TOC-inspection API (`toc_num_entries`, `toc_entry`,
//! `frame_data_size`, `frame_data_offset`) and for `num_completed_passes`.
//!
//! The API exists so a progressive streamer can split a JXL frame at section
//! boundaries without decoding it. That makes the byte offsets load-bearing:
//! an offset that is off by even one byte silently corrupts every split. The
//! incremental-parse tests below are the ones that matter: the frame header
//! and TOC can be read across several `process()` calls, and a parser that
//! recomputes rather than accumulates its byte counters gets the offset wrong
//! only on that path.

use std::collections::HashSet;

use crate::api::{Event, JxlDecoder, JxlDecoderOptions, TocEntry, TocGroupKind};
use crate::tests::api::wrap_with_jxlp_chunks;

/// Single-frame files with no container, for which
/// `frame_data_offset + frame_data_size` must be exactly the file length.
const BARE_SINGLE_FRAME_FILES: &[&str] = &[
    "resources/test/basic.jxl",
    "resources/test/multiple_lf_420.jxl",
    "resources/test/has_permutation.jxl",
];

const ALL_FILES: &[&str] = &[
    "resources/test/basic.jxl",
    "resources/test/multiple_lf_420.jxl",
    "resources/test/has_permutation.jxl",
    "resources/test/has_permutation_with_container.jxl",
];

/// A decoder fed `chunk_size` more bytes per `process()` call, so that the
/// frame header and TOC parse can be forced across several calls.
struct Fed<'a> {
    decoder: JxlDecoder,
    remaining: &'a [u8],
    window: &'a [u8],
    chunk_size: usize,
    process_calls: usize,
}

impl<'a> Fed<'a> {
    fn new(file: &'a [u8], chunk_size: usize, options: JxlDecoderOptions) -> Self {
        Fed {
            decoder: JxlDecoder::new(options),
            remaining: file,
            window: &file[0..0],
            chunk_size,
            process_calls: 0,
        }
    }

    /// Runs one `process()` call after growing the window by one chunk.
    fn step(&mut self) -> Event {
        self.window =
            &self.remaining[..(self.window.len() + self.chunk_size).min(self.remaining.len())];
        let before = self.window.len();
        let event = self.decoder.process(&mut self.window, None, None).unwrap();
        self.process_calls += 1;
        self.remaining = &self.remaining[(before - self.window.len())..];
        event
    }

    /// Steps until `want` is returned, feeding more input on `NeedMoreInput`.
    fn run_to(&mut self, want: fn(&Event) -> bool) -> Event {
        loop {
            let event = self.step();
            if want(&event) {
                return event;
            }
            match event {
                Event::NeedMoreInput { .. } => {
                    assert!(!self.remaining.is_empty(), "ran out of input");
                }
                Event::Complete => panic!("reached Complete before the wanted event"),
                _ => {}
            }
        }
    }

    fn run_to_frame_header(&mut self) {
        self.run_to(|e| matches!(e, Event::FrameHeader));
    }

    fn run_to_frame_complete(&mut self) {
        self.run_to(|e| matches!(e, Event::FrameComplete { .. }));
    }
}

/// Decodes to the first `Event::FrameHeader`, feeding the whole file at once.
fn decode_to_frame_header(file: &[u8]) -> JxlDecoder {
    decode_to_frame_header_chunked(file, file.len().max(1)).0
}

/// Like [`decode_to_frame_header`], revealing `chunk_size` more bytes per
/// call. Also returns the number of `process()` calls made.
fn decode_to_frame_header_chunked(file: &[u8], chunk_size: usize) -> (JxlDecoder, usize) {
    let mut fed = Fed::new(file, chunk_size, JxlDecoderOptions::default());
    fed.run_to_frame_header();
    (fed.decoder, fed.process_calls)
}

fn collect_toc(d: &JxlDecoder) -> Vec<TocEntry> {
    (0..d.toc_num_entries())
        .map(|i| d.toc_entry(i).expect("in-range TOC entry"))
        .collect()
}

/// Invariants that must hold for any valid frame's TOC, derived purely from the
/// entries themselves (no external reference decoder).
fn assert_toc_invariants(d: &JxlDecoder) {
    let entries = collect_toc(d);
    assert!(!entries.is_empty(), "a frame always has >= 1 TOC entry");
    assert!(d.toc_entry(entries.len()).is_none(), "out-of-range entry");

    // Offsets are relative to frame_data_offset, start at 0, and are the running
    // sum of preceding section sizes (contiguous layout).
    let mut acc = 0u64;
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(e.offset, acc, "entry {i} offset is not contiguous");
        acc += e.size as u64;
    }
    assert_eq!(d.frame_data_size(), acc, "frame_data_size != sum of sizes");
    assert!(d.frame_data_offset() > 0, "frame_data_offset should be > 0");

    // The multiset of kinds must be exactly the JPEG XL section layout — this
    // validates the (possibly permuted) bitstream-order -> spec-kind mapping is
    // internally consistent.
    if entries.len() == 1 {
        assert_eq!(entries[0].kind, TocGroupKind::All);
        return;
    }
    let mut lf_globals = 0;
    let mut hf_globals = 0;
    let mut lf_group_idxs = Vec::new();
    let mut group_passes = Vec::new();
    for e in &entries {
        match e.kind {
            TocGroupKind::All => panic!("All kind in a multi-entry frame"),
            TocGroupKind::LfGlobal => lf_globals += 1,
            TocGroupKind::HfGlobal => hf_globals += 1,
            TocGroupKind::LfGroup(i) => lf_group_idxs.push(i),
            TocGroupKind::GroupPass {
                pass_idx,
                group_idx,
            } => group_passes.push((pass_idx, group_idx)),
        }
    }
    assert_eq!(lf_globals, 1, "exactly one LfGlobal expected");
    assert_eq!(hf_globals, 1, "exactly one HfGlobal expected");

    lf_group_idxs.sort_unstable();
    for (i, idx) in lf_group_idxs.iter().enumerate() {
        assert_eq!(*idx as usize, i, "LfGroup indices not a contiguous 0..n");
    }

    let num_groups = group_passes.iter().map(|&(_, g)| g).max().unwrap_or(0) as usize + 1;
    let num_passes = group_passes.len() / num_groups.max(1);
    assert_eq!(
        group_passes.len(),
        num_groups * num_passes,
        "GroupPass count is not num_groups * num_passes"
    );
    let mut seen = HashSet::new();
    for &(p, g) in &group_passes {
        assert!(seen.insert((p, g)), "duplicate GroupPass ({p}, {g})");
        assert!((g as usize) < num_groups, "group_idx out of range");
        assert!((p as usize) < num_passes, "pass_idx out of range");
    }
}

#[test]
fn toc_invariants_basic() {
    let file = std::fs::read("resources/test/basic.jxl").unwrap();
    assert_toc_invariants(&decode_to_frame_header(&file));
}

#[test]
fn toc_invariants_multigroup() {
    // A multi-group image exercises the LfGroup / HfGlobal / GroupPass layout
    // rather than the single "All" entry.
    let file = std::fs::read("resources/test/multiple_lf_420.jxl").unwrap();
    assert_toc_invariants(&decode_to_frame_header(&file));
}

#[test]
fn toc_invariants_permuted() {
    // has_permutation.jxl has a permuted TOC; the invariants check that the
    // bitstream-order -> spec-kind mapping survives the permutation.
    let file = std::fs::read("resources/test/has_permutation.jxl").unwrap();
    assert_toc_invariants(&decode_to_frame_header(&file));
}

#[test]
fn toc_permutation_container_consistency() {
    // The bare and containerised variants of the same permuted image must report
    // identical TOC structure and frame_data_size. Only frame_data_offset
    // differs — the containerised file's section data starts later by the box
    // overhead. This is the check that frame_data_offset is file-absolute
    // rather than codestream-relative.
    let bare = std::fs::read("resources/test/has_permutation.jxl").unwrap();
    let cont = std::fs::read("resources/test/has_permutation_with_container.jxl").unwrap();

    let db = decode_to_frame_header(&bare);
    let dc = decode_to_frame_header(&cont);

    assert_eq!(collect_toc(&db), collect_toc(&dc), "TOC differs");
    assert_eq!(
        db.frame_data_size(),
        dc.frame_data_size(),
        "frame_data_size differs bare vs container"
    );
    assert!(
        dc.frame_data_offset() > db.frame_data_offset(),
        "containerised frame_data_offset ({}) should exceed bare ({}) by the \
         container box overhead",
        dc.frame_data_offset(),
        db.frame_data_offset(),
    );
}

/// Ground truth for the absolute byte anchor: for a bare, single-frame file the
/// section data runs to the end of the file, so
/// `frame_data_offset + frame_data_size == file_length`.
///
/// This is what catches an under- or over-counted frame-header/TOC byte total
/// in absolute terms, rather than only relative to another parse.
fn assert_offset_reaches_end_of_file(d: JxlDecoder, file: &[u8], label: &str) {
    let offset = d.frame_data_offset();
    let size = d.frame_data_size();
    assert_eq!(
        offset + size,
        file.len() as u64,
        "{label}: frame_data_offset ({offset}) + frame_data_size ({size}) \
         should be exactly the file length ({})",
        file.len()
    );
}

#[test]
fn frame_data_offset_reaches_end_of_bare_file() {
    for name in BARE_SINGLE_FRAME_FILES {
        let file = std::fs::read(name).unwrap();
        assert_offset_reaches_end_of_file(decode_to_frame_header(&file), &file, name);
    }
}

#[test]
fn frame_data_offset_reaches_end_of_bare_file_chunked() {
    // Same absolute check, but with the header + TOC parsed across many calls.
    for name in BARE_SINGLE_FRAME_FILES {
        let file = std::fs::read(name).unwrap();
        for chunk in [1usize, 3, 7, 17] {
            let (d, calls) = decode_to_frame_header_chunked(&file, chunk);
            // At one byte per reveal every chunk boundary lands mid-header
            // and mid-TOC, so this must take several calls; the larger chunk
            // sizes are extra coverage and may legitimately finish in one.
            assert!(
                chunk > 1 || calls > 1,
                "{name}: chunk_size={chunk} did not split the frame-info parse \
                 across multiple process() calls, so it does not exercise the \
                 incremental path"
            );
            assert_offset_reaches_end_of_file(d, &file, &format!("{name} chunk={chunk}"));
        }
    }
}

#[test]
fn toc_chunked_matches_single_shot() {
    // Regression (originally fixed downstream as "accumulate TOC byte size
    // across incremental parses"): feeding a frame in tiny chunks splits
    // frame-header + TOC parsing across multiple process() calls. The resulting
    // frame_data_offset / frame_data_size / TOC entries MUST match the
    // single-shot feed — otherwise the section-data byte anchor undercounts by
    // the bytes consumed in earlier retries and every progressive split
    // boundary shifts.
    for name in ALL_FILES {
        let file = std::fs::read(name).unwrap();
        let single = decode_to_frame_header(&file);
        let single_offset = single.frame_data_offset();
        let single_size = single.frame_data_size();
        let single_entries = collect_toc(&single);

        // 1 byte at a time is the most aggressive split; the other sizes vary
        // where the chunk boundary lands, including mid-TOC.
        for chunk in [1usize, 3, 7, 17] {
            let (chunked, calls) = decode_to_frame_header_chunked(&file, chunk);
            assert!(
                chunk > 1 || calls > 1,
                "{name}: chunk_size={chunk} was not incremental"
            );
            assert_eq!(
                chunked.frame_data_offset(),
                single_offset,
                "{name}: frame_data_offset differs at chunk_size={chunk}"
            );
            assert_eq!(
                chunked.frame_data_size(),
                single_size,
                "{name}: frame_data_size differs at chunk_size={chunk}"
            );
            assert_eq!(
                collect_toc(&chunked),
                single_entries,
                "{name}: TOC entries differ at chunk_size={chunk}"
            );
        }
    }
}

/// Feeds a file in `chunk_size` pieces through the whole first frame, sampling
/// `num_completed_passes()` after the header and each time the decoder asks for
/// more input.
fn completed_pass_sequence(file: &[u8], chunk_size: usize) -> Vec<usize> {
    let mut fed = Fed::new(file, chunk_size, JxlDecoderOptions::default());
    fed.run_to_frame_header();
    let mut seq = vec![fed.decoder.num_completed_passes()];
    loop {
        match fed.step() {
            Event::FrameComplete { .. } => break,
            Event::NeedMoreInput { .. } => {
                assert!(
                    !fed.remaining.is_empty(),
                    "ran out of input before frame end"
                );
                seq.push(fed.decoder.num_completed_passes());
            }
            e => panic!("unexpected event {e:?}"),
        }
    }
    seq
}

#[test]
fn num_completed_passes_is_monotone_across_incremental_feeds() {
    // Progressive consumers re-render only when this counter grows, so it must
    // start at zero, never go backwards while a frame is fed in pieces, and
    // actually advance for a multi-pass image. Both fixtures below are encoded
    // with several passes.
    for name in [
        "resources/test/conformance_test_images/progressive.jxl",
        "resources/test/progressive_ac.jxl",
    ] {
        let file = std::fs::read(name).unwrap();
        for chunk in [4096usize, 65536] {
            let seq = completed_pass_sequence(&file, chunk);
            assert_eq!(seq[0], 0, "{name}: chunk={chunk}: should start at 0");
            assert!(
                seq.windows(2).all(|w| w[0] <= w[1]),
                "{name}: chunk={chunk}: num_completed_passes went backwards: {seq:?}"
            );
            assert!(
                seq.len() > 1,
                "{name}: chunk={chunk} did not feed the frame incrementally"
            );
            assert!(
                *seq.last().unwrap() >= 1,
                "{name}: chunk={chunk}: never observed a completed pass: {seq:?}"
            );
        }
    }
}

#[test]
fn toc_api_reports_nothing_outside_frame_window() {
    let file = std::fs::read("resources/test/multiple_lf_420.jxl").unwrap();
    let mut fed = Fed::new(&file, file.len(), JxlDecoderOptions::default());
    let check_empty = |d: &JxlDecoder, when: &str| {
        assert_eq!(d.toc_num_entries(), 0, "{when}");
        assert!(d.toc_entry(0).is_none(), "{when}");
        assert_eq!(d.frame_data_size(), 0, "{when}");
        assert_eq!(d.frame_data_offset(), 0, "{when}");
        assert_eq!(d.num_completed_passes(), 0, "{when}");
    };
    check_empty(&fed.decoder, "before BasicInfo");
    fed.run_to(|e| matches!(e, Event::BasicInfo));
    check_empty(&fed.decoder, "after BasicInfo");
    fed.run_to_frame_header();
    assert!(fed.decoder.toc_num_entries() > 0);
    fed.run_to_frame_complete();
    check_empty(&fed.decoder, "after FrameComplete");
}

#[test]
fn frame_data_offsets_advance_across_frames() {
    // Multi-frame files: each frame's section data must start at or after the
    // end of the previous frame's, and stay inside the file. This is the check
    // that the offset is re-anchored per frame rather than left at the first
    // frame's value.
    for name in [
        "resources/test/conformance_test_images/animation_spline.jxl",
        "tests/testdata/5_frames_numbered_jxli.jxl",
    ] {
        let file = std::fs::read(name).unwrap();
        let mut fed = Fed::new(&file, file.len(), JxlDecoderOptions::default());
        let mut frames = Vec::new();
        loop {
            fed.run_to_frame_header();
            let offset = fed.decoder.frame_data_offset();
            let size = fed.decoder.frame_data_size();
            assert!(
                offset + size <= file.len() as u64,
                "{name}: frame {} data runs past the end of the file",
                frames.len()
            );
            frames.push((offset, size));
            let Event::FrameComplete { has_more_frames } =
                fed.run_to(|e| matches!(e, Event::FrameComplete { .. }))
            else {
                unreachable!()
            };
            if !has_more_frames {
                break;
            }
        }

        assert!(frames.len() > 1, "{name}: expected several frames");
        for w in frames.windows(2) {
            let ((o0, s0), (o1, _)) = (w[0], w[1]);
            assert!(
                o1 >= o0 + s0,
                "{name}: frame data offsets go backwards: {o1} < {o0} + {s0}"
            );
        }
    }
}

#[test]
fn toc_api_reports_nothing_for_skipped_frames() {
    // Under `scan_frames_only` no Frame is built, but the decoder still emits
    // FrameHeader. The previous frame's Frame may still be held internally for
    // reference-frame chaining, so the TOC accessors must report "nothing"
    // rather than that stale frame's entries.
    let file = std::fs::read("tests/testdata/5_frames_numbered_jxli.jxl").unwrap();
    let options = JxlDecoderOptions {
        scan_frames_only: true,
        ..Default::default()
    };
    let mut fed = Fed::new(&file, file.len(), options);
    let mut seen = 0;
    loop {
        fed.run_to_frame_header();
        seen += 1;
        let d = &fed.decoder;
        assert_eq!(d.toc_num_entries(), 0, "frame {seen}");
        assert!(d.toc_entry(0).is_none(), "frame {seen}");
        assert_eq!(d.frame_data_size(), 0, "frame {seen}");
        assert_eq!(d.frame_data_offset(), 0, "frame {seen}");
        let Event::FrameComplete { has_more_frames } =
            fed.run_to(|e| matches!(e, Event::FrameComplete { .. }))
        else {
            unreachable!()
        };
        if !has_more_frames {
            break;
        }
    }
    assert!(seen > 1, "expected several scanned frames");
}

/// Bare codestream plus its TOC as seen at `FrameHeader`: for a bare file the
/// codestream and file positions coincide.
fn bare_sections(name: &str) -> (Vec<u8>, u64, Vec<TocEntry>) {
    let bare = std::fs::read(name).unwrap();
    let d = decode_to_frame_header(&bare);
    let start = d.frame_data_codestream_offset();
    assert_eq!(start, d.frame_data_offset());
    let entries = collect_toc(&d);
    (bare, start, entries)
}

/// Re-boxes `bare` into `jxlp` boxes with boundaries inside the frame data:
/// at the start of every odd-numbered section and in the middle of every
/// even-numbered one that is big enough.
fn rebox_inside_frame_data(bare: &[u8], start: u64, entries: &[TocEntry]) -> Vec<u8> {
    let mut boundaries = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let pos = (start + e.offset) as usize;
        if i % 2 == 1 {
            boundaries.push(pos);
        } else if e.size >= 16 {
            boundaries.push(pos + e.size as usize / 2);
        }
    }
    wrap_with_jxlp_chunks(bare, &boundaries)
}

/// Checks that every currently mappable entry of `d` points at the bytes of
/// the corresponding bare section. Returns how many entries could be mapped.
fn check_mapped_entries(
    d: &JxlDecoder,
    container: &[u8],
    bare: &[u8],
    start: u64,
    entries: &[TocEntry],
) -> usize {
    assert_eq!(collect_toc(d), entries, "TOC differs from the bare file's");
    assert_eq!(d.frame_data_codestream_offset(), start);
    let mut mapped = 0;
    for (i, e) in entries.iter().enumerate() {
        let Some(file_offset) = d.toc_entry_file_offset(i) else {
            continue;
        };
        mapped += 1;
        let file_offset = file_offset as usize;
        let bare_start = (start + e.offset) as usize;
        // A box boundary may fall inside the section; compare only the bytes up
        // to it, which is what shows the offset is right for the section start.
        let n = (e.size as usize).min(4);
        assert_eq!(
            &container[file_offset..file_offset + n],
            &bare[bare_start..bare_start + n],
            "entry {i} ({:?}) maps to the wrong file offset {file_offset}",
            e.kind
        );
    }
    mapped
}

#[test]
fn toc_entry_file_offsets_across_jxlp_boundaries() {
    for name in [
        "resources/test/green_queen_vardct_e3.jxl",
        "resources/test/has_permutation.jxl",
    ] {
        let (bare, start, entries) = bare_sections(name);
        let container = rebox_inside_frame_data(&bare, start, &entries);

        // All input at once.
        let d = decode_to_frame_header(&container);
        let mapped = check_mapped_entries(&d, &container, &bare, start, &entries);
        assert!(mapped > 0, "{name}: nothing mapped");

        // The naive file-absolute formula is wrong for at least one entry; if
        // it were right everywhere, this test would not exercise the fix.
        let naive_wrong = (0..entries.len()).any(|i| {
            d.toc_entry_file_offset(i)
                .is_some_and(|o| o != d.frame_data_offset() + entries[i].offset)
        });
        assert!(naive_wrong, "{name}: no box header inside the frame data");

        // Fed in small chunks: entries whose box has not been parsed yet map to
        // `None`, never to a wrong offset, and all map once the frame is in.
        for chunk in [1usize, 5, 64, 1000] {
            let mut fed = Fed::new(&container, chunk, JxlDecoderOptions::default());
            fed.run_to_frame_header();
            let mut saw_unmapped = false;
            let mut seen_offsets = vec![None; entries.len()];
            loop {
                let mapped = check_mapped_entries(&fed.decoder, &container, &bare, start, &entries);
                saw_unmapped |= mapped < entries.len();
                // Once an entry maps, its offset must not change or vanish.
                for (i, seen) in seen_offsets.iter_mut().enumerate() {
                    let now = fed.decoder.toc_entry_file_offset(i);
                    assert!(
                        seen.is_none() || *seen == now,
                        "{name}: chunk={chunk}: entry {i} mapping changed"
                    );
                    *seen = now;
                }
                match fed.step() {
                    Event::FrameComplete { .. } => break,
                    Event::NeedMoreInput { .. } => {}
                    e => panic!("unexpected event {e:?}"),
                }
            }
            if chunk == 1 {
                assert!(saw_unmapped, "{name}: chunk=1 never saw an unparsed box");
            }
        }
    }
}

#[test]
fn codestream_offset_mapping_matches_jxlp_layout() {
    // Directly check the mapper against the container layout: every codestream
    // byte of the frame data maps to the same byte in the container.
    let (bare, start, entries) = bare_sections("resources/test/green_queen_vardct_e3.jxl");
    let container = rebox_inside_frame_data(&bare, start, &entries);
    let d = decode_to_frame_header(&container);
    let end = start + d.frame_data_size();
    for pos in start..end {
        if let Some(file_offset) = d.codestream_offset_to_file_offset(pos) {
            assert_eq!(container[file_offset as usize], bare[pos as usize], "{pos}");
        }
    }
    // Past the end of the codestream there is no box.
    assert_eq!(d.codestream_offset_to_file_offset(bare.len() as u64), None);
}
