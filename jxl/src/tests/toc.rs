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
//! incremental-parse tests below are the ones that matter — the frame header
//! and TOC can be read across several `process()` calls, and a parser that
//! recomputes rather than accumulates its byte counters gets the offset wrong
//! only on that path.

use std::collections::HashSet;

use crate::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, ProcessingResult, TocEntry, TocGroupKind,
    states,
};
use crate::image::{Image, Rect};

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

/// Drive a fresh decoder over `file` to the `WithFrameInfo` state, where the
/// TOC API is available, feeding the whole buffer at once.
fn decode_to_frame_info(file: &[u8]) -> JxlDecoder<states::WithFrameInfo> {
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut input = file;
    let mut with_info = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    loop {
        match with_info.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => with_info = fallback,
        }
    }
}

/// Drive a decoder to `WithFrameInfo` by revealing `chunk_size` more bytes at a
/// time. This forces the frame header and TOC to be parsed across many
/// `process()` calls (the `OutOfBounds` retry path), including chunk boundaries
/// that land mid-TOC.
fn decode_to_frame_info_chunked(
    file: &[u8],
    chunk_size: usize,
) -> (JxlDecoder<states::WithFrameInfo>, usize) {
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut remaining = file;
    let mut window = &remaining[0..0];
    let mut with_info = loop {
        window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
        let before = window.len();
        let res = decoder.process(&mut window, None).unwrap();
        remaining = &remaining[(before - window.len())..];
        match res {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!remaining.is_empty(), "ran out of input before image info");
                decoder = fallback;
            }
        }
    };
    // This second stage is where the frame header and TOC are read.
    let mut process_calls = 0usize;
    let with_frame = loop {
        window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
        let before = window.len();
        let res = with_info.process(&mut window, None).unwrap();
        process_calls += 1;
        remaining = &remaining[(before - window.len())..];
        match res {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!remaining.is_empty(), "ran out of input before frame info");
                with_info = fallback;
            }
        }
    };
    (with_frame, process_calls)
}

fn collect_toc(d: &JxlDecoder<states::WithFrameInfo>) -> Vec<TocEntry> {
    (0..d.toc_num_entries())
        .map(|i| d.toc_entry(i).expect("in-range TOC entry"))
        .collect()
}

/// Invariants that must hold for any valid frame's TOC, derived purely from the
/// entries themselves (no external reference decoder).
fn assert_toc_invariants(d: &JxlDecoder<states::WithFrameInfo>) {
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
    assert_toc_invariants(&decode_to_frame_info(&file));
}

#[test]
fn toc_invariants_multigroup() {
    // A multi-group image exercises the LfGroup / HfGlobal / GroupPass layout
    // rather than the single "All" entry.
    let file = std::fs::read("resources/test/multiple_lf_420.jxl").unwrap();
    assert_toc_invariants(&decode_to_frame_info(&file));
}

#[test]
fn toc_invariants_permuted() {
    // has_permutation.jxl has a permuted TOC; the invariants check that the
    // bitstream-order -> spec-kind mapping survives the permutation.
    let file = std::fs::read("resources/test/has_permutation.jxl").unwrap();
    assert_toc_invariants(&decode_to_frame_info(&file));
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

    let db = decode_to_frame_info(&bare);
    let dc = decode_to_frame_info(&cont);

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
fn assert_offset_reaches_end_of_file(
    d: JxlDecoder<states::WithFrameInfo>,
    file: &[u8],
    label: &str,
) {
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
        assert_offset_reaches_end_of_file(decode_to_frame_info(&file), &file, name);
    }
}

#[test]
fn frame_data_offset_reaches_end_of_bare_file_chunked() {
    // Same absolute check, but with the header + TOC parsed across many calls.
    for name in BARE_SINGLE_FRAME_FILES {
        let file = std::fs::read(name).unwrap();
        for chunk in [1usize, 3, 7, 17] {
            let (d, calls) = decode_to_frame_info_chunked(&file, chunk);
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
        let single = decode_to_frame_info(&file);
        let single_offset = single.frame_data_offset();
        let single_size = single.frame_data_size();
        let single_entries = collect_toc(&single);

        // 1 byte at a time is the most aggressive split; the other sizes vary
        // where the chunk boundary lands, including mid-TOC.
        for chunk in [1usize, 3, 7, 17] {
            let (chunked, calls) = decode_to_frame_info_chunked(&file, chunk);
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

/// Feed a file in `chunk_size` pieces all the way through frame decoding,
/// sampling `num_completed_passes()` at every point where the decoder asks for
/// more input. Returns the observed sequence.
fn completed_pass_sequence(file: &[u8], chunk_size: usize) -> Vec<usize> {
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut remaining = file;
    let mut window = &remaining[0..0];
    let mut with_info = loop {
        window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
        let before = window.len();
        let res = decoder.process(&mut window, None).unwrap();
        remaining = &remaining[(before - window.len())..];
        match res {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!remaining.is_empty(), "ran out of input before image info");
                decoder = fallback;
            }
        }
    };
    let size = with_info.basic_info().size;
    let mut with_frame = loop {
        window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
        let before = window.len();
        let res = with_info.process(&mut window, None).unwrap();
        remaining = &remaining[(before - window.len())..];
        match res {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!remaining.is_empty(), "ran out of input before frame info");
                with_info = fallback;
            }
        }
    };

    let mut seq = vec![with_frame.num_completed_passes()];
    let mut output = Image::<f32>::new((size.0 * 3, size.1)).unwrap();
    let rect = Rect {
        size: output.size(),
        origin: (0, 0),
    };
    loop {
        window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
        let before = window.len();
        let mut bufs = [JxlOutputBuffer::from_image_rect_mut(
            output.get_rect_mut(rect).into_raw(),
        )];
        let res = with_frame.process(&mut window, &mut bufs, None).unwrap();
        remaining = &remaining[(before - window.len())..];
        match res {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                assert!(!remaining.is_empty(), "ran out of input before frame end");
                with_frame = fallback;
                seq.push(with_frame.num_completed_passes());
            }
        }
    }
    seq
}

#[test]
fn num_completed_passes_is_monotone_across_incremental_feeds() {
    // Progressive consumers (Hikaru's medical-image streamer) re-render only
    // when this counter grows, so it must start at zero, never go backwards
    // while a frame is fed in pieces, and actually advance for a multi-pass
    // image. Both fixtures below are encoded with several passes.
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
        let mut input = file.as_slice();
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut with_info = loop {
            match decoder.process(&mut input, None).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };

        let mut frames = Vec::new();
        loop {
            let mut with_frame = loop {
                match with_info.process(&mut input, None).unwrap() {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => with_info = fallback,
                }
            };
            let offset = with_frame.frame_data_offset();
            let size = with_frame.frame_data_size();
            assert!(
                offset + size <= file.len() as u64,
                "{name}: frame {} data runs past the end of the file",
                frames.len()
            );
            frames.push((offset, size));
            with_info = loop {
                match with_frame.skip_frame(&mut input).unwrap() {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => with_frame = fallback,
                }
            };
            if !with_info.has_more_frames() {
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
