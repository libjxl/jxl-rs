// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;

use super::*;
use crate::api::codestream_parser::{frame_scan_info::used_slots, stored_slot};

// No allocation or reference counting on the ordinary streaming path.
pub(super) enum ReadySection {
    Owned(SectionBuffer),
    Shared(Arc<SectionBuffer>),
}
impl Deref for ReadySection {
    type Target = SectionBuffer;
    fn deref(&self) -> &SectionBuffer {
        match self {
            Self::Owned(s) => s,
            Self::Shared(s) => s,
        }
    }
}

struct Record {
    header: FrameHeader,
    toc: Toc,
    before: (usize, usize),
    level5: bool,
    dependencies: Vec<usize>,
    sections: Vec<Arc<SectionBuffer>>,
}

#[derive(Default)]
pub(super) struct ReplayCache {
    records: BTreeMap<usize, Record>,
    slots: [Option<usize>; 8],
    pending: Option<Record>,
    target: Option<usize>,
}

impl ReplayCache {
    fn reachable(&self, roots: impl Iterator<Item = usize>) -> BTreeSet<usize> {
        let mut stack: Vec<_> = roots.collect();
        let mut live = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if live.insert(id) {
                stack.extend(&self.records[&id].dependencies);
            }
        }
        live
    }
    fn prune(&mut self) {
        let live = self.reachable(self.slots.iter().flatten().copied().chain(self.target));
        self.records.retain(|id, _| live.contains(id));
    }
    pub(super) fn reset(&mut self, keep: [bool; 8]) {
        self.pending = None;
        self.target = None;
        for (slot, keep) in self.slots.iter_mut().zip(keep) {
            if !keep {
                *slot = None;
            }
        }
        self.prune();
    }
    pub(super) fn collect(&mut self, section: SectionBuffer) -> ReadySection {
        if let Some(record) = &mut self.pending {
            let section = Arc::new(section);
            record.sections.push(section.clone());
            ReadySection::Shared(section)
        } else {
            ReadySection::Owned(section)
        }
    }
}

impl FrameInfo {
    pub(in crate::api::codestream_parser) fn begin_record(
        &mut self,
        before: (usize, usize),
        level5: bool,
    ) {
        let frame = self.frame.as_ref().unwrap();
        if frame.supports_repeated_render() {
            return;
        }
        let dependencies = used_slots(frame.header())
            .into_iter()
            .zip(self.replay.slots)
            .filter_map(|(used, id)| if used { id } else { None })
            .collect();
        self.replay.pending = Some(Record {
            header: frame.header().clone(),
            toc: frame.toc().clone(),
            before,
            level5,
            dependencies,
            sections: vec![],
        });
    }
    pub(in crate::api::codestream_parser) fn finish_record(&mut self, visible: bool) {
        if let Some(record) = self.replay.pending.take() {
            let id = self
                .replay
                .records
                .last_key_value()
                .map_or(0, |(k, _)| k + 1);
            if let Some(slot) = stored_slot(&record.header) {
                self.replay.slots[slot] = Some(id);
            }
            self.replay.target = visible.then_some(id);
            self.replay.records.insert(id, record);
            self.replay.prune();
        }
    }
    pub(super) fn forget_replay_slots(&mut self) {
        self.replay.slots = [None; 8];
        self.replay.prune();
    }

    pub(in crate::api::codestream_parser) fn release_target(&mut self) {
        self.repeat_ready = false;
        if self.replay.target.take().is_some() {
            self.replay.prune();
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rerender_frame(
        &mut self,
        file: &FileHeader,
        options: &JxlDecoderOptions,
        shift: u8,
        viewport: Rect,
        format: &JxlPixelFormat,
        profile: &JxlColorProfile,
        buffers: &mut [JxlOutputBuffer],
        runner: &mut dyn JxlParallelRunner,
    ) -> Result<()> {
        if !self.repeat_ready {
            return Err(Error::ApiUsageError(
                "no completed frame available for repeated decoding",
            ));
        }
        validate_output_buffers(buffers, Some(format))?;
        let Some(target) = self.replay.target else {
            return self
                .frame
                .as_mut()
                .unwrap()
                .rerender_frame(shift, viewport, format, profile, buffers, runner);
        };
        // Replay in stream order into a private slot table. Frame counters seed noise;
        // skipped unrelated frames must still count towards them.
        let mut replay = FrameInfo::new();
        replay.buffer_recycler = self.buffer_recycler.clone();
        let mut empty = SmallBuffer::new(0);
        let order = self.replay.reachable(std::iter::once(target));
        let mut last_use = BTreeMap::new();
        for &id in &order {
            for &dep in &self.replay.records[&id].dependencies {
                last_use.insert(dep, id);
            }
        }
        let mut slots = [None; 8];
        for id in order {
            replay.keep_slots = slots
                .map(|slot| slot.is_some_and(|s| last_use.get(&s).is_some_and(|&last| last >= id)));
            let r = &self.replay.records[&id];
            replay.frame_header = Some(r.header.clone());
            replay.frame_counters = r.before;
            replay.make_frame(
                &mut empty,
                r.toc.clone(),
                file,
                options,
                format,
                profile,
                ProcessMode::Process,
                r.level5,
                shift,
                viewport,
            )?;
            if let Some(slot) = stored_slot(&r.header) {
                slots[slot] = Some(id);
            }
            replay.sections.clear();
            for section in &r.sections {
                replay.queue_ready(ReadySection::Shared(section.clone()));
            }
            if replay
                .process_sections(
                    (id == target).then_some(&mut *buffers),
                    profile,
                    format,
                    runner,
                )?
                .is_some()
            {
                return Err(Error::SectionTooShort);
            }
        }
        Ok(())
    }
}
