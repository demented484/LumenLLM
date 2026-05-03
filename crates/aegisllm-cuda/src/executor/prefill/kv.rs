pub(super) const PREFILL_KV_PAGE_TOKENS: usize = 256;

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillKvPageSpec {
    pub(super) page_tokens: usize,
    pub(super) max_pages_per_sequence: usize,
}

impl PrefillKvPageSpec {
    #[allow(dead_code)]
    pub(super) fn for_context(context_size: usize) -> Self {
        Self {
            page_tokens: PREFILL_KV_PAGE_TOKENS,
            max_pages_per_sequence: context_size.div_ceil(PREFILL_KV_PAGE_TOKENS).max(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(super) struct PrefillSequenceId(pub(super) u64);

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillKvSequencePages {
    pub(super) sequence: PrefillSequenceId,
    pub(super) logical_pages: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillSlotMapper {
    page_tokens: usize,
    pages: PrefillKvSequencePages,
}

#[allow(dead_code)]
impl PrefillSlotMapper {
    pub(super) fn new(page_tokens: usize, pages: PrefillKvSequencePages) -> Self {
        Self { page_tokens, pages }
    }

    pub(super) fn physical_slot(&self, logical_position: usize) -> Option<u32> {
        let page = logical_position / self.page_tokens;
        let offset = logical_position % self.page_tokens;
        let physical_page = *self.pages.logical_pages.get(page)?;
        physical_page
            .checked_mul(self.page_tokens as u32)?
            .checked_add(offset as u32)
    }

    pub(super) fn slot_mapping(&self, start_position: usize, tokens: usize) -> Option<Vec<u32>> {
        (0..tokens)
            .map(|idx| self.physical_slot(start_position + idx))
            .collect()
    }
}

/// Outcome of a `allocate_or_evict` call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum KvAllocResult {
    /// Pages allocated from the free list — no eviction needed.
    Allocated(PrefillKvSequencePages),
    /// Free list was exhausted; the given sequence was evicted to make room.
    EvictedAndAllocated {
        evicted: PrefillSequenceId,
        pages: PrefillKvSequencePages,
    },
    /// Even after eviction, not enough pages are available (caller's request too large).
    OutOfMemory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillKvPageAllocator {
    page_tokens: usize,
    free_pages: Vec<u32>,
    active: HashMap<PrefillSequenceId, PrefillKvSequencePages>,
    /// LRU clock: monotonically increasing tick incremented on each allocation and touch.
    lru_clock: u64,
    /// Last-used tick per active sequence (higher = more recently used).
    lru_ticks: HashMap<PrefillSequenceId, u64>,
}

#[allow(dead_code)]
impl PrefillKvPageAllocator {
    pub(super) fn new(num_pages: usize, page_tokens: usize) -> Self {
        let mut free_pages = (0..num_pages as u32).collect::<Vec<_>>();
        free_pages.reverse();
        Self {
            page_tokens,
            free_pages,
            active: HashMap::new(),
            lru_clock: 0,
            lru_ticks: HashMap::new(),
        }
    }

    pub(super) fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub(super) fn active_sequences(&self) -> usize {
        self.active.len()
    }

    pub(super) fn pages_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_tokens).max(1)
    }

    /// Touch a sequence to update its LRU position.
    pub(super) fn touch(&mut self, sequence: PrefillSequenceId) {
        if self.active.contains_key(&sequence) {
            self.lru_clock += 1;
            self.lru_ticks.insert(sequence, self.lru_clock);
        }
    }

    pub(super) fn allocate(
        &mut self,
        sequence: PrefillSequenceId,
        tokens: usize,
    ) -> Option<PrefillKvSequencePages> {
        if self.active.contains_key(&sequence) {
            return None;
        }
        let need = self.pages_for_tokens(tokens);
        if self.free_pages.len() < need {
            return None;
        }
        let pages = self.alloc_pages(sequence, need);
        Some(pages)
    }

    /// Try to allocate; if the free list is exhausted, evict the LRU non-active
    /// sequence to reclaim its pages. Returns the allocation outcome.
    pub(super) fn allocate_or_evict(
        &mut self,
        sequence: PrefillSequenceId,
        tokens: usize,
        active_sequences: &HashSet<PrefillSequenceId>,
    ) -> KvAllocResult {
        if self.active.contains_key(&sequence) {
            return KvAllocResult::OutOfMemory;
        }
        let need = self.pages_for_tokens(tokens);
        if self.free_pages.len() >= need {
            let pages = self.alloc_pages(sequence, need);
            return KvAllocResult::Allocated(pages);
        }
        // LRU eviction: pick the oldest sequence not in `active_sequences`
        let victim = self
            .lru_ticks
            .iter()
            .filter(|(seq, _)| !active_sequences.contains(seq) && **seq != sequence)
            .min_by_key(|(_, tick)| **tick)
            .map(|(seq, _)| *seq);
        let Some(victim_id) = victim else {
            return KvAllocResult::OutOfMemory;
        };
        self.release_sequence(victim_id);
        if self.free_pages.len() < need {
            return KvAllocResult::OutOfMemory;
        }
        let pages = self.alloc_pages(sequence, need);
        KvAllocResult::EvictedAndAllocated {
            evicted: victim_id,
            pages,
        }
    }

    fn alloc_pages(&mut self, sequence: PrefillSequenceId, need: usize) -> PrefillKvSequencePages {
        self.lru_clock += 1;
        let tick = self.lru_clock;
        let mut logical_pages = Vec::with_capacity(need);
        for _ in 0..need {
            logical_pages.push(self.free_pages.pop().expect("checked above"));
        }
        let pages = PrefillKvSequencePages { sequence, logical_pages };
        self.active.insert(sequence, pages.clone());
        self.lru_ticks.insert(sequence, tick);
        pages
    }

    pub(super) fn release(&mut self, pages: PrefillKvSequencePages) -> bool {
        let Some(active) = self.active.remove(&pages.sequence) else {
            return false;
        };
        if active.logical_pages != pages.logical_pages {
            self.active.insert(active.sequence, active);
            return false;
        }
        self.lru_ticks.remove(&pages.sequence);
        self.release_pages(active.logical_pages);
        true
    }

    pub(super) fn release_sequence(&mut self, sequence: PrefillSequenceId) -> bool {
        let Some(active) = self.active.remove(&sequence) else {
            return false;
        };
        self.lru_ticks.remove(&sequence);
        self.release_pages(active.logical_pages);
        true
    }

    fn release_pages(&mut self, mut pages: Vec<u32>) {
        self.free_pages.append(&mut pages);
        let mut seen = HashSet::new();
        self.free_pages.retain(|page| seen.insert(*page));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::{
        KvAllocResult, PREFILL_KV_PAGE_TOKENS, PrefillKvPageAllocator, PrefillSequenceId,
        PrefillSlotMapper,
    };

    #[test]
    fn page_allocator_reuses_released_pages() {
        let mut allocator = PrefillKvPageAllocator::new(4, PREFILL_KV_PAGE_TOKENS);
        let seq = allocator.allocate(PrefillSequenceId(7), 513).unwrap();
        assert_eq!(seq.logical_pages, [0, 1, 2]);
        assert_eq!(allocator.free_pages(), 1);
        assert!(allocator.release(seq));
        assert_eq!(allocator.free_pages(), 4);
    }

    #[test]
    fn page_allocator_rejects_double_release_and_duplicate_owner() {
        let mut allocator = PrefillKvPageAllocator::new(2, PREFILL_KV_PAGE_TOKENS);
        let seq = allocator.allocate(PrefillSequenceId(7), 1).unwrap();
        assert!(allocator.allocate(PrefillSequenceId(7), 1).is_none());
        assert!(allocator.release(seq.clone()));
        assert!(!allocator.release(seq));
        assert_eq!(allocator.free_pages(), 2);
    }

    #[test]
    fn page_allocator_rejects_oversized_sequence() {
        let mut allocator = PrefillKvPageAllocator::new(2, PREFILL_KV_PAGE_TOKENS);
        assert!(allocator.allocate(PrefillSequenceId(1), 513).is_none());
        assert_eq!(allocator.free_pages(), 2);
    }

    #[test]
    fn slot_mapper_translates_logical_positions_to_physical_slots() {
        let pages = super::PrefillKvSequencePages {
            sequence: PrefillSequenceId(9),
            logical_pages: vec![5, 2],
        };
        let mapper = PrefillSlotMapper::new(4, pages);
        assert_eq!(mapper.physical_slot(0), Some(20));
        assert_eq!(mapper.physical_slot(3), Some(23));
        assert_eq!(mapper.physical_slot(4), Some(8));
        assert_eq!(mapper.slot_mapping(2, 4).unwrap(), [22, 23, 8, 9]);
    }

    #[test]
    fn lru_eviction_evicts_oldest_non_active_sequence() {
        // 3 pages total. Allocate seq 1 (1 page) then seq 2 (1 page) then seq 3 (1 page).
        // seq 1 is oldest. Now try to allocate seq 4 (1 page) with seq 2 and 3 marked active.
        // Expected: seq 1 gets evicted.
        let mut allocator = PrefillKvPageAllocator::new(3, PREFILL_KV_PAGE_TOKENS);
        allocator.allocate(PrefillSequenceId(1), 1).unwrap();
        allocator.allocate(PrefillSequenceId(2), 1).unwrap();
        allocator.allocate(PrefillSequenceId(3), 1).unwrap();
        assert_eq!(allocator.free_pages(), 0);

        let mut active: HashSet<PrefillSequenceId> = HashSet::new();
        active.insert(PrefillSequenceId(2));
        active.insert(PrefillSequenceId(3));
        match allocator.allocate_or_evict(PrefillSequenceId(4), 1, &active) {
            KvAllocResult::EvictedAndAllocated { evicted, pages } => {
                assert_eq!(evicted, PrefillSequenceId(1));
                assert_eq!(pages.sequence, PrefillSequenceId(4));
            }
            other => panic!("expected EvictedAndAllocated, got {other:?}"),
        }
        // seq 1 pages freed, seq 4 now active
        assert_eq!(allocator.active_sequences(), 3); // seq 2, 3, 4
    }

    #[test]
    fn lru_eviction_respects_touch_order() {
        // 2 pages total. Allocate seq 1 then seq 2. Touch seq 1 (makes it newer).
        // seq 2 should now be the LRU victim.
        let mut allocator = PrefillKvPageAllocator::new(2, PREFILL_KV_PAGE_TOKENS);
        allocator.allocate(PrefillSequenceId(1), 1).unwrap();
        allocator.allocate(PrefillSequenceId(2), 1).unwrap();
        allocator.touch(PrefillSequenceId(1)); // seq 1 is now newer
        assert_eq!(allocator.free_pages(), 0);

        match allocator.allocate_or_evict(PrefillSequenceId(3), 1, &HashSet::new()) {
            KvAllocResult::EvictedAndAllocated { evicted, .. } => {
                assert_eq!(evicted, PrefillSequenceId(2), "seq 2 is LRU after touching seq 1");
            }
            other => panic!("expected EvictedAndAllocated, got {other:?}"),
        }
    }

    #[test]
    fn lru_returns_out_of_memory_when_all_sequences_active() {
        let mut allocator = PrefillKvPageAllocator::new(2, PREFILL_KV_PAGE_TOKENS);
        allocator.allocate(PrefillSequenceId(1), 1).unwrap();
        allocator.allocate(PrefillSequenceId(2), 1).unwrap();

        // Both sequences are "active" — cannot evict either
        let mut active = HashSet::new();
        active.insert(PrefillSequenceId(1));
        active.insert(PrefillSequenceId(2));
        assert_eq!(
            allocator.allocate_or_evict(PrefillSequenceId(3), 1, &active),
            KvAllocResult::OutOfMemory,
        );
    }
}
