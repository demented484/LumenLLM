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

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrefillKvPageAllocator {
    page_tokens: usize,
    free_pages: Vec<u32>,
    active: HashMap<PrefillSequenceId, PrefillKvSequencePages>,
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
        }
    }

    pub(super) fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub(super) fn pages_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_tokens).max(1)
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
        let mut logical_pages = Vec::with_capacity(need);
        for _ in 0..need {
            logical_pages.push(self.free_pages.pop()?);
        }
        let pages = PrefillKvSequencePages {
            sequence,
            logical_pages,
        };
        self.active.insert(sequence, pages.clone());
        Some(pages)
    }

    pub(super) fn release(&mut self, pages: PrefillKvSequencePages) -> bool {
        let Some(active) = self.active.remove(&pages.sequence) else {
            return false;
        };
        if active.logical_pages != pages.logical_pages {
            self.active.insert(active.sequence, active);
            return false;
        }
        self.release_pages(active.logical_pages);
        true
    }

    pub(super) fn release_sequence(&mut self, sequence: PrefillSequenceId) -> bool {
        let Some(active) = self.active.remove(&sequence) else {
            return false;
        };
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
    use super::{
        PREFILL_KV_PAGE_TOKENS, PrefillKvPageAllocator, PrefillSequenceId, PrefillSlotMapper,
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
}
