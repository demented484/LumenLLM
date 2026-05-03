/// Radix-tree based prefix cache for KV pages.
///
/// Maps token-sequence prefixes to their KV page tables so that repeated
/// prefixes (e.g. the system prompt) can be re-used without re-prefilling.
///
/// Design:
/// - Keys are `Vec<u32>` token sequences (prefixes).
/// - Values are `Vec<u32>` physical KV page indices.
/// - Matching is longest-common-prefix: on lookup we walk the trie as far as
///   the query tokens allow and return the longest match found.
/// - Entries are evicted LRU-style when `capacity_pages` is reached.
use std::collections::HashMap;

/// One node in the radix trie.
#[derive(Debug)]
struct RadixNode {
    /// Tokens stored at this edge (the edge label from parent to here).
    edge: Vec<u32>,
    /// KV pages held by this prefix (cumulative from root to here).
    pages: Vec<u32>,
    /// LRU access tick (higher = more recently used).
    lru_tick: u64,
    children: HashMap<u32, Box<RadixNode>>,
}

impl RadixNode {
    fn new(edge: Vec<u32>, pages: Vec<u32>, tick: u64) -> Self {
        Self { edge, pages, lru_tick: tick, children: HashMap::new() }
    }
}

/// Lookup result from `PrefixCache::lookup`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PrefixHit {
    /// Number of tokens matched (tokens that can be skipped in prefill).
    pub(super) matched_tokens: usize,
    /// KV page table for the matched prefix (already populated in VRAM).
    pub(super) pages: Vec<u32>,
}

/// Radix-tree prefix cache.
#[derive(Debug)]
pub(super) struct PrefixCache {
    root: HashMap<u32, Box<RadixNode>>,
    /// Total KV pages currently pinned by the cache.
    total_pages: usize,
    /// Soft limit on pinned pages — entries are evicted once this is exceeded.
    capacity_pages: usize,
    /// Monotonically increasing clock for LRU ordering.
    clock: u64,
}

impl PrefixCache {
    pub(super) fn new(capacity_pages: usize) -> Self {
        Self {
            root: HashMap::new(),
            total_pages: 0,
            capacity_pages,
            clock: 0,
        }
    }

    /// Clear all entries (e.g. on model reload).
    pub(super) fn clear(&mut self) {
        self.root.clear();
        self.total_pages = 0;
    }

    /// Number of KV pages pinned by the cache.
    pub(super) fn pinned_pages(&self) -> usize {
        self.total_pages
    }

    /// Look up the longest matching prefix for `tokens`.
    /// Returns `None` if no prefix is cached.
    pub(super) fn lookup(&mut self, tokens: &[u32]) -> Option<PrefixHit> {
        if tokens.is_empty() {
            return None;
        }
        self.clock += 1;
        let tick = self.clock;

        // First pass: immutable walk to find the best match.
        let (best_matched, best_pages, hit_node_ptr) =
            lookup_immutable(&self.root, tokens);

        if best_matched == 0 {
            return None;
        }

        // Update LRU tick on the matched node using the raw pointer collected above.
        if let Some(ptr) = hit_node_ptr {
            unsafe { (*ptr).lru_tick = tick };
        }

        Some(PrefixHit {
            matched_tokens: best_matched,
            pages: best_pages,
        })
    }

    /// Insert a completed prefix → pages mapping.
    /// Evicts LRU entries if `capacity_pages` would be exceeded.
    pub(super) fn insert(&mut self, tokens: Vec<u32>, pages: Vec<u32>) {
        if tokens.is_empty() || pages.is_empty() {
            return;
        }
        self.clock += 1;
        let tick = self.clock;

        // Evict until we have room
        let new_pages = pages.len();
        while self.total_pages + new_pages > self.capacity_pages {
            if !self.evict_lru() {
                break;
            }
        }

        // Insert into the trie
        self.total_pages += new_pages;
        insert_into(&mut self.root, &tokens, pages, tick);
    }

    /// Evict the single least-recently-used leaf. Returns `true` if something was evicted.
    fn evict_lru(&mut self) -> bool {
        let victim = find_lru_leaf(&self.root, u64::MAX);
        let Some((path, freed_pages)) = victim else {
            return false;
        };
        self.total_pages = self.total_pages.saturating_sub(freed_pages);
        remove_path(&mut self.root, &path);
        true
    }
}

/// Immutable trie walk: returns (matched_tokens, pages_clone, raw_ptr_to_deepest_matched_node).
/// The raw pointer is safe to dereference as `*mut RadixNode` in the calling `lookup()` because
/// `lookup()` holds `&mut self` for its entire duration, so no concurrent mutation is possible.
fn lookup_immutable(
    children: &HashMap<u32, Box<RadixNode>>,
    tokens: &[u32],
) -> (usize, Vec<u32>, Option<*mut RadixNode>) {
    let Some(&first) = tokens.first() else {
        return (0, Vec::new(), None);
    };
    let Some(node) = children.get(&first) else {
        return (0, Vec::new(), None);
    };

    let edge = &node.edge;
    let shared = edge
        .iter()
        .zip(tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();

    if shared < edge.len() {
        // Partial edge match — no complete node reached.
        return (0, Vec::new(), None);
    }

    // We consumed the whole edge.  Try to go deeper.
    let rest = &tokens[shared..];
    if !rest.is_empty() {
        let (sub_matched, sub_pages, sub_ptr) = lookup_immutable(&node.children, rest);
        if sub_matched > 0 {
            return (shared + sub_matched, sub_pages, sub_ptr);
        }
    }

    // This node is the deepest match.
    let ptr = node.as_ref() as *const RadixNode as *mut RadixNode;
    (shared, node.pages.clone(), Some(ptr))
}

/// Recursively insert a token sequence into the trie.
fn insert_into(
    children: &mut HashMap<u32, Box<RadixNode>>,
    tokens: &[u32],
    pages: Vec<u32>,
    tick: u64,
) {
    let Some(&first) = tokens.first() else { return };
    if let Some(node) = children.get_mut(&first) {
        let edge = node.edge.clone();
        let shared = edge
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        if shared == edge.len() && shared == tokens.len() {
            // Exact match — update pages and tick
            node.pages = pages;
            node.lru_tick = tick;
        } else if shared == edge.len() {
            // Edge fully consumed — recurse into children
            insert_into(&mut node.children, &tokens[shared..], pages, tick);
        } else {
            // Split this edge at `shared`
            let (common, rest_edge) = edge.split_at(shared);
            let (_, rest_tokens) = tokens.split_at(shared);
            // Existing child gets a shorter edge, moves down
            let mut old_child = children.remove(&first).unwrap();
            old_child.edge = rest_edge.to_vec();
            let common_pages = old_child.pages[..old_child.pages.len().min(shared)].to_vec();
            let mut split_node = RadixNode::new(common.to_vec(), common_pages, tick);
            split_node.children.insert(rest_edge[0], old_child);
            if !rest_tokens.is_empty() {
                insert_into(&mut split_node.children, rest_tokens, pages, tick);
            } else {
                split_node.pages = pages;
            }
            children.insert(first, Box::new(split_node));
        }
    } else {
        children.insert(first, Box::new(RadixNode::new(tokens.to_vec(), pages, tick)));
    }
}

/// Find the LRU leaf path. Returns (path of first-tokens, pages_freed).
fn find_lru_leaf(
    children: &HashMap<u32, Box<RadixNode>>,
    parent_tick: u64,
) -> Option<(Vec<u32>, usize)> {
    let mut best_tick = parent_tick;
    let mut best: Option<(Vec<u32>, usize)> = None;
    for (first, node) in children {
        let (path, pages) = if node.children.is_empty() {
            (vec![*first], node.pages.len())
        } else if let Some((mut sub_path, sub_pages)) = find_lru_leaf(&node.children, node.lru_tick) {
            sub_path.insert(0, *first);
            (sub_path, sub_pages)
        } else {
            (vec![*first], node.pages.len())
        };
        if node.lru_tick < best_tick {
            best_tick = node.lru_tick;
            best = Some((path, pages));
        }
    }
    best
}

/// Remove a node at `path[0]` (and prune empty ancestors).
fn remove_path(children: &mut HashMap<u32, Box<RadixNode>>, path: &[u32]) {
    let Some(&first) = path.first() else { return };
    if path.len() == 1 {
        children.remove(&first);
        return;
    }
    if let Some(node) = children.get_mut(&first) {
        remove_path(&mut node.children, &path[1..]);
        if node.children.is_empty() {
            children.remove(&first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixHit};

    #[test]
    fn prefix_cache_hit_on_exact_match() {
        let mut cache = PrefixCache::new(1000);
        cache.insert(vec![1, 2, 3, 4], vec![10, 11]);
        let hit = cache.lookup(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(hit.matched_tokens, 4);
        assert_eq!(hit.pages, [10, 11]);
    }

    #[test]
    fn prefix_cache_no_hit_on_disjoint_sequence() {
        let mut cache = PrefixCache::new(1000);
        cache.insert(vec![1, 2, 3], vec![5]);
        assert!(cache.lookup(&[4, 5, 6]).is_none());
    }

    #[test]
    fn prefix_cache_returns_none_when_empty() {
        let mut cache = PrefixCache::new(1000);
        assert!(cache.lookup(&[1, 2, 3]).is_none());
    }

    #[test]
    fn prefix_cache_partial_match_returns_best_prefix() {
        let mut cache = PrefixCache::new(1000);
        cache.insert(vec![1, 2, 3], vec![10]);
        // Query with extra tokens — should still match the 3-token prefix
        let hit = cache.lookup(&[1, 2, 3, 99, 100]).unwrap();
        assert_eq!(hit.matched_tokens, 3);
    }

    #[test]
    fn prefix_cache_evicts_lru_when_over_capacity() {
        // Capacity for 2 pages. Insert two 1-page entries, then a third.
        // The first (oldest, LRU) should be evicted.
        let mut cache = PrefixCache::new(2);
        cache.insert(vec![1], vec![0]);  // oldest
        cache.insert(vec![2], vec![1]);  // second
        // Force eviction by inserting a third entry needing 1 page
        cache.insert(vec![3], vec![2]);
        // Token [1] (oldest) should have been evicted
        assert!(cache.lookup(&[1]).is_none());
        // [2] and [3] should still be present
        assert!(cache.lookup(&[2]).is_some());
        assert!(cache.lookup(&[3]).is_some());
    }

    #[test]
    fn prefix_cache_pinned_pages_tracks_inserts() {
        let mut cache = PrefixCache::new(1000);
        assert_eq!(cache.pinned_pages(), 0);
        cache.insert(vec![1, 2], vec![0, 1, 2]);
        assert_eq!(cache.pinned_pages(), 3);
        cache.clear();
        assert_eq!(cache.pinned_pages(), 0);
    }

    #[test]
    fn prefix_cache_lru_touch_on_lookup_prevents_eviction() {
        // Insert A then B. Touch A. Now B is LRU. Insert C to trigger eviction.
        let mut cache = PrefixCache::new(2);
        cache.insert(vec![1], vec![0]); // A
        cache.insert(vec![2], vec![1]); // B
        // Touch A so it's the newer entry
        cache.lookup(&[1]);
        // Insert C — B should be evicted (LRU), not A
        cache.insert(vec![3], vec![2]); // C
        assert!(cache.lookup(&[1]).is_some(), "A should survive");
        assert!(cache.lookup(&[2]).is_none(), "B should be evicted");
    }
}
