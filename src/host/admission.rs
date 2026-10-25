//! Request admission reserves enough full pages to finish before any input runs.
//! This prevents every active long prompt from waiting on a page nobody can free.
use super::runtime::{Blocks, Workspace};

pub(crate) struct Reservation {
    pub(crate) blocks: Blocks,
    epoch: u64,
}

pub(crate) struct PageReservations {
    max_positions: usize,
    total_pages: usize,
    free_slots: Vec<usize>,
    free_pages: Vec<usize>,
    live: Vec<Option<(u64, Vec<usize>)>>,
    next_epoch: u64,
}
impl PageReservations {
    pub(crate) fn new(
        slots: usize,
        pages: usize,
        max_positions: usize,
    ) -> Result<Self, &'static str> {
        if slots == 0 || pages == 0 || max_positions == 0 {
            return Err("admission capacities must be positive");
        }
        if Workspace::blocks_for(max_positions) > pages {
            return Err("page budget cannot finish one maximum-length request");
        }
        Ok(Self {
            max_positions,
            total_pages: pages,
            free_slots: (0..slots).rev().collect(),
            free_pages: (0..pages).rev().collect(),
            live: vec![None; slots],
            next_epoch: 1,
        })
    }

    /// `positions` includes the complete prompt and maximum generated length.
    /// None leaves all resources untouched; keep that request queued and run
    /// already admitted requests until one finishes and releases its reservation.
    pub(crate) fn try_reserve(
        &mut self,
        positions: usize,
    ) -> Result<Option<Reservation>, &'static str> {
        if positions == 0 || positions > self.max_positions {
            return Err("request positions exceed the configured context");
        }
        let pages = Workspace::blocks_for(positions);
        if self.free_slots.is_empty() || self.free_pages.len() < pages {
            return Ok(None);
        }
        let epoch = self.next_epoch;
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or("reservation epoch exhausted")?;
        let slot = self.free_slots.pop().unwrap();
        let full: Vec<_> = (0..pages).map(|_| self.free_pages.pop().unwrap()).collect();
        self.live[slot] = Some((epoch, full.clone()));
        Ok(Some(Reservation {
            blocks: Blocks {
                full,
                sliding: slot,
            },
            epoch,
        }))
    }

    /// Verify ownership before returning any resources. A cloned/stale Blocks
    /// handle cannot release the same pages twice or release another request.
    pub(crate) fn release(&mut self, reservation: Reservation) -> Result<(), &'static str> {
        let Reservation { mut blocks, epoch } = reservation;
        let Some(entry) = self.live.get_mut(blocks.sliding) else {
            return Err("request slot is outside the admission pool");
        };
        if !entry
            .as_ref()
            .is_some_and(|(live_epoch, pages)| *live_epoch == epoch && pages == &blocks.full)
        {
            return Err("request does not own this page reservation");
        }
        *entry = None;
        self.free_pages.append(&mut blocks.full);
        self.free_slots.push(blocks.sliding);
        Ok(())
    }

    pub(crate) fn available_pages(&self) -> usize {
        self.free_pages.len()
    }
    pub(crate) fn total_pages(&self) -> usize {
        self.total_pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn physical32_queues_next_owner_without_changing_context_or_full_pages() {
        let mut pool = PageReservations::new(32, 256, 8192).unwrap();
        let mut admitted: Vec<_> = (0..32)
            .map(|_| pool.try_reserve(513).unwrap().unwrap())
            .collect();
        assert_eq!(admitted.last().unwrap().blocks.sliding, 31);
        assert_eq!(pool.available_pages(), 192);
        assert!(pool.try_reserve(8192).unwrap().is_none());
        assert_eq!(pool.available_pages(), 192);
        let completed = admitted.pop().unwrap();
        let stale = Reservation { blocks: completed.blocks.clone(), epoch: completed.epoch };
        pool.release(completed).unwrap();
        let next = pool.try_reserve(8192).unwrap().unwrap();
        assert_eq!(next.blocks.sliding, 31);
        assert_eq!(next.blocks.full.len(), 16);
        assert_eq!(pool.available_pages(), 178);
        assert!(pool.release(stale).is_err());
        pool.release(next).unwrap();
        for owner in admitted { pool.release(owner).unwrap(); }
        assert_eq!(pool.available_pages(), 256);
    }
    #[test]
    fn long_requests_queue_before_execution_without_page_deadlock() {
        let mut pool = PageReservations::new(64, 256, 8192).unwrap();
        let mut admitted = Vec::new();
        for _ in 0..16 {
            admitted.push(pool.try_reserve(8192).unwrap().unwrap());
        }
        assert_eq!(pool.available_pages(), 0);
        for _ in 0..3 {
            assert!(pool.try_reserve(8192).unwrap().is_none());
        }
        // Any admitted request has all16 pages needed to finish. Completion frees
        // exactly enough for the next queued maximum-length request.
        let first = admitted.remove(0);
        let slot = first.blocks.sliding;
        pool.release(first).unwrap();
        let next = pool.try_reserve(8192).unwrap().unwrap();
        assert_eq!(next.blocks.sliding, slot);
        assert_eq!(next.blocks.full.len(), 16);
        admitted.push(next);
        let pages: BTreeSet<_> = admitted
            .iter()
            .flat_map(|b| b.blocks.full.iter().copied())
            .collect();
        assert_eq!(pages.len(), pool.total_pages());
        for allocation in admitted {
            pool.release(allocation).unwrap();
        }
        assert_eq!(pool.available_pages(), 256);
    }
    #[test]
    fn short_requests_use_all64_slots_and_failed_reservation_is_atomic() {
        let mut pool = PageReservations::new(64, 256, 8192).unwrap();
        let admitted: Vec<_> = (0..64)
            .map(|_| pool.try_reserve(513).unwrap().unwrap())
            .collect();
        assert_eq!(pool.available_pages(), 128);
        assert!(pool.try_reserve(1).unwrap().is_none());
        assert_eq!(pool.available_pages(), 128);
        let pages: BTreeSet<_> = admitted
            .iter()
            .flat_map(|b| b.blocks.full.iter().copied())
            .collect();
        assert_eq!(pages.len(), 128);
        for allocation in admitted {
            pool.release(allocation).unwrap();
        }
        assert_eq!(pool.available_pages(), 256);
    }
    #[test]
    fn stale_foreign_and_oversized_requests_cannot_change_allocator_state() {
        let mut pool = PageReservations::new(64, 256, 8192).unwrap();
        assert!(pool.try_reserve(8193).is_err());
        assert!(pool.try_reserve(0).is_err());
        let first = pool.try_reserve(100).unwrap().unwrap();
        let stale = Reservation {
            blocks: first.blocks.clone(),
            epoch: first.epoch,
        };
        let mut foreign = Reservation {
            blocks: first.blocks.clone(),
            epoch: first.epoch,
        };
        foreign.blocks.full[0] += 1;
        assert!(pool.release(foreign).is_err());
        assert_eq!(pool.available_pages(), 255);
        pool.release(first).unwrap();
        let replacement = pool.try_reserve(100).unwrap().unwrap();
        assert_eq!(replacement.blocks.sliding, stale.blocks.sliding);
        assert_eq!(replacement.blocks.full, stale.blocks.full);
        assert!(
            pool.release(stale).is_err(),
            "old epoch cannot release the next request"
        );
        assert_eq!(pool.available_pages(), 255);
        pool.release(replacement).unwrap();
        assert_eq!(pool.available_pages(), 256);
    }
}
