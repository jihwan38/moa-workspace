//! Bounded FIFO collection before an idle worker begins its first model chunk.
//! The first receive fixes the deadline; later arrivals never extend it.
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub(super) const WINDOW: Duration = Duration::from_millis(2);

pub(super) fn receive<T>(jobs: &mpsc::Receiver<T>, capacity: usize) -> Option<Vec<(T, Instant)>> {
    let first = jobs.recv().ok()?;
    let received = Instant::now();
    Some(collect(first, received, capacity, WINDOW, Instant::now, |remaining| {
        jobs.recv_timeout(remaining).ok()
    }))
}

fn collect<T>(
    first: T,
    received: Instant,
    capacity: usize,
    window: Duration,
    mut now: impl FnMut() -> Instant,
    mut receive: impl FnMut(Duration) -> Option<T>,
) -> Vec<(T, Instant)> {
    assert!(capacity > 0);
    let deadline = received + window;
    let mut batch = Vec::with_capacity(capacity);
    batch.push((first, received));
    while batch.len() < capacity {
        let remaining = deadline.saturating_duration_since(now());
        if remaining.is_zero() { break; }
        let Some(job) = receive(remaining) else { break; };
        batch.push((job, now()));
    }
    batch
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn fifo_capacity_preserves_uncollected_jobs_and_disconnect_flushes() {
        let (tx, rx) = mpsc::channel();
        for i in 0..6 { tx.send(i).unwrap(); }
        drop(tx);
        let a = receive(&rx, 4).unwrap();
        let b = receive(&rx, 4).unwrap();
        assert_eq!(a.iter().map(|r| r.0).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!(b.iter().map(|r| r.0).collect::<Vec<_>>(), [4, 5]);
        assert!(receive(&rx, 4).is_none());
        assert!(a.windows(2).all(|r| r[0].1 <= r[1].1));
    }

    #[test]
    fn continuous_arrivals_do_not_extend_first_receipt_deadline() {
        let start = Instant::now();
        let clock = Cell::new(start);
        let mut waits = Vec::new();
        let mut next = 1;
        let batch = collect(0, start, 64, WINDOW, || clock.get(), |remaining| {
            waits.push(remaining);
            clock.set(clock.get() + Duration::from_millis(1));
            let value = next; next += 1; Some(value)
        });
        assert_eq!(waits, [Duration::from_millis(2), Duration::from_millis(1)]);
        assert_eq!(batch.iter().map(|r| r.0).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(batch[0].1, start);
        assert_eq!(batch[2].1, start + WINDOW);
    }

    #[test]
    fn timeout_or_single_capacity_returns_first_job_without_loss() {
        let start = Instant::now();
        let batch = collect(7, start, 64, WINDOW, || start, |_| None);
        assert_eq!(batch, [(7, start)]);
        let batch = collect(8, start, 1, WINDOW, || start, |_| panic!("capacity full"));
        assert_eq!(batch, [(8, start)]);
    }
}
