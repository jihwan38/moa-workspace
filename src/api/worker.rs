use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use crate::host::generate::{GenerationOutput, GenerationRequest};
use crate::host::tokenizer::Tokenizer;
use crate::host::worker::Worker;

pub enum StreamEvent {
    Delta(String),
    ReasoningDelta(String),
    Done(GenerationOutput),
    Error(String),
}

pub struct Job {
    pub request: GenerationRequest,
    pub stream: bool,
    pub events: mpsc::Sender<StreamEvent>,
}

#[derive(Clone)]
pub struct WorkerHandle {
    jobs: mpsc::Sender<Job>,
    alive: Arc<AtomicBool>,
}

impl WorkerHandle {
    pub fn submit(&self, job: Job) {
        if let Err(mpsc::SendError(job)) = self.jobs.send(job) {
            let _ = job
                .events
                .send(StreamEvent::Error("generation worker is not running".to_owned()));
        }
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }
}

pub fn spawn(tokenizer: Tokenizer) -> Result<WorkerHandle, String> {
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let alive = Arc::new(AtomicBool::new(true));
    let worker_alive = Arc::clone(&alive);
    std::thread::Builder::new()
        .name("gemma4-worker".into())
        .spawn(move || {
            let _alive = AliveFlag(worker_alive);
            match Worker::new(tokenizer) {
                Ok(worker) => {
                    let _ = ready_tx.send(Ok(()));
                    worker.run(jobs_rx);
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.to_string()));
                }
            }
        })
        .map_err(|err| format!("failed to spawn the gemma4 generation worker thread: {err}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(WorkerHandle { jobs: jobs_tx, alive }),
        Ok(Err(message)) => Err(message),
        Err(_) => Err("generation worker thread exited before finishing startup".to_owned()),
    }
}

struct AliveFlag(Arc<AtomicBool>);

impl Drop for AliveFlag {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}
