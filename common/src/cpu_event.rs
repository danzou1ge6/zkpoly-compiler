use event_listener::{Event, Listener};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct CpuEvent {
    event: Event,
    notified: AtomicBool,
}

impl CpuEvent {
    pub fn reset(&mut self) {
        self.notified.store(false, Ordering::SeqCst);
    }
    
    pub fn new() -> Self {
        Self {
            event: Event::new(),
            notified: AtomicBool::new(false),
        }
    }

    pub fn notify(&self) {
        if !self.notified.swap(true, Ordering::SeqCst) {
            self.event.notify(usize::MAX); // Wake all listeners
        }
    }

    pub fn wait(&self) {
        if !self.notified.load(Ordering::SeqCst) {
            let listener = self.event.listen();
            if !self.notified.load(Ordering::SeqCst) {
                listener.wait();
            }
        }
    }
}