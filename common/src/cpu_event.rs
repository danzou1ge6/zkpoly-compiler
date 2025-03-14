use std::sync::{Condvar, Mutex};

// note that this event can only be used once
pub struct CpuEvent {
    cond_var: Condvar,
    mutex: Mutex<bool>,
}

impl CpuEvent {
    pub fn new() -> Self {
        Self {
            cond_var: Condvar::new(),
            mutex: Mutex::new(false),
        }
    }

    pub fn notify(&self) {
        let mut notified = self.mutex.lock().unwrap();
        assert!(*notified == false); // Only notify once
        *notified = true;
        self.cond_var.notify_all();
    }

    pub fn wait(&self) {
        let mut notified = self.mutex.lock().unwrap();
        while !*notified {
            notified = self.cond_var.wait(notified).unwrap();
        }
    }
}
