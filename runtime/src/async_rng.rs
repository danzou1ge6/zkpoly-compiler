use crossbeam_channel::{bounded, Receiver};
use rand::{rngs::OsRng, RngCore};
use std::thread;

/// 异步随机数生成器
/// 在一个独立的线程中不断生成随机数，多个消费者可以通过subscribe获取随机数
#[derive(Clone)]
pub struct AsyncRng {
    rx: Receiver<u64>,
}

impl AsyncRng {
    /// 创建一个新的异步随机数生成器
    ///
    /// # Returns
    /// * `AsyncRng` - 异步随机数生成器实例
    pub fn new(cap: usize) -> Self {
        let (tx, rx) = bounded(cap);

        // 启动生产者线程
        thread::spawn(move || {
            let mut rng = OsRng::new().unwrap();
            loop {
                let random_u64 = rng.next_u64();
                if let Err(_) = tx.send(random_u64) {
                    break;
                };
            }
        });

        Self { rx }
    }

    /// 非阻塞地尝试获取一个随机数
    ///
    /// # Returns
    /// * `Option<u64>` - 如果有可用的随机数则返回Some，否则返回None
    pub fn try_recv(&self) -> Option<u64> {
        self.rx.try_recv().ok()
    }

    /// 阻塞地等待并获取一个随机数
    ///
    /// # Returns
    /// * `Option<u64>` - 如果成功获取到随机数则返回Some，否则返回None
    pub fn recv(&self) -> Option<u64> {
        self.rx.recv().ok()
    }
}

impl rand_core::RngCore for AsyncRng {
    fn next_u32(&mut self) -> u32 {
        let val = self
            .recv()
            .unwrap_or_else(|| OsRng::new().unwrap().next_u64());
        (val >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.recv()
            .unwrap_or_else(|| OsRng::new().unwrap().next_u64())
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&val[..len]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), rand_core::Error> {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&val[..len]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_async_rng() {
        let rng1 = AsyncRng::new(10);
        let rng2 = rng1.clone();

        // 测试两个消费者是否都能接收到随机数
        let v1 = rng1.recv().unwrap();
        let v2 = rng2.recv().unwrap();

        assert_ne!(v1, 0);
        assert_ne!(v2, 0);

        // 测试克隆和多线程访问
        let rng3 = rng1.clone();

        thread::sleep(Duration::from_millis(100));
        let v3 = rng3.recv().unwrap();
        assert_ne!(v3, 0);
    }

    #[test]
    fn test_non_blocking() {
        let rng = AsyncRng::new(10);

        // 测试非阻塞接收
        match rng.try_recv() {
            Some(v) => assert_ne!(v, 0),
            _ => (), // 这是正常的，因为可能还没有生成随机数
        }
    }

    #[test]
    fn test_multi_thread() {
        let rng = AsyncRng::new(10);
        let rng2 = rng.clone();

        let handle = thread::spawn(move || {
            let val = rng2.recv().unwrap();
            assert_ne!(val, 0);
        });

        let val = rng.recv().unwrap();
        assert_ne!(val, 0);

        handle.join().unwrap();
    }
}
