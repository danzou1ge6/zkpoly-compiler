#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MsmConfig {
    pub window_size: u32,
    pub target_window: u32,
    pub cards: Vec<u32>,
    pub debug: bool,
    pub batch_per_run: u32,
    pub parts: u32,
    pub stage_scalers: u32,
    pub stage_points: u32,
    pub bits: u32,
}

impl MsmConfig {
    pub fn get_precompute(&self) -> u32 {
        let actual_windows = self.bits.div_ceil(self.window_size);
        let n_windows = if actual_windows < self.target_window {
            actual_windows
        } else {
            self.target_window
        };
        actual_windows.div_ceil(n_windows)
    }

    pub fn new(
        window_size: u32,
        target_window: u32,
        cards: Vec<u32>,
        debug: bool,
        batch_per_run: u32,
        parts: u32,
        stage_scalers: u32,
        stage_points: u32,
        bits: u32,
    ) -> Self {
        Self {
            window_size,
            target_window,
            cards,
            debug,
            batch_per_run,
            parts,
            stage_scalers,
            stage_points,
            bits,
        }
    }
}

impl Default for MsmConfig {
    fn default() -> Self {
        MsmConfig::new(16, 4, vec![0], false, 8, 2, 2, 2, 254)
    }
}
