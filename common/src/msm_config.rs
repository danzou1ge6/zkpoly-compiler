use pasta_curves::arithmetic::CurveAffine;

#[derive(Debug, Clone)]
pub struct MSMConfig<P: CurveAffine> {
    pub _marker: std::marker::PhantomData<P>,
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

impl<P: CurveAffine> MSMConfig<P> {
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
            _marker: std::marker::PhantomData,
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
