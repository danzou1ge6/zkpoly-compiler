use zkpoly_cuda_api::stream::CudaStream;

pub trait Transport {
    fn cpu2gpu(&self, target: &mut Self, stream: &CudaStream);
    fn gpu2cpu(&self, target: &mut Self, stream: &CudaStream);
    fn gpu2gpu(&self, target: &mut Self, stream: &CudaStream);
    fn cpu2cpu(&self, target: &mut Self);
    fn cpu2disk(&self, target: &mut Self);
    fn disk2cpu(&self, target: &mut Self);
}
