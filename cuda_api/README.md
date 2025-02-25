# CUDA API for Rust

This library provides a safe Rust interface to NVIDIA's CUDA runtime API, enabling GPU programming in Rust with CUDA.

## Features

- **Automatic Bindings**: Auto-generated bindings to CUDA runtime API using bindgen
- **Memory Management**: Utilities for GPU memory allocation, freeing, and data transfers
- **Stream Management**: Support for CUDA streams and events for asynchronous operations
- **Error Handling**: Convenient macros for CUDA error checking
- **Safe API**: Rust-friendly wrappers around unsafe CUDA functions
- **Multi-device Support**: APIs that work with multiple GPUs

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Rust 1.45.0 or higher

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
cuda_api = { path = "path/to/cuda_api" }
```

The library requires the CUDA Toolkit to be installed. The build process will look for CUDA in the following locations:

1. The path specified by the `CUDA_PATH` environment variable
2. Default CUDA installation path (`/usr/local/cuda`)

## Usage

### Basic Memory Operations

```rust
use cuda_api::mem::{alloc_pinned, free_pinned, cuda_h2d, cuda_d2h};
use std::ffi::c_void;

// Allocate pinned memory
let host_array: *mut i32 = alloc_pinned(1024);

// Allocate GPU memory using CudaAllocator
let device_id = 0;
let allocator = CudaAllocator::new(device_id, 1024 * std::mem::size_of::<i32>());
let device_array = allocator.allocate::<i32>(0);

// Copy data from host to device
cuda_h2d(
    device_array as *mut c_void, 
    host_array as *const c_void, 
    1024 * std::mem::size_of::<i32>()
);

// Copy data back from device to host
let result_array: *mut i32 = alloc_pinned(1024);
cuda_d2h(
    result_array as *mut c_void, 
    device_array as *const c_void, 
    1024 * std::mem::size_of::<i32>()
);

// Free memory
free_pinned(host_array);
free_pinned(result_array);
// Allocator will automatically free device memory when dropped
```

### Using CUDA Streams

```rust
use cuda_api::stream::{CudaStream, CudaEvent};

// Create a stream for a specific device
let device_id = 0;
let stream = CudaStream::new(device_id);

// Allocate memory asynchronously
let device_array = stream.allocate::<f32>(1024);

// Create events for synchronization
let event = CudaEvent::new();

// Asynchronous memory operations
let host_data: Vec<f32> = vec![1.0; 1024];
stream.memcpy_h2d(device_array, host_data.as_ptr(), 1024);

// Record an event
stream.record(&event);

// Wait for operations to complete
event.sync();
// or
stream.sync();

// Free memory asynchronously
stream.free(device_array);
```

## API Reference

### Modules

- **bindings**: Auto-generated raw bindings to CUDA runtime API
- **error**: Error handling utilities for CUDA operations
- **mem**: Memory management functions and allocators
- **stream**: CUDA streams and events for asynchronous operations

### Key Components

#### Memory Management

- `alloc_pinned<T>`: Allocates pinned host memory
- `free_pinned<T>`: Frees pinned host memory
- `CudaAllocator`: Manages device memory allocation
- `cuda_h2d`: Copies data from host to device
- `cuda_d2h`: Copies data from device to host

#### Stream Management

- `CudaStream`: Manages a CUDA stream for asynchronous operations
- `CudaEvent`: Provides synchronization points within streams
- Stream-based memory operations: `allocate`, `free`, `memcpy_h2d`, `memcpy_d2h`, `memcpy_d2d`
- Synchronization methods: `sync`, `record`, `wait`

## Error Handling

The library provides a convenient `cuda_check!` macro for error handling:

```rust
use cuda_api::cuda_check;

unsafe {
    // Will automatically check for errors and panic with detailed information if one occurs
    cuda_check!(cudaMemcpy(dst, src, size, cudaMemcpyKind_cudaMemcpyHostToDevice));
}
```

## Building from Source

1. Ensure CUDA Toolkit is installed
2. Set the `CUDA_PATH` environment variable if CUDA is not in the default location
3. Run `cargo build`

