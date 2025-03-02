# ZKPoly Runtime

## Overview

The ZKPoly Runtime is a specialized execution environment designed for zero-knowledge proof (ZKP) applications. It provides a unified instruction set and execution model that abstracts hardware-specific details, enabling efficient computation across different device types (CPU, GPU, and in future development, disk storage).

The runtime serves as the execution layer for the ZKPoly compiler, interpreting and executing the generated intermediate representation to perform complex cryptographic operations required for zero-knowledge proofs.

## Key Features

- **Cross-device Computation**: Seamlessly execute code across CPU and GPU devices
- **Fine-grained Memory Management**: Efficient allocation and deallocation of memory resources
- **Data Transfer Abstraction**: Simplified data movement between different memory hierarchies
- **Parallel Execution Model**: Support for multi-threaded operations and asynchronous processing
- **Synchronization Primitives**: Event-based coordination between operations and devices
- **Specialized for ZKP Applications**: Optimized for cryptographic operations commonly used in ZKPs

## Architecture

The ZKPoly Runtime consists of several key components:

### Core Components

1. **Runtime Engine**: Central execution engine that processes instructions
2. **Instruction Set**: A comprehensive set of operations covering memory management, data transfer, function calls, and control flow
3. **Memory Management**: Handles allocation and deallocation across different device types
4. **Variable Management**: Tracks and manages variables through their lifecycle
5. **Function Registry**: Maintains callable functions for cryptographic operations
6. **Event System**: Coordinates synchronization between operations
7. **Thread Pool**: Manages parallel execution of instructions

### Data Types

The runtime supports several specialized data types:

- **ScalarArray**: Arrays of field elements, the fundamental building blocks for polynomials
- **PointArray**: Arrays of elliptic curve points
- **Scalar**: Single field elements
- **Point**: Single elliptic curve points
- **Transcript**: For Fiat-Shamir transformations in ZKP protocols
- **Tuple**: Composite data structure consisting of other variables

## Instruction Set

The runtime operates on a set of instructions that provide fine-grained control over computation. These instructions are grouped into several categories:

### Memory Management
- **Allocate**: Reserve memory on a specific device
- **Deallocate**: Release previously allocated memory
- **RemoveRegister**: Remove a variable reference without freeing the underlying memory

### Data Transfer
- **Transfer**: Move data between devices (CPU/GPU)

### Function Execution
- **FuncCall**: Execute registered functions with mutable and immutable parameters

### Synchronization
- **Wait**: Block until a specific event completes
- **Record**: Mark the occurrence of an event

### Multi-threading
- **Fork**: Create a new execution thread
- **Join**: Wait for a thread to complete

### Data Operations
- **Rotation**: Circular shift of a scalar array
- **Slice**: Extract a segment from an array
- **SetSliceMeta**: Modify metadata of an array slice
- **LoadConstant**: Load a value from the constant table
- **AssembleTuple**: Create a tuple from individual variables
- **Blind**: Obfuscate a portion of data with random values

### Control Flow
- **Return**: Return a value from the execution

## Memory Management

The runtime provides a sophisticated memory management system to efficiently handle data across different devices:

- **Pinned Memory Pool**: For CPU operations, using page-locked memory for efficient CPU-GPU transfers
- **CUDA Allocators**: For GPU memory allocation with support for multiple devices
- **Slicing**: Zero-copy view into existing memory regions
- **Reference Counting**: Implicit through Rust's ownership system

## Device Support

### CPU
- Provides the baseline execution environment
- Handles control flow and orchestration
- Executes scalar operations when GPU acceleration is not available

### GPU
- Accelerates compute-intensive operations
- Supports multiple GPUs through device IDs
- Includes stream management for concurrent operations
- Optimized for cryptographic primitives like MSM (Multi-Scalar Multiplication) and NTT (Number Theoretic Transform)

### Disk (Future Development)
- Support for larger-than-memory datasets
- Persistent storage for intermediate results

## Multi-threading and Synchronization

The runtime supports parallel execution through:

1. **Thread Pool**: For CPU parallelism
2. **CUDA Streams**: For GPU operation concurrency
3. **Events**: For synchronization between operations
   - GPU events for device synchronization
   - Thread events for CPU thread coordination
4. **Fork/Join Model**: For expressing parallel execution patterns

## Usage Example

Below is a simplified example demonstrating usage of the runtime:

```rust
// Create runtime components
let mut variable = VariableTable::new();
let mut instructions = Vec::new();
let mut events = EventTable::new();

// Allocate memory
instructions.push(Instruction::Allocate {
    device: DeviceType::CPU,
    typ: Typ::scalar_array(1024),
    id: var_id,
    offset: None,
});

// Transfer data to GPU
instructions.push(Instruction::Transfer {
    src_device: DeviceType::CPU,
    dst_device: DeviceType::GPU { device_id: 0 },
    stream: Some(stream_id),
    src_id: cpu_var_id,
    dst_id: gpu_var_id,
});

// Execute function
instructions.push(Instruction::FuncCall {
    func_id: function_id,
    arg_mut: vec![output_id],
    arg: vec![input_id, stream_id],
});

// Create and run the runtime
let runtime = Runtime::new(
    instructions, variable, constants, thread_pool,
    functions, events, threads, cpu_allocator, gpu_allocators,
    rng, libs
);
let result = runtime.run();
```

## Performance Considerations

1. **Memory Transfer Overhead**: Data transfer between devices can be a bottleneck; minimize when possible
2. **Proper Synchronization**: Ensure correct event ordering to avoid race conditions
3. **Device Selection**: Choose appropriate device for each operation based on workload characteristics
4. **Memory Management**: Reuse allocated memory when possible to reduce allocation overhead

## Integration with ZKPoly Compiler

The runtime is designed to execute the intermediate representation generated by the ZKPoly compiler:

1. The compiler analyzes ZK circuit descriptions
2. It generates an optimized instruction sequence
3. The runtime executes these instructions efficiently

## Future Development

- **Disk Storage Support**: For handling larger-than-memory datasets
- **Distributed Execution**: Running computations across multiple machines
- **Additional Hardware Accelerators**: Support for specialized hardware beyond GPUs
- **Dynamic Instruction Optimization**: Runtime adaptation based on execution patterns

## Related Modules

- **memory_pool**: Provides efficient memory allocation strategies
- **cuda_api**: Low-level interface to CUDA operations
- **core**: Implements fundamental cryptographic operations
- **compiler**: Generates instruction sequences for the runtime

