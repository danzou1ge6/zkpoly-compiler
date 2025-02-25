# Type3 IR Layer

Type3 IR is the lowest level intermediate representation in the ZKPoly compiler. This IR is designed to be closer to execution, focusing on memory management, device-specific optimizations, and execution streams.

## Core Components

### Chunk

A Chunk is the central execution unit in Type3 IR, representing a sequence of instructions to be executed. It contains:

- **Instructions**: Sequential list of operations to execute
- **Register Types**: Type information for each register
- **Register Devices**: Mapping of registers to execution devices
- **Address Mapping**: Memory address assignments for GPU memory
- **Register Allocator**: Manages register IDs

### Instructions

Instructions are the basic operations executed within a Chunk. Each instruction consists of:

1. **InstructionNode**: The operation to perform
2. **SourceInfo**: Optional source location for debugging

### Tracks

Type3 IR organizes instructions into execution tracks based on their device requirements and dependencies:

- **MemoryManagement**: Memory allocation and deallocation operations
- **CoProcess**: Operations that involve coordination between CPU and GPU
- **Gpu**: Operations executed on the GPU
- **Cpu**: Operations executed on the CPU
- **ToGpu**: Data transfer operations from CPU to GPU
- **FromGpu**: Data transfer operations from GPU to CPU
- **GpuMemory**: GPU memory operations

### Memory Management

Type3 IR introduces sophisticated memory management:

- **Addr**: Represents a memory address (typically on GPU)
- **Size**: Memory size in two forms:
  - **IntegralSize**: Power-of-two sizes for efficient allocation
  - **SmithereenSize**: Arbitrary sizes for exact memory requirements
- **Device-specific allocations**: Separate memory management for GPU, CPU, and stack

## Instruction Types

### Memory Operations
- **GpuMalloc**: Allocates memory on the GPU
- **GpuFree**: Releases GPU memory
- **CpuMalloc**: Allocates memory on the CPU
- **CpuFree**: Releases CPU memory
- **StackFree**: Releases stack memory

### Data Movement
- **Transfer**: Transfers data between devices
- **Move**: Marks an object as moved (logical operation for tracking)
- **SetPolyMeta**: Updates polynomial metadata

### Computation
- **Type2**: Wraps a Type2 vertex operation for execution
- **Tuple**: Creates a tuple from components

## Type System

Type3 IR uses a simplified type system compared to Type2 IR:

- **ScalarArray**: Arrays of field elements with metadata
- **PointBase**: Base points for MSM operations
- **Scalar**: Field elements
- **Transcript**: Cryptographic transcripts
- **Point**: Curve points
- **Tuple/Stream/GpuBuffer**: Various container types

The type system adds support for:

- **PolyMeta**: Metadata for polynomials, including:
  - **Sliced**: View into a section of a polynomial
  - **Rotated**: Polynomial with rotated indices

## Optimization Passes

### Track Splitting

The track splitting pass separates operations into different execution tracks based on their device requirements. This enables:

- Parallel execution across devices
- Minimizing device synchronization
- Efficient memory management per device

### Rotation Fusion

Combines multiple polynomial rotations into more efficient operations to reduce:
- Memory bandwidth
- Computation redundancy

## Lowering from Type2 to Type3

The lowering process converts Type2 IR to Type3 IR through several steps:

1. Register allocation for all values
2. Device assignment based on operation characteristics
3. Memory planning for all devices
4. Instruction generation
5. Track assignment for instructions
6. Optimization passes (track splitting, rotation fusion)

## Type3 vs Type2

Key differences from Type2 IR:

- **Execution Focus**: Type3 IR is concerned with concrete execution rather than representation
- **Memory Addressing**: Explicit memory addresses instead of abstract identifiers
- **Device Specificity**: Clear separation of operations by execution device
- **Instruction Sequence**: Linear instruction sequence rather than a graph
- **Register-Based**: Uses registers rather than graph vertices for data flow
- **Metadata**: Enhanced support for polynomial views (sliced, rotated)
- **Multiple Tracks**: Operations separated into execution tracks for parallelism

## Chunk Execution Model

The execution of a Chunk follows these steps:

1. Memory is allocated according to the preplanned scheme
2. Instructions are executed in sequence within their respective tracks
3. Data is transferred between devices as needed
4. Results are produced in the specified output registers
5. Temporary memory is freed according to the preplanned schedule

This approach minimizes runtime overhead by performing as much planning as possible at compile time.

## Device Communication

Type3 IR explicitly models data movement between devices:

- CPU → GPU transfers (ToGpu track)
- GPU → CPU transfers (FromGpu track)
- Device-internal operations (Gpu and Cpu tracks)
- GPU memory management operations (GpuMemory track)

This explicit modeling allows for the overlapped data movement, which is often a bottleneck in heterogeneous computing.

## Stream Management

Type3 IR introduces the concept of execution streams for overlapping computation and data transfer. Stream operations are sequenced to maximize parallelism while respecting dependencies.