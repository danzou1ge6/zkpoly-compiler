# ZKPoly - Zero Knowledge Proof Polynomial Compiler

ZKPoly is a specialized compiler and runtime system designed for efficient zero-knowledge proof computations, with a focus on polynomial operations and cryptographic primitives. It provides a comprehensive stack for developing, compiling, and executing high-performance zero-knowledge proof applications on both CPU and GPU.

## Overview

Zero-knowledge proofs (ZKPs) are cryptographic protocols that allow one party to prove to another that a statement is true without revealing any information beyond the validity of the statement itself. These proofs are fundamental to many blockchain and cryptographic applications, including privacy-preserving transactions, confidential computing, and verifiable computation.

ZKPoly addresses the computational challenges of ZKPs by:

1. Providing high-level abstractions for polynomial operations
2. Implementing efficient GPU accelerated cryptographic primitives
3. Optimizing memory usage and execution across heterogeneous devices
4. Automating the compilation and optimization of proof circuits

## System Architecture

ZKPoly is structured as a multi-layered system:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    User Code (ZKP Application)              │
  └───────────────────────────────┬─────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────┐
 ┌┤                         Compiler                            │
 |├─────────────────┬─────────────────────────┬─────────────────┤
 |│      AST        │        Type2 IR         │     Type3 IR    │
 |│  (Front-end)    │   (Middle-end + Opts)   │  (Back-end)     │
 |└─────────────────┴───────────────┬─────────┴─────────────────┘
 |                                  │
 |┌─────────────────────────────────▼─────────────────────────────┐
 |│                           Runtime                             ├─┐
 |├──────────────────┬───────────────────────┬────────────────────┤ |
 |│  Instruction Set │   Memory Management   │ Device Coordination│ |
 |└──────────────────┴──────────┬────────────┴────────────────────┘ |
 |                              │                                   |
 |┌─────────────────────────────▼───────────────────────────────┐   |
 └>                        Core Libraries                       │   |
  ├───────────────┬───────────────┬─────────────┬───────────────┤   |
  │   NTT Ops     │   Polynomial  │     MSM     │  Fused Kernels│   |
  │               │      Ops      │             │               │   |
  └───────────────┴───────────────┴─────────────┴───────────────┘   |
                                                                    |
  ┌──────────────────────────────────────────────────────────────┐  |
  │                    Low-level Infrastructure                  <──┘
  ├────────────────────────────┬─────────────────────────────────┤
  │        CUDA API            │        Memory Pool              │
  └────────────────────────────┴─────────────────────────────────┘
```

## Key Components

### Compiler

The compiler takes high-level ZKP specifications and progressively lowers them through three intermediate representations (IRs):

1. **AST (Abstract Syntax Tree)**: Front-end representation that preserves the structure of user code.
   - Type system for polynomials (coefficient and Lagrange forms), scalars, points, arrays, and tuples
   - Expression representation for arithmetic operations
   - Common cryptographic primitives like MSM and NTT

2. **Type2 IR**: Mid-level representation with explicit typing and memory considerations.
   - Computation graph representation for optimizations
   - Precise type system with memory size information
   - Device targeting (CPU/GPU)
   - Multiple optimization passes including memory planning, kernel fusion, and graph scheduling

3. **Type3 IR**: Low-level representation focused on execution.
   - Instruction sequence organized into execution tracks
   - Explicit memory addressing and management
   - Stream-based execution model
   - Device-specific optimization

### Runtime

The runtime is responsible for executing the compiled programs efficiently:

- **Instruction Set**: Fine-grained control over memory, computation, and synchronization
- **Cross-device Execution**: Seamless operation across CPU and GPU
- **Memory Management**: Efficient allocation and deallocation across devices
- **Synchronization**: Event-based coordination between operations
- **Parallel Execution**: Multi-threaded operations and stream-based concurrency

### Core Libraries

The core module implements fundamental cryptographic and mathematical operations:

- **NTT Operations**: Fast Number Theoretic Transform for polynomial operations
- **Polynomial Operations**: Addition, multiplication, evaluation, division, and other operations
- **MSM (Multi-Scalar Multiplication)**: Efficient batched operations for elliptic curve points
- **Fused Kernels**: Dynamically generated operations for optimal performance

### Support Libraries

- **CUDA API**: Safe Rust interface to NVIDIA's CUDA runtime API
- **Memory Pool**: High-performance slab allocator for CUDA pinned memory
- **Common Utilities**: Shared data structures and algorithms used throughout the system

## Common Operations

ZKPoly efficiently supports common operations required in zero-knowledge proofs:

- **Polynomial Arithmetic**: Addition, subtraction, multiplication in both coefficient and Lagrange forms
- **NTT Transforms**: Converting between coefficient and Lagrange representations
- **Polynomial Evaluation**: Evaluating polynomials at specific points
- **Kate Polynomial Division**: Computing quotients for polynomial commitments
- **Multi-Scalar Multiplication**: Efficient implementation of ∑(scalar_i · point_i)
- **Batch Operations**: Optimized parallel versions of common operations
- **Fiat-Shamir Transformations**: Transcript management for non-interactive proofs

## Environment Setup

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Rust 1.45.0 or higher

### Environment Variables

```
export CUDA_PATH=/path/to/cuda  # Tell Cargo where CUDA is installed
```

### Building

```bash
# Clone the repository
git clone https://github.com/danzou1ge6/zkpoly-compiler.git
cd zkpoly-compiler

# Build the project
cargo build --release
```

## Usage Examples

### Simple Polynomial Operations

```rust
use zkpoly_compiler::ast::{PolyCoef, Scalar};

// Create polynomials
let a = PolyCoef::new([1, 2, 3]);  // 1 + 2x + 3x²
let b = PolyCoef::new([4, 5]);     // 4 + 5x

// Perform operations
let c = a + b;                     // 5 + 7x + 3x²
let point = Scalar::from_u64(2);
let result = c.evaluate_at(point); // 5 + 7*2 + 3*2² = 5 + 14 + 12 = 31
```

### Zero-Knowledge Proof Circuit

```rust
use zkpoly_compiler::{ast::*, transit::*};

// Define a simple ZKP computation function
fn create_zkp_circuit() {
    // Create inputs
    let secret = Scalar::new("secret");
    let public_value = Scalar::new("public");
    
    // Compute commitment
    let g = Point::generator();
    let commitment = g * secret;
    
    // Verify relationship
    let computed = commitment * public_value;
    let expected = g * (secret * public_value);
    
    // Create constraint ensuring computed == expected
    let is_valid = computed.equals(expected);
    
    return is_valid;
}

// Compile the circuit
let compiler = Compiler::new();
let program = compiler.compile(create_zkp_circuit);

// Execute the compiled program
let runtime = Runtime::new();
let result = runtime.execute(program, {"public": 42});
```

## Performance Considerations

- **Memory Transfer**: Data transfer between CPU and GPU can be a bottleneck; minimize when possible
- **Kernel Fusion**: The compiler automatically fuses compatible operations to reduce kernel launches
- **Memory Planning**: Efficient memory allocation and reuse is critical for performance
- **Device Selection**: Operations are automatically assigned to the most appropriate device
- **Stream Management**: Use concurrent streams for overlapped execution

## Future Development

- **Disk Storage Support**: For handling larger-than-memory datasets
- **Distributed Execution**: Running computations across multiple machines
- **Additional Hardware Accelerators**: Support for specialized hardware beyond GPUs
- **Dynamic Instruction Optimization**: Runtime adaptation based on execution patterns

