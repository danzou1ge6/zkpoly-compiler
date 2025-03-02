# ZKPoly Common Module

The common module (`zkpoly_common`) provides a collection of foundational data structures and utilities used throughout the ZKPoly compiler project. This module serves as a shared library of components for zero-knowledge proof polynomial operations and manipulations.

## Overview

The common module implements various low-level utilities that are essential for the ZKPoly compiler pipeline, including:

- Type definitions and representations for polynomials and scalars
- Directed graph data structures for dependency tracking and computation scheduling
- Mathematical operation abstractions for polynomial arithmetic
- Utilities for memory management and allocation
- Configuration structures for multi-scalar multiplication (MSM)

## Key Components

### Type System (`typ.rs`)

Defines fundamental types used throughout the project:

- `Slice`: Represents a range with a start and length
- `PolyType`: Enum for polynomial representation types (Coefficient or Lagrange form)
- `PolyMeta`: Metadata for polynomials, supporting slicing and rotation
- Type template system for working with different type contexts

### Directed Graph (`digraph`)

A generic directed graph implementation with:

- Vertex and edge management
- Graph traversal algorithms (DFS, topological sort)
- Cycle detection
- Forward and backward traversal

### Arithmetic Expressions (`arith.rs`)

Data structures for representing and manipulating arithmetic expressions:

- Binary and unary operations
- Computation graphs for expressions
- Mutability control for operands
- Relabeling and traversal utilities

### Memory Management

- `heap.rs`: ID allocation and management
- `mm_heap.rs`: Memory management heap implementation

### Multi-Scalar Multiplication (`msm_config.rs`)

Configuration for multi-scalar multiplication operations in cryptographic contexts:

- Window size and target parameters
- Debug options
- Batching configurations
- Resource staging parameters

### Other Utilities

- `bijection.rs`: Bijective mapping utilities
- `interval_tree.rs`: Interval tree data structure
- `load_dynamic.rs`: Dynamic loading functionality

## Usage

The common module is used as a dependency by other components in the ZKPoly compiler project. Import specific components as needed:

```rust
use zkpoly_common::digraph::external::Digraph;
use zkpoly_common::typ::{PolyType, Slice};
use zkpoly_common::arith::{ArithBinOp, ArithGraph};
// etc.
```

## Dependencies

This module has minimal external dependencies:

- `libloading` (0.8): For dynamic library loading
- `pasta_curves` (0.5.1): Cryptographic curve implementations

## Integration

The common module is designed to integrate with the compiler, runtime, and core modules to provide shared functionality throughout the ZKPoly toolchain.