# ZKPoly Compiler

The ZKPoly Compiler is a specialized compiler designed for zero-knowledge proof (ZKP) computations, particularly focused on polynomial operations. This document describes the overall architecture of the compiler and its three-level intermediate representation (IR) structure.

## Overall Architecture

The compiler follows a multi-stage compilation pipeline:

1. **Front-end (AST)**: Parses user code into an Abstract Syntax Tree representation
2. **Type2 IR**: Transforms AST into a typed intermediate representation with additional semantic information
3. **Type3 IR**: Optimizes the IR for execution, focusing on memory management and stream management

This multi-level IR design allows for progressive lowering of high-level polynomial operations into efficient, executable code with various optimizations applied at each level.

## IR Levels in Detail

### 1. AST (Abstract Syntax Tree)

The AST layer represents the initial parsed form of user code, preserving the structure and semantics of the source.

#### Key Features:
- **Type System**: Supports various data types like polynomials (in both coefficient and Lagrange forms), scalars, points, arrays, and tuples
- **Expression Representation**: Organizes computations as a graph of nodes with source location tracking
- **Type Erasure**: Uses the `TypeEraseable` trait to enable generic operations across different types
- **Common Operations**: Defines arithmetic operations for different polynomial and scalar types

#### Core Types:
- `Poly`: Represents polynomials (both coefficient and Lagrange forms)
- `Scalar`: Represents field elements
- `Point`: Represents curve points
- `Array`, `Tuple`: Composite data structures
- `CommonNode`: Shared operations across different types (function calls, tuple/array access)
- `AstVertex`: Type-erased representation of any AST node

#### AST to Type2 Conversion:
The `lowering` module contains the type inference mechanism that converts AST nodes into Type2 IR, adding type information and validating operations.

### 2. Type2 IR

The Type2 IR is a mid-level representation that introduces explicit typing, memory considerations, and graph-based optimizations.

#### Key Features:
- **Precise Type System**: Enhances the AST types with size and memory information
- **Computation Graph**: Represents computations as an explicit directed graph
- **Memory Planning**: Introduces memory management for polynomial operations
- **Optimization Passes**: Includes several optimization passes:
  - `graph_scheduling`: Determines execution order
  - `kernel_fusion`: Combines compatible operations
  - `memory_planning`: Efficiently allocates memory
  - `precompute`: Identifies values that can be computed ahead of time
  - `manage_inverse`: Optimizes inversion operations

#### Core Types:
- `Typ<Rt>`: Detailed type representation with runtime type information
- `Size`: Memory size information for different types
- `Vertex`: Nodes in the computation graph
- `Cg`: Complete computation graph with inputs and outputs

### 3. Type3 IR

The Type3 IR is the lowest level representation, closest to execution. It focuses on memory efficiency and operation fusion.

#### Key Features:
- **Simplified Type System**: Uses the `PolyMeta` type to represent polynomial metadata
- **Memory Optimization**: Further optimizes memory usage and access patterns
- **Specialized Operations**: Adds support for sliced and rotated polynomial operations
- **Track Splitting**: Separates computations into independent tracks for better parallelism
- **Rotation Fusion**: Combines multiple rotations into a single operation

#### Core Types:
- `Typ`: The Type3 type system using `PolyMeta` for polynomial information
- `Slice`: Represents a view into a section of a polynomial
- `PolyMeta`: Contains metadata about polynomials (slice information, rotation amount)

## Compilation Process

1. User code is parsed into the AST representation
2. The type inferencer analyzes the AST and creates a typed representation
3. The AST is lowered to the Type2 IR with full type information
4. Various optimization passes are applied to the Type2 IR
5. The Type2 IR is lowered to the Type3 IR
6. Further optimizations are applied at the Type3 level
7. The final code is generated for execution

## Error Handling

Each IR level includes its own error types to handle issues specific to that level:
- AST level: Syntax and basic semantic errors
- Type2 level: Type compatibility issues, degree mismatches, and operation constraints
- Type3 level: Memory layout and execution errors

## Usage

The compiler module is typically used as a library within the larger ZKPoly framework, which handles the end-to-end process of compiling and executing zero-knowledge proof computations.