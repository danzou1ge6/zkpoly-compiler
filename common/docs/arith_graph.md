# Arithmetic Computation Graph

This document provides a formal description of the arithmetic computation graph system implemented in `arith.rs`.

## Overview

The arithmetic computation graph (`ArithGraph`) is a directed acyclic graph (DAG) that represents mathematical operations and their dependencies. It is designed to model computation flows involving polynomials and scalar values within the zkpoly compiler ecosystem.

## Graph Structure

An `ArithGraph<OuterId, ArithIndex>` consists of:

- A directed graph (`Digraph`) containing vertices representing operations
- A list of input node IDs
- A list of output node IDs

Where:
- `OuterId`: An identifier type for external/global references (e.g., variable names)
- `ArithIndex`: An identifier type for internal graph nodes (typically a `UsizeId`)

## Node Types

Each vertex in the graph is an `ArithVertex` that contains an `Operation`. There are three primary types of operations:

### 1. Input Operation

```rust
Operation::Input {
    outer_id: OuterId,
    typ: FusedType,
    mutability: Mutability,
}
```

**Purpose**: Represents input data from outside the computation graph.

**Properties**:
- `outer_id`: External reference to the input data
- `typ`: Data type (`FusedType::Scalar` or `FusedType::ScalarArray`)
- `mutability`: Whether the input can be modified (`Mutability::Const` or `Mutability::Mut`)

**Dependencies**: None (leaf nodes in the graph)

### 2. Arithmetic Operation

```rust
Operation::Arith(Arith<ArithIndex>)
```

**Purpose**: Represents a mathematical operation on one or more inputs.

**Arith Types**:
1. Binary Operations:
   ```rust
   Arith::Bin(BinOp, lhs_index, rhs_index)
   ```
   - `BinOp`: The operation type, which can be:
     - `BinOp::Pp(ArithBinOp)`: Polynomial-Polynomial operations
     - `BinOp::Ss(ArithBinOp)`: Scalar-Scalar operations
     - `BinOp::Sp(SpOp)`: Scalar-Polynomial operations
   - `lhs_index`: Index of left-hand operand node
   - `rhs_index`: Index of right-hand operand node

2. Unary Operations:
   ```rust
   Arith::Unr(UnrOp, operand_index)
   ```
   - `UnrOp`: The operation type, which can be:
     - `UnrOp::P(ArithUnrOp)`: Polynomial unary operations
     - `UnrOp::S(ArithUnrOp)`: Scalar unary operations
   - `operand_index`: Index of the operand node

**Dependencies**: References to other nodes in the graph through indices

### 3. Output Operation

```rust
Operation::Output {
    outer_id: OuterId,
    typ: FusedType,
    store_node: ArithIndex,
    in_node: Vec<ArithIndex>,
}
```

**Purpose**: Represents the output or result of a computation.

**Properties**:
- `outer_id`: External reference for the output
- `typ`: Data type (`FusedType::Scalar` or `FusedType::ScalarArray`) 
- `store_node`: Index of the node whose value will be stored
- `in_node`: List of input nodes that this output depends on, the dependency is typically because of the reuse of input buffer to be an output buffer

**Dependencies**: References to the storing node and possibly other input nodes

## Arithmetic Operations

### Binary Operations

The following binary operations are supported:

#### ArithBinOp (for Polynomial-Polynomial and Scalar-Scalar)
- `ArithBinOp::Add` - Addition
- `ArithBinOp::Sub` - Subtraction
- `ArithBinOp::Mul` - Multiplication
- `ArithBinOp::Div` - Division

#### SpOp (for Scalar-Polynomial)
- `SpOp::Add` - Add scalar to polynomial
- `SpOp::Sub` - Subtract scalar from polynomial
- `SpOp::SubBy` - Subtract polynomial from scalar
- `SpOp::Mul` - Multiply polynomial by scalar
- `SpOp::Div` - Divide polynomial by scalar
- `SpOp::DivBy` - Divide scalar by polynomial

### Unary Operations

The following unary operations are supported:

#### ArithUnrOp
- `ArithUnrOp::Neg` - Negation
- `ArithUnrOp::Inv` - Multiplicative inverse

## Dependency Representation

Dependencies in the graph are represented in several ways:

1. **Direct References**: Arithmetic operations directly reference their operand nodes through indices.

2. **Uses Method**: Each vertex provides a `uses()` method that returns an iterator over all nodes it depends on.

3. **Graph Structure**: The underlying `Digraph` maintains the edges between nodes, allowing for traversal and analysis.

## Data Flow

Data flows through the graph as follows:

1. **Inputs**: Data enters the graph through Input operations.

2. **Processing**: Arithmetic operations consume data from their dependencies and produce new values.

3. **Outputs**: Results are captured by Output operations, which reference the final computational nodes.

4. **Execution Order**: The graph can be topologically sorted to determine the correct execution order that respects all dependencies.

## Graph Manipulation

The `ArithGraph` provides several methods for analysis and manipulation:

- `topology_sort()`: Returns nodes in dependency-respecting order
- `uses()`: Gets all external references used in the graph
- `mutable_uses()`: Gets mutable external references
- `relabeled()`: Creates a new graph with remapped external references

## Polynomial Type Support

Not all operations are supported for all polynomial types (coefficient form vs. Lagrange form). Methods like `support_coef()` indicate whether an operation can be applied to polynomials in coefficient form.

## Example Usage

A typical flow for using the arithmetic graph:

1. Create a new graph
2. Add input nodes for external data
3. Build computation by adding arithmetic operations
4. Define output nodes to capture results
5. Perform a topological sort to determine evaluation order
6. Execute the operations in order

## Relation to Other Components

The arithmetic graph relies on other components in the common module:

- `digraph`: Provides the underlying graph structure
- `heap`: Handles ID allocation for nodes
- `typ`: Defines data types for values in the graph

## Usage in the Compiler Pipeline

The arithmetic graph serves as an intermediate representation in the compiler pipeline, allowing for:

- Computation scheduling
- Optimization of operations
- Analysis of data dependencies
- Code generation for efficient execution (kernel fusion)