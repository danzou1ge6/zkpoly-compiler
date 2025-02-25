# Type2 IR Layer

Type2 IR is the middle-level intermediate representation in the ZKPoly compiler. This IR introduces explicit typing, memory considerations, and computation graph optimizations.

## Core Components

### Computation Graph (Cg)

The Computation Graph (`Cg`) is the central data structure of Type2 IR, representing a complete computation as a directed graph of operations.

Key characteristics:
- Directed graph structure with vertices representing operations
- Explicit typing information for all vertices
- Support for various optimization passes
- Tracks input and output relationships between operations

### Vertex

A Vertex represents a single operation in the computation graph. Each vertex consists of:

1. **VertexNode**: The operation itself
2. **Type**: The data type produced by the operation
3. **SourceInfo**: Source location tracking for debugging and error reporting

### VertexNode Types

VertexNode defines all possible operations in the Type2 IR. These include:

#### Memory Operations
- **NewPoly**: Creates a new polynomial (zeros or ones)
- **Constant**: Represents a constant value
- **Extend**: Extends a polynomial to a higher degree

#### Arithmetic Operations
- **SingleArith**: A single arithmetic expression
- **Arith**: Complex arithmetic expression graph
- **ScalarInvert**: Computes the multiplicative inverse of a scalar

#### Transformation Operations
- **Ntt**: Converts between polynomial representations (Coefficient ↔ Lagrange)
- **RotateIdx**: Rotates a polynomial by shifting indices
- **Slice**: Extracts a segment of a polynomial
- **BatchedInvert**: Computes inverses for all elements in a polynomial
- **KateDivision**: Polynomial division for Kate commitments
- **EvaluatePoly**: Evaluates a polynomial at a point
- **ScanMul**: Computes running products across a polynomial
- **DistributePowers**: Distributes powers across a polynomial

#### Cryptographic Operations
- **Msm**: Multi-scalar multiplication
- **HashTranscript**: Adds a value to a cryptographic transcript
- **SqueezeScalar**: Extracts a challenge from a transcript
- **Blind**: Blinds a polynomial with random factors

#### Data Structure Operations
- **Array**: Creates an array from elements
- **ArrayGet**: Accesses an element from an array
- **TupleGet**: Accesses an element from a tuple
- **AssmblePoly**: Assembles a polynomial from coefficients

#### Control Flow
- **Entry**: Represents an input to the computation
- **Return**: Designates the output of the computation
- **UserFunction**: Calls a user-defined function

## Type System

Type2 IR introduces a more detailed type system:

- **Scalars**: Field elements
- **Points**: Curve points
- **Polynomials**: Specified with type (Coefficient/Lagrange) and degree
- **PointBase**: Base points for MSM operations
- **Transcript**: Cryptographic transcripts
- **Arrays**: Homogeneous collections with known length
- **Tuples**: Heterogeneous collections

The type system also tracks memory size information to facilitate efficient allocation.

## Device Targeting

Operations in Type2 IR are annotated with their preferred execution device:
- **GPU**: Operations that benefit from GPU acceleration
- **CPU**: Operations better suited for CPU execution
- **PreferGpu**: Operations that can run on either but prefer GPU when available

## Optimization Passes

Type2 IR supports several optimization passes:

### Graph Scheduling
Determines the optimal order of operations based on dependencies and reduce the memory footprint to reduce the need of transfering between devices

### Kernel Fusion
Combines compatible operations(mainly arith ops) to reduce global memory transfers and kernel launches.

### Memory Planning
Allocates gpu memory efficiently, reusing memory when possible, with several sub-components:
- **Object Analysis**: Analyzes object lifetimes
- **Integral Allocator**: Allocates memory in power-of-two chunks

This pass eliminate the need to dynamically allocate memory on gpu in the runtime, reducing both the memory allocation overhead and the fragmentations.
### Precompute
Identifies values that can be computed ahead of time.

### Manage Inverse
Sepreating poly and scalar inversion operations from kernel fusion to reduce the amount or computation.

## Example Computation Graph

A simple computation like `c = a + b` where `a` and `b` are polynomials would be represented as:

1. Entry vertices for inputs `a` and `b`
2. A vertex with `Arith` operation to perform addition
3. A Return vertex pointing to the result

More complex operations get broken down into a graph of these primitive operations.

## Building and Manipulating Type2 IR

Type2 IR is not typically constructed directly by users but is generated from the AST through lowering. The lowering process:

1. Traverses the AST
2. Performs type inference
3. Converts AST operations to Type2 operations
4. Constructs the computation graph

Once constructed, the Type2 IR can be optimized through the various passes before being lowered to Type3 IR.

## Type2 vs AST

Unlike the AST, which focuses on representing the user's code structure, Type2 IR is designed for optimization and execution planning. Key differences:

- Type2 has explicit types and memory information
- Type2 represents computation as a directed graph rather than a tree
- Type2 includes specialized operations for efficient execution
- Type2 can be optimized through various passes
- Type2 includes device targeting information