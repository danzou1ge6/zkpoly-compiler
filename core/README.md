# Core Module

The Core module is a fundamental component of the zkpoly-compiler project, providing essential cryptographic and mathematical operations required for zero-knowledge proof systems. It implements various operators and algorithms optimized for high performance on both GPU and CPU platforms.

## Implemented Operators

### NTT (Number Theoretic Transform) Operations

The NTT module implements efficient Number Theoretic Transform algorithms that are essential for polynomial operations in finite fields:

- **SsipNtt**: Space-saving in-place NTT implementation optimized for GPU computation
- **SsipPrecompute**: Precomputation of twiddle factors for the NTT algorithm
- **RecomputeNtt**: Memory-efficient NTT implementation that recomputes values as needed rather than storing them
- **DistributePowers**: Distribution of powers for polynomial operations
- **GenPqOmegas**: Generation of PQ and Omega values for NTT stages

### Polynomial Operations

The polynomial module provides extensive functionality for manipulating polynomials in different representations:

- **PolyAdd**: Addition of polynomials (element-wise)
- **PolySub**: Subtraction of polynomials (element-wise)
- **PolyMul**: Pointwise multiplication of polynomials (element-wise)
- **PolyZero**: Generation of zero polynomials
- **PolyOneLagrange**: Generation of unit polynomials in Lagrange basis representation
- **PolyOneCoef**: Generation of unit polynomials in coefficient form
- **PolyEval**: Polynomial evaluation at a given point
- **KateDivision**: Kate  division for polynomial commitment schemes, calculateing $p(x)/(x-b)$
- **PolyScan**: Scan multiplication operation on polynomials
- **PolyInvert**: Batch inversion of polynomials
- **ScalarInv**: Inversion of individual scalar values

### MSM (Multi-Scalar Multiplication)

MSM is a critical operation in elliptic curve cryptography and zero-knowledge proofs:

- **MSM**: Efficient batched multi-scalar multiplication on GPU
- **MSMPrecompute**: Precomputation for MSM to optimize subsequent operations

### Fused Kernels

The fused kernels system allows dynamic generation and execution of fused operations as CUDA kernels:

- Arithmetic binary operations: Addition, Subtraction, Multiplication, Division
- Arithmetic unary operations: Negation, Inversion
- Support for both scalar and array operations
- noteice that division and inversion are supported but at a high cost, it is recommended to complete the invert sepreately using `ScalarInv` or `PolyInvert`

### CPU Kernels

CPU implementations of various algorithms:

- **InterpolateKernel**: Lagrange polynomial interpolation
- **AssmblePoly**: Assembly of polynomial coefficients
- **HashTranscript**: Hashing operations for transcript generation
- **HashTranscriptWrite**: Writing operations for transcript
- **SqueezeScalar**: Extracting challenge scalars from a transcript

## Architecture

The Core module is designed with a flexible architecture:

- Most GPU operations are implemented in CUDA/C++ and exposed through a Rust interface
- Uses jit compiling and dynamic library loading to support different field types and curves across the ffi
- Uses abstractions to handle memory and device management
- Supports both coefficient and Lagrange representations of polynomials

## Performance Considerations

- GPU implementations are optimized for parallel computation
- Batch operations are provided where applicable for improved throughput
- Fused operations reduce kernel launches and memory accesses

This core module serves as the computational foundation for the zkpoly-compiler project, enabling efficient zero-knowledge proof generation and verification.