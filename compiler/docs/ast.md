# AST (Abstract Syntax Tree) Layer

The AST layer provides the basic building blocks for constructing computational expressions in ZKPoly. This document describes the node types and how to use them to build computation graphs.

## Available Node Types

### Primitive Types

#### Scalar
Represents a field element for arithmetic operations.

Usage examples:
- Creating a scalar: `Scalar::from_u64(42)`
- Arithmetic: `scalar1 + scalar2`, `scalar1 * scalar2`

#### Point
Represents a point on an elliptic curve, used for cryptographic operations.

Usage examples:
- Creating a point: `Point::from_coordinates(x, y)`
- Point operations: `point.double()`, `point1 + point2`

#### PolyCoef (Polynomial in Coefficient form)
Represents a polynomial where coefficients are stored directly.

Usage examples:
- Creating a polynomial: `PolyCoef::new([1, 2, 3])` (creates x² + 2x + 3)
- Evaluation: `poly.evaluate_at(x)`

#### PolyLagrange (Polynomial in Lagrange form)
Represents a polynomial through its evaluations at specific points.

Usage examples:
- Creating from evaluations: `PolyLagrange::from_evaluations(values)`
- Interpolation: `PolyLagrange::interpolate(points, values)`

#### Transcript
Manages cryptographic transcripts for proof systems.

Usage examples:
- Creating a transcript: `Transcript::new()`
- Adding data: `transcript.append("label", value)`
- Generating challenges: `transcript.squeeze_scalar()`

### Composite Types

#### Array
Fixed-size collection of elements with the same type.

Usage examples:
- Creating an array: `Array::new([elem1, elem2, elem3])`
- Accessing elements: `array.get(index)`

#### Tuple
Fixed-size collection of elements with potentially different types.

Usage examples:
- Creating tuples:
  - `Tuple2::new(scalar, point)`
  - `Tuple3::new(poly1, poly2, scalar)`
- Accessing elements: `tuple.get_0()`, `tuple.get_1()`

### Functions

#### User-Defined Functions
Custom functions can be defined and called within the computation graph.

Usage examples:
- Defining a function: `Function::new("add", |a, b| a + b)`
- Calling a function: `function.call(arg1, arg2)`

## Operations

### Arithmetic Operations

#### Scalar Arithmetic
- Addition: `scalar1 + scalar2`
- Subtraction: `scalar1 - scalar2`
- Multiplication: `scalar1 * scalar2`
- Division: `scalar1 / scalar2`
- Inversion: `scalar.invert()`

#### Polynomial Arithmetic (Lagrange Form)
- Addition: `poly1 + poly2`
- Subtraction: `poly1 - poly2`
- Multiplication with scalar: `scalar * poly`
- Negation: `-poly`

#### Polynomial Arithmetic (Coefficient Form)
- Addition: `poly1 + poly2`
- Subtraction: `poly1 - poly2`
- Addition with scalar: `poly + scalar`
- Subtraction with scalar: `poly - scalar` or `scalar - poly`
- Negation: `-poly`

### Polynomial-Specific Operations

#### NTT (Number Theoretic Transform)
Converts between coefficient and Lagrange forms.

Usage examples:
- Forward NTT: `poly_coef.to_lagrange()`
- Inverse NTT: `poly_lagrange.to_coef()`

#### Evaluation
Evaluates a polynomial at a specific point.

Usage example: `poly.evaluate_at(point)`

#### Interpolation
Creates a polynomial passing through given points.

Usage example: `PolyCoef::interpolate(xs, ys)`

### Cryptographic Operations

#### MSM (Multi-Scalar Multiplication)
Performs the sum of scalar-point products.

Usage example: `msm(scalars, points)`

#### Hash to Transcript
Adds values to a cryptographic transcript.

Usage example: `transcript.append("label", value)`

### Data Structure Operations

#### Array/Tuple Access
Retrieves elements from composite data structures.

Usage examples:
- Array access: `array.get(index)`
- Tuple access: `tuple.get_0()`, `tuple.get_1()`

## Building Computations

To build a computation using the AST:

1. Create primitive values (scalars, points, polynomials)
2. Combine them using arithmetic operations
3. Apply specialized operations (NTT, evaluation, MSM)
4. Return final results

Example of building a simple computation:
```
// Calculate (a*x^2 + b*x + c) evaluated at point p
let a = Scalar::from_u64(1);
let b = Scalar::from_u64(2);
let c = Scalar::from_u64(3);
let p = Scalar::from_u64(5);

// Build polynomial in coefficient form
let poly = PolyCoef::new([c, b, a]);  // 1*x^2 + 2*x + 3

// Evaluate at point p
let result = poly.evaluate_at(p);  // 1*5^2 + 2*5 + 3 = 38
```

## Node Type Semantics

Each node type has specific semantic meaning in the ZKPoly system:

- **Scalar**: Represents elements from a finite field (Zp)
- **Point**: Represents a point on an elliptic curve (E)
- **PolyCoef**: Represents a polynomial in coefficient form (f(x) = a₀ + a₁x + ... + aₙx^n)
- **PolyLagrange**: Represents a polynomial in Lagrange basis (through evaluations at roots of unity)
- **Transcript**: Represents a Fiat-Shamir transcript for non-interactive proofs
- **Array**: Represents a fixed-size homogeneous collection
- **Tuple**: Represents a fixed-size heterogeneous collection

Understanding these semantics is crucial for building correct zero-knowledge proof computations.