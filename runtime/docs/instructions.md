# Runtime Instruction Definition

## Instruction Set Overview

The ZKPoly compiler's runtime instruction set is an intermediate representation designed for zero-knowledge proof applications. These instructions allow fine-grained control over memory allocation, data transfers, function calls, and multi-threaded operations.

## Instruction Types

### Memory Management
- **Allocate**: Allocate memory on the specified device
  - `device`: Type of device for memory allocation (CPU, GPU, or Disk)
  - `typ`: Data type for allocated memory
  - `id`: Unique identifier assigned to the variable
  - `offset`: Optional offset for GPU allocations

- **Deallocate**: Free previously allocated memory
  - `id`: Identifier of the variable to free

- **RemoveRegister**: Remove reference from register file only without freeing underlying memory
  - `id`: Identifier of the variable to remove

### Data Transfer
- **Transfer**: Transfer data between devices
  - `src_device`: Source device type
  - `dst_device`: Destination device type
  - `stream`: Optional GPU stream identifier
  - `src_id`: Source variable identifier
  - `dst_id`: Destination variable identifier

### Function Calls
- **FuncCall**: Call a function
  - `func_id`: Function identifier
  - `arg_mut`: List of mutable arguments
  - `arg`: List of immutable arguments

### Synchronization
- **Wait**: Wait for an event to complete
  - `slave`: Device type waiting for the event
  - `stream`: Optional GPU stream identifier, this is used when a gpu stream is the slave
  - `event`: Event identifier to wait for

- **Record**: Record an event
  - `stream`: Optional GPU stream identifier
  - `event`: Event identifier to record

### Multi-threading
- **Fork**: Create a new thread to execute a set of instructions
  - `new_thread`: Identifier for the new thread
  - `instructions`: List of instructions for the new thread to execute

- **Join**: Wait for a thread to complete
  - `thread`: Identifier of the thread to wait for

### Data Operations
- **Rotation**: Circular shift of a scalar array
  - `src`: Source variable identifier
  - `dst`: Destination variable identifier
  - `shift`: Amount to shift (can be negative)

- **Slice**: Extract a segment from a scalar array
  - `src`: Source variable identifier
  - `dst`: Destination variable identifier
  - `start`: Start index
  - `end`: End index

- **SetSliceMeta**: Directly operate on metadata
  - `src`: Source variable identifier
  - `dst`: Destination variable identifier
  - `offset`: Offset
  - `len`: Length

- **LoadConstant**: Load a value from the constant table
  - `src`: Constant identifier
  - `dst`: Destination variable identifier

- **AssembleTuple**: Assemble a tuple from separate variables
  - `vars`: List of variable identifiers
  - `dst`: Destination tuple variable identifier

- **Blind**: Obfuscate part of a scalar array with random data
  - `dst`: Destination variable identifier
  - `start`: Start index
  - `end`: End index

### Control Flow
- **Return**: Return a variable value
  - Parameter: Identifier of the variable to return

## Usage Example

A simple instruction sequence can be found in `tests/simple_instruction.rs`

## Design Considerations

The instruction set is designed to balance:
1. Fine-grained control over underlying resources
2. Support for cross-device execution (CPU/GPU/Disk)
3. Efficient memory management
4. Support for parallel computing models
5. Special requirements of zero-knowledge proof systems