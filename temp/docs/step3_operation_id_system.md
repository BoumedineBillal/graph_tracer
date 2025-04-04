# Operation ID System

The Operation ID System is the third component of our PyTorch graph tracer implementation. This system provides a robust way to uniquely identify and track operations in a computational graph while preserving the relationship between operations and tensors.

## Design Overview

The operation ID system is designed to:

1. Assign unique identifiers to operations based on their type and inputs
2. Create a mapping between complex, descriptive IDs and simpler IDs for reference
3. Automatically handle initial tensor loading operations
4. Track tensor flow between operations in the computational graph
5. Maintain consistent indexing for operations with the same signature

## Core Components

The system consists of three main classes:

### 1. OperationIDManager

Responsible for creating and mapping operation IDs:

```python
class OperationIDManager:
    def __init__(self):
        self.op_type_counters = defaultdict(int)
        self.id_mapping = {}
        self.next_op_id = 0
        self.op_signature_counts = defaultdict(int)
```

Key methods:
- `get_load_operation_id(op_type)`: Creates IDs for load operations (initial tensors)
- `get_operation_id(op_name, input_op_ids)`: Creates IDs for operations based on inputs

### 2. Operation

Represents a single operation in the computational graph:

```python
class Operation:
    def __init__(self, op_name):
        self.op_name = op_name
        self.complex_id = None
        self.unique_id = None
        self.input_tensor_infos = []
        self.output_tensor_infos = []
        self.input_op_ids = []
        self.attributes = {}
```

Key methods:
- `add_input_tensor(tensor_info, source_op_id)`: Adds an input tensor and its source
- `add_output_tensor(tensor_info)`: Adds an output tensor
- `generate_operation_id(id_manager)`: Generates IDs based on inputs

### 3. OperationCollection

Manages a collection of operations and their relationships:

```python
class OperationCollection:
    def __init__(self):
        self.id_manager = OperationIDManager()
        self.operations = {}
        self.tensor_to_op = {}
        self.known_tensors = set()
```

Key methods:
- `create_operation(op_name, input_tensor_infos, output_tensor_infos)`: Creates an operation
- `create_load_operation(tensor_info)`: Creates a load operation for an initial tensor
- `is_known_tensor(tensor_info)`: Checks if a tensor has a source operation

## Operation ID Format

The operation ID system uses two types of IDs:

### 1. Complex ID

A descriptive ID that encodes operation type, index, and input dependencies:

- For load operations: `{op_type}*{index}`
  - Example: `load*0`

- For other operations: `{op_name}*{index}|{input_op_id1}|{input_op_id2}|...`
  - Example: `add*0|op*0|op*1`

The index is specific to operations with the same name and input signature, ensuring that identical operations get different indices.

### 2. Unique ID

A simpler ID for reference by other operations:

- Format: `op*{index}`
- Example: `op*2`

## Key Features

### Automatic Load Operations

When a tensor without a known source is encountered, a load operation is automatically created:

```python
def create_operation(self, op_name, input_tensor_infos, output_tensor_infos):
    # ...
    for tensor_info in input_tensor_infos:
        if not self.is_known_tensor(tensor_info):
            self.create_load_operation(tensor_info)
    # ...
```

### Input Signature-Based Indexing

Operations with the same name and input signature get incrementing indices:

```python
def get_operation_index(self, op_name, input_signature):
    signature_key = f"{op_name}|{input_signature}"
    index = self.op_signature_counts[signature_key]
    self.op_signature_counts[signature_key] += 1
    return index
```

This ensures that:
- `add*0|op*0|op*1` is the first addition of tensors from operations `op*0` and `op*1`
- `add*1|op*0|op*1` is the second addition of the same tensors
- `add*0|op*1|op*2` is a different addition with different inputs

### Tensor Source Tracking

The system maintains a mapping from tensors to their source operations:

```python
def register_tensor_source(self, tensor_info, op_id):
    self.tensor_to_op[tensor_info.tensor_id] = op_id
    self.known_tensors.add(tensor_info.tensor_id)
```

This allows for efficient lookup of a tensor's origin, supporting the construction of connected graphs.

## Usage Example

Here's how the system is used in practice:

```python
# Create a collection
collection = OperationCollection()

# Create tensors and TensorInfo objects
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
a_info = TensorInfo(a)
b_info = TensorInfo(b)

# Create an addition operation: c = a + b
c = a + b
c_info = TensorInfo(c)
op_add = collection.create_operation("add", [a_info, b_info], [c_info])

# Load operations for a and b will be created automatically
print(op_add)  # Operation(add, id=op*2, complex_id=add*0|op*0|op*1)

# Create a multiplication operation: d = c * c
d = c * c
d_info = TensorInfo(d)
op_mul = collection.create_operation("mul", [c_info, c_info], [d_info])
print(op_mul)  # Operation(mul, id=op*3, complex_id=mul*0|op*2)
```

## Integration with Tensor Analysis

The operation ID system works closely with the TensorInfo objects from the Tensor Analyzer (step 2):

1. TensorInfo objects identify specific tensors in the graph
2. Operations store references to their input and output TensorInfo objects
3. The system automatically tracks which operation produced each tensor

This integration enables the construction of a complete computational graph that accurately represents the flow of tensors through operations.

## Advantages

1. **Deterministic IDs**: Operations with the same inputs always get the same complex ID pattern
2. **Automatic Load Handling**: Initial tensors are automatically detected and handled
3. **Index Management**: Operations with the same signature get incrementing indices
4. **Memory Efficiency**: Uses tensor IDs rather than storing the tensors themselves
5. **Input Preservation**: The complex ID encodes the full history of tensor sources

## Next Steps

This operation ID system provides the foundation for:

1. Building decorators that automatically track tensor operations
2. Constructing a complete computational graph during model execution
3. Visualizing the tensor flow through PyTorch models
4. Analyzing patterns and optimizing model execution

The next step will focus on integrating this system with decorators to track operations during model execution.
