# Tensor Operation Analyzer

The Tensor Operation Analyzer is the second step in our PyTorch graph tracer implementation. This component examines function calls in PyTorch to determine if they involve tensor operations and tracks tensors as they flow through nested data structures.

**Key Implementation Update**: The main change in this implementation is that both `input_tensors` and `output_tensors` now contain **lists of TensorInfo objects** rather than dictionaries. This provides a more structured, object-oriented approach to storing tensor information.

## Design Overview

The tensor analyzer is designed to:

1. Track tensors through arbitrary nesting of data structures
2. Identify which operations involve tensors as inputs or outputs
3. Preserve exact paths to tensors within complex data structures
4. Execute operations safely and capture results or errors
5. Provide a detailed analysis that can be used for graph construction

## Core Components

### TensorInfo Class

A class that encapsulates information about a tensor and its path:

```python
class TensorInfo:
    def __init__(self, tensor=None)
    def set_path(self, path)
    def set_tensor(self, tensor)
```

The TensorInfo class stores:
- The path to the tensor (a list of path elements)
- The tensor's unique ID (for identifying the same tensor in different contexts)
- The tensor's shape

### TensorOperationAnalyzer Class

The main class that provides analysis functionality:

```python
class TensorOperationAnalyzer:
    def find_tensors_in_structure(self, obj, path=None)
    def analyze_operation(self, func, args, kwargs)
```

### Tensor Path Tracking

Tensors are tracked using a structured path representation:

- Each path is a list of dictionaries
- Each dictionary represents one level of nesting
- Dictionaries contain:
  - `type`: Container type ('list', 'dict', 'tuple', 'arg', 'kwarg', 'output')
  - Index or key: `'index'` for sequences, `'key'` for mappings

Example path to a tensor:

```python
[
  {'type': 'arg', 'index': 0},           # First positional argument
  {'type': 'dict', 'key': 'params'},     # Dictionary with key 'params'
  {'type': 'list', 'index': 2}           # List at index 2
]
```

This structured representation allows for precise tracking of tensor locations, even in deeply nested structures.

## Key Methods

### TensorInfo Methods

```python
def set_path(self, path):
    """
    Set the path to this tensor
    
    Args:
        path: List of path elements leading to this tensor
        
    Returns:
        self: For method chaining
    """
```

```python
def set_tensor(self, tensor):
    """
    Update tensor information
    
    Args:
        tensor: The tensor to capture information about
        
    Returns:
        self: For method chaining
    """
```

### find_tensors_in_structure

```python
def find_tensors_in_structure(self, obj, path=None):
    """
    Recursively find tensors in a nested structure and record their paths.
    
    Args:
        obj: The object to search for tensors
        path: Current path in the structure (default: empty list)
        
    Returns:
        list: List of TensorInfo objects, each containing:
            - path: List of dicts describing location
            - tensor_id: Unique identifier for the tensor
            - shape: Tensor shape as a list
    """
```

This method recursively explores a nested data structure, finding all tensors and creating TensorInfo objects for each one containing:
- The exact path to the tensor
- The tensor ID (to identify the same tensor in different locations)
- The shape of the tensor

It handles:
- Tensors (base case)
- Lists and tuples (recursive case with indexed access)
- Dictionaries (recursive case with key-based access)

### analyze_operation

```python
def analyze_operation(self, func, args, kwargs):
    """
    Analyze if a function call involves tensor operations.
    
    Args:
        func: The function/method being called
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        
    Returns:
        dict: Analysis results containing:
            - has_tensor_inputs: Boolean
            - has_tensor_outputs: Boolean
            - input_tensors: List of TensorInfo objects for inputs
            - output_tensors: List of TensorInfo objects for outputs
            - result: The original result of the function
            - error: Any error that occurred, or None
    """
```

This method:
1. Analyzes function arguments to find input tensors and creates TensorInfo objects for them
2. Executes the function with the provided arguments
3. Analyzes the result to find output tensors and creates TensorInfo objects for them
4. Handles errors that may occur during execution
5. Returns a comprehensive analysis dictionary

## Analysis Result Structure

The analysis result is a dictionary containing:

```python
{
    'has_tensor_inputs': bool,  # Whether any inputs are tensors
    'has_tensor_outputs': bool,  # Whether any outputs are tensors
    'input_tensors': [           # List of TensorInfo objects for inputs
        TensorInfo(),           # TensorInfo instance containing path, tensor_id, and shape
        ...
    ],
    'output_tensors': [          # List of TensorInfo objects for outputs
        TensorInfo(),           # TensorInfo instance containing path, tensor_id, and shape
        ...
    ],
    'result': value,             # The original result returned by the function
    'error': str or None         # Error message if an exception occurred
}
```

This structure clearly shows that the `input_tensors` and `output_tensors` fields contain **lists of TensorInfo objects**, not dictionaries as in the previous implementation.

## Usage Examples

### Finding Tensors in Nested Structure

```python
analyzer = TensorOperationAnalyzer()
nested_data = {
    'tensors': [tensor1, tensor2],
    'config': {'value': tensor3}
}

# Get a list of TensorInfo objects
tensor_infos = analyzer.find_tensors_in_structure(nested_data)

# Access information about each tensor
for info in tensor_infos:
    print(f"Tensor with shape {info.shape} found at path: {info.path}")
    print(f"Tensor ID: {info.tensor_id}")
```

### Analyzing a Simple Operation

```python
analyzer = TensorOperationAnalyzer()
analysis = analyzer.analyze_operation(
    torch.add, (tensor1, tensor2), {}
)

# Check if this is a tensor operation
if analysis['has_tensor_inputs'] and analysis['has_tensor_outputs']:
    print("This is a tensor-to-tensor operation")
    
# Access the result
result = analysis['result']  # Same as tensor1 + tensor2

# Access information from TensorInfo objects
for input_tensor in analysis['input_tensors']:
    print(f"Input tensor with ID {input_tensor.tensor_id}")
    print(f"Shape: {input_tensor.shape}")
    print(f"Path: {input_tensor.path}")
```

### Analyzing a Complex Operation

```python
def process_data(config):
    tensors = config['tensors']
    return torch.cat(tensors, dim=config['params']['dim'])

analysis = analyzer.analyze_operation(
    process_data, 
    ({"tensors": [t1, t2], "params": {"dim": 0}},),
    {}
)

# Check inputs and outputs
print(f"Input tensors: {len(analysis['input_tensors'])}")
print(f"Output tensors: {len(analysis['output_tensors'])}")

# Get tensor paths from TensorInfo objects
for i, tensor_info in enumerate(analysis['input_tensors']):
    # Format path for display
    path_elements = []
    for p in tensor_info.path:
        if p['type'] == 'arg':
            path_elements.append(f"args[{p['index']}]")
        elif p['type'] == 'dict' and 'key' in p:
            path_elements.append(f"['{p['key']}']")
        elif p['type'] == 'list' and 'index' in p:
            path_elements.append(f"[{p['index']}]")
    
    path_str = ''.join(path_elements)
    print(f"Input tensor #{i+1} at: {path_str}")
```

## Error Handling

The analyzer handles exceptions that might occur during function execution:

```python
def problematic_function(tensor):
    return tensor / 0  # Will raise exception

analysis = analyzer.analyze_operation(
    problematic_function, (tensor,), {}
)

if analysis['error'] is not None:
    print(f"Error occurred: {analysis['error']}")
    # Input tensors are still available
    print(f"Function was called with {len(analysis['input_tensors'])} tensor inputs")
```

## Key Advantages

1. **Object-Oriented Design**: Uses dedicated TensorInfo class for tensor data
2. **Non-Invasive Tracing**: Analyzes operations without modifying the original functions
3. **Path-Based Tracking**: Records precise paths to tensors in complex data structures
4. **Error Resilience**: Continues analysis even when operations fail
5. **Result Preservation**: Maintains original operation results for use in decorators
6. **Comprehensive Analysis**: Provides detailed information about tensor flow

## Limitations

1. **Tensor Methods**: Cannot directly detect 'self' tensor in method calls
2. **Custom Tensor-Like Objects**: Only detects standard PyTorch tensors
3. **External Effects**: Cannot track tensors modified by in-place operations
4. **Performance Overhead**: Analysis adds computational overhead to operations

## Extensibility

The object-oriented design allows for easy extension:

1. **Enhanced TensorInfo**: Additional methods can be added to TensorInfo as needed
2. **Custom Analysis**: The analyzer can be extended to perform more complex analysis
3. **Specialization**: Subclasses can provide domain-specific functionality
4. **Integration**: Can be combined with other components through composition

## Next Steps

This tensor operation analyzer provides the foundation for:

1. Building a comprehensive tensor flow graph
2. Creating decorators that track tensor operations during model execution
3. Visualizing how tensors move through PyTorch operations
4. Optimizing model execution by identifying bottlenecks

The next step will focus on using this analyzer to decorate PyTorch operations and build a connected graph that shows how tensors flow through a model.
