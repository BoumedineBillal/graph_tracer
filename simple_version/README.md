# Connected Tensor Tracer for PyTorch

A lightweight, non-invasive tool for tracing tensor operations in PyTorch models and visualizing computation graphs.

## Overview

The Connected Tensor Tracer provides a way to:
1. Track tensor flow through PyTorch operations
2. Build a connected graph of operations
3. Visualize model architecture using ONNX format
4. Handle tensor ID reuse in PyTorch

Unlike PyTorch's built-in tracing tools, this tracer focuses specifically on tracking tensor connections between operations without modifying the model's behavior.

## How It Works

### Basic Operation Tracing

The tracer uses Python's context managers and function decoration to intercept PyTorch operations:

1. **Function Interception**: Temporarily replaces PyTorch functions (in `torch`, `torch.nn.functional`, and `torch.Tensor` methods) with decorated versions.

2. **ID Extraction**: Records tensor IDs and shapes before and after each operation.

3. **Connection Tracking**: Builds a graph of operations by tracking which tensors flow between operations.

4. **ONNX Visualization**: Converts the tracked operation graph into ONNX format for visualization with tools like Netron.

### Example Usage

```python
import torch
from connected_tensor_tracer import trace_tensor_ids, create_onnx_visualization

# Create your model
model = YourModel()
input_tensor = torch.randn(1, 3, 224, 224)

# Trace operations
with trace_tensor_ids(input_tensors=[input_tensor]) as tracer:
    output = model(input_tensor)

# Get operation sequence and visualize
operations = tracer.get_operation_sequence()
onnx_path = create_onnx_visualization(operations)
```

## Solving the Tensor ID Reuse Problem

### The Challenge

PyTorch reuses memory addresses (tensor IDs) when tensors go out of scope. This presents a significant challenge for tracing:

1. **ID Collisions**: When PyTorch reuses a tensor ID, it breaks the connection tracking because the same ID now refers to different tensors in different parts of the computation.

2. **False Connections**: Without handling ID reuse, the tracer would create incorrect connections between operations that aren't actually related.

3. **Graph Inconsistency**: This leads to a misleading computation graph that doesn't accurately represent the model's architecture.

4. **In-place Operations**: In-place operations like `relu(inplace=True)` reuse the same tensor ID for both input and output, creating additional challenges for correct connection tracking.

### Our Solution

The Connected Tensor Tracer solves this problem with a novel ID conflict resolution approach:

```python
def record_operation(self, op_name, input_tensor_ids, output_tensor_ids):
    # ...
    
    # Check for ID conflicts in outputs
    for tid, shape in output_tensor_ids:
        if tid in self.known_tensors:
            # This ID is already in use - create a new one for previous operations
            new_id = f"generated_{self.next_replacement_id}"
            self.next_replacement_id += 1
            
            # For in-place operations: update the current operation's input IDs first
            # if this output ID is also in the inputs of this operation
            for i, (input_tid, input_shape) in enumerate(input_tensor_ids):
                if input_tid == tid:
                    input_tensor_ids[i] = (new_id, input_shape)
            
            # Update all previous operations that use this ID
            for op in self.operations.values():
                # Update inputs
                for i, (input_tid, input_shape) in enumerate(op["inputs"]):
                    if input_tid == tid:
                        op["inputs"][i] = (new_id, input_shape)
                
                # Update outputs
                for i, (output_tid, output_shape) in enumerate(op["outputs"]):
                    if output_tid == tid:
                        op["outputs"][i] = (new_id, output_shape)
            
            # Replace old ID with new ID in known tensors
            self.known_tensors.remove(tid)
            self.known_tensors.add(new_id)
    
    # ...
```

The solution works as follows:

1. **Conflict Detection**: When a new operation produces a tensor with an ID that already exists in our known tensor set, we've detected a reuse case.
   ```python
   # Check for ID conflicts in outputs
   for tid, shape in output_tensor_ids:
       if tid in self.known_tensors:
           # Conflict detected
   ```

2. **In-place Operation Handling**: For in-place operations where the output tensor has the same ID as one of the input tensors, we update the input tensor ID in the current operation to maintain correct connections.
   ```python
   # For in-place operations: update the current operation's input IDs first
   for i, (input_tid, input_shape) in enumerate(input_tensor_ids):
       if input_tid == tid:
           input_tensor_ids[i] = (new_id, input_shape)
   ```

3. **Prior References Update**: We update all previous operations that referenced the old ID to use a newly generated unique ID.
   ```python
   # Update all previous operations that use this ID
   for op in self.operations.values():
       # Update inputs
       for i, (input_tid, input_shape) in enumerate(op["inputs"]):
           if input_tid == tid:
               op["inputs"][i] = (new_id, input_shape)
       
       # Update outputs
       for i, (output_tid, output_shape) in enumerate(op["outputs"]):
           if output_tid == tid:
               op["outputs"][i] = (new_id, output_shape)
   ```

4. **ID Recycling**: The current operation can then safely use the original ID without conflict, as PyTorch intended.
   ```python
   # No explicit code needed here - by not changing the current output tensor ID,
   # we allow the natural recycling of the ID as PyTorch intended
   ```

5. **Continuous Adaptation**: This approach continuously adapts to ID reuse as the model executes, maintaining a consistent graph structure.
   ```python
   # Replace old ID with new ID in known tensors
   self.known_tensors.remove(tid)
   self.known_tensors.add(new_id)
   ```

### Advantages of This Approach

1. **Non-invasive**: No changes to PyTorch's memory management or tensor creation.

2. **Accuracy**: Correctly represents actual tensor connections without false links.

3. **In-place Support**: Properly handles in-place operations like ReLU with inplace=True.

4. **Efficiency**: Minimal overhead since we only update operations when conflicts occur.

5. **Simplicity**: No need to maintain complex mapping structures or timestamp-based IDs.

## Implementation Details

The tracer consists of several key components:

1. **Tensor ID Extraction**: Recursively extracts tensor IDs from nested structures (lists, tuples, dictionaries).

2. **Operation Tracer**: Maintains a registry of operations and their tensor connections.

3. **Context Manager**: Provides a clean API for tracing specific sections of code.

4. **ONNX Converter**: Transforms the operation sequence into a visualizable graph format.

## Limitations

- Only traces operations that are explicitly intercepted by the tracer (must be in the functions_to_trace dictionary)
- Visualization is simplified compared to full ONNX export from PyTorch
- Does not trace custom operations unless specifically added
- While in-place operations are now handled correctly, complex models with multiple in-place operations may require careful analysis

## Future Improvements

- Add support for custom operations
- Improve visualization with tensor shape information
- Track gradient flow during backpropagation
- Add quantitative analysis of computation and memory usage
- Further enhance in-place operation handling for edge cases

## Acknowledgements

This tool was developed as a learning project to better understand PyTorch's internal operation and tensor management system. It's not intended to replace PyTorch's official tools but to provide a different perspective on model architecture.