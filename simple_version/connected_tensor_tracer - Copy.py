"""
Connected Tensor Tracer - A comprehensive utility for tracing tensor flow through PyTorch operations,
with operation tracking and ONNX visualization capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import contextlib
from typing import Any, List, Dict, Tuple, Optional, Set, Union
import onnx
from onnx import helper, TensorProto

#####################################################
# PART 1: Define helper functions for tensor ID extraction
#####################################################

def extract_tensor_ids(args):
    """
    Extract tensor IDs from arguments, handling nested structures.
    
    Args:
        args: The arguments to search for tensors
        
    Returns:
        List of tuples (tensor_id, tensor_shape)
    """
    tensor_ids = []
    
    def _extract_from_item(item):
        if isinstance(item, torch.Tensor):
            tensor_ids.append((id(item), list(item.shape)))
        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                _extract_from_item(sub_item)
        elif isinstance(item, dict):
            for sub_item in item.values():
                _extract_from_item(sub_item)
    
    for arg in args:
        _extract_from_item(arg)
    
    return tensor_ids


def format_tensor_ids(tensor_ids):
    """Format tensor IDs for printing"""
    return [f"Tensor(id={tid}, shape={shape})" for tid, shape in tensor_ids]

#####################################################
# PART 2: Connected Operation Tracing System
#####################################################

class OperationTracer:
    """Traces operations by their connections through tensors"""
    
    def __init__(self):
        # Dictionary of known operations {op_id: operation_record}
        self.operations = {}
        
        # Set of known tensor IDs
        self.known_tensors = set()
        
        # Next operation ID
        self.next_op_id = 0
    
    def register_input_tensors(self, tensors):
        """Register initial input tensors"""
        for tensor in tensors:
            self.known_tensors.add(id(tensor))
    
    def record_operation(self, op_name, input_tensor_ids, output_tensor_ids):
        """Record an operation if it uses known tensors"""
        # Check if connected to known tensors
        connected = False
        for tid, _ in input_tensor_ids:
            if tid in self.known_tensors:
                connected = True
                break
                
        if not connected:
            return None
            
        # Create operation record
        op_id = self.next_op_id
        self.next_op_id += 1
        
        operation = {
            "id": op_id,
            "name": op_name,
            "inputs": input_tensor_ids,
            "outputs": output_tensor_ids
        }
        
        # Store operation
        self.operations[op_id] = operation
        
        # Add output tensors to known set
        for tid, _ in output_tensor_ids:
            self.known_tensors.add(tid)
            
        return op_id
        
    def get_operation_sequence(self):
        """Get all operations in recording order"""
        return list(self.operations.values())
    
    def find_connected_operations(self, op_id):
        """Find operations connected to a specific operation"""
        if op_id not in self.operations:
            return []
            
        op = self.operations[op_id]
        
        # Find operations that consume this operation's outputs
        next_ops = []
        
        # Output tensor IDs from this operation
        output_tids = set(tid for tid, _ in op["outputs"])
        
        # Find operations that use these output tensors as inputs
        for other_id, other_op in self.operations.items():
            if other_id == op_id:
                continue
                
            # Check if any input tensor of other_op is in our outputs
            for tid, _ in other_op["inputs"]:
                if tid in output_tids:
                    next_ops.append(other_id)
                    break
                    
        return next_ops

# Create a global tracer
operation_tracer = OperationTracer()

#####################################################
# PART 3: Define the tensor ID tracer
#####################################################

class TensorIdTracer(contextlib.ContextDecorator):
    """
    A context decorator for tracing tensor IDs through PyTorch operations.
    """
    def __init__(self, op_name):
        self.op_name = op_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract tensor IDs from inputs
            input_tensor_ids = extract_tensor_ids(args)
            input_tensor_ids.extend(extract_tensor_ids(list(kwargs.values())))
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Extract tensor IDs from output
            output_tensor_ids = []
            if isinstance(result, torch.Tensor):
                output_tensor_ids.append((id(result), list(result.shape)))
            else:
                output_tensor_ids.extend(extract_tensor_ids([result]))
            
            # Print input and output tensor IDs together
            print(f"Operation {self.op_name} input tensors: {format_tensor_ids(input_tensor_ids)} output tensors: {format_tensor_ids(output_tensor_ids)}")
            
            # Record operation if connected
            op_id = operation_tracer.record_operation(self.op_name, input_tensor_ids, output_tensor_ids)
            if op_id is not None:
                print(f"Connected operation: {self.op_name} (ID: {op_id})")
            
            return result
        
        return wrapper

#####################################################
# PART 4: Create a context manager for tracing
#####################################################

@contextlib.contextmanager
def trace_tensor_ids(functions_to_trace=None, input_tensors=None):
    """
    Context manager to trace tensor IDs in PyTorch operations.
    
    Args:
        functions_to_trace: Dictionary with keys 'torch', 'F', and 'tensor'
                           containing lists of function names to trace
        input_tensors: List of input tensors to register for connection tracking
    """
    print("Start tracing tensor IDs")
    
    # Default operations to trace if none specified
    if functions_to_trace is None:
        functions_to_trace = {
            'torch': ['cat', 'add', 'matmul'],
            'F': ['relu', 'conv2d', 'linear', 'batch_norm', 'adaptive_avg_pool2d'],
            'tensor': ['view', '__add__']
        }
    
    # Register input tensors for connection tracking if provided
    if input_tensors:
        operation_tracer.register_input_tensors(input_tensors)
    
    # Store original functions
    original_torch_functions = {}
    original_F_functions = {}
    original_tensor_methods = {}
    
    # Patch torch functions
    for func_name in functions_to_trace.get('torch', []):
        if hasattr(torch, func_name) and callable(getattr(torch, func_name)):
            try:
                original_torch_functions[func_name] = getattr(torch, func_name)
                setattr(torch, func_name, 
                        TensorIdTracer(f"torch.{func_name}")(original_torch_functions[func_name]))
            except Exception as e:
                print(f"Error patching torch.{func_name}: {e}")
    
    # Patch F functions
    for func_name in functions_to_trace.get('F', []):
        if hasattr(F, func_name) and callable(getattr(F, func_name)):
            try:
                original_F_functions[func_name] = getattr(F, func_name)
                setattr(F, func_name, 
                        TensorIdTracer(f"F.{func_name}")(original_F_functions[func_name]))
            except Exception as e:
                print(f"Error patching F.{func_name}: {e}")
    
    # Patch tensor methods
    for method_name in functions_to_trace.get('tensor', []):
        if hasattr(torch.Tensor, method_name) and callable(getattr(torch.Tensor, method_name)):
            try:
                original_tensor_methods[method_name] = getattr(torch.Tensor, method_name)
                setattr(torch.Tensor, method_name, 
                        TensorIdTracer(f"tensor.{method_name}")(original_tensor_methods[method_name]))
            except Exception as e:
                print(f"Error patching tensor.{method_name}: {e}")
    
    try:
        # Yield control back to the with block
        yield operation_tracer
    finally:
        # Restore original functions
        for func_name, original_func in original_torch_functions.items():
            setattr(torch, func_name, original_func)
        for func_name, original_func in original_F_functions.items():
            setattr(F, func_name, original_func)
        for method_name, original_method in original_tensor_methods.items():
            setattr(torch.Tensor, method_name, original_method)
        print("End tracing tensor IDs")

#####################################################
# PART 5: ONNX Visualization
#####################################################

def create_onnx_visualization(operations, output_path="model_visualization.onnx"):
    """
    Create an ONNX visualization from traced operations
    
    Args:
        operations: List of operation records from the tracer
        output_path: Path to save the ONNX file
    """
    # Create mappings for tensors
    tensor_names = {}  # Maps tensor IDs to names
    nodes = []
    
    # Process all tensors first to create consistent naming
    for op in operations:
        for tid, shape in op["inputs"] + op["outputs"]:
            if tid not in tensor_names:
                tensor_names[tid] = f"tensor_{tid}"
    
    # Create a node for each operation
    for i, op in enumerate(operations):
        # Get operation name
        op_name = op["name"]
        
        # Get input and output names
        input_names = [tensor_names[tid] for tid, _ in op["inputs"]]
        output_names = [tensor_names[tid] for tid, _ in op["outputs"]]
        
        # Create node (simplified, just for visualization)
        node = helper.make_node(
            op_name,  # Use the original operation name
            input_names,
            output_names,
            name=f"{op_name}_{i}"
        )
        
        nodes.append(node)
    
    # Create dummy input/output for the graph
    # Just choose the first and last tensors from operations
    inputs = []
    outputs = []
    
    # First operation's inputs as graph inputs
    if operations:
        for tid, shape in operations[0]["inputs"]:
            inputs.append(helper.make_tensor_value_info(
                tensor_names[tid], TensorProto.FLOAT, shape
            ))
    
    # Last operation's outputs as graph outputs
    if operations:
        for tid, shape in operations[-1]["outputs"]:
            outputs.append(helper.make_tensor_value_info(
                tensor_names[tid], TensorProto.FLOAT, shape
            ))
    
    # Create graph
    graph = helper.make_graph(
        nodes,
        "OperationVisualization",
        inputs,
        outputs
    )
    
    # Create model
    model = helper.make_model(graph, producer_name="TensorTracer")
    model.opset_import[0].version = 13  # Use a recent opset version
    
    # Save model
    onnx.save(model, output_path)
    print(f"Visualization saved to {output_path}")
    print(f"View it with Netron: https://netron.app/")
    
    return output_path

#####################################################
# PART 6: Example usage and testing
#####################################################

# Create a model for testing
class CustomModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CustomModel, self).__init__()
        
        # Initial convolutional block
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block - path A
        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        
        # Second convolutional block - path B
        self.conv2b = nn.Conv2d(16, 32, kernel_size=1)
        self.bn2b = nn.BatchNorm2d(32)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fourth convolutional block - residual connection
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Final classification layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Initial block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Parallel paths
        path_a = F.relu(self.bn2a(self.conv2a(x)))
        path_b = F.relu(self.bn2b(self.conv2b(x)))
        
        # Concatenate paths
        x = torch.cat([path_a, path_b], dim=1)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual connection
        identity = x
        x = self.bn4(self.conv4(x))
        x = x + identity  # Add operation (residual connection)
        x = F.relu(x)
        
        # Classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Only run this if the file is executed directly
if __name__ == "__main__":
    # Create model and test input
    model = CustomModel()
    test_input = torch.randn(1, 3, 8, 8)
    
    # Trace tensor IDs
    with trace_tensor_ids(input_tensors=[test_input]) as tracer:
        output = model(test_input)
    
    # Get the operation sequence
    operations = tracer.get_operation_sequence()
    
    # Print connected operations
    print("\nConnected Operations:")
    for op in operations:
        print(f"Operation: {op['name']}")
        print(f"  Inputs: {format_tensor_ids(op['inputs'])}")
        print(f"  Outputs: {format_tensor_ids(op['outputs'])}")
        
        # Find next connected operations
        next_ops = tracer.find_connected_operations(op['id'])
        if next_ops:
            next_op_names = [tracer.operations[op_id]['name'] for op_id in next_ops]
            print(f"  Next operations: {', '.join(next_op_names)}")
        print()
    
    # Create ONNX visualization
    onnx_path = create_onnx_visualization(operations)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
