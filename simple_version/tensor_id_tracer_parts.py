"""
Tensor ID Tracer - A utility for tracing tensor IDs through PyTorch operations,
organized in distinct parts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import contextlib
from typing import Any, List, Dict, Tuple, Optional, Set, Union

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
# PART 2: Define the tensor ID tracer
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
            
            return result
        
        return wrapper

#####################################################
# PART 3: Create a context manager for tracing
#####################################################

@contextlib.contextmanager
def trace_tensor_ids(functions_to_trace=None):
    """
    Context manager to trace tensor IDs in PyTorch operations.
    
    Args:
        functions_to_trace: Dictionary with keys 'torch', 'F', and 'tensor'
                           containing lists of function names to trace
    """
    print("Start tracing tensor IDs")
    
    # Default operations to trace if none specified
    if functions_to_trace is None:
        functions_to_trace = {
            'torch': ['cat', 'add', 'matmul'],
            'F': ['relu', 'conv2d', 'linear', 'batch_norm', 'adaptive_avg_pool2d'],
            'tensor': ['view', '__add__']
        }
    
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
        yield
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
# PART 4: Example usage and testing
#####################################################

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x







# Only run this if the file is executed directly
if __name__ == "__main__":
    # Create model and test input
    model = SimpleModel()
    test_input = torch.randn(1, 3, 8, 8)
    
    # Trace tensor IDs
    with trace_tensor_ids():
        output = model(test_input)
        
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

















