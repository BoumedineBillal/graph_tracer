import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import types
import contextlib
import functools
from collections import defaultdict
import os
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable

# Import our custom modules
from tensor_operation_analyzer import TensorOperationAnalyzer, TensorInfo
from operation_id_system import Operation, OperationCollection, OperationIDManager
from pytorch_operation_collector import PyTorchOperationCollector


class OperationTracer:
    """
    Main class for tracing PyTorch operations and building a computational graph.
    This class uses all our custom modules to collect operations on-the-fly.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize the operation tracer.
        
        Args:
            verbose (bool): Whether to print verbose output during tracing
        """
        # Initialize our components
        self.operation_collector = PyTorchOperationCollector()
        self.tensor_analyzer = TensorOperationAnalyzer()
        self.operation_collection = OperationCollection()
        
        # Tracing state
        self.is_tracing = False
        self.verbose = verbose
        self.errors = []
        self.execution_order = 0
    
    def start_tracing(self):
        """Start tracing operations."""
        self.is_tracing = True
        if self.verbose:
            print("Operation tracing started")
    
    def stop_tracing(self):
        """Stop tracing operations."""
        self.is_tracing = False
        if self.verbose:
            print("Operation tracing stopped")
            print(f"Collected {len(self.operation_collection.operations)} operations")
    
    def trace_operation(self, op_name):
        """
        Create a decorator for tracing operations with the given name.
        
        Args:
            op_name (str): The name of the operation to trace
            
        Returns:
            function: A decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Only trace if tracing is enabled
                if not self.is_tracing:
                    return func(*args, **kwargs)
                
                # Analyze the operation
                analysis = self.tensor_analyzer.analyze_operation(func, args, kwargs)
                
                # If there was an error, log it and return
                if analysis['error'] is not None:
                    self.log_error(op_name, analysis['error'])
                    return analysis['result']
                
                # Only process operations with tensor inputs or outputs
                if analysis['has_tensor_inputs'] or analysis['has_tensor_outputs']:
                    # Create the operation in our graph
                    operation = self.operation_collection.create_operation(
                        op_name,
                        analysis['input_tensors'],
                        analysis['output_tensors']
                    )
                    
                    # Set execution order
                    operation.set_attribute('execution_order', self.execution_order)
                    self.execution_order += 1
                    
                    if self.verbose:
                        print(f"Traced operation: {operation}")
                
                # Return the original result
                return analysis['result']
            
            return wrapper
        
        return decorator
    
    def log_error(self, op_name, error_message):
        """
        Log an error that occurred during operation execution.
        
        Args:
            op_name (str): Name of the operation
            error_message (str): Error message
        """
        self.errors.append((op_name, error_message))
        if self.verbose:
            print(f"Error in operation {op_name}: {error_message}")
    
    def get_operations(self):
        """Get all operations in the graph."""
        return self.operation_collection.get_all_operations()
    
    def get_operations_by_type(self, op_type):
        """Get operations of a specific type."""
        return self.operation_collection.get_operations_by_type(op_type)
    
    def get_errors(self):
        """Get all recorded errors."""
        return self.errors
    
    def get_operation_count_by_type(self):
        """Get count of operations by type."""
        counts = defaultdict(int)
        for op in self.get_operations():
            counts[op.op_name] += 1
        return counts
    
    def patch_pytorch_operations(self):
        """
        Patch all PyTorch operations we've collected.
        
        Returns:
            dict: Original operations that can be restored later
        """
        originals = {}
        
        # Categorize operations by namespace
        torch_ops = {}
        functional_ops = {}
        tensor_ops = {}
        
        for op_name, op_info in self.operation_collector.all_operations.items():
            if op_name.startswith('torch.'):
                torch_ops[op_name] = op_info
            elif op_name.startswith('F.'):
                functional_ops[op_name] = op_info
            elif op_name.startswith('tensor.'):
                tensor_ops[op_name] = op_info
        
        # Patch torch module functions
        originals['torch'] = {}
        for op_name, op_info in torch_ops.items():
            # Strip the 'torch.' prefix
            name = op_name[6:]
            original = op_info['callable']
            
            # Skip if not accessible or non-callable
            if not hasattr(torch, name) or not callable(getattr(torch, name)):
                continue
                
            # Store original and set patched version
            originals['torch'][name] = original
            setattr(torch, name, self.trace_operation(op_name)(original))
        
        # Patch torch.nn.functional functions
        originals['F'] = {}
        for op_name, op_info in functional_ops.items():
            # Strip the 'F.' prefix
            name = op_name[2:]
            original = op_info['callable']
            
            # Skip if not accessible or non-callable
            if not hasattr(F, name) or not callable(getattr(F, name)):
                continue
                
            # Store original and set patched version
            originals['F'][name] = original
            setattr(F, name, self.trace_operation(op_name)(original))
        
        # Patch tensor methods
        originals['tensor'] = {}
        for op_name, op_info in tensor_ops.items():
            # Strip the 'tensor.' prefix
            name = op_name[7:]
            original = op_info['callable']
            
            # Skip if not accessible or non-callable
            if not hasattr(torch.Tensor, name) or not callable(getattr(torch.Tensor, name)):
                continue
                
            # Store original and set patched version
            originals['tensor'][name] = original
            setattr(torch.Tensor, name, self.trace_operation(op_name)(original))
        
        return originals
    
    def restore_pytorch_operations(self, originals):
        """
        Restore original PyTorch operations.
        
        Args:
            originals (dict): Original operations to restore
        """
        # Restore torch module functions
        for name, original in originals.get('torch', {}).items():
            setattr(torch, name, original)
        
        # Restore torch.nn.functional functions
        for name, original in originals.get('F', {}).items():
            setattr(F, name, original)
        
        # Restore tensor methods
        for name, original in originals.get('tensor', {}).items():
            setattr(torch.Tensor, name, original)


@contextlib.contextmanager
def trace_pytorch_operations(verbose=False):
    """
    Context manager for tracing PyTorch operations.
    
    Args:
        verbose (bool): Whether to print verbose output during tracing
        
    Yields:
        OperationTracer: The operation tracer instance
    """
    tracer = OperationTracer(verbose=verbose)
    
    # Start tracing
    tracer.start_tracing()
    
    # Patch all PyTorch operations
    originals = tracer.patch_pytorch_operations()
    
    try:
        yield tracer
    finally:
        # Restore original PyTorch operations
        tracer.restore_pytorch_operations(originals)
        
        # Stop tracing
        tracer.stop_tracing()


class SimpleConvModel(nn.Module):
    """
    A simple convolutional model with two inputs and three outputs.
    Includes various operations like Conv2d, SiLU, MaxPool2d, addition, and concatenation.
    """
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        # Input branch 1
        self.conv1a = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Input branch 2
        self.conv2a = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Middle branch after concatenation
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Output branches
        self.conv_out1 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_out2 = nn.Conv2d(32, 8, kernel_size=1)
        self.conv_out3 = nn.Conv2d(32, 4, kernel_size=1)
    
    def forward(self, x1, x2):
        # Process input branch 1
        f1 = self.conv1a(x1)
        f1 = F.silu(f1)
        f1 = F.max_pool2d(f1, kernel_size=2, stride=2)
        f1 = self.conv1b(f1)
        
        # Process input branch 2
        f2 = self.conv2a(x2)
        f2 = F.silu(f2)
        f2 = F.max_pool2d(f2, kernel_size=2, stride=2)
        f2 = self.conv2b(f2)
        
        # Add branch outputs
        f_add = f1 + f2
        
        # Create another path with concatenation
        f_cat = torch.cat([f1, f2], dim=1)
        
        # Process concatenated features
        f_cat = self.conv3(f_cat)
        f_cat = F.silu(f_cat)
        
        # Generate three different outputs
        out1 = self.conv_out1(f_cat)  # 16 channels
        out2 = self.conv_out2(f_cat)  # 8 channels
        out3 = self.conv_out3(f_add)  # 4 channels from the addition path
        
        return out1, out2, out3


def test_operation_tracer():
    """Test the operation tracer with a simple convolutional model."""
    print("\n===== Testing Operation Tracer with SimpleConvModel =====")
    
    # Create model and inputs
    model = SimpleConvModel()
    input1 = torch.randn(1, 3, 32, 32)
    input2 = torch.randn(1, 3, 32, 32)
    
    # Trace all PyTorch operations during model forward pass
    with trace_pytorch_operations(verbose=True) as tracer:
        # Run the model
        outputs = model(input1, input2)
        
        # Print output shapes
        print("\nModel outputs:")
        for i, output in enumerate(outputs):
            print(f"  Output {i+1} shape: {list(output.shape)}")
    
    # Print statistics about the traced operations
    op_counts = tracer.get_operation_count_by_type()
    total_ops = len(tracer.get_operations())
    
    print(f"\nTraced {total_ops} operations:")
    for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_name}: {count}")
    
    # Print errors if any
    errors = tracer.get_errors()
    if errors:
        print("\nErrors during tracing:")
        for op_name, error in errors:
            print(f"  {op_name}: {error}")
    
    return tracer


if __name__ == "__main__":
    # Run the test
    tracer = test_operation_tracer()
