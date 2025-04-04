import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import types
import contextlib
import functools
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable

# Import our custom modules
try:
    from tensor_operation_analyzer import TensorOperationAnalyzer, TensorInfo
    from operation_id_system import Operation, OperationCollection, OperationIDManager
except ImportError:
    # Fallback definitions for testing
    class TensorInfo:
        """Simplified TensorInfo for testing purposes"""
        def __init__(self, tensor=None):
            self.tensor_id = id(tensor) if tensor is not None else None
            self.shape = list(tensor.shape) if tensor is not None and hasattr(tensor, 'shape') else []
            self.path = []
            
        def __eq__(self, other):
            if not isinstance(other, TensorInfo):
                return False
            return self.tensor_id == other.tensor_id
            
        def __hash__(self):
            return hash(self.tensor_id)

    class TensorOperationAnalyzer:
        """Simplified TensorOperationAnalyzer for testing purposes"""
        def analyze_operation(self, func, args, kwargs):
            # Find tensors in args
            input_tensors = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_tensors.append(TensorInfo(arg))
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
                
                # Find tensors in result
                output_tensors = []
                if isinstance(result, torch.Tensor):
                    output_tensors.append(TensorInfo(result))
                elif isinstance(result, (list, tuple)):
                    for item in result:
                        if isinstance(item, torch.Tensor):
                            output_tensors.append(TensorInfo(item))
                
                return {
                    'has_tensor_inputs': bool(input_tensors),
                    'has_tensor_outputs': bool(output_tensors),
                    'input_tensors': input_tensors,
                    'output_tensors': output_tensors,
                    'result': result,
                    'error': None
                }
            except Exception as e:
                return {
                    'has_tensor_inputs': bool(input_tensors),
                    'has_tensor_outputs': False,
                    'input_tensors': input_tensors,
                    'output_tensors': [],
                    'result': None,
                    'error': str(e)
                }

    class Operation:
        def __init__(self, op_name):
            self.op_name = op_name
            self.complex_id = None
            self.unique_id = None
            self.input_tensor_infos = []
            self.output_tensor_infos = []
            self.input_op_ids = []
            self.attributes = {}

    class OperationIDManager:
        def __init__(self):
            pass

    class OperationCollection:
        def __init__(self):
            self.operations = {}
            self.known_tensors = set()
            
        def create_operation(self, op_name, input_tensor_infos, output_tensor_infos):
            op = Operation(op_name)
            return op
            
        def is_known_tensor(self, tensor_info):
            return False


class OperationDecorator:
    """
    A class that provides decorators for tracing tensor operations.
    This is responsible for analyzing operations and building the operation graph.
    """
    def __init__(self, graph_tracer=None):
        """
        Initialize with an optional graph tracer.
        
        Args:
            graph_tracer: The GraphTracer instance that will collect the operations
        """
        self.graph_tracer = graph_tracer
        self.analyzer = TensorOperationAnalyzer()
    
    def set_graph_tracer(self, graph_tracer):
        """Set the graph tracer to use with this decorator."""
        self.graph_tracer = graph_tracer
    
    def trace_operation(self, op_name=None):
        """
        Create a decorator for tracing a function or method.
        
        Args:
            op_name (str, optional): Custom name for the operation. 
                                     If None, function name will be used.
            
        Returns:
            decorator: A decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Only trace if we have a graph tracer
                if not self.graph_tracer or not self.graph_tracer.is_tracing:
                    return func(*args, **kwargs)
                
                # Get operation name from function if not provided
                function_name = op_name if op_name is not None else func.__name__
                
                # Use TensorOperationAnalyzer to examine the operation
                analysis = self.analyzer.analyze_operation(func, args, kwargs)
                
                # If there was an error, log it and return
                if analysis['error'] is not None:
                    self.graph_tracer.log_error(function_name, analysis['error'])
                    return analysis['result']
                
                # Only process operations with tensor inputs or outputs
                if analysis['has_tensor_inputs'] or analysis['has_tensor_outputs']:
                    # Create the operation in our graph
                    self.graph_tracer.record_operation(
                        op_name=function_name,
                        input_tensors=analysis['input_tensors'],
                        output_tensors=analysis['output_tensors']
                    )
                
                # Return the original result
                return analysis['result']
            
            return wrapper
        
        return decorator


class GraphTracer:
    """
    Main class for tracing PyTorch operations and building a computational graph.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize the graph tracer.
        
        Args:
            verbose (bool): Whether to print verbose output during tracing
        """
        self.operation_collection = OperationCollection()
        self.operation_decorator = OperationDecorator(self)
        self.is_tracing = False
        self.verbose = verbose
        self.errors = []
    
    def start_tracing(self):
        """Start tracing operations."""
        self.is_tracing = True
        if self.verbose:
            print("Graph tracing started")
    
    def stop_tracing(self):
        """Stop tracing operations."""
        self.is_tracing = False
        if self.verbose:
            print("Graph tracing stopped")
            print(f"Collected {len(self.operation_collection.operations)} operations")
    
    def record_operation(self, op_name, input_tensors, output_tensors):
        """
        Record an operation in the graph.
        
        Args:
            op_name (str): Name of the operation
            input_tensors (list): List of input TensorInfo objects
            output_tensors (list): List of output TensorInfo objects
        """
        operation = self.operation_collection.create_operation(
            op_name, input_tensors, output_tensors
        )
        
        if self.verbose:
            print(f"Recorded operation: {operation}")
        
        return operation
    
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
    
    def trace(self, func_name=None):
        """
        Create a decorator for tracing a function or method.
        This is a shortcut for operation_decorator.trace_operation.
        
        Args:
            func_name (str, optional): Custom name for the function. 
                                      If None, function name will be used.
            
        Returns:
            decorator: A decorator function
        """
        return self.operation_decorator.trace_operation(func_name)


@contextlib.contextmanager
def trace_graph(verbose=False):
    """
    Context manager for tracing PyTorch operations and building a computational graph.
    
    Args:
        verbose (bool): Whether to print verbose output during tracing
        
    Yields:
        GraphTracer: The graph tracer instance
    """
    tracer = GraphTracer(verbose=verbose)
    tracer.start_tracing()
    
    try:
        yield tracer
    finally:
        tracer.stop_tracing()


def patch_module(module, tracer):
    """
    Patch all functions in a module with tracing decorators.
    
    Args:
        module: The module to patch (e.g., torch.nn.functional)
        tracer: The GraphTracer instance to use
        
    Returns:
        dict: Map of original functions to restore later
    """
    originals = {}
    
    for attr_name in dir(module):
        if attr_name.startswith("_"):  # Skip internal functions
            continue
        
        attr = getattr(module, attr_name)
        if isinstance(attr, (types.FunctionType, types.BuiltinFunctionType)):
            originals[attr_name] = attr
            setattr(module, attr_name, tracer.trace(attr_name)(attr))
    
    return originals


def patch_tensor_methods(tracer):
    """
    Patch tensor methods with tracing decorators.
    
    Args:
        tracer: The GraphTracer instance to use
        
    Returns:
        dict: Map of original methods to restore later
    """
    originals = {}
    
    # List of methods to patch (common tensor operations)
    methods_to_patch = [
        # Arithmetic operations
        "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__",
        "__mod__", "__pow__", "add", "sub", "mul", "matmul", "div", "floor_divide",
        "remainder", "pow", "add_", "sub_", "mul_", "div_",
        # Activation functions
        "relu", "sigmoid", "tanh", "softmax",
        # Shape operations
        "view", "reshape", "permute", "transpose", "squeeze", "unsqueeze",
        # Reduction operations
        "sum", "mean", "max", "min"
    ]
    
    for method_name in methods_to_patch:
        if hasattr(torch.Tensor, method_name):
            originals[method_name] = getattr(torch.Tensor, method_name)
            setattr(torch.Tensor, method_name, 
                   tracer.trace(f"tensor.{method_name}")(originals[method_name]))
    
    return originals


@contextlib.contextmanager
def trace_pytorch(tracer=None, patch_functional=True, patch_tensor=True):
    """
    Context manager for tracing PyTorch operations by patching modules and tensor methods.
    
    Args:
        tracer: The GraphTracer instance to use. If None, a new one will be created.
        patch_functional: Whether to patch torch.nn.functional
        patch_tensor: Whether to patch torch.Tensor methods
        
    Yields:
        GraphTracer: The graph tracer instance
    """
    # Create a tracer if one wasn't provided
    if tracer is None:
        tracer = GraphTracer()
    
    # Start tracing
    tracer.start_tracing()
    
    # Dictionaries to store original functions/methods
    originals = {}
    
    try:
        # Patch torch.nn.functional if requested
        if patch_functional:
            originals['functional'] = patch_module(F, tracer)
        
        # Patch tensor methods if requested
        if patch_tensor:
            originals['tensor'] = patch_tensor_methods(tracer)
        
        yield tracer
    finally:
        # Restore original functions/methods
        for module_name, module_originals in originals.items():
            if module_name == 'functional':
                for attr_name, orig_func in module_originals.items():
                    setattr(F, attr_name, orig_func)
            elif module_name == 'tensor':
                for method_name, orig_method in module_originals.items():
                    setattr(torch.Tensor, method_name, orig_method)
        
        # Stop tracing
        tracer.stop_tracing()


# Test model with two inputs and three outputs
class TestConvModel(nn.Module):
    def __init__(self):
        super(TestConvModel, self).__init__()
        # Input branch 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Input branch 2
        self.conv2 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.silu2 = nn.SiLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Middle branch
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.silu3 = nn.SiLU()
        
        # Output branches
        self.conv_out1 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_out2 = nn.Conv2d(32, 8, kernel_size=1)
        self.conv_out3 = nn.Conv2d(32, 4, kernel_size=1)
    
    def forward(self, x1, x2):
        # Process input branches
        branch1 = self.maxpool1(self.silu1(self.conv1(x1)))
        branch2 = self.maxpool2(self.silu2(self.conv2(x2)))
        
        # Concatenate branches
        x = torch.cat([branch1, branch2], dim=1)
        
        # Middle processing
        x = self.silu3(self.conv3(x))
        
        # Generate outputs
        out1 = self.conv_out1(x)
        out2 = self.conv_out2(x)
        out3 = self.conv_out3(x)
        
        return out1, out2, out3


def test_graph_tracer():
    print("===== Testing Graph Tracer with Conv Model =====")
    
    # Create model and inputs
    model = TestConvModel()
    input1 = torch.randn(1, 3, 32, 32)
    input2 = torch.randn(1, 3, 32, 32)
    
    # Trace with context manager
    with trace_pytorch(patch_functional=True, patch_tensor=True) as tracer:
        # Run model inference
        outputs = model(input1, input2)
        
        print(f"\nModel executed successfully with {len(outputs)} outputs")
        for i, output in enumerate(outputs):
            print(f"Output {i+1} shape: {output.shape}")
    
    # Print statistics
    operations = tracer.get_operations()
    print(f"\nCollected {len(operations)} operations:")
    
    # Count by operation type
    op_types = defaultdict(int)
    for op in operations:
        op_types[op.op_name] += 1
    
    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")
    
    return tracer


if __name__ == "__main__":
    tracer = test_graph_tracer()
