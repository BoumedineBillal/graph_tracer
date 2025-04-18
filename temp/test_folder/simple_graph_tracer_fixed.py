import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from collections import defaultdict
import functools

# Add parent directory to path to import graph tracer components
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import tensor analyzer components directly
from tensor_operation_analyzer import TensorOperationAnalyzer, TensorInfo
from operation_id_system import Operation, OperationCollection, OperationIDManager
from pytorch_operation_collector import PyTorchOperationCollector

# Define a simple context manager for tracing that doesn't try to patch Tensor methods
class CustomOperationTracer:
    """Simplified operation tracer that doesn't patch tensor methods to avoid errors"""
    
    def __init__(self, verbose=False):
        self.operation_collector = PyTorchOperationCollector()
        self.tensor_analyzer = TensorOperationAnalyzer()
        self.operation_collection = OperationCollection()
        self.is_tracing = False
        self.verbose = verbose
        self.errors = []
        self.execution_order = 0
    
    def start_tracing(self):
        self.is_tracing = True
        if self.verbose:
            print("Operation tracing started")
    
    def stop_tracing(self):
        self.is_tracing = False
        if self.verbose:
            print("Operation tracing stopped")
            print(f"Collected {len(self.operation_collection.operations)} operations")
    
    def trace_operation(self, op_name):
        """Create a decorator for tracing operations with the given name"""
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
        self.errors.append((op_name, error_message))
        if self.verbose:
            print(f"Error in operation {op_name}: {error_message}")
    
    def get_operations(self):
        """Get all operations in the graph"""
        return self.operation_collection.get_all_operations()
    
    def get_operations_by_type(self, op_type):
        """Get operations of a specific type"""
        return self.operation_collection.get_operations_by_type(op_type)
    
    def get_errors(self):
        """Get all recorded errors"""
        return self.errors
    
    def get_operation_count_by_type(self):
        """Get count of operations by type"""
        counts = defaultdict(int)
        for op in self.get_operations():
            counts[op.op_name] += 1
        return counts
    
    def patch_torch_and_functional(self):
        """Patch torch and F functions but not tensor methods (to avoid the error)"""
        originals = {}
        
        # Categorize operations by namespace (excluding tensor methods)
        torch_ops = {}
        functional_ops = {}
        
        for op_name, op_info in self.operation_collector.all_operations.items():
            if op_name.startswith('torch.'):
                torch_ops[op_name] = op_info
            elif op_name.startswith('F.'):
                functional_ops[op_name] = op_info
        
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
        
        return originals
    
    def restore_torch_and_functional(self, originals):
        """Restore original torch and F functions"""
        # Restore torch module functions
        for name, original in originals.get('torch', {}).items():
            setattr(torch, name, original)
        
        # Restore torch.nn.functional functions
        for name, original in originals.get('F', {}).items():
            setattr(F, name, original)


# Define a context manager for tracing operations
class trace_operations:
    def __init__(self, verbose=False):
        self.tracer = CustomOperationTracer(verbose=verbose)
        
    def __enter__(self):
        self.tracer.start_tracing()
        self.originals = self.tracer.patch_torch_and_functional()
        return self.tracer
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracer.restore_torch_and_functional(self.originals)
        self.tracer.stop_tracing()


# Define a simple CNN model for demonstration
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 16 * 16, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def trace_and_create_onnx_graph():
    # Create the model and test input
    model = SimpleConvNet()
    input_tensor = torch.randn(1, 3, 64, 64)
    
    print("===== Tracing PyTorch Operations =====")
    
    # Trace operations during model execution
    with trace_operations(verbose=True) as tracer:
        output = model(input_tensor)
        
        # Get operations in execution order 
        operations = sorted(tracer.get_operations(), 
                          key=lambda op: op.get_attribute('execution_order', float('inf')))
    
    # Print operation statistics
    op_counts = defaultdict(int)
    for op in operations:
        op_counts[op.op_name] += 1
    
    print("\n===== Operation Statistics =====")
    print(f"Total operations: {len(operations)}")
    for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_name}: {count}")
    
    # Create a simple ONNX graph to visualize the operations
    print("\n===== Creating ONNX Graph for Visualization =====")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Export the model to ONNX (just for visualization)
    onnx_file = "output/traced_model.onnx"
    
    # Export to ONNX
    torch.onnx.export(
        model,
        input_tensor,
        onnx_file,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    
    print(f"ONNX model saved to: {os.path.abspath(onnx_file)}")
    print("\nTo visualize the model graph:")
    print("1. Install Netron: pip install netron")
    print("2. Use the following Python code:")
    print("   >>> import netron")
    print(f"   >>> netron.start('{onnx_file}')")
    print("3. Or visit https://netron.app/ and upload the ONNX file")
    
    print("\n===== Traced Operations Flow =====")
    print("Operations in the execution sequence:")
    for i, op in enumerate(operations):
        if op.op_name != "load":
            inputs = [f"tensor_{info.tensor_id}" for info in op.input_tensor_infos]
            outputs = [f"tensor_{info.tensor_id}" for info in op.output_tensor_infos]
            print(f"{i+1}. {op.op_name}: {inputs} -> {outputs}")
    
    return model, tracer, onnx_file


if __name__ == "__main__":
    model, tracer, onnx_file = trace_and_create_onnx_graph()
