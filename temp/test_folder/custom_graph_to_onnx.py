import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import helper, TensorProto
from collections import defaultdict
import functools

# Add parent directory to path to import graph tracer modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import tensor analyzer components directly
from tensor_operation_analyzer import TensorOperationAnalyzer, TensorInfo
from operation_id_system import Operation, OperationCollection, OperationIDManager
from pytorch_operation_collector import PyTorchOperationCollector

# Custom operation tracer (simplified to avoid patching Tensor methods)
class CustomOperationTracer:
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
        return self.operation_collection.get_all_operations()
    
    def get_operations_by_type(self, op_type):
        return self.operation_collection.get_operations_by_type(op_type)
    
    def get_errors(self):
        return self.errors
    
    def get_operation_count_by_type(self):
        counts = defaultdict(int)
        for op in self.get_operations():
            counts[op.op_name] += 1
        return counts
    
    def patch_torch_and_functional(self):
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
        # Restore torch module functions
        for name, original in originals.get('torch', {}).items():
            setattr(torch, name, original)
        
        # Restore torch.nn.functional functions
        for name, original in originals.get('F', {}).items():
            setattr(F, name, original)
    
    def convert_traced_ops_to_onnx(self, output_file='operation_graph.onnx'):
        """
        Convert traced operations to ONNX format for visualization
        This creates a custom ONNX graph based on the traced operations
        """
        print("\nConverting traced operations to ONNX graph...")
        
        # Get operations in execution order
        operations = sorted(self.get_operations(), 
                          key=lambda op: op.get_attribute('execution_order', float('inf')))
        
        # Extract operations as a list of dictionaries for easier processing
        op_list = []
        for op in operations:
            if op.op_name == "load":
                continue  # Skip load operations
                
            op_dict = {
                'op_name': op.op_name,
                'inputs': [info.tensor_id for info in op.input_tensor_infos],
                'outputs': [info.tensor_id for info in op.output_tensor_infos],
                'execution_order': op.get_attribute('execution_order')
            }
            op_list.append(op_dict)
        
        # Find all used tensor IDs
        used_tensor_ids = set()
        for op in op_list:
            used_tensor_ids.update(op['inputs'])
            used_tensor_ids.update(op['outputs'])
        
        # Create tensor name mapping
        tensor_names = {tid: f"tensor_{tid}" for tid in used_tensor_ids}
        
        # Identify input tensors (tensors that are only inputs, never outputs)
        input_only = set()
        for op in op_list:
            input_only.update(op['inputs'])
        for op in op_list:
            input_only.difference_update(op['outputs'])
        
        # Identify output tensors (tensors that are only outputs, never inputs)
        output_only = set()
        for op in op_list:
            output_only.update(op['outputs'])
        for op in op_list:
            output_only.difference_update(op['inputs'])
        
        # Create ONNX nodes
        nodes = []
        for i, op in enumerate(op_list):
            input_names = [tensor_names[tid] for tid in op['inputs']]
            output_names = [tensor_names[tid] for tid in op['outputs']]
            
            # Use an ONNX-supported operation name or fallback to "Custom"
            op_type = op['op_name']
            if op_type not in ['Relu', 'Conv', 'MatMul', 'Add', 'MaxPool', 'Flatten', 'BatchNormalization']:
                # Map PyTorch operations to ONNX operations where possible
                if op_type == 'relu':
                    op_type = 'Relu'
                elif op_type == 'conv2d':
                    op_type = 'Conv'
                elif op_type == 'matmul':
                    op_type = 'MatMul'
                elif op_type == 'add':
                    op_type = 'Add'
                elif op_type == 'max_pool2d':
                    op_type = 'MaxPool'
                elif op_type == 'batch_norm':
                    op_type = 'BatchNormalization'
                elif op_type == 'view' or op_type == 'flatten':
                    op_type = 'Flatten'
                elif op_type == 'cat':
                    op_type = 'Concat'
                elif op_type == 'mul':
                    op_type = 'Mul'
                else:
                    op_type = 'Custom'  # Fallback for unsupported operations
            
            # Create node
            node = helper.make_node(
                op_type,
                name=f"{op['op_name']}_{i}",
                inputs=input_names,
                outputs=output_names
            )
            nodes.append(node)
        
        # Create inputs for the graph
        graph_inputs = []
        for tid in input_only:
            # Try to find shape information for the tensor
            shape = None
            for op in operations:
                for info in op.input_tensor_infos:
                    if info.tensor_id == tid and info.shape:
                        shape = info.shape
                        break
                if shape:
                    break
            
            if not shape:
                shape = [1]  # Default shape if none found
                
            # Create input value info
            input_value_info = helper.make_tensor_value_info(
                tensor_names[tid],
                TensorProto.FLOAT,  # Assume float type
                shape
            )
            graph_inputs.append(input_value_info)
        
        # Create outputs for the graph
        graph_outputs = []
        for tid in output_only:
            # Try to find shape information for the tensor
            shape = None
            for op in operations:
                for info in op.output_tensor_infos:
                    if info.tensor_id == tid and info.shape:
                        shape = info.shape
                        break
                if shape:
                    break
            
            if not shape:
                shape = [1]  # Default shape if none found
                
            # Create output value info
            output_value_info = helper.make_tensor_value_info(
                tensor_names[tid],
                TensorProto.FLOAT,  # Assume float type
                shape
            )
            graph_outputs.append(output_value_info)
        
        # If no inputs/outputs found, create dummy ones to make a valid graph
        if not graph_inputs:
            graph_inputs = [helper.make_tensor_value_info('dummy_input', TensorProto.FLOAT, [1])]
            
        if not graph_outputs:
            graph_outputs = [helper.make_tensor_value_info('dummy_output', TensorProto.FLOAT, [1])]
        
        # Create graph definition
        graph_def = helper.make_graph(
            nodes,
            'PyTorchTracedOperations',
            graph_inputs,
            graph_outputs
        )
        
        # Create model
        model_def = helper.make_model(
            graph_def,
            producer_name='PyTorch-GraphTracer'
        )
        
        # Save model
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        onnx.save(model_def, output_file)
        print(f"ONNX graph saved to: {os.path.abspath(output_file)}")
        
        return output_file


# Context manager for tracing operations
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


# Simple CNN model for demonstration
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


def main():
    # Create the model and test input
    model = SimpleConvNet()
    input_tensor = torch.randn(1, 3, 64, 64)
    
    print("===== Tracing Model Operations =====")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Trace operations during model execution
    with trace_operations(verbose=True) as tracer:
        output = model(input_tensor)
        
        # Print operation statistics
        op_counts = tracer.get_operation_count_by_type()
        total_ops = len(tracer.get_operations())
        
        print(f"\nTraced {total_ops} operations:")
        for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_name}: {count}")
        
        # Convert traced operations to ONNX graph
        onnx_path = tracer.convert_traced_ops_to_onnx("output/traced_operations.onnx")
    
    print("\nTo visualize the operation graph:")
    print("1. Install Netron: pip install netron")
    print("2. Use the following Python code:")
    print("   >>> import netron")
    print(f"   >>> netron.start('{onnx_path}')")
    print("3. Or visit https://netron.app/ and upload the ONNX file")
    
    return model, tracer, onnx_path


if __name__ == "__main__":
    model, tracer, onnx_path = main()
