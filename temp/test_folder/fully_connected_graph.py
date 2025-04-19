import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import helper, TensorProto
from collections import defaultdict, OrderedDict
import functools
import json

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
        
        # Store explicit tracking of tensor flow
        self.tensor_flow = {}  # tensor_id -> list of operations that use it
        self.tensor_sources = {}  # tensor_id -> operation that created it
    
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
                    
                    # Track tensor sources and flows
                    for tensor_info in operation.input_tensor_infos:
                        if tensor_info.tensor_id not in self.tensor_flow:
                            self.tensor_flow[tensor_info.tensor_id] = []
                        self.tensor_flow[tensor_info.tensor_id].append(operation.unique_id)
                        
                    for tensor_info in operation.output_tensor_infos:
                        self.tensor_sources[tensor_info.tensor_id] = operation.unique_id
                    
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
    
    def print_graph_proto(self, graph_def, indent=0):
        """
        Print the graph definition in a human-readable format for debugging
        """
        print("\n" + "=" * 80)
        print("GRAPH DEFINITION (for debugging)")
        print("=" * 80)
        
        # Print graph name
        print(f"Graph: {graph_def.name}")
        
        # Print inputs
        print("\nInputs:")
        for i, input_info in enumerate(graph_def.input):
            print(f"  {i+1}. {input_info.name}")
        
        # Print outputs
        print("\nOutputs:")
        for i, output_info in enumerate(graph_def.output):
            print(f"  {i+1}. {output_info.name}")
        
        # Print nodes
        print("\nNodes:")
        for i, node in enumerate(graph_def.node):
            print(f"  {i+1}. {node.op_type} (name: {node.name})")
            print(f"     Inputs: {', '.join(node.input)}")
            print(f"     Outputs: {', '.join(node.output)}")
        
        print("=" * 80)
    
    def export_graph_json(self, graph_def, output_file):
        """
        Export the graph definition as JSON for easier debugging
        """
        # Convert graph to dictionary
        graph_dict = {
            'name': graph_def.name,
            'inputs': [i.name for i in graph_def.input],
            'outputs': [o.name for o in graph_def.output],
            'nodes': []
        }
        
        for node in graph_def.node:
            node_dict = {
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {}
            }
            
            # Add attributes if any
            for attr in node.attribute:
                node_dict['attributes'][attr.name] = f"<attr of type {attr.type}>"
            
            graph_dict['nodes'].append(node_dict)
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(graph_dict, f, indent=2)
        
        print(f"Graph exported as JSON to: {os.path.abspath(output_file)}")
    
    def create_fully_connected_graph(self, output_file='fully_connected_graph.onnx', 
                                   print_debug=True, export_json=True):
        """
        Create a fully connected visualization graph that preserves all operation connections.
        """
        print("\nCreating fully connected visualization graph...")
        
        # Get operations in execution order
        operations = sorted(self.get_operations(), 
                          key=lambda op: op.get_attribute('execution_order', float('inf')))
        
        # Extract key operations and their inputs/outputs
        op_dict = OrderedDict()
        critical_ops = ["conv2d", "batch_norm", "relu", "max_pool2d", "view", "linear"]
        secondary_ops = ["add", "mul", "concat", "flatten"]
        
        # First pass: identify important operations
        for op in operations:
            # Skip load operations
            if op.op_name == "load" or op.op_name.startswith("has_torch"):
                continue
            
            # Get the simplified name without namespace
            simple_name = op.op_name
            if simple_name.startswith("torch."):
                simple_name = simple_name[6:]
            elif simple_name.startswith("F."):
                simple_name = simple_name[2:]
            
            # Check if this is a critical operation
            is_critical = False
            for critical_op in critical_ops:
                if critical_op in simple_name:
                    is_critical = True
                    break
            
            # Also include secondary operations
            if not is_critical:
                for secondary_op in secondary_ops:
                    if secondary_op in simple_name:
                        is_critical = True
                        break
            
            # Always include tensor.view which is needed for connection
            if "view" in simple_name:
                is_critical = True
            
            # Skip non-critical operations with no outputs
            if not is_critical and not op.output_tensor_infos:
                continue
            
            # Record this operation
            op_key = f"{simple_name}_{op.get_attribute('execution_order')}"
            op_dict[op_key] = {
                'op_name': simple_name,
                'inputs': [info.tensor_id for info in op.input_tensor_infos],
                'outputs': [info.tensor_id for info in op.output_tensor_infos],
                'execution_order': op.get_attribute('execution_order')
            }
        
        # Build a mapping from tensors to producers and consumers
        tensor_producers = {}
        tensor_consumers = defaultdict(list)
        
        for op_key, op_info in op_dict.items():
            # Record this op as producer of its output tensors
            for tensor_id in op_info['outputs']:
                tensor_producers[tensor_id] = op_key
            
            # Record this op as consumer of its input tensors
            for tensor_id in op_info['inputs']:
                tensor_consumers[tensor_id].append(op_key)
        
        # Create tensor name mapping
        all_tensors = set()
        for op_info in op_dict.values():
            all_tensors.update(op_info['inputs'])
            all_tensors.update(op_info['outputs'])
        
        tensor_names = {tid: f"tensor_{tid}" for tid in all_tensors}
        
        # Create nodes
        nodes = []
        for op_key, op_info in op_dict.items():
            input_names = [tensor_names[tid] for tid in op_info['inputs']]
            output_names = [tensor_names[tid] for tid in op_info['outputs']]
            
            node = helper.make_node(
                op_info['op_name'],
                name=op_key,
                inputs=input_names,
                outputs=output_names
            )
            nodes.append(node)
        
        # Find inputs and outputs of the graph
        graph_input_tensors = set()
        for op_info in op_dict.values():
            for tensor_id in op_info['inputs']:
                if tensor_id not in tensor_producers:
                    graph_input_tensors.add(tensor_id)
        
        graph_output_tensors = set()
        for op_info in op_dict.values():
            for tensor_id in op_info['outputs']:
                if tensor_id not in tensor_consumers or not tensor_consumers[tensor_id]:
                    graph_output_tensors.add(tensor_id)
        
        # Create graph inputs and outputs
        graph_inputs = []
        for tensor_id in graph_input_tensors:
            # Try to find shape information for this tensor
            shape = None
            for op in operations:
                for info in op.input_tensor_infos:
                    if info.tensor_id == tensor_id and info.shape:
                        shape = info.shape
                        break
                if shape:
                    break
            
            # Use default shape if none found
            if not shape:
                shape = [1]
                
            graph_inputs.append(helper.make_tensor_value_info(
                tensor_names[tensor_id],
                TensorProto.FLOAT,
                shape
            ))
        
        graph_outputs = []
        for tensor_id in graph_output_tensors:
            # Try to find shape information for this tensor
            shape = None
            for op in operations:
                for info in op.output_tensor_infos:
                    if info.tensor_id == tensor_id and info.shape:
                        shape = info.shape
                        break
                if shape:
                    break
            
            # Use default shape if none found
            if not shape:
                shape = [1]
                
            graph_outputs.append(helper.make_tensor_value_info(
                tensor_names[tensor_id],
                TensorProto.FLOAT,
                shape
            ))
        
        # Ensure we have at least one input and output
        if not graph_inputs:
            graph_inputs = [helper.make_tensor_value_info('dummy_input', TensorProto.FLOAT, [1])]
        if not graph_outputs:
            graph_outputs = [helper.make_tensor_value_info('dummy_output', TensorProto.FLOAT, [1])]
        
        # Create the graph definition
        graph_def = helper.make_graph(
            nodes,
            'FullyConnectedModelGraph',
            graph_inputs,
            graph_outputs
        )
        
        # Print debug information if requested
        if print_debug:
            self.print_graph_proto(graph_def)
        
        # Export as JSON for debugging if requested
        if export_json:
            json_path = output_file.replace('.onnx', '.json')
            self.export_graph_json(graph_def, json_path)
        
        # Create model
        model_def = helper.make_model(
            graph_def,
            producer_name='PyTorch-OperationTracer'
        )
        
        # Save model
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        onnx.save(model_def, output_file)
        print(f"Fully connected graph saved to: {os.path.abspath(output_file)}")
        
        return output_file
    
    def trace_operation_flow(self):
        """
        Print a textual representation of the operation flow for debugging
        """
        operations = sorted(self.get_operations(), 
                           key=lambda op: op.get_attribute('execution_order', float('inf')))
        
        print("\n" + "=" * 80)
        print("OPERATION FLOW")
        print("=" * 80)
        
        for op in operations:
            if op.op_name == "load":
                continue
                
            # Get input tensor info
            input_sources = []
            for tensor_info in op.input_tensor_infos:
                source_op_id = self.operation_collection.get_source_op_id(tensor_info)
                if source_op_id:
                    source_op = self.operation_collection.get_operation(source_op_id)
                    input_sources.append(f"{source_op.op_name}")
                else:
                    input_sources.append("model_input")
            
            # Print operation details
            print(f"Operation: {op.op_name}")
            print(f"  Inputs from: {', '.join(input_sources)}")
            print(f"  Input tensors: {[info.tensor_id for info in op.input_tensor_infos]}")
            print(f"  Output tensors: {[info.tensor_id for info in op.output_tensor_infos]}")
            print()
        
        print("=" * 80)


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
        x = x.view(x.size(0), -1)  # Flatten operation 
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
        
        # Print text representation of the operation flow
        tracer.trace_operation_flow()
        
        # Print operation statistics
        op_counts = tracer.get_operation_count_by_type()
        total_ops = len(tracer.get_operations())
        
        print(f"\nTraced {total_ops} operations:")
        for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_name}: {count}")
        
        # Create fully connected visualization graph
        graph_path = tracer.create_fully_connected_graph(
            "output/fully_connected_graph.onnx",
            print_debug=True,
            export_json=True
        )
    
    print("\nTo visualize the operation graph:")
    print("1. Install Netron: pip install netron")
    print("2. Use the following Python code:")
    print("   >>> import netron")
    print(f"   >>> netron.start('{graph_path}')")
    print("3. Or visit https://netron.app/ and upload the ONNX file")
    
    return model, tracer, graph_path


if __name__ == "__main__":
    model, tracer, graph_path = main()
