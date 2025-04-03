import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import types
import contextlib
import uuid
import onnx
from onnx import helper, TensorProto
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Original utility functions
def find_model_instance():
    """Finds the nearest `self` reference in the call stack that is an instance of `nn.Module`."""
    stack = inspect.stack()
    for frame_info in stack:
        local_vars = frame_info.frame.f_locals
        if "self" in local_vars and isinstance(local_vars["self"], nn.Module):
            return local_vars["self"]  # Return the model instance
    return None  # No model found

def my_decorator(func):
    def wrapper(*args, **kwargs):
        model_instance = find_model_instance()  # Get the model dynamically
        model_name = type(model_instance).__name__ if model_instance else "Unknown"
        print(f"Calling function: {func.__name__} from model: {model_name}")
        return func(*args, **kwargs)
    return wrapper

@contextlib.contextmanager
def patch_module_functions(module, decorator):
    print("start decorating")
    originals = {}
    for attr_name in dir(module):
        if attr_name.startswith("_"):  # Skip internal functions
            continue
        
        attr = getattr(module, attr_name)
        if isinstance(attr, (types.FunctionType, types.BuiltinFunctionType)):
            #if attr.__name__.startswith("_"): print(attr.__name__)
            
            if not attr.__name__.startswith("_"):
                originals[attr_name] = attr
                setattr(module, attr_name, decorator(attr))
    try:
        yield module
    finally:
        for attr_name, orig_func in originals.items():
            setattr(module, attr_name, orig_func)
        print("end decorating")

# New Graph Tracking functionality
class TensorGraphTracker:
    def __init__(self):
        self.graph = []
        self.tensor_ids = {}  # Map actual tensor objects to our tracking IDs
        self.next_id = 0
        self.input_tensors = set()  # Track initial inputs
        
    def get_tensor_id(self, tensor):
        """Get or create ID for a tensor"""
        tensor_hash = id(tensor)
        if tensor_hash not in self.tensor_ids:
            self.tensor_ids[tensor_hash] = self.next_id
            self.next_id += 1
        return self.tensor_ids[tensor_hash]
    
    def register_operation(self, op_name, inputs, outputs):
        """Register an operation with its inputs and outputs"""
        # Handle single tensor or list/tuple of tensors
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
            
        # Get IDs for all tensors
        input_ids = [self.get_tensor_id(x) for x in inputs if isinstance(x, torch.Tensor)]
        output_ids = [self.get_tensor_id(x) for x in outputs if isinstance(x, torch.Tensor)]
        
        # Mark first seen tensors as inputs
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                tensor_id = self.get_tensor_id(tensor)
                is_new = True
                for op in self.graph:
                    if tensor_id in op['outputs']:
                        is_new = False
                        break
                if is_new:
                    self.input_tensors.add(tensor_id)
        
        # Add operation to graph
        if input_ids and output_ids:  # Only add if we have valid tensors
            self.graph.append({
                'op_name': op_name,
                'inputs': input_ids,
                'outputs': output_ids
            })
    
    def filter_connected_graph(self):
        """Filter the graph to keep only nodes connected to input tensors"""
        # Build a forward graph from inputs to outputs
        connected = set(self.input_tensors)
        queue = list(self.input_tensors)
        
        # BFS to find all connected nodes
        while queue:
            current = queue.pop(0)
            for op in self.graph:
                if current in op['inputs']:
                    for output_id in op['outputs']:
                        if output_id not in connected:
                            connected.add(output_id)
                            queue.append(output_id)
        
        # Filter graph to only include connected operations
        filtered_graph = []
        for op in self.graph:
            if any(i in connected for i in op['inputs']) and any(o in connected for o in op['outputs']):
                filtered_graph.append(op)
        
        return filtered_graph
    
    def to_onnx(self, output_file='model_graph.onnx'):
        """Convert tracking graph to ONNX for visualization"""
        # Filter graph to keep only connected components
        filtered_graph = self.filter_connected_graph()
        
        # Create ONNX nodes
        nodes = []
        # Find used tensor IDs
        used_tensor_ids = set()
        for op in filtered_graph:
            used_tensor_ids.update(op['inputs'])
            used_tensor_ids.update(op['outputs'])
            
        # Create unique tensor names
        tensor_names = {tid: f"tensor_{tid}" for tid in used_tensor_ids}
        
        # Input tensors for the graph
        inputs = [helper.make_tensor_value_info(tensor_names[i], TensorProto.FLOAT, [1]) 
                 for i in self.input_tensors if i in used_tensor_ids]
        
        # Output tensors (tensors that are outputs but not inputs to other ops)
        output_only = set()
        for op in filtered_graph:
            output_only.update(op['outputs'])
        for op in filtered_graph:
            output_only.difference_update(op['inputs'])
        
        outputs = [helper.make_tensor_value_info(tensor_names[i], TensorProto.FLOAT, [1]) 
                  for i in output_only]
        
        # Create nodes
        for i, op in enumerate(filtered_graph):
            input_names = [tensor_names[tid] for tid in op['inputs']]
            output_names = [tensor_names[tid] for tid in op['outputs']]
            
            # Create ONNX node
            node = helper.make_node(
                op['op_name'],
                name=f"{op['op_name']}_{i}",
                inputs=input_names,
                outputs=output_names
            )
            nodes.append(node)
        
        # Create the graph
        graph_def = helper.make_graph(
            nodes,
            'TensorFlowGraph',
            inputs,
            outputs
        )
        
        # Create the model
        model_def = helper.make_model(graph_def, producer_name='PyTorch-Tracer')
        
        # Save the model
        onnx.save(model_def, output_file)
        print(f"ONNX graph saved to {output_file}")
        return model_def

# Enhanced tensor operation decorator
def tensor_graph_decorator(tracker, op_name):
    """Creates a decorator for tensor operations that builds a graph"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get the model instance
            model_instance = find_model_instance()
            model_name = type(model_instance).__name__ if model_instance else "Unknown"
            
            # Extract tensor inputs
            tensor_inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Handle the result
            if isinstance(result, torch.Tensor):
                tensor_outputs = [result]
            elif isinstance(result, (list, tuple)) and all(isinstance(r, torch.Tensor) for r in result):
                tensor_outputs = list(result)
            else:
                tensor_outputs = []
            
            # Register the operation in our graph if we have tensor outputs
            if tensor_outputs:
                tracker.register_operation(op_name, tensor_inputs, tensor_outputs)
                print(f"Traced {op_name} from {model_name}: {len(tensor_inputs)} inputs → {len(tensor_outputs)} outputs")
            
            return result
        return wrapper
    return decorator

@contextlib.contextmanager
def trace_tensor_graph(ops_to_trace=None):
    """Context manager to trace tensor operations and build a graph"""
    print("Start tracing tensor graph")
    
    # Create a tracker
    tracker = TensorGraphTracker()
    
    # Default operations to trace
    if ops_to_trace is None:
        ops_to_trace = [
            "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__",
            "__mod__", "__pow__", "add", "sub", "mul", "matmul", "div", "floor_divide",
            "remainder", "pow", "add_", "sub_", "mul_", "div_"
        ]
    
    # Store original methods
    originals = {}
    
    # Patch each operation
    for op_name in ops_to_trace:
        if hasattr(torch.Tensor, op_name):
            originals[op_name] = getattr(torch.Tensor, op_name)
            setattr(torch.Tensor, op_name, 
                   tensor_graph_decorator(tracker, op_name)(originals[op_name]))
    
    try:
        yield tracker
    finally:
        # Restore original methods
        for op_name, original_op in originals.items():
            setattr(torch.Tensor, op_name, original_op)
        print("End tracing tensor graph")

# Example usage
if __name__ == "__main__":
    # Simple tensor operations example
    with trace_tensor_graph() as tracker:
        # Create input tensors
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        
        # Perform operations
        c = a + b
        d = a * c
        e = d / b
        
        # Create unconnected operations
        x = torch.tensor([7, 8, 9])
        y = x * 2
        
        # Generate ONNX graph
        tracker.to_onnx('simple_tensor_graph.onnx')
        
        # Print the tracked graph
        print("\nTracked operations:")
        for op in tracker.graph:
            print(f"Operation: {op['op_name']}")
            print(f"  Inputs: {op['inputs']}")
            print(f"  Outputs: {op['outputs']}")
        
        # Print filtered graph
        filtered = tracker.filter_connected_graph()
        print("\nFiltered connected operations:")
        for op in filtered:
            print(f"Operation: {op['op_name']}")
            print(f"  Inputs: {op['inputs']}")
            print(f"  Outputs: {op['outputs']}")
    
    # YOLO model example
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    
    # Load YOLOv5n model
    model = YOLO('yolov5n.pt')
    
    # Generate a random image
    random_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(random_img)
    
    # Trace tensor operations during model inference
    with trace_tensor_graph() as tracker:
        # Run inference (make sure tensors are created inside the context)
        results = model(img)
        
        # Generate ONNX graph
        tracker.to_onnx('yolo_tensor_graph.onnx')