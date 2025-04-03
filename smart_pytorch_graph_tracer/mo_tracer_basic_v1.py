import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import types
import contextlib

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

# New functions for tensor operations

def tensor_op_decorator(op_name):
    """Creates a decorator for tensor operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            model_instance = find_model_instance()  # Get the model dynamically
            model_name = type(model_instance).__name__ if model_instance else "Unknown"
            print(f"Tensor operation: {op_name} from model: {model_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@contextlib.contextmanager
def patch_tensor_operations(ops_to_patch=None):
    """Context manager to patch tensor arithmetic operations"""
    print("start patching tensor operations")
    
    # Default operations to patch if none specified
    if ops_to_patch is None:
        ops_to_patch = [
            "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__",
            "__mod__", "__pow__", "add", "sub", "mul", "matmul", "div", "floor_divide",
            "remainder", "pow", "add_", "sub_", "mul_", "div_"
        ]
    
    # Store original methods
    originals = {}
    
    # Patch each operation
    for op_name in ops_to_patch:
        if hasattr(torch.Tensor, op_name):
            originals[op_name] = getattr(torch.Tensor, op_name)
            setattr(torch.Tensor, op_name, tensor_op_decorator(op_name)(originals[op_name]))
    
    try:
        yield
    finally:
        # Restore original methods
        for op_name, original_op in originals.items():
            setattr(torch.Tensor, op_name, original_op)
        print("end patching tensor operations")

# Combined context manager for both module functions and tensor operations
@contextlib.contextmanager
def patch_pytorch(module=None, tensor_ops=None):
    """Context manager to patch both module functions and tensor operations"""
    # First patch tensor operations
    with patch_tensor_operations(tensor_ops):
        # Then patch module functions if a module is provided
        if module is not None:
            with patch_module_functions(module, my_decorator):
                yield
        else:
            yield

# Example usage
if __name__ == "__main__":
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    
    # Load YOLOv5n model
    model = YOLO('yolov5n.pt')
    
    # Generate a random image (3 channels, 640x640 pixels)
    random_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(random_img)
    
    # Patch both F module and tensor operations
    with patch_pytorch(module=F):
        # Run inference
        results = model(img)
    
    # Or patch just tensor operations
    with patch_tensor_operations():
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        c = a + b  # Will print tensor operation message