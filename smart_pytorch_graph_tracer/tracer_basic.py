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

# def my_decorator(func):
#     seen_instances = set()
#     def wrapper(*args, **kwargs):
#         model_instance = find_model_instance()
#         if model_instance:
#             # Create a unique identifier using object id
#             instance_id = id(model_instance)
#             model_name = type(model_instance).__name__
#             # Only print if we haven't seen this exact instance before
#             if instance_id not in seen_instances:
#                 print(f"Calling function: {func.__name__} from model: {model_name}")
#                 seen_instances.add(instance_id)
#         return func(*args, **kwargs)
    
#     return wrapper

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



from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load YOLOv5n model
model = YOLO('yolov5n.pt')

# Generate a random image (3 channels, 640x640 pixels)
random_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
img = Image.fromarray(random_img)



with patch_module_functions(F, my_decorator):
    # Run inference
    results = model(img)



















