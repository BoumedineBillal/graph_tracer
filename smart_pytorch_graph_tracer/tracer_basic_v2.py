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

# Define a submodule that also uses F.relu
class SubModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.relu(x)  # This should log the submodule name
        return x * 2  # Just an arbitrary transformation

# Define the main model that uses the submodule
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = SubModule()

    def forward(self, x):
        x = F.relu(x)  # This should log "MyModel"
        x = self.submodule(x)  # Calls the submodule, which also uses relu
        return x

# Usage
model = MyModel()
x = torch.tensor([-1.0, 0.0, 1.0])

with patch_module_functions(F, my_decorator):
    y = model(x)  # Calls F.relu() inside both MyModel and SubModule



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



















