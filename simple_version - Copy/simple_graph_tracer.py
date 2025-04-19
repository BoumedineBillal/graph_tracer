"""
Simple PyTorch Graph Tracer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################
# PART 1: Create a medium PyTorch model
#####################################################

class CustomModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CustomModel, self).__init__()
        
        # Initial convolutional block
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block - path A
        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        
        # Second convolutional block - path B
        self.conv2b = nn.Conv2d(16, 32, kernel_size=1)
        self.bn2b = nn.BatchNorm2d(32)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fourth convolutional block - residual connection
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Final classification layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # Initial block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Parallel paths
        path_a = F.relu(self.bn2a(self.conv2a(x)))
        path_b = F.relu(self.bn2b(self.conv2b(x)))
        
        # Concatenate paths
        x = torch.cat([path_a, path_b], dim=1)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual connection
        identity = x
        x = self.bn4(self.conv4(x))
        x = x + identity  # Add operation (residual connection)
        x = F.relu(x)
        
        # Classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

#####################################################
# PART 2: Test the model with a sample input
#####################################################

# Create model instance
model = CustomModel()

# Sample input (batch_size=1, channels=3, height=32, width=32)
sample_input = torch.randn(1, 3, 32, 32)

# Forward pass
output = model(sample_input)

print(f"Model input shape: {sample_input.shape}")
print(f"Model output shape: {output.shape}")
print("Model architecture:")
print(model)

#####################################################
# PART 3: Get references to functions in torch, F, and tensor methods
#####################################################

def get_function_references():
    # Get functions from torch module
    torch_functions = [name for name in dir(torch) if callable(getattr(torch, name))]
    
    # Get functions from torch.nn.functional
    F_functions = [name for name in dir(F) if callable(getattr(F, name))]
    
    # Get methods from tensor class
    tensor_methods = [name for name in dir(torch.Tensor) if callable(getattr(torch.Tensor, name))]
    
    return torch_functions, F_functions, tensor_methods

def print_function_references(torch_functions, F_functions, tensor_methods, save_to_file=None):
    output_lines = []
    
    output_lines.append("\nTorch functions:")
    output_lines.append(", ".join(sorted(torch_functions)))  # Print first 20 to avoid too much output
    output_lines.append(f"Total torch functions: {len(torch_functions)}")
    
    output_lines.append("\nTorch.nn.functional functions:")
    output_lines.append(", ".join(sorted(F_functions)))
    output_lines.append(f"Total F functions: {len(F_functions)}")
    
    output_lines.append("\nTorch.Tensor methods:")
    output_lines.append(", ".join(sorted(tensor_methods)))  # Print first 20 to avoid too much output
    output_lines.append(f"Total tensor methods: {len(tensor_methods)}")
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Save to file if requested
    if save_to_file:
        with open(save_to_file, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"Function references saved to {save_to_file}")

# Get function references
torch_funcs, F_funcs, tensor_methods = get_function_references()

# Print the function references
print_function_references(torch_funcs, F_funcs, tensor_methods, save_to_file="function_references.txt")

#####################################################
# PART 3.2: Get only operations used in the model
#####################################################

def get_model_operations():
    """Identify operations specifically used in our model"""
    # Operations we know the model uses (from inspecting the model code)
    model_torch_ops = ['cat', 'add']
    
    model_F_ops = [
        'relu',
        'conv2d',
        'batch_norm',
        'adaptive_avg_pool2d',
        'linear'
    ]
    
    model_tensor_ops = [
        'view',
        '__add__'
    ]
    
    # Make sure these operations exist in our collected references
    verified_torch_ops = [op for op in model_torch_ops if op in torch_funcs]
    verified_F_ops = [op for op in model_F_ops if op in F_funcs]
    verified_tensor_ops = [op for op in model_tensor_ops if op in tensor_methods]
    
    # Print the verified operations
    print("\nOperations used in the model:")
    print(f"torch operations: {', '.join(verified_torch_ops)}")
    print(f"F operations: {', '.join(verified_F_ops)}")
    print(f"tensor operations: {', '.join(verified_tensor_ops)}")
    
    return verified_torch_ops, verified_F_ops, verified_tensor_ops

# Get operations used in the model
model_torch_ops, model_F_ops, model_tensor_ops = get_model_operations()

#####################################################
# PART 4: Create operation tracer using context manager
#####################################################

import functools
import contextlib

class OperationTracer(contextlib.ContextDecorator):
    def __init__(self, op_name):
        self.op_name = op_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Format args to show tensor IDs instead of full tensors
            formatted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    formatted_args.append(f"Tensor(id={id(arg)}, shape={list(arg.shape)})")
                else:
                    formatted_args.append(arg)
                    
            print(f"Hello I'm the operation {self.op_name} with args {formatted_args}")
            outputs = func(*args, **kwargs)
            
            # Format outputs to show tensor IDs
            if isinstance(outputs, torch.Tensor):
                output_repr = f"Tensor(id={id(outputs)}, shape={list(outputs.shape)})"
            else:
                output_repr = outputs
                
            print(f"Hello I'm the operation {self.op_name} with outputs {output_repr}")
            return outputs
        return wrapper

#####################################################
# PART 4: Create a simple graph tracer using context manager
#####################################################

@contextlib.contextmanager
def trace_operations(functions_to_trace=None):
    """Context manager to trace operations based on Part 3 references"""
    print("Start tracing operations")
    
    # If no specific functions provided, use the model-specific operations from Part 3.2
    if functions_to_trace is None:
        # Use only the operations actually used in our model
        functions_to_trace = {
            'torch': model_torch_ops,
            'F': model_F_ops,
            'tensor': model_tensor_ops
        }
    
    
    
    # Dictionaries to store original functions
    original_torch_functions = {}
    original_F_functions = {}
    original_tensor_methods = {}
    
    # Patch torch functions
    for func_name in functions_to_trace['torch']:
        if hasattr(torch, func_name) and callable(getattr(torch, func_name)):
            try:
                original_torch_functions[func_name] = getattr(torch, func_name)
                setattr(torch, func_name, 
                        OperationTracer(f"torch.{func_name}")(original_torch_functions[func_name]))
            except Exception as e:
                print(f"Error patching torch.{func_name}: {e}")
    
    # Patch F functions
    for func_name in functions_to_trace['F']:
        if hasattr(F, func_name) and callable(getattr(F, func_name)):
            try:
                original_F_functions[func_name] = getattr(F, func_name)
                setattr(F, func_name, 
                        OperationTracer(f"F.{func_name}")(original_F_functions[func_name]))
            except Exception as e:
                print(f"Error patching F.{func_name}: {e}")
    
    # Patch tensor methods
    for method_name in functions_to_trace['tensor']:
        if hasattr(torch.Tensor, method_name) and callable(getattr(torch.Tensor, method_name)):
            try:
                original_tensor_methods[method_name] = getattr(torch.Tensor, method_name)
                setattr(torch.Tensor, method_name, 
                        OperationTracer(f"tensor.{method_name}")(original_tensor_methods[method_name]))
            except Exception as e:
                print(f"Error patching tensor.{method_name}: {e}")
    
    try:
        # Yield control back to the code inside the with block
        yield
    finally:
        # Restore original functions
        for func_name, original_func in original_torch_functions.items():
            setattr(torch, func_name, original_func)
        for func_name, original_func in original_F_functions.items():
            setattr(F, func_name, original_func)
        for method_name, original_method in original_tensor_methods.items():
            setattr(torch.Tensor, method_name, original_method)
        print("End tracing operations")
        

# Test the context manager with our model
print("\n\nTesting with context manager for operation tracing:")
test_input = torch.randn(1, 3, 8, 8)  # Smaller input for brevity

# Use the context manager
with trace_operations():
    model_output = model(test_input)

print("Model execution completed with traced operations")
