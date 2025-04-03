import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#############################################
# Building Blocks for Complex Models
#############################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(self.conv(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class CustomArithmeticBlock(nn.Module):
    def __init__(self, channels):
        super(CustomArithmeticBlock, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        # First multiply by scale
        x = x * self.scale
        # Then add bias
        return x + self.bias

#############################################
# Models for Testing
#############################################

class ComplexTestModel(nn.Module):
    """
    A complex model with various operations including convolutions, 
    upsampling, pooling, and element-wise operations
    
    Input shape: (batch_size, 32, 128, 128)
    Output shape: (batch_size, 1, 128, 128)
    """
    def __init__(self):
        super(ComplexTestModel, self).__init__()
        
        # Initial convolution
        self.initial_conv = ConvBlock(32, 16)
        
        # Down path
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        
        # Custom arithmetic operations
        self.arithmetic1 = CustomArithmeticBlock(64)
        
        # Middle convolutions
        self.mid_conv1 = ConvBlock(64, 64)
        self.mid_conv2 = ConvBlock(64, 64)
        
        # Up path
        self.up1 = UpBlock(64, 32)
        self.up2 = UpBlock(32, 16)
        
        # Final convolution
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
    
    def forward(self, x):
        # Down path with skip connections
        x1 = self.initial_conv(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Middle processing with custom arithmetic
        x3 = self.arithmetic1(x3)
        
        # Non-sequential behavior: we'll apply convolutions and add results
        mid1 = self.mid_conv1(x3)
        mid2 = self.mid_conv2(x3)
        x3 = mid1 + mid2  # Element-wise addition
        
        # Up path with skip connections
        x2_up = self.up1(x3)
        x1_up = self.up2(x2_up)
        
        # Connect with skip connection from first layer
        x_combined = x1 * x1_up  # Element-wise multiplication
        
        # Final processing
        output = self.final_conv(x_combined)
        return output


class TensorOperationsModel(nn.Module):
    """
    A model that focuses only on tensor operations like addition,
    multiplication, division, etc. without any neural network layers.
    
    Input shape: (batch_size, 10)
    Output shape: (batch_size, 10)
    """
    def __init__(self):
        super(TensorOperationsModel, self).__init__()
        # Create some learnable parameters
        self.weights1 = nn.Parameter(torch.ones(10))
        self.weights2 = nn.Parameter(torch.ones(10) * 0.5)
        self.bias1 = nn.Parameter(torch.zeros(10))
        self.bias2 = nn.Parameter(torch.zeros(10) + 0.1)
        
        # Constants
        self.register_buffer('constant1', torch.ones(10) * 2.0)
        self.register_buffer('constant2', torch.ones(10) * 0.3)
    
    def forward(self, x):
        # x shape: (batch_size, 10)
        
        # Basic operations
        a = x + self.bias1                # Addition
        b = x * self.weights1             # Multiplication
        c = x / (self.constant1 + 0.1)    # Division (with small constant for stability)
        d = x - self.bias2                # Subtraction
        
        # Combined operations
        e = a + b                         # Addition of tensors
        f = c * d                         # Multiplication of tensors
        g = e / (f + 0.1)                 # Division of tensors
        
        # More complex operations
        h = torch.pow(g, 2)               # Power operation
        i = torch.sqrt(h + 0.1)           # Square root (with small constant for stability)
        j = torch.exp(i * self.weights2)  # Exponential
        
        # Final operations
        k = j + self.bias1                # Addition
        result = k * self.constant2       # Final scaling
        
        return result


#############################################
# Model Registry and Utilities
#############################################

# Registry of all available models
MODEL_REGISTRY = {
    'complex': {
        'class': ComplexTestModel,
        'input_shape': (1, 32, 128, 128),
        'description': 'Complex model with convolutions, pooling, and various tensor operations'
    },
    'tensor_ops': {
        'class': TensorOperationsModel,
        'input_shape': (1, 10),
        'description': 'Model focusing only on tensor operations like addition, multiplication, etc.'
    }
}


def get_model(model_name):
    """
    Get a model by name from the registry
    
    Args:
        model_name (str): Name of the model to retrieve
        
    Returns:
        tuple: (model, input_shape)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_name]
    model = model_info['class']()
    model.eval()  # Set to evaluation mode
    
    return model, model_info['input_shape']


def test_model_inference(model_name):
    """
    Test that the model works with the expected input shape
    
    Args:
        model_name (str): Name of the model to test
        
    Returns:
        tuple: Output shape
    """
    model, input_shape = get_model(model_name)
    
    # Create test input
    x = torch.randn(input_shape)
    
    # Run forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Model: {model_name}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    return output.shape


def convert_model_to_onnx(model_name, output_dir="onnx_models"):
    """
    Convert a model to ONNX format
    
    Args:
        model_name (str): Name of the model to convert
        output_dir (str): Directory to save the ONNX model
        
    Returns:
        str: Path to the ONNX file
    """
    from model_to_onnx import convert_to_onnx, visualize_onnx_graph
    
    # Get model and input shape
    model, input_shape = get_model(model_name)
    
    # Convert
    output_path = os.path.join(os.path.dirname(__file__), output_dir)
    onnx_file = convert_to_onnx(model, input_shape, output_path, f"{model_name}_model")
    
    # Visualize
    visualize_onnx_graph(onnx_file)
    return onnx_file


def list_available_models():
    """List all available models with their descriptions"""
    print("Available Models:")
    print("-" * 80)
    for name, info in MODEL_REGISTRY.items():
        print(f"{name}: {info['description']}")
        print(f"  Input shape: {info['input_shape']}")
    print("-" * 80)


if __name__ == "__main__":
    # List available models
    list_available_models()
    
    # Test all models
    for model_name in MODEL_REGISTRY.keys():
        print(f"\nTesting model: {model_name}")
        test_model_inference(model_name)
    
    # Convert models to ONNX
    convert_model_to_onnx('complex')
    convert_model_to_onnx('tensor_ops')
