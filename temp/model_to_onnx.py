import torch
import os
import onnx
from onnx import helper
import onnxruntime as ort

def convert_to_onnx(model, input_shape, output_path, model_name='model', 
                    dynamic_axes=None, opset_version=15, check_model=True, 
                    check_runtime=True):
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model (nn.Module): PyTorch model to convert
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width)
        output_path (str): Directory path to save the ONNX model
        model_name (str): Name for the saved model file (without extension)
        dynamic_axes (dict, optional): Dynamic axes for variable size inputs/outputs
        opset_version (int): ONNX opset version to use
        check_model (bool): Whether to check the exported model with ONNX checker
        check_runtime (bool): Whether to perform a runtime check with ONNXRuntime
        
    Returns:
        str: Full path to the exported ONNX model
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape, requires_grad=False)
    
    # Full path for the output file
    onnx_file_path = os.path.join(output_path, f"{model_name}.onnx")
    
    # Default dynamic axes if None is provided
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export the model
    print(f"Converting model to ONNX format with opset version {opset_version}...")
    torch.onnx.export(
        model,               # Model being exported
        dummy_input,         # Model input
        onnx_file_path,      # Output file
        export_params=True,  # Export model parameters
        opset_version=opset_version,  # ONNX opset version
        do_constant_folding=True,    # Optimize constant expressions
        input_names=['input'],      # Input node name
        output_names=['output'],    # Output node name
        dynamic_axes=dynamic_axes,   # Support dynamic batch size
        verbose=False               # Detailed export info
    )
    
    # Check the exported model
    if check_model:
        print("Checking ONNX model...")
        onnx_model = onnx.load(onnx_file_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed!")
    
    # Check with ONNX Runtime
    if check_runtime:
        print("Checking ONNX model with ONNX Runtime...")
        try:
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_file_path)
            
            # Prepare inputs
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Run the same input through PyTorch model
            with torch.no_grad():
                torch_output = model(dummy_input).numpy()
            
            # Compare outputs
            import numpy as np
            if np.allclose(ort_outputs[0], torch_output, rtol=1e-3, atol=1e-5):
                print("ONNX Runtime test passed! PyTorch and ONNX Runtime outputs match.")
            else:
                print("Warning: PyTorch and ONNX Runtime outputs don't match closely.")
                print(f"Max absolute difference: {np.max(np.abs(ort_outputs[0] - torch_output))}")
        except Exception as e:
            print(f"ONNX Runtime check failed: {e}")
    
    print(f"Model exported to: {onnx_file_path}")
    return onnx_file_path

def visualize_onnx_graph(onnx_file_path, output_image_path=None):
    """
    Visualize ONNX model graph structure.
    This function requires netron package to be installed.
    
    Args:
        onnx_file_path (str): Path to the ONNX model file
        output_image_path (str, optional): Path to save the visualization image
            If None, will save in the same directory with '.png' extension
    
    Returns:
        str: Path to the output image
    """
    try:
        import netron
    except ImportError:
        print("Netron package not found. Please install with: pip install netron")
        return None
    
    if output_image_path is None:
        output_image_path = onnx_file_path.replace('.onnx', '.png')
    
    # Start the server and save the image
    print(f"Generating visualization of ONNX model to {output_image_path}")
    netron.export_file(onnx_file_path, output_image_path)
    
    return output_image_path
