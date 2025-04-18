import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from collections import defaultdict

# Add parent directory to path to import graph tracer components
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import graph tracer components
from graph_tracer_integrated import trace_pytorch_operations

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
    operations_list = []
    
    # Trace operations during model execution
    with trace_pytorch_operations(verbose=True) as tracer:
        output = model(input_tensor)
        
        # Get operations in execution order 
        operations = sorted(tracer.get_operations(), 
                          key=lambda op: op.get_attribute('execution_order', float('inf')))
        
        # Store operation information for ONNX creation
        for op in operations:
            if op.op_name != "load":  # Skip initial tensor load operations
                op_info = {
                    'name': op.op_name,
                    'inputs': [{'name': f'tensor_{info.tensor_id}', 
                               'shape': info.shape} for info in op.input_tensor_infos],
                    'outputs': [{'name': f'tensor_{info.tensor_id}', 
                                'shape': info.shape} for info in op.output_tensor_infos]
                }
                operations_list.append(op_info)
    
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
    
    # Create ONNX model from traced operations
    onnx_nodes = []
    onnx_inputs = []
    onnx_outputs = []
    onnx_initializers = []
    
    # Track tensors to avoid duplicates
    tensor_map = {}
    
    # Export the model normally to get proper ONNX structure
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
    print("First 10 operations in the execution sequence:")
    for i, op in enumerate(operations[:10]):
        if op.op_name != "load":
            inputs = [f"tensor_{info.tensor_id}" for info in op.input_tensor_infos]
            outputs = [f"tensor_{info.tensor_id}" for info in op.output_tensor_infos]
            print(f"{i+1}. {op.op_name}: {inputs} -> {outputs}")
    
    return model, tracer, onnx_file

if __name__ == "__main__":
    model, tracer, onnx_file = trace_and_create_onnx_graph()
