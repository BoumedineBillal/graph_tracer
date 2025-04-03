# PyTorch Operation Collection

This module provides tools for collecting and exploring PyTorch operations across different namespaces.

## Overview

The implementation consists of two main components:

1. **PyTorchOperationCollector**: Discovers all callable operations from PyTorch namespaces
2. **PyTorchOperationSummarizer**: Generates human-readable summaries of the collected operations

## PyTorchOperationCollector

This class collects all callable operations from:
- `torch` namespace
- `torch.nn.functional` namespace 
- `torch.Tensor` methods

For each operation, it collects:
- The callable object itself
- Signature information when available
- Whether it's an in-place operation
- Whether it's a dunder method (like `__add__`)
- Whether it's a built-in function

### Usage

```python
from pytorch_operation_collector import PyTorchOperationCollector

# Create collector and automatically collect operations
collector = PyTorchOperationCollector()

# Access all operations
all_ops = collector.all_operations

# Get counts by namespace
counts = collector.get_operation_count_by_namespace()
print(f"Found {counts['torch']} operations in torch namespace")
```

## PyTorchOperationSummarizer

This class generates readable summaries of the collected operations. It can:
- Print a concise summary to the console
- Export detailed information to a folder structure

### Usage

```python
from pytorch_operation_collector import PyTorchOperationCollector, PyTorchOperationSummarizer

# Create collector
collector = PyTorchOperationCollector()

# Create summarizer with the collector
summarizer = PyTorchOperationSummarizer(collector)

# Print summary to console
summarizer.print_summary()

# Export detailed information to a folder
summarizer.export_full_summary('pytorch_ops')
```

## Output Format

When exporting to a folder, the summarizer creates:
- `summary.txt`: Overview of all operations
- `torch_operations.txt`: Detailed list of torch namespace operations
- `F_operations.txt`: Detailed list of nn.functional operations
- `tensor_operations.txt`: Detailed list of tensor methods

## Next Steps

This implementation is the first step toward building a comprehensive PyTorch operation tracer. The collected operations will be used to:

1. Create decorators for each operation type
2. Track tensor flow during model execution
3. Build a computational graph of operations
4. Visualize the execution flow

These next steps will build upon the collected operation information to provide insights into how PyTorch models execute internally.
