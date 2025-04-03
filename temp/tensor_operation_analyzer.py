import torch
import numpy as np
from typing import Any, List, Dict, Tuple, Optional, Union


class TensorInfo:
    """
    Class that captures information about a tensor and its path within a data structure.
    """
    def __init__(self, tensor=None):
        """
        Initialize with an optional tensor.
        
        Args:
            tensor: The tensor to capture information about
        """
        # Path to the tensor in the data structure
        self.path = []
        
        # Tensor metadata
        self.tensor_id = id(tensor) if tensor is not None else None
        self.shape = list(tensor.shape) if tensor is not None and hasattr(tensor, 'shape') else []
    
    def set_path(self, path):
        """Set the path to this tensor"""
        self.path = path
        return self
    
    def set_tensor(self, tensor):
        """Update tensor information"""
        self.tensor_id = id(tensor)
        self.shape = list(tensor.shape) if hasattr(tensor, 'shape') else []
        return self

    def __repr__(self):
        """String representation for debugging"""
        path_str = str(self.path)
        shape_str = 'x'.join(str(dim) for dim in self.shape) if self.shape else 'scalar'
        return f"TensorInfo(shape={shape_str}, id={self.tensor_id}, path={path_str})"


class TensorOperationAnalyzer:
    """
    Analyzes function calls to determine if they involve tensor operations,
    including when tensors are nested in containers.
    """
    
    def find_tensors_in_structure(self, obj: Any, path: Optional[List] = None) -> List[TensorInfo]:
        """
        Recursively find tensors in a nested structure and record their paths.
        
        Args:
            obj: The object to search for tensors
            path: Current path in the structure (default: empty list)
            
        Returns:
            list: List of TensorInfo objects with paths to tensors
        """
        if path is None:
            path = []
            
        tensor_infos = []
        
        if isinstance(obj, torch.Tensor):
            # Create a TensorInfo object for this tensor
            info = TensorInfo(obj)
            info.set_path(path.copy())
            tensor_infos.append(info)
        elif isinstance(obj, (list, tuple)):
            # Record that we're inside a sequence and continue searching
            container_type = 'list' if isinstance(obj, list) else 'tuple'
            for i, item in enumerate(obj):
                sub_path = path + [{'type': container_type, 'index': i}]
                tensor_infos.extend(self.find_tensors_in_structure(item, sub_path))
        elif isinstance(obj, dict):
            # Record that we're inside a dict and continue searching
            for key, value in obj.items():
                sub_path = path + [{'type': 'dict', 'key': key}]
                tensor_infos.extend(self.find_tensors_in_structure(value, sub_path))
                
        return tensor_infos
    
    def analyze_operation(self, func, args, kwargs):
        """
        Analyze if a function call involves tensor operations.
        
        Args:
            func: The function/method being called
            args: Positional arguments passed to the function
            kwargs: Keyword arguments passed to the function
            
        Returns:
            dict: Analysis results containing tensor inputs, outputs, and the original result
        """
        # Find tensors in positional args
        input_tensors = []
        for i, arg in enumerate(args):
            paths = self.find_tensors_in_structure(arg, [{'type': 'arg', 'index': i}])
            input_tensors.extend(paths)
        
        # Find tensors in keyword args
        for key, value in kwargs.items():
            paths = self.find_tensors_in_structure(value, [{'type': 'kwarg', 'key': key}])
            input_tensors.extend(paths)
        
        has_tensor_inputs = len(input_tensors) > 0
        
        # Call the function to get the result
        try:
            result = func(*args, **kwargs)
            
            # Check output for tensors
            output_tensors = self.find_tensors_in_structure(result, [{'type': 'output'}])
            has_tensor_outputs = len(output_tensors) > 0
            
            # Create the analysis result
            analysis = {
                'has_tensor_inputs': has_tensor_inputs,
                'has_tensor_outputs': has_tensor_outputs,
                'input_tensors': input_tensors,  # Now a list of TensorInfo objects
                'output_tensors': output_tensors,  # Now a list of TensorInfo objects
                'result': result,  # Store the original result
                'error': None
            }
        except Exception as e:
            # If there was an error, record it
            analysis = {
                'has_tensor_inputs': has_tensor_inputs,
                'has_tensor_outputs': False,
                'input_tensors': input_tensors,  # List of TensorInfo objects
                'output_tensors': [],  # Empty list (no outputs due to error)
                'result': None,
                'error': str(e)
            }
            
        return analysis
