import inspect
import torch
import torch.nn as nn
from typing import Optional, Type, Any, List, Tuple


class ModelInstanceFinder:
    """
    A class to find model instances in the Python call stack.
    Useful for identifying which PyTorch module is currently executing.
    """
    
    def __init__(self, module_type: Type = nn.Module):
        """
        Initialize the finder with the target module type.
        
        Args:
            module_type: The type of module to look for (default: nn.Module)
        """
        self.module_type = module_type
    
    def find_nearest_instance(self) -> Optional[nn.Module]:
        """
        Finds the nearest `self` reference in the call stack that is an instance of the target module type.
        
        Returns:
            The model instance if found, None otherwise
        """
        stack = inspect.stack()
        try:
            # Skip the first frame which is this method itself
            for frame_info in stack[1:]:
                local_vars = frame_info.frame.f_locals
                if "self" in local_vars and isinstance(local_vars["self"], self.module_type):
                    return local_vars["self"]  # Return the model instance
            return None  # No model found
        finally:
            # Ensure frames are properly cleaned up
            del stack
    
    def find_all_instances(self) -> List[nn.Module]:
        """
        Finds all instances of the target module type in the call stack.
        
        Returns:
            List of model instances found in the call stack
        """
        stack = inspect.stack()
        instances = []
        try:
            for frame_info in stack[1:]:
                local_vars = frame_info.frame.f_locals
                if "self" in local_vars and isinstance(local_vars["self"], self.module_type):
                    instances.append(local_vars["self"])
            return instances
        finally:
            # Ensure frames are properly cleaned up
            del stack
    
    def get_instance_hierarchy(self) -> List[Tuple[nn.Module, str]]:
        """
        Returns the hierarchy of model instances in the call stack with their frame names.
        
        Returns:
            List of tuples containing (instance, frame_name)
        """
        stack = inspect.stack()
        hierarchy = []
        try:
            for frame_info in stack[1:]:
                local_vars = frame_info.frame.f_locals
                if "self" in local_vars and isinstance(local_vars["self"], self.module_type):
                    frame_name = frame_info.function
                    hierarchy.append((local_vars["self"], frame_name))
            return hierarchy
        finally:
            # Ensure frames are properly cleaned up
            del stack
    
    @staticmethod
    def find_in_stack(predicate: callable) -> Optional[Any]:
        """
        General utility to find any object in the call stack based on a custom predicate.
        
        Args:
            predicate: A function that takes a local variable dictionary and returns True if the
                      desired object is found
                      
        Returns:
            The found object or None
        """
        stack = inspect.stack()
        try:
            for frame_info in stack[1:]:
                local_vars = frame_info.frame.f_locals
                result = predicate(local_vars)
                if result is not None:
                    return result
            return None
        finally:
            # Ensure frames are properly cleaned up
            del stack


# Test Models
class SubModule(nn.Module):
    """A simple submodule that finds its parent module in the call stack"""
    def __init__(self, name):
        super(SubModule, self).__init__()
        self.name = name
        self.finder = ModelInstanceFinder()
    
    def forward(self, x):
        # Find the nearest model instance (should be the parent calling this submodule)
        parent_module = self.finder.find_nearest_instance()
        if parent_module is not self:  # Skip itself
            print(f"SubModule '{self.name}' found parent: {type(parent_module).__name__}")
        
        # Get full hierarchy
        hierarchy = self.finder.get_instance_hierarchy()
        print(f"Hierarchy from '{self.name}':")
        for i, (instance, frame) in enumerate(hierarchy):
            if instance is not self:  # Skip itself in output for clarity
                print(f"  {i}. {type(instance).__name__} in frame '{frame}'")
        
        # Apply some operation to show this is a real module
        return x + 1


class MidModule(nn.Module):
    """A middle-level module that contains submodules"""
    def __init__(self, name):
        super(MidModule, self).__init__()
        self.name = name
        self.sub1 = SubModule(f"{name}_sub1")
        self.sub2 = SubModule(f"{name}_sub2")
        self.finder = ModelInstanceFinder()
    
    def forward(self, x):
        print(f"\nExecuting MidModule '{self.name}'")
        
        # Process through submodules
        x = self.sub1(x)
        x = self.sub2(x)
        
        # Get parent
        parent = self.finder.find_nearest_instance()
        if parent is not self:  # Skip itself
            print(f"MidModule '{self.name}' found parent: {type(parent).__name__}")
        
        return x * 2


class MainModel(nn.Module):
    """Top-level model containing multiple mid-level modules"""
    def __init__(self):
        super(MainModel, self).__init__()
        self.mid1 = MidModule("mid1")
        self.mid2 = MidModule("mid2")
        self.finder = ModelInstanceFinder()
    
    def forward(self, x):
        print("\n========= Starting MainModel Forward Pass =========")
        
        # Try to find parent (should be None at the top level)
        parent = self.finder.find_nearest_instance()
        if parent is not self:
            print(f"MainModel found parent: {type(parent).__name__}")
        else:
            print("MainModel is the top-level module (no parent found)")
        
        # Process through mid modules
        x = self.mid1(x)
        x = self.mid2(x)
        
        # A more complex example finding submodules
        all_modules = self.finder.find_all_instances()
        print("\nAll modules in call stack:")
        for i, module in enumerate(all_modules):
            if module is not self:  # Skip itself for clarity
                print(f"  {i}. {type(module).__name__}: {getattr(module, 'name', 'unnamed')}")
        
        print("========= Completed MainModel Forward Pass =========\n")
        return x + 3


# Custom module type to demonstrate the module_type parameter
class CustomModule(nn.Module):
    """A custom module type for testing specific type finding"""
    def __init__(self, name):
        super(CustomModule, self).__init__()
        self.name = name
        self.finder = ModelInstanceFinder(module_type=CustomModule)  # Only find CustomModule instances
    
    def forward(self, x):
        # Find only CustomModule instances
        instances = self.finder.find_all_instances()
        print(f"\nCustomModule '{self.name}' found {len(instances)} CustomModule instances")
        return x + 5


class HybridModel(nn.Module):
    """A model that mixes regular and custom modules"""
    def __init__(self):
        super(HybridModel, self).__init__()
        self.mid = MidModule("hybrid_mid")
        self.custom1 = CustomModule("custom1")
        self.custom2 = CustomModule("custom2")
    
    def forward(self, x):
        print("\n========= Starting HybridModel Forward Pass =========")
        
        # Process through different types of modules
        x = self.mid(x)
        x = self.custom1(x)
        x = self.custom2(x)
        
        print("========= Completed HybridModel Forward Pass =========\n")
        return x


# Test functions
def test_module_finding():
    """Test basic module finding functionality"""
    print("\n===== Testing Basic Module Finding =====")
    
    # Create and run a model
    model = MainModel()
    x = torch.tensor([1.0, 2.0, 3.0])
    output = model(x)
    
    print(f"Input: {x}")
    print(f"Output: {output}")


def test_custom_module_finding():
    """Test finding specific module types"""
    print("\n===== Testing Custom Module Finding =====")
    
    # Create and run a hybrid model
    model = HybridModel()
    x = torch.tensor([1.0, 2.0, 3.0])
    output = model(x)
    
    print(f"Input: {x}")
    print(f"Output: {output}")


def test_custom_predicate():
    """Test finding with a custom predicate"""
    print("\n===== Testing Custom Predicate =====")
    
    # Define a custom predicate function
    def find_tensor_with_value_gt_2(locals_dict):
        for var_name, var_value in locals_dict.items():
            if isinstance(var_value, torch.Tensor) and var_value.numel() > 0 and var_value.max() > 2:
                return var_name, var_value
        return None
    
    # Create a simple function that has a tensor in its locals
    def func_with_tensor():
        t1 = torch.tensor([1.0, 3.0, 2.0])  # Has value > 2
        t2 = torch.tensor([1.0, 2.0, 0.5])  # Doesn't have value > 2
        
        # Find tensor using custom predicate
        finder = ModelInstanceFinder()
        result = finder.find_in_stack(find_tensor_with_value_gt_2)
        
        if result:
            var_name, tensor = result
            print(f"Found tensor '{var_name}' with value > 2: {tensor}")
        else:
            print("No tensor with value > 2 found")
    
    # Run the test
    func_with_tensor()


if __name__ == "__main__":
    # Run all tests
    test_module_finding()
    test_custom_module_finding()
    test_custom_predicate()
