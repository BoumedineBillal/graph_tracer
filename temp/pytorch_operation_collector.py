import torch
import torch.nn.functional as F
import inspect
import types
import os

class PyTorchOperationCollector:
    """Collects all callable operations from PyTorch namespaces"""
    
    def __init__(self, collect_on_init=True):
        # Central storage for all operations
        self.all_operations = {}
        
        # Collect operations during initialization if requested
        if collect_on_init:
            self.collect_all()
    
    def collect_from_namespace(self, namespace, prefix):
        """
        Collect operations from a given namespace
        
        Args:
            namespace: The module to inspect
            prefix: String prefix for the operation name (e.g., "torch.", "F.")
        """
        for name in dir(namespace):
            # Skip private attributes unless they're operator dunders
            if name.startswith('_') and not (name.startswith('__') and name.endswith('__')):
                continue
                
            attr = getattr(namespace, name)
            if callable(attr) and not inspect.isclass(attr):
                # Full operation name with prefix
                full_name = f"{prefix}{name}"
                
                # Try to get the signature
                try:
                    sig = inspect.signature(attr)
                    signature_str = str(sig)
                except (ValueError, TypeError):
                    signature_str = None
                
                # Store operation info
                self.all_operations[full_name] = {
                    'callable': attr,
                    'signature': signature_str,
                    'in_place': name.endswith('_'),
                    'is_dunder': name.startswith('__') and name.endswith('__'),
                    'is_builtin': isinstance(attr, types.BuiltinFunctionType)
                }
    
    def collect_tensor_methods(self):
        """Collect methods from torch.Tensor class"""
        for name in dir(torch.Tensor):
            # Skip private methods unless they're operator dunders
            if name.startswith('_') and not (name.startswith('__') and name.endswith('__')):
                continue
                
            method = getattr(torch.Tensor, name)
            if callable(method):
                # Full operation name
                full_name = f"tensor.{name}"
                
                # Try to get signature if possible
                try:
                    sig = inspect.signature(method)
                    signature_str = str(sig)
                except (ValueError, TypeError):
                    signature_str = None
                
                # Store operation info
                self.all_operations[full_name] = {
                    'callable': method,
                    'signature': signature_str,
                    'in_place': name.endswith('_'),
                    'is_dunder': name.startswith('__') and name.endswith('__'),
                    'is_builtin': isinstance(method, types.BuiltinFunctionType)
                }
        
    def collect_all(self):
        """Collect operations from all relevant PyTorch namespaces"""
        # Clear previous collection
        self.all_operations = {}
        
        # Collect from torch namespace
        self.collect_from_namespace(torch, 'torch.')
        
        # Collect from torch.nn.functional
        self.collect_from_namespace(F, 'F.')
        
        # Collect tensor methods
        self.collect_tensor_methods()
        
        return self.all_operations
    
    def get_operation_count_by_namespace(self):
        """Get count of operations by namespace"""
        counts = {
            'torch': 0,
            'F': 0,
            'tensor': 0
        }
        
        for op_name in self.all_operations:
            if op_name.startswith('torch.'):
                counts['torch'] += 1
            elif op_name.startswith('F.'):
                counts['F'] += 1
            elif op_name.startswith('tensor.'):
                counts['tensor'] += 1
        
        return counts
    
class PyTorchOperationSummarizer:
    """Generates summaries of PyTorch operations collected by PyTorchOperationCollector"""
    
    def __init__(self, collector=None):
        """Initialize with an optional collector instance"""
        self.collector = collector
    
    def set_collector(self, collector):
        """Set the operation collector to use"""
        self.collector = collector
    
    def print_summary(self):
        """Print a summary of collected operations to console"""
        if not self.collector:
            raise ValueError("No collector set. Use set_collector() first.")
            
        counts = self.collector.get_operation_count_by_namespace()
        print(f"Collected {len(self.collector.all_operations)} PyTorch operations:")
        print(f"  - torch namespace: {counts['torch']}")
        print(f"  - F namespace: {counts['F']}")
        print(f"  - tensor methods: {counts['tensor']}")
        
        # Show some examples from each namespace
        namespaces = ['torch.', 'F.', 'tensor.']
        for prefix in namespaces:
            examples = [op for op in self.collector.all_operations.keys() 
                        if op.startswith(prefix)][:5]  # First 5 ops
            print(f"\nSample operations from {prefix}:")
            for ex in examples:
                info = self.collector.all_operations[ex]
                print(f"  - {ex} {info['signature'] or ''}")
    
    def export_full_summary(self, output_folder='pytorch_operations'):
        """Export a full summary of all collected operations to text files in a folder"""
        if not self.collector:
            raise ValueError("No collector set. Use set_collector() first.")
            
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Main summary file
        main_file = os.path.join(output_folder, 'summary.txt')
        with open(main_file, 'w') as f:
            # Write header
            counts = self.collector.get_operation_count_by_namespace()
            f.write(f"PyTorch Operations Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Total operations: {len(self.collector.all_operations)}\n")
            f.write(f"  - torch namespace: {counts['torch']}\n")
            f.write(f"  - F namespace: {counts['F']}\n")
            f.write(f"  - tensor methods: {counts['tensor']}\n\n")
        
        # Export each namespace to a separate file
        namespaces = ['torch.', 'F.', 'tensor.']
        for prefix in namespaces:
            # Get operations for this namespace
            namespace_short = prefix.replace('.', '')
            namespace_file = os.path.join(output_folder, f"{namespace_short}_operations.txt")
            
            # Sort operations for this namespace
            namespace_ops = [(name, info) for name, info in sorted(self.collector.all_operations.items()) 
                            if name.startswith(prefix)]
            
            with open(namespace_file, 'w') as f:
                f.write(f"{prefix} Operations\n")
                f.write(f"{'-' * (len(prefix) + 10)}\n\n")
                
                for name, info in namespace_ops:
                    f.write(f"{name}")
                    if info['signature']:
                        f.write(f" {info['signature']}")
                    f.write("\n")
                    
                    # Write additional metadata
                    if info['in_place']:
                        f.write("  In-place operation\n")
                    if info['is_dunder']:
                        f.write("  Dunder method\n")
                    if info['is_builtin']:
                        f.write("  Built-in function\n")
                    f.write("\n")
            
        print(f"Full summary exported to {os.path.abspath(output_folder)}")

# Usage example
if __name__ == "__main__":
    # Create and use the collector
    collector = PyTorchOperationCollector()  # Now collects on init by default
    
    # Create the summarizer and set the collector
    summarizer = PyTorchOperationSummarizer(collector)
    summarizer.print_summary()
    
    # Export full summary to a folder
    summarizer.export_full_summary('pytorch_ops')
