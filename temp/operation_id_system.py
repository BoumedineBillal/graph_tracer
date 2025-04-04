import torch
from typing import Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict

# Importing TensorInfo from the analyzer
try:
    from tensor_operation_analyzer import TensorInfo
except ImportError:
    # Fallback definition for testing
    class TensorInfo:
        """Simplified TensorInfo for testing purposes"""
        def __init__(self, tensor=None):
            self.tensor_id = id(tensor) if tensor is not None else None
            self.shape = list(tensor.shape) if tensor is not None and hasattr(tensor, 'shape') else []
            self.path = []
            
        def __eq__(self, other):
            if not isinstance(other, TensorInfo):
                return False
            return self.tensor_id == other.tensor_id
            
        def __hash__(self):
            return hash(self.tensor_id)


class OperationIDManager:
    """
    Manages the creation and mapping of operation IDs.
    
    This class handles:
    1. Creating smart IDs for operations based on their inputs
    2. Mapping complex IDs to simpler unique IDs
    3. Tracking operation indices for operations with the same name and input signature
    """
    
    def __init__(self):
        # Counter for each operation type
        self.op_type_counters = defaultdict(int)
        
        # Map from complex ID to simpler unique ID
        self.id_mapping = {}
        
        # Counter for generating unique operation IDs
        self.next_op_id = 0
        
        # Track operations by name and input signature
        self.op_signature_counts = defaultdict(int)
    
    def get_unique_op_id(self) -> str:
        """Get a new unique operation ID."""
        op_id = f"op*{self.next_op_id}"
        self.next_op_id += 1
        return op_id
    
    def get_load_operation_id(self, op_type: str) -> Tuple[str, str]:
        """
        Generate ID for a load operation (one that creates initial tensors).
        
        Args:
            op_type: Type of load operation
            
        Returns:
            Tuple[str, str]: (complex_id, unique_id)
        """
        # For load operations, we use a global counter by type
        index = self.op_type_counters[op_type]
        self.op_type_counters[op_type] += 1
        
        # Create the complex ID for this load operation
        complex_id = f"{op_type}*{index}"
        
        # Check if we already have a mapping for this ID
        if complex_id in self.id_mapping:
            return complex_id, self.id_mapping[complex_id]
        
        # Create a new unique ID and store the mapping
        unique_id = self.get_unique_op_id()
        self.id_mapping[complex_id] = unique_id
        
        return complex_id, unique_id
    
    def get_operation_index(self, op_name: str, input_signature: str) -> int:
        """
        Get the index for an operation with the given name and input signature.
        
        Args:
            op_name: Operation name
            input_signature: Signature of input operations
            
        Returns:
            int: Index for this operation
        """
        signature_key = f"{op_name}|{input_signature}"
        index = self.op_signature_counts[signature_key]
        self.op_signature_counts[signature_key] += 1
        return index
    
    def get_operation_id(self, op_name: str, input_op_ids: List[str]) -> Tuple[str, str]:
        """
        Generate ID for an operation based on its type and input operations.
        
        Args:
            op_name: Type of operation
            input_op_ids: List of unique IDs of operations that produced input tensors
            
        Returns:
            Tuple[str, str]: (complex_id, unique_id)
        """
        # Create a sorted input signature string
        input_signature = "|".join(sorted(input_op_ids))
        
        # Get index specific to this operation name and input signature
        index = self.get_operation_index(op_name, input_signature)
        
        # Create the complex ID
        if input_op_ids:
            complex_id = f"{op_name}*{index}|{input_signature}"
        else:
            complex_id = f"{op_name}*{index}"
        
        # Check if we already have a mapping for this ID
        if complex_id in self.id_mapping:
            return complex_id, self.id_mapping[complex_id]
        
        # Create a new unique ID and store the mapping
        unique_id = self.get_unique_op_id()
        self.id_mapping[complex_id] = unique_id
        
        return complex_id, unique_id


class Operation:
    """
    Represents a tensor operation in the computational graph.
    
    This class stores information about a PyTorch operation including its
    input and output tensors, operation type, and additional metadata.
    """
    
    def __init__(self, op_name: str):
        """
        Initialize an operation.
        
        Args:
            op_name: Name of the operation (e.g., 'load', 'add', 'matmul')
        """
        self.op_name = op_name
        
        # Will be set when generate_ids is called
        self.complex_id = None
        self.unique_id = None
        
        # Tensor tracking
        self.input_tensor_infos = []   # TensorInfo objects for input tensors
        self.output_tensor_infos = []  # TensorInfo objects for output tensors
        self.input_op_ids = []         # IDs of operations that produced input tensors
        
        # Additional metadata
        self.attributes = {}          # Operation attributes
        self.execution_order = None   # Position in execution sequence
    
    def add_input_tensor(self, tensor_info: TensorInfo, source_op_id: Optional[str] = None) -> None:
        """
        Add an input tensor and its source operation.
        
        Args:
            tensor_info: TensorInfo object for the input tensor
            source_op_id: Unique ID of the operation that produced this tensor
        """
        if tensor_info not in self.input_tensor_infos:
            self.input_tensor_infos.append(tensor_info)
            
        if source_op_id and source_op_id not in self.input_op_ids:
            self.input_op_ids.append(source_op_id)
    
    def add_output_tensor(self, tensor_info: TensorInfo) -> None:
        """Add an output tensor to this operation."""
        if tensor_info not in self.output_tensor_infos:
            self.output_tensor_infos.append(tensor_info)
    
    def generate_load_id(self, id_manager: OperationIDManager) -> None:
        """
        Generate ID for a load operation.
        
        Args:
            id_manager: The ID manager to use
        """
        self.complex_id, self.unique_id = id_manager.get_load_operation_id(self.op_name)
    
    def generate_operation_id(self, id_manager: OperationIDManager) -> None:
        """
        Generate ID for an operation based on its inputs.
        
        Args:
            id_manager: The ID manager to use
        """
        self.complex_id, self.unique_id = id_manager.get_operation_id(
            self.op_name, self.input_op_ids
        )
    
    def set_attribute(self, key: str, value: any) -> None:
        """Set an additional attribute for this operation."""
        self.attributes[key] = value
    
    def get_attribute(self, key: str, default: any = None) -> any:
        """Get an attribute value with an optional default."""
        return self.attributes.get(key, default)
    
    def __repr__(self) -> str:
        """String representation of the operation."""
        if self.complex_id:
            return f"Operation({self.op_name}, id={self.unique_id}, complex_id={self.complex_id})"
        else:
            return f"Operation({self.op_name}, id=<not_generated>)"


class OperationCollection:
    """
    A simple collection class to store and manage operations.
    """
    
    def __init__(self):
        self.id_manager = OperationIDManager()
        self.operations = {}  # Maps unique_id to Operation
        self.tensor_to_op = {}  # Maps tensor_id to operation unique_id
        self.known_tensors = set()  # Set of all tensor IDs we've seen
    
    def register_tensor_source(self, tensor_info: TensorInfo, op_id: str) -> None:
        """
        Register which operation produced a tensor.
        
        Args:
            tensor_info: TensorInfo object for the tensor
            op_id: Unique ID of the source operation
        """
        self.tensor_to_op[tensor_info.tensor_id] = op_id
        self.known_tensors.add(tensor_info.tensor_id)
    
    def get_source_op_id(self, tensor_info: TensorInfo) -> Optional[str]:
        """
        Get the operation that produced a tensor.
        
        Args:
            tensor_info: TensorInfo object for the tensor
            
        Returns:
            str: Unique ID of the source operation, or None if not found
        """
        return self.tensor_to_op.get(tensor_info.tensor_id)
    
    def is_known_tensor(self, tensor_info: TensorInfo) -> bool:
        """
        Check if a tensor is already known to the collection.
        
        Args:
            tensor_info: TensorInfo object for the tensor
            
        Returns:
            bool: True if the tensor is known, False otherwise
        """
        return tensor_info.tensor_id in self.known_tensors
    
    def add_operation(self, operation: Operation) -> None:
        """
        Add an operation to the collection.
        
        Args:
            operation: The operation to add
        """
        if operation.unique_id is None:
            raise ValueError("Operation must have a unique ID before adding to collection")
        
        self.operations[operation.unique_id] = operation
        
        # Register this operation as the source for all output tensors
        for tensor_info in operation.output_tensor_infos:
            self.register_tensor_source(tensor_info, operation.unique_id)
    
    def create_load_operation(self, tensor_info: TensorInfo) -> Operation:
        """
        Create a load operation for a tensor with no source.
        
        Args:
            tensor_info: TensorInfo object for the tensor
            
        Returns:
            Operation: The created load operation
        """
        operation = Operation("load")
        operation.add_output_tensor(tensor_info)
        operation.generate_load_id(self.id_manager)
        
        # Add to collection
        self.add_operation(operation)
        return operation
    
    def create_operation(self, op_name: str, input_tensor_infos: List[TensorInfo], 
                        output_tensor_infos: List[TensorInfo]) -> Operation:
        """
        Create an operation with input and output tensors.
        
        Args:
            op_name: Name of the operation
            input_tensor_infos: List of TensorInfo objects for input tensors
            output_tensor_infos: List of TensorInfo objects for output tensors
            
        Returns:
            Operation: The created operation
        """
        operation = Operation(op_name)
        
        # Process input tensors
        for tensor_info in input_tensor_infos:
            # If tensor is not known, create a load operation for it
            if not self.is_known_tensor(tensor_info):
                self.create_load_operation(tensor_info)
            
            # Get source operation ID (now it must exist)
            source_op_id = self.get_source_op_id(tensor_info)
            operation.add_input_tensor(tensor_info, source_op_id)
        
        # Add output tensors
        for tensor_info in output_tensor_infos:
            operation.add_output_tensor(tensor_info)
        
        # Generate ID based on inputs
        operation.generate_operation_id(self.id_manager)
        
        # Add to collection
        self.add_operation(operation)
        return operation
    
    def get_operation(self, unique_id: str) -> Optional[Operation]:
        """Get an operation by its unique ID."""
        return self.operations.get(unique_id)
    
    def get_all_operations(self) -> List[Operation]:
        """Get all operations in the collection."""
        return list(self.operations.values())
    
    def get_operations_by_type(self, op_type: str) -> List[Operation]:
        """Get all operations of a specific type."""
        return [op for op in self.operations.values() if op.op_name == op_type]
    
    def __len__(self) -> int:
        """Get the number of operations in the collection."""
        return len(self.operations)


# Test function to demonstrate the system
def test_operation_id_system():
    print("Testing Operation ID System")
    print("--------------------------")
    
    # Create a collection to store operations
    collection = OperationCollection()
    
    # Create test tensors and TensorInfo objects
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    # Create TensorInfo objects manually for testing
    # In a real system, these would come from TensorOperationAnalyzer
    a_info = TensorInfo(a)
    b_info = TensorInfo(b)
    
    # First operation: c = a + b
    print("\n1. Addition operation: c = a + b")
    c = a + b  # This is just to get a valid tensor for the TensorInfo
    c_info = TensorInfo(c)
    
    # Create the operation (load operations for a and b will be created automatically)
    op_add = collection.create_operation("add", [a_info, b_info], [c_info])
    
    print(f"  Operation: {op_add}")
    print(f"  Input tensors: {len(op_add.input_tensor_infos)}")
    print(f"  Input operations: {op_add.input_op_ids}")
    print(f"  Complex ID: {op_add.complex_id}")
    print(f"  Unique ID: {op_add.unique_id}")
    
    # Check that load operations were automatically created
    print("\n  Automatically generated load operations:")
    for op in collection.get_operations_by_type("load"):
        print(f"    {op}")
    
    # Second operation: d = c * c
    print("\n2. Multiplication operation: d = c * c")
    d = c * c  # Just to get a tensor
    d_info = TensorInfo(d)
    
    op_mul = collection.create_operation("mul", [c_info, c_info], [d_info])
    
    print(f"  Operation: {op_mul}")
    print(f"  Input tensors: {len(op_mul.input_tensor_infos)}")
    print(f"  Input operations: {op_mul.input_op_ids}")
    print(f"  Complex ID: {op_mul.complex_id}")
    print(f"  Unique ID: {op_mul.unique_id}")
    
    # Third operation: e = torch.cat([c, d])
    print("\n3. Concatenation operation: e = torch.cat([c, d])")
    e = torch.cat([c.unsqueeze(0), d.unsqueeze(0)])  # Just to get a tensor
    e_info = TensorInfo(e)
    
    op_cat = collection.create_operation("cat", [c_info, d_info], [e_info])
    
    print(f"  Operation: {op_cat}")
    print(f"  Input tensors: {len(op_cat.input_tensor_infos)}")
    print(f"  Input operations: {op_cat.input_op_ids}")
    print(f"  Complex ID: {op_cat.complex_id}")
    print(f"  Unique ID: {op_cat.unique_id}")
    
    # Fourth operation: another addition with same inputs
    print("\n4. Another 'add' operation with same inputs: f = torch.add(a, b)")
    f = torch.add(a, b)  # Pure tensor operation
    f_info = TensorInfo(f)
    
    # This should use index 1 since it's the second operation with same signature
    op_add2 = collection.create_operation("add", [a_info, b_info], [f_info])
    
    print(f"  Operation: {op_add2}")
    print(f"  Input tensors: {len(op_add2.input_tensor_infos)}")
    print(f"  Input operations: {op_add2.input_op_ids}")
    print(f"  Complex ID: {op_add2.complex_id}")
    print(f"  Unique ID: {op_add2.unique_id}")
    
    # Fifth operation: different operation with same inputs
    print("\n5. Different operation type with same inputs: g = a - b")
    g = a - b  # Subtraction operation
    g_info = TensorInfo(g)
    
    # This should use index 0 since it's the first 'sub' operation with this signature
    op_sub = collection.create_operation("sub", [a_info, b_info], [g_info])
    
    print(f"  Operation: {op_sub}")
    print(f"  Input tensors: {len(op_sub.input_tensor_infos)}")
    print(f"  Input operations: {op_sub.input_op_ids}")
    print(f"  Complex ID: {op_sub.complex_id}")
    print(f"  Unique ID: {op_sub.unique_id}")
    
    # Show the full ID mapping
    print("\nFull ID Mapping:")
    for complex_id, unique_id in collection.id_manager.id_mapping.items():
        print(f"  {complex_id} -> {unique_id}")
    
    # Show all operations
    print("\nAll Operations:")
    for op in collection.get_all_operations():
        print(f"  {op}")
    
    return collection  # Return for further testing if needed


if __name__ == "__main__":
    # Run the test
    test_operation_id_system()
