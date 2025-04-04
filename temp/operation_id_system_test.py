import unittest
import torch
import sys
import os

# Add parent directory to path so we can import operation_id_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temp.operation_id_system import Operation, OperationCollection, OperationIDManager, TensorInfo


class TestOperationIDSystem(unittest.TestCase):
    """
    Comprehensive test suite for the Operation ID System.
    This test suite verifies all aspects of the system with assertions.
    """
    
    def setUp(self):
        """Set up test tensors and objects"""
        self.collection = OperationCollection()
        
        # Create test tensors
        self.a = torch.tensor([1, 2, 3])
        self.b = torch.tensor([4, 5, 6])
        self.c = torch.tensor([7, 8, 9])
        self.d = torch.tensor([10, 11, 12])
        
        # Create TensorInfo objects
        self.a_info = TensorInfo(self.a)
        self.b_info = TensorInfo(self.b)
        self.c_info = TensorInfo(self.c)
        self.d_info = TensorInfo(self.d)
    
    def test_load_operations(self):
        """Test automatic creation of load operations for tensors"""
        # Create an operation that uses tensors a and b
        output = self.a + self.b
        output_info = TensorInfo(output)
        
        # Create the operation
        op = self.collection.create_operation("add", [self.a_info, self.b_info], [output_info])
        
        # Check that load operations were created for a and b
        load_ops = self.collection.get_operations_by_type("load")
        self.assertEqual(len(load_ops), 2, "Should have created 2 load operations")
        
        # Check that load operations have the correct format
        for i, load_op in enumerate(load_ops):
            self.assertTrue(load_op.complex_id.startswith("load*"), 
                            f"Load operation should have complex ID starting with 'load*', got {load_op.complex_id}")
            self.assertTrue(load_op.unique_id.startswith("op*"), 
                            f"Load operation should have unique ID starting with 'op*', got {load_op.unique_id}")
    
    def test_operation_id_format(self):
        """Test the format of operation IDs"""
        # Create operations
        add_out = self.a + self.b
        add_info = TensorInfo(add_out)
        add_op = self.collection.create_operation("add", [self.a_info, self.b_info], [add_info])
        
        mul_out = self.c * self.d
        mul_info = TensorInfo(mul_out)
        mul_op = self.collection.create_operation("mul", [self.c_info, self.d_info], [mul_info])
        
        # Check complex ID format for add operation
        self.assertTrue(add_op.complex_id.startswith("add*0|"), 
                        f"Add operation should have complex ID starting with 'add*0|', got {add_op.complex_id}")
        
        # Check unique ID format
        self.assertTrue(add_op.unique_id.startswith("op*"), 
                        f"Operation should have unique ID starting with 'op*', got {add_op.unique_id}")
        
        # Extract input operation IDs from complex ID
        complex_parts = add_op.complex_id.split("|", 1)
        self.assertEqual(len(complex_parts), 2, f"Complex ID should have 2 parts: {add_op.complex_id}")
        op_part, inputs_part = complex_parts
        
        # Check that the inputs part contains all input operation IDs
        for input_op_id in add_op.input_op_ids:
            self.assertIn(input_op_id, inputs_part, 
                          f"Input op ID {input_op_id} should be in complex ID {add_op.complex_id}")
    
    def test_same_operation_different_inputs(self):
        """Test that same operation type with different inputs get different indices"""
        # Create add operations with different input combinations
        add_ab_out = self.a + self.b
        add_ab_info = TensorInfo(add_ab_out)
        add_ab_op = self.collection.create_operation("add", [self.a_info, self.b_info], [add_ab_info])
        
        add_cd_out = self.c + self.d
        add_cd_info = TensorInfo(add_cd_out)
        add_cd_op = self.collection.create_operation("add", [self.c_info, self.d_info], [add_cd_info])
        
        # Both should have index 0 since they're the first add operations with their respective inputs
        self.assertTrue(add_ab_op.complex_id.startswith("add*0|"), 
                        f"First add operation should have index 0, got {add_ab_op.complex_id}")
        self.assertTrue(add_cd_op.complex_id.startswith("add*0|"), 
                        f"First add operation with different inputs should also have index 0, got {add_cd_op.complex_id}")
        
        # But they should have different complex IDs
        self.assertNotEqual(add_ab_op.complex_id, add_cd_op.complex_id, 
                            "Add operations with different inputs should have different complex IDs")
    
    def test_same_operation_same_inputs(self):
        """Test that same operation type with same inputs get incrementing indices"""
        # Create three add operations with the same inputs
        add_ops = []
        for i in range(3):
            out = self.a + self.b  # Just to get a tensor
            out_info = TensorInfo(out)
            op = self.collection.create_operation("add", [self.a_info, self.b_info], [out_info])
            add_ops.append(op)
        
        # Check that the indices increment
        for i, op in enumerate(add_ops):
            self.assertTrue(op.complex_id.startswith(f"add*{i}|"), 
                            f"Add operation {i} should have index {i}, got {op.complex_id}")
    
    def test_different_operation_same_inputs(self):
        """Test that different operation types with same inputs get their own indices"""
        # Create operations of different types with the same inputs
        ops = []
        
        # Addition
        add_out = self.a + self.b
        add_info = TensorInfo(add_out)
        add_op = self.collection.create_operation("add", [self.a_info, self.b_info], [add_info])
        ops.append(add_op)
        
        # Subtraction
        sub_out = self.a - self.b
        sub_info = TensorInfo(sub_out)
        sub_op = self.collection.create_operation("sub", [self.a_info, self.b_info], [sub_info])
        ops.append(sub_op)
        
        # Multiplication
        mul_out = self.a * self.b
        mul_info = TensorInfo(mul_out)
        mul_op = self.collection.create_operation("mul", [self.a_info, self.b_info], [mul_info])
        ops.append(mul_op)
        
        # Each should have index 0 since they're the first of their type with these inputs
        for op in ops:
            self.assertTrue(op.complex_id.startswith(f"{op.op_name}*0|"), 
                            f"{op.op_name} operation should have index 0, got {op.complex_id}")
        
        # Create a second subtraction with the same inputs
        sub2_out = self.a - self.b
        sub2_info = TensorInfo(sub2_out)
        sub2_op = self.collection.create_operation("sub", [self.a_info, self.b_info], [sub2_info])
        
        # This should have index 1
        self.assertTrue(sub2_op.complex_id.startswith("sub*1|"), 
                        f"Second subtraction should have index 1, got {sub2_op.complex_id}")
    
    def test_operation_chaining(self):
        """Test operations that use outputs from other operations"""
        # Create a chain of operations: a + b -> c, c * c -> d, d - a -> e
        ab_out = self.a + self.b
        ab_info = TensorInfo(ab_out)
        ab_op = self.collection.create_operation("add", [self.a_info, self.b_info], [ab_info])
        
        c_squared = ab_out * ab_out
        c_squared_info = TensorInfo(c_squared)
        mul_op = self.collection.create_operation("mul", [ab_info, ab_info], [c_squared_info])
        
        final_out = c_squared - self.a
        final_info = TensorInfo(final_out)
        sub_op = self.collection.create_operation("sub", [c_squared_info, self.a_info], [final_info])
        
        # Check that the operations are chained correctly
        self.assertEqual(len(mul_op.input_op_ids), 1, 
                         f"Mul operation should have 1 input operation, got {len(mul_op.input_op_ids)}")
        self.assertEqual(mul_op.input_op_ids[0], ab_op.unique_id, 
                         f"Mul operation should use add operation as input, got {mul_op.input_op_ids}")
        
        self.assertEqual(len(sub_op.input_op_ids), 2, 
                         f"Sub operation should have 2 input operations, got {len(sub_op.input_op_ids)}")
        self.assertIn(mul_op.unique_id, sub_op.input_op_ids, 
                      f"Sub operation should use mul operation as input, got {sub_op.input_op_ids}")
    
    def test_complex_graph(self):
        """Test a more complex graph of operations"""
        # Create a series of operations that form a complex graph
        
        # First layer of operations
        ab_out = self.a + self.b
        ab_info = TensorInfo(ab_out)
        ab_op = self.collection.create_operation("add", [self.a_info, self.b_info], [ab_info])
        
        cd_out = self.c * self.d
        cd_info = TensorInfo(cd_out)
        cd_op = self.collection.create_operation("mul", [self.c_info, self.d_info], [cd_info])
        
        # Second layer that combines results from first layer
        combined_out = ab_out + cd_out
        combined_info = TensorInfo(combined_out)
        combined_op = self.collection.create_operation("add", [ab_info, cd_info], [combined_info])
        
        # Another branch that uses the original inputs
        alt_out = self.a * self.d
        alt_info = TensorInfo(alt_out)
        alt_op = self.collection.create_operation("mul", [self.a_info, self.d_info], [alt_info])
        
        # Final operations that combine everything
        final_out = combined_out * alt_out
        final_info = TensorInfo(final_out)
        final_op = self.collection.create_operation("mul", [combined_info, alt_info], [final_info])
        
        # Verify operation count
        self.assertEqual(len(self.collection), 9, 
                         f"Should have 9 operations (4 load + 5 compute), got {len(self.collection)}")
        
        # Verify correct input/output chaining
        self.assertEqual(len(combined_op.input_op_ids), 2, 
                         f"Combined operation should have 2 input operations, got {len(combined_op.input_op_ids)}")
        self.assertIn(ab_op.unique_id, combined_op.input_op_ids, 
                      f"Combined operation should use ab operation as input, got {combined_op.input_op_ids}")
        self.assertIn(cd_op.unique_id, combined_op.input_op_ids, 
                      f"Combined operation should use cd operation as input, got {combined_op.input_op_ids}")
        
        self.assertEqual(len(final_op.input_op_ids), 2, 
                         f"Final operation should have 2 input operations, got {len(final_op.input_op_ids)}")
        self.assertIn(combined_op.unique_id, final_op.input_op_ids, 
                      f"Final operation should use combined operation as input, got {final_op.input_op_ids}")
        self.assertIn(alt_op.unique_id, final_op.input_op_ids, 
                      f"Final operation should use alt operation as input, got {final_op.input_op_ids}")
    
    def test_unknown_tensor_handling(self):
        """Test handling of unknown tensors"""
        # Create operations with unknown tensors
        # They should automatically create load operations
        
        # Get initial operation count
        initial_count = len(self.collection)
        
        # Create operation with new tensors
        add_out = self.a + self.b
        add_info = TensorInfo(add_out)
        add_op = self.collection.create_operation("add", [self.a_info, self.b_info], [add_info])
        
        # Should have created 3 operations (2 load + 1 add)
        self.assertEqual(len(self.collection), initial_count + 3,
                         f"Should have created 3 new operations (2 load + 1 add), got {len(self.collection) - initial_count}")
        
        # Check that the tensors are now known
        self.assertTrue(self.collection.is_known_tensor(self.a_info), "Tensor 'a' should be known")
        self.assertTrue(self.collection.is_known_tensor(self.b_info), "Tensor 'b' should be known")
        self.assertTrue(self.collection.is_known_tensor(add_info), "Result tensor should be known")
        
        # Now create an operation that uses the output of the previous operation
        # This should not create any new load operations
        mul_out = add_out * self.c
        mul_info = TensorInfo(mul_out)
        
        prev_count = len(self.collection)
        mul_op = self.collection.create_operation("mul", [add_info, self.c_info], [mul_info])
        
        # Should have created 2 operations (1 load for c + 1 mul)
        self.assertEqual(len(self.collection), prev_count + 2,
                         f"Should have created 2 new operations (1 load + 1 mul), got {len(self.collection) - prev_count}")
    
    def test_tensor_reuse(self):
        """Test reusing the same tensor in multiple operations"""
        # Create multiple operations that reuse the same tensors
        
        # First operation
        add_out = self.a + self.b
        add_info = TensorInfo(add_out)
        add_op = self.collection.create_operation("add", [self.a_info, self.b_info], [add_info])
        
        # Second operation reusing a
        mul_out = self.a * self.c
        mul_info = TensorInfo(mul_out)
        mul_op = self.collection.create_operation("mul", [self.a_info, self.c_info], [mul_info])
        
        # Third operation reusing a again
        sub_out = self.a - self.d
        sub_info = TensorInfo(sub_out)
        sub_op = self.collection.create_operation("sub", [self.a_info, self.d_info], [sub_info])
        
        # Check that tensor a is associated with the same load operation in all three operations
        a_load_id = None
        for op in [add_op, mul_op, sub_op]:
            # Find the input op ID corresponding to tensor a
            for i, tensor_info in enumerate(op.input_tensor_infos):
                if tensor_info.tensor_id == self.a_info.tensor_id:
                    if a_load_id is None:
                        a_load_id = op.input_op_ids[i]
                    else:
                        self.assertEqual(op.input_op_ids[i], a_load_id,
                                      f"Tensor 'a' should be associated with the same load operation in all operations")
    
    def test_id_manager_consistency(self):
        """Test consistency of the ID manager"""
        # Create operations and check ID mapping
        add_out = self.a + self.b
        add_info = TensorInfo(add_out)
        add_op = self.collection.create_operation("add", [self.a_info, self.b_info], [add_info])
        
        # Create a second identical operation
        add_out2 = self.a + self.b
        add_info2 = TensorInfo(add_out2)
        add_op2 = self.collection.create_operation("add", [self.a_info, self.b_info], [add_info2])
        
        # Check ID mapping
        for complex_id, unique_id in self.collection.id_manager.id_mapping.items():
            # Find the operation with this unique ID
            op = self.collection.get_operation(unique_id)
            if op:
                self.assertEqual(op.complex_id, complex_id,
                               f"Operation's complex ID {op.complex_id} should match complex ID in mapping {complex_id}")
                self.assertEqual(op.unique_id, unique_id,
                               f"Operation's unique ID {op.unique_id} should match unique ID in mapping {unique_id}")


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases in the Operation ID System."""
    
    def setUp(self):
        """Set up test objects"""
        self.collection = OperationCollection()
        
        # Create test tensors
        self.a = torch.tensor([1, 2, 3])
        self.b = torch.tensor([4, 5, 6])
        
        # Create TensorInfo objects
        self.a_info = TensorInfo(self.a)
        self.b_info = TensorInfo(self.b)
    
    def test_same_tensor_twice(self):
        """Test using the same tensor twice in one operation"""
        # Create operation with the same tensor as both inputs: a * a
        mul_out = self.a * self.a
        mul_info = TensorInfo(mul_out)
        mul_op = self.collection.create_operation("mul", [self.a_info, self.a_info], [mul_info])
        
        # Check that the operation has one unique input operation ID
        self.assertEqual(len(mul_op.input_op_ids), 1,
                       f"Operation should have 1 unique input operation ID, got {len(mul_op.input_op_ids)}")
        
        # Check complex ID format - should reference the same operation ID twice
        complex_parts = mul_op.complex_id.split("|", 1)
        self.assertEqual(len(complex_parts), 2, f"Complex ID should have 2 parts: {mul_op.complex_id}")
        input_parts = complex_parts[1].split("|")
        
        # Should have one unique input operation ID repeated twice
        self.assertEqual(len(set(input_parts)), 1,
                       f"Complex ID should reference one unique input operation, got {input_parts}")
    
    def test_empty_inputs(self):
        """Test operations with no inputs"""
        # Create a made-up operation with no inputs
        out = torch.randn(3)
        out_info = TensorInfo(out)
        
        # This is not a real scenario but tests the code
        op = Operation("const")
        op.add_output_tensor(out_info)
        op.generate_operation_id(self.collection.id_manager)
        self.collection.add_operation(op)
        
        # Check that the operation has a valid complex ID without input parts
        self.assertEqual(op.complex_id, "const*0",
                       f"Operation with no inputs should have complex ID 'const*0', got {op.complex_id}")
    
    def test_many_operations_same_signature(self):
        """Test creating many operations with the same signature"""
        # Create 100 add operations with the same inputs
        ops = []
        for i in range(100):
            out = self.a + self.b
            out_info = TensorInfo(out)
            op = self.collection.create_operation("add", [self.a_info, self.b_info], [out_info])
            ops.append(op)
        
        # Check that the indices increment correctly
        for i, op in enumerate(ops):
            self.assertTrue(op.complex_id.startswith(f"add*{i}|"),
                          f"Operation {i} should have index {i}, got {op.complex_id}")
    
    def test_many_operation_types(self):
        """Test creating operations of many different types"""
        # Create 20 operations of different types with the same inputs
        op_types = [f"op_type_{i}" for i in range(20)]
        ops = []
        
        for op_type in op_types:
            out = self.a + self.b  # Just to get a tensor
            out_info = TensorInfo(out)
            op = self.collection.create_operation(op_type, [self.a_info, self.b_info], [out_info])
            ops.append(op)
        
        # Check that each operation has the correct type and index 0
        for i, op in enumerate(ops):
            self.assertEqual(op.op_name, op_types[i],
                          f"Operation should have type {op_types[i]}, got {op.op_name}")
            self.assertTrue(op.complex_id.startswith(f"{op_types[i]}*0|"),
                          f"Operation should have index 0, got {op.complex_id}")


if __name__ == "__main__":
    unittest.main()
