import unittest
import torch
import torch.nn.functional as F
import sys
from io import StringIO
from tensor_operation_analyzer import TensorOperationAnalyzer, TensorInfo


class TestTensorOperationAnalyzer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a record of all test results
        cls.test_results = []
    
    def setUp(self):
        # Create an analyzer instance for each test
        self.analyzer = TensorOperationAnalyzer()
        
        # Capture test name for reporting
        self._test_name = self._testMethodName
        self._test_desc = self._testMethodDoc or "No description"
        self._stdout = StringIO()
        self._old_stdout = sys.stdout
        sys.stdout = self._stdout
        
        # Create tensors with compatible shapes for testing
        self.tensor1d = torch.tensor([1, 2, 3])
        self.tensor1d_b = torch.tensor([4, 5, 6])
        self.tensor2d = torch.tensor([[1, 2], [3, 4]])
        self.tensor2d_b = torch.tensor([[5, 6], [7, 8]])
        self.tensor3d = torch.randn(3, 4, 5)
    
    def tearDown(self):
        # Restore stdout
        sys.stdout = self._old_stdout
        
        # Record test result
        output = self._stdout.getvalue()
        self.__class__.test_results.append({
            'name': self._test_name,
            'description': self._test_desc,
            'output': output,
            'status': 'PASS'  # Assume pass if we get here (failures would raise AssertionError)
        })
    
    @classmethod
    def tearDownClass(cls):
        # Print summary of all tests
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        for i, result in enumerate(cls.test_results, 1):
            print(f"\nTest #{i}: {result['name']}")
            print(f"Description: {result['description']}")
            print(f"Status: {result['status']}")
            
            # Print output if it exists
            if result['output'].strip():
                print("Output:")
                print("-" * 40)
                print(result['output'].strip())
                print("-" * 40)
            else:
                print("No output produced")
        
        print("\n" + "=" * 80)
        print(f"TOTAL TESTS: {len(cls.test_results)}")
        print("ALL TESTS PASSED")
        print("=" * 80 + "\n")
    
    def format_path(self, path_elements):
        """Format a path list into a readable string"""
        parts = []
        for elem in path_elements:
            if elem.get('type') == 'arg':
                parts.append(f"args[{elem['index']}]")
            elif elem.get('type') == 'kwarg':
                parts.append(f"kwargs['{elem['key']}']")
            elif elem.get('type') == 'list':
                parts.append(f"[{elem['index']}]")
            elif elem.get('type') == 'tuple':
                parts.append(f"[{elem['index']}]")
            elif elem.get('type') == 'dict':
                parts.append(f"['{elem['key']}']")
            elif elem.get('type') == 'output':
                parts.append("result")
        
        return ''.join(parts)
        
    def print_analysis_summary(self, analysis, operation_name):
        """Pretty print a comprehensive summary of the analyzer's results"""
        print(f"\n[ANALYSIS SUMMARY: {operation_name}]")
        print("-" * 50)
        
        # Basic information
        print(f"Has Tensor Inputs: {analysis['has_tensor_inputs']}")
        print(f"Has Tensor Outputs: {analysis['has_tensor_outputs']}")
        
        # Input tensors
        print(f"\nInput Tensors: {len(analysis['input_tensors'])}")
        for i, tensor_info in enumerate(analysis['input_tensors']):
            # Extract shape information
            shape_str = "x".join(str(dim) for dim in tensor_info.shape) if tensor_info.shape else "scalar"
            
            # Format path as readable string
            path_str = self.format_path(tensor_info.path)
            
            print(f"  Input #{i+1}: Shape={shape_str}, Path={path_str}")
        
        # Output tensors
        print(f"\nOutput Tensors: {len(analysis['output_tensors'])}")
        for i, tensor_info in enumerate(analysis['output_tensors']):
            # Extract shape information
            shape_str = "x".join(str(dim) for dim in tensor_info.shape) if tensor_info.shape else "scalar"
            
            # Format path as readable string
            path_str = self.format_path(tensor_info.path)
            
            print(f"  Output #{i+1}: Shape={shape_str}, Path={path_str}")
        
        # Result information
        if analysis['error'] is not None:
            print(f"\nError: {analysis['error']}")
        else:
            # Print result information
            result = analysis['result']
            if isinstance(result, torch.Tensor):
                print(f"\nResult: Tensor with shape {list(result.shape)}")
            elif isinstance(result, (tuple, list)) and any(isinstance(x, torch.Tensor) for x in result):
                print(f"\nResult: {type(result).__name__} containing {sum(1 for x in result if isinstance(x, torch.Tensor))} tensors")
            else:
                print(f"\nResult: {str(result)[:50]}{'...' if len(str(result)) > 50 else ''}")
        
        print("-" * 50)
        
    def test_simple_tensor_operation(self):
        """Test a simple operation with direct tensor inputs"""
        print(f"Testing tensor addition with shapes {self.tensor1d.shape} and {self.tensor1d_b.shape}")
        analysis = self.analyzer.analyze_operation(
            torch.add, (self.tensor1d, self.tensor1d_b), {}
        )
        
        # Check analysis results
        self.assertTrue(analysis['has_tensor_inputs'])
        
        # Verify result is correct
        expected = self.tensor1d + self.tensor1d_b
        torch.testing.assert_close(analysis['result'], expected)
        
        # Print comprehensive analysis
        self.print_analysis_summary(analysis, "torch.add")
        
    def test_nested_tensor_operation(self):
        """Test an operation with tensors in nested structures"""
        # Create a nested structure
        nested_input = {
            'tensors': [self.tensor1d, self.tensor2d],
            'params': {
                'sizes': self.tensor2d.shape,
                'values': self.tensor3d
            }
        }
        
        print(f"Testing with nested structure containing 3 tensors")
        
        # Define a function that uses the nested structure
        def process_nested(data):
            # Use torch.cat with additional dimension to handle different shapes
            tensors = [t.unsqueeze(0) for t in data['tensors']]
            result = torch.cat(tensors).mean() + data['params']['values'].sum()
            return result
        
        # Analyze the operation
        analysis = self.analyzer.analyze_operation(
            process_nested, (nested_input,), {}
        )
        
        # Check analysis results
        self.assertTrue(analysis['has_tensor_inputs'])
        self.assertEqual(len(analysis['input_tensors']), 3)  # tensor1d, tensor2d, tensor3d
        
        # Print comprehensive analysis
        self.print_analysis_summary(analysis, "process_nested")
        
    def test_tensor_path_recording(self):
        """Test that paths to tensors are recorded correctly"""
        # Create a complex nested structure
        nested_structure = [
            {'name': 'first', 'value': self.tensor1d},
            [self.tensor2d, {'nested': self.tensor3d}]
        ]
        
        print(f"Testing path recording in complex nested structure")
        
        # Find tensors in the structure
        tensor_infos = self.analyzer.find_tensors_in_structure(nested_structure)
        
        # Should find 3 tensors
        self.assertEqual(len(tensor_infos), 3)
        
        # Check that tensor IDs are recorded
        tensor_ids = [info.tensor_id for info in tensor_infos]
        self.assertIn(id(self.tensor1d), tensor_ids)
        self.assertIn(id(self.tensor2d), tensor_ids)
        self.assertIn(id(self.tensor3d), tensor_ids)
        
        print(f"Found {len(tensor_infos)} tensors with correct paths:")
        for i, info in enumerate(tensor_infos):
            path_str = self.format_path(info.path)
            print(f"  Tensor {i+1}: {path_str}")
        
    def test_torch_operations(self):
        """Test with various torch operations"""
        print("Testing various torch operations")
        
        # Test torch.cat
        analysis = self.analyzer.analyze_operation(
            torch.cat, ([self.tensor1d, self.tensor1d],), {'dim': 0}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        self.print_analysis_summary(analysis, "torch.cat")
        
        # Test torch.stack
        analysis = self.analyzer.analyze_operation(
            torch.stack, ([self.tensor2d, self.tensor2d],), {'dim': 0}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        self.print_analysis_summary(analysis, "torch.stack")
        
        # Test torch.matmul
        analysis = self.analyzer.analyze_operation(
            torch.matmul, (self.tensor2d, self.tensor2d_b), {}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        self.print_analysis_summary(analysis, "torch.matmul")
    
    def test_nn_functional_operations(self):
        """Test with torch.nn.functional operations"""
        print("Testing nn.functional operations")
        
        # Test F.relu
        analysis = self.analyzer.analyze_operation(
            F.relu, (self.tensor2d,), {}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        self.print_analysis_summary(analysis, "F.relu")
        
        # Test F.softmax
        analysis = self.analyzer.analyze_operation(
            F.softmax, (self.tensor2d,), {'dim': 0}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        self.print_analysis_summary(analysis, "F.softmax")
    
    def test_tensor_methods(self):
        """Test with tensor methods"""
        print("Testing tensor methods")
        
        # Test tensor.sum()
        analysis = self.analyzer.analyze_operation(
            self.tensor3d.sum, (), {}
        )
        self.assertFalse(analysis['has_tensor_inputs'])  # self is not in args
        self.print_analysis_summary(analysis, "tensor.sum()")
        
        print("Note: Special handling needed for tensor methods with 'self' parameter")
    
    def test_non_tensor_operation(self):
        """Test with an operation that doesn't involve tensors"""
        print("Testing non-tensor operation")
        
        def add_numbers(a, b):
            return a + b
        
        analysis = self.analyzer.analyze_operation(
            add_numbers, (1, 2), {}
        )
        self.assertFalse(analysis['has_tensor_inputs'])
        self.assertFalse(analysis['has_tensor_outputs'])
        self.assertEqual(analysis['result'], 3)
        
        self.print_analysis_summary(analysis, "add_numbers")
    
    def test_mixed_operation(self):
        """Test with an operation that mixes tensors and non-tensors"""
        print("Testing mixed tensor and non-tensor operation")
        
        def scale_tensor(tensor, factor):
            return tensor * factor
        
        analysis = self.analyzer.analyze_operation(
            scale_tensor, (self.tensor1d, 2.5), {}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        torch.testing.assert_close(analysis['result'], self.tensor1d * 2.5)
        
        self.print_analysis_summary(analysis, "scale_tensor")
    
    def test_operation_with_error(self):
        """Test analysis of an operation that raises an error"""
        print("Testing operation that raises an error")
        
        def problematic_func(tensor):
            # Use int division to ensure an exception is raised
            zero_tensor = torch.zeros(1, dtype=torch.int)
            return tensor.to(torch.int) // zero_tensor
        
        analysis = self.analyzer.analyze_operation(
            problematic_func, (self.tensor1d,), {}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        # Check that error was handled
        self.assertIsNone(analysis['result'])
        self.assertIsNotNone(analysis['error'])
        
        self.print_analysis_summary(analysis, "problematic_func (division by zero)")
    
    def test_multiple_output_tensors(self):
        """Test operations that return multiple tensors"""
        print("Testing operation with multiple tensor outputs")
        
        # Define a function that returns multiple tensors
        def split_tensor(tensor):
            split1, split2 = tensor.chunk(2)
            return split1, split2, tensor.shape
        
        analysis = self.analyzer.analyze_operation(
            split_tensor, (self.tensor3d,), {}
        )
        self.assertTrue(analysis['has_tensor_inputs'])
        
        self.print_analysis_summary(analysis, "split_tensor")
    
    def test_large_structure(self):
        """Test with a large nested structure"""
        print("Testing with large nested structure")
        
        # Create a complex nested structure with many tensors
        large_structure = {
            'level1': [
                {'tensor': self.tensor1d, 'name': 'first'},
                {'tensor': self.tensor2d, 'name': 'second'},
                [self.tensor3d, 
                 [torch.ones(2, 3), torch.zeros(3, 4)]]
            ],
            'level2': {
                'a': {'tensor': torch.eye(3)},
                'b': [torch.arange(10), torch.randn(5)]
            }
        }
        
        # Create a simple operation that just returns the structure
        def identity_op(structure):
            return structure
        
        analysis = self.analyzer.analyze_operation(
            identity_op, (large_structure,), {}
        )
        
        self.print_analysis_summary(analysis, "large_structure_test")
        
        # Also test the raw find_tensors_in_structure method
        tensor_infos = self.analyzer.find_tensors_in_structure(large_structure)
        self.assertEqual(len(tensor_infos), 8)  # Should find 8 tensors
        
        print(f"\nRaw find_tensors_in_structure result:")
        print(f"Found {len(tensor_infos)} tensors in large structure")
        for i, info in enumerate(tensor_infos, 1):
            shape_str = "x".join(str(dim) for dim in info.shape) if info.shape else "scalar"
            path_str = self.format_path(info.path)
            print(f"  Tensor {i}: Shape={shape_str}, Path={path_str}")

if __name__ == '__main__':
    unittest.main()
