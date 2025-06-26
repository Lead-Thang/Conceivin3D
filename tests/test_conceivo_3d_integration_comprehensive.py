"""
Comprehensive tests for Conceivo3DIntegration
This test suite includes edge cases and complex workflows.
"""

import unittest
import tempfile
import shutil
import os
import torch
from pathlib import Path
from conceivin3d.ai.integrations.conceivo_3d_integration import Conceivo3DIntegration

class TestConceivo3DIntegrationComprehensive(unittest.TestCase):
    """Comprehensive tests for Conceivo3DIntegration including edge cases and complex workflows."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.conceivo_3d = Conceivo3DIntegration()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Handle read-only directories in Windows
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            # If directory is read-only, first change permissions
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    os.chmod(os.path.join(root, name), 0o777)
                for name in dirs:
                    os.chmod(os.path.join(root, name), 0o777)
            os.chmod(self.test_dir, 0o777)
            shutil.rmtree(self.test_dir)
        
    def test_edge_case_empty_command(self):
        """Test handling of empty commands."""
        result = self.conceivo_3d.process_3d_command("")
        self.assertIsNone(result, "Should return None for empty command")
        
    def test_edge_case_invalid_position(self):
        """Test handling of invalid positions in commands."""
        # Test invalid position format
        result = self.conceivo_3d.process_3d_command("Create a cube at position (invalid, 0, 0)")
        # The command should return None for invalid positions
        self.assertIsNone(
            result, 
            "Process command should return None when position contains invalid numeric values"
        )
        
        # Test negative coordinates - these are valid!
        result = self.conceivo_3d.process_3d_command("Create a cube at position (-1, -2, -3)")
        self.assertIsNotNone(
            result, 
            "Process command should accept negative coordinates as valid input"
        )
        
        # Extract location from result using string containment first for better error messages
        self.assertIn(
            "bpy.context.object.location", result,
            "Generated script should contain location assignment for negative coordinates"
        )
        # Then verify the actual values are correctly formatted
        self.assertRegex(
            result, 
            r"location\s*=\s*$\s*-1\.0\s*,\s*-2\.0\s*,\s*-3\.0\s*$",
            "Negative coordinates should be formatted with correct decimal precision"
        )
        
        # Test floating point coordinates
        result = self.conceivo_3d.process_3d_command("Create a cube at position (1.5, 2.7, 3.2)")
        self.assertIsNotNone(
            result, 
            "Process command should accept floating point coordinates as valid input"
        )
        
        self.assertIn(
            "bpy.context.object.location", result,
            "Generated script should contain location assignment for floating point coordinates"
        )
        # Verify the actual values are correctly formatted with proper spacing around commas
        self.assertRegex(
            result, 
            r"location\s*=\s*$\s*1\.5\s*,\s*2\.7\s*,\s*3\.2\s*$",
            "Floating point coordinates should maintain their precision in generated script"
        )
        
    def test_error_handling_model_save_failure(self):
        """Test error handling when model saving fails due to permission issues."""
        # Create a read-only directory to trigger permission error
        readonly_dir = os.path.join(self.test_dir, 'readonly')
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only permissions
        
        self.conceivo_3d.trained_models_dir = readonly_dir
        success = self.conceivo_3d.save_model("test_model.pt")
        self.assertFalse(success, "Should fail gracefully when saving to read-only directory")
        
        # Clean up read-only directory
        os.chmod(readonly_dir, 0o777)  # Make it writable again for cleanup
        shutil.rmtree(readonly_dir, ignore_errors=True)  # Remove even if not empty
        
    def test_error_handling_model_load_missing(self):
        """Test error handling when trying to load a non-existent model."""
        missing_path = os.path.join(self.test_dir, 'missing_model.pt')
        success = self.conceivo_3d.load_model(missing_path)
        self.assertFalse(success, "Should handle missing model files gracefully")
        
    def test_complex_workflow_multiple_operations(self):
        """Test a complex workflow with multiple 3D operations."""
        # Process multiple commands in sequence
        commands = [
            "Create a base cube at position (0, 0, 0)",
            "Add a sphere on top of the cube",
            "Rotate the sphere 45 degrees around X-axis",
            "Scale the combined object by 2x"
        ]
        
        results = []
        for cmd in commands:
            result = self.conceivo_3d.process_3d_command(cmd)
            results.append(result)
        
        # Verify all operations were processed
        self.assertEqual(len(results), len(commands), "All commands should be processed")
        # Verify none of the results are None (assuming successful processing)
        self.assertTrue(all(result is not None for result in results), "All operations should succeed")
        
    def test_model_switching(self):
        """Test switching between different models."""
        # Save current model
        original_model = self.conceivo_3d.model
        
        # Create and use a new model
        self.conceivo_3d.model = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
        
        # Test saving and loading with new model
        temp_path = os.path.join(self.test_dir, "temp_model.pt")
        success_save = self.conceivo_3d.save_model(temp_path)
        self.assertTrue(success_save, "Should save new model type successfully")
        
        success_load = self.conceivo_3d.load_model(temp_path)
        self.assertTrue(success_load, "Should load new model type successfully")
        
        # Restore original model
        self.conceivo_3d.model = original_model

if __name__ == '__main__':
    unittest.main()