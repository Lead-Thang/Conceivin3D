"""
Test script for validating Conceivo 3D Integration
This script tests the ability to import and use the Conceivo 3D Integration
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to Python path to help with module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the Conceivo 3D integration module
from conceivin3d.ai.integrations.conceivo_3d_integration import Conceivo3DIntegration


class TestConceivo3DIntegration(unittest.TestCase):
    """Test suite for Conceivo3DIntegration class"""

    def setUp(self):
        """Setup test environment before each test method"""
        self.conceivo_3d = Conceivo3DIntegration()
        self.test_dir = tempfile.mkdtemp()
        
    def test_initialization(self):
        """Test proper initialization of Conceivo3DIntegration"""
        self.assertIsNotNone(self.conceivo_3d)
        self.assertTrue(hasattr(self.conceivo_3d, 'device'))
        self.assertTrue(hasattr(self.conceivo_3d, 'tokenizer'))
        self.assertTrue(hasattr(self.conceivo_3d, 'model'))
    
    def test_process_3d_command(self):
        """Test 3D command processing"""
        cmd = "Create a simple cube at position (0, 0, 0)"
        result = self.conceivo_3d.process_3d_command(cmd)
        
        # Basic validation that we got some code
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Save generated code to temporary file for inspection
        temp_code_path = os.path.join(self.test_dir, 'test_cube.py')
        with open(temp_code_path, 'w') as f:
            f.write(result)
            
        # Check if file was created
        self.assertTrue(os.path.exists(temp_code_path))
        
    def test_execute_3d_code(self):
        """Test execution of generated 3D code"""
        # Create a simple test script
        test_script = '''
from pythreejs import Scene, SphereGeometry, MeshStandardMaterial, Mesh, DirectionalLight
from IPython.display import display

# Create a scene
scene = Scene()

# Create a sphere geometry with radius 1
geometry = SphereGeometry(radius=1)

# Create a material
material = MeshStandardMaterial(color='red')

# Create a mesh and add it to the scene
sphere = Mesh(geometry, material)
scene.add(sphere)

# Add lighting
light = DirectionalLight(color='white', intensity=1)
scene.add(light)

# Display the scene (this would work in Jupyter notebook)
# display(scene)
'''
        
        # Save test script to temporary file
        test_script_path = os.path.join(self.test_dir, 'test_scene.py')
        with open(test_script_path, 'w') as f:
            f.write(test_script)
            
        # Execute the test script
        result = self.conceivo_3d.execute_3d_code(test_script_path)
        self.assertTrue(result['success'])
        
    def test_analyze_3d_model(self):
        """Test model analysis functionality"""
        analysis_prompt = "A futuristic chair with ergonomic design using carbon fiber material, featuring adjustable height and lumbar support."
        analysis = self.conceivo_3d.analyze_3d_model(analysis_prompt)
        
        # Validate output
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 0)
        
    def test_save_load_model(self):
        """Test model saving and loading functionality"""
        # Test save
        save_path = os.path.join(self.test_dir, 'test_model.pt')
        success = self.conceivo_3d.save_model(save_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))
        
        # Test load
        loaded = self.conceivo_3d.load_model(save_path)
        self.assertTrue(loaded)
        

if __name__ == '__main__':
    unittest.main()