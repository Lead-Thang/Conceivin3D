"""
Test script for validating Conceivo3DIntegration import and basic functionality
"""

import os
import sys
# Add project root to Python path to help with module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conceivin3d.ai.integrations.conceivo_3d_integration import Conceivo3DIntegration

print("Successfully imported Conceivo3DIntegration")

# Initialize the integration
test_conceivo = Conceivo3DIntegration()
print("Successfully created Conceivo3DIntegration instance")

# Test a method
response = test_conceivo.process_3d_command("Create a simple sphere")
print(f"\nTest response from process_3d_command():")
print(response)