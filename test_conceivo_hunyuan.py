test_conceivo_hunyuan.py
"""
Test script for verifying Conceivo AI + Hunyuan3D integration
"""

import os
import sys
sys.path.append('.')

from conceivin3d.ai.integrations.conceivo_3d_integration import Conceivo3DIntegration

def test_3d_generation():
    print("Initializing Conceivo 3D Integration...")
    conceivo_3d = Conceivo3DIntegration()

    # Test command for 3D generation
    test_cmd = "Create a modern ergonomic chair with wooden legs and fabric seat"
    print(f"\n===== TESTING 3D GENERATION =====")
    print(f"Processing command: '{test_cmd}'")

    try:
        result = conceivo_3d.process_3d_command(test_cmd)
        if result:
            print("\n3D generation request completed successfully!")
        else:
            print("\n3D generation request failed.")
    except Exception as e:
        print(f"\nError during 3D generation: {str(e)}")

if __name__ == "__main__":
    test_3d_generation()