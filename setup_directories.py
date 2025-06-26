setup_directories.py
"""
Script to set up directory structure for 3D generation testing
"""

import os
import sys

def setup_directories():
    # Get the current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to create directories using absolute paths
    try:
        os.makedirs("sample_data", exist_ok=True)
        os.makedirs("sample_data\\images", exist_ok=True)
        os.makedirs("sample_data\\meshes", exist_ok=True)
        print("Successfully created directory structure:")
        print("- sample_data/ (root directory)")
        print("  - images/ (for input images)")
        print("  - meshes/ (for input meshes)")
    except Exception as e:
        print(f"Error creating directories: {str(e)}")

if __name__ == "__main__":
    setup_directories()