test_dir_create.py
"""
Simple script to test directory creation
"""

import os

def test_dir_create():
    try:
        os.makedirs("test_dir", exist_ok=True)
        print("Successfully created test_dir")
    except Exception as e:
        print(f"Error creating directory: {str(e)}")

if __name__ == "__main__":
    test_dir_create()