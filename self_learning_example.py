"""
Self-Learning 3D Example
This script demonstrates the self-learning capabilities of the Conceivo 3D integration.
"""

import os
import sys
import time
from datetime import datetime
# Add the current directory to Python path to help with module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the Conceivo 3D integration module
from conceivin3d.ai.integrations.gemma_3d_integration import Conceivo3DIntegration

# Create directories for training data and model storage
os.makedirs("generated_models/training_data", exist_ok=True)
os.makedirs("generated_models/saved_models", exist_ok=True)

# Initialize the Conceivo 3D integration
conceivo_3d = Conceivo3DIntegration()

# First example command - basic chair design
cmd1 = "Create an office chair with ergonomic design, using carbon fiber material. The chair should have adjustable height features."

print("\n===== FIRST DESIGN ITERATION =====")
print("Processing command:", cmd1)

# Process the command and generate 3D modeling code
try:
    generated_code1 = conceivo_3d.process_3d_command(cmd1)
    
    # Save the generated code
    example_path1 = os.path.join("generated_models", "chair_design_v1.py")
    with open(example_path1, "w") as f:
        f.write(generated_code1)
    
    print(f"\n3D modeling code generated successfully! Code saved to {example_path1}")
    
    # Simulate user feedback
    feedback1 = "The chair looks good, but I would like to see more pronounced lumbar support and add a headrest."
    print(f"\nProcessing user feedback: {feedback1}")
    loss1 = conceivo_3d.process_feedback(cmd1, feedback1)
    print(f"Feedback processing complete. Training loss: {loss1:.4f}")
    
    # Save the trained model
    conceivo_3d.save_model("conceivo_3d_feedback_model_v1.pt")
    
    # Second example command - improved chair design based on feedback (moved inside try block)
    cmd2 = "Create an advanced office chair with superior ergonomic design, carbon fiber construction, and featuring both lumbar support and headrest."

    print("\n===== SECOND DESIGN ITERATION =====")
    print("Processing improved command:", cmd2)

    # Process the updated command
    generated_code2 = conceivo_3d.process_3d_command(cmd2)

    # Save the improved code
    example_path2 = os.path.join("generated_models", "chair_design_v2.py")
    with open(example_path2, "w") as f:
        f.write(generated_code2)

    print(f"\nImproved model generated and saved to {example_path2}")

    # Show the improvement
    print("\n=== COMPARISON ===")
    print("First design focused on basic features:")
    print("=" * 50)
    print(generated_code1[:500] + "..." if len(generated_code1) > 500 else generated_code1)
    print("\nSecond design incorporates learned feedback:")
    print("=" * 50)
    print(generated_code2[:500] + "..." if len(generated_code2) > 500 else generated_code2)

    # Retrain periodically from saved data
    print("\nRetraining model from saved feedback data...")
    time.sleep(2)  # Simulate passage of time
    conceivo_3d.retrain_from_saved_data()

    # Save the retrained model
    conceivo_3d.save_model("conceivo_3d_feedback_model_v2.pt")

    # Analyze a model to demonstrate understanding
    analysis_prompt = "A futuristic chair with ergonomic design using carbon fiber material, featuring adjustable height and lumbar support."
    print("\nAnalyzing the 3D model structure...")
    analysis = conceivo_3d.analyze_3d_model(analysis_prompt)
    print("\nModel Analysis:")
    print(analysis)

    print("\nSelf-learning demonstration completed successfully!")

except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
