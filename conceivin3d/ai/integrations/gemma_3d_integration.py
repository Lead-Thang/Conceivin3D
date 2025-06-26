"""
Full Conceivo 3D Integration Example
This script demonstrates the comprehensive capabilities of the Conceivo 3D integration,
including initial design, feedback processing, model saving/loading, retraining,
and model analysis.
"""

import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path to help with module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Conceivo 3D integration module
from conceivin3d.ai.integrations.gemma_3d_integration import Conceivo3DIntegration

# --- Setup Directories ---
print("Setting up necessary directories...")
os.makedirs("generated_models/training_data", exist_ok=True)
os.makedirs("generated_models/saved_models", exist_ok=True)
os.makedirs("generated_models/generated_scripts", exist_ok=True) # New directory for generated Python scripts
print("Directories created: generated_models/, generated_models/training_data/, generated_models/saved_models/, generated_models/generated_scripts/")

# --- Initialize Conceivo 3D Integration ---
print("\nInitializing Conceivo 3D Integration (this may download model weights)...")
# Note: The model "tencent/conceivo-3d-large" is a placeholder.
# You might need to replace it with an actual accessible model or configure your environment.
conceivo_3d = Conceivo3DIntegration()
print("Conceivo 3D Integration initialized.")

# --- First Design Iteration ---
cmd1 = "Create an office chair with ergonomic design, using carbon fiber material. The chair should have adjustable height features."
print(f"\n===== FIRST DESIGN ITERATION =====")
print(f"Processing command: \"{cmd1}\"")

try:
    generated_code1 = conceivo_3d.process_3d_command(cmd1)
    
    # Save the generated code to a specific directory
    script_path1 = os.path.join("generated_models", "generated_scripts", "chair_design_v1.py")
    with open(script_path1, "w") as f:
        f.write(generated_code1)
    
    print(f"\n3D modeling code generated successfully! Code saved to {script_path1}")
    print("\nGenerated code snippet (first 500 chars):")
    print("=" * 50)
    print(generated_code1[:500] + "..." if len(generated_code1) > 500 else generated_code1)
    print("=" * 50)

    # Simulate execution of the generated code
    print(f"\nSimulating execution of {script_path1}...")
    execution_result1 = conceivo_3d.execute_3d_code(script_path1)
    if execution_result1["success"]:
        print("\nExecution successful!")
        print("Script Output:\n", execution_result1["output"])
    else:
        print("\nExecution failed!")
        print("Script Error:\n", execution_result1["error"])

    # --- Simulate User Feedback for First Iteration ---
    feedback1 = "The chair looks good, but I would like to see more pronounced lumbar support and add a headrest."
    print(f"\nProcessing user feedback: \"{feedback1}\"")
    loss1 = conceivo_3d.process_feedback(cmd1, feedback1)
    print(f"Feedback processing complete. Training loss: {loss1:.4f}")
    
    # --- Save the Trained Model after First Feedback ---
    model_filename_v1 = "conceivo_3d_feedback_model_v1.pt"
    conceivo_3d.save_model(model_filename_v1)
    print(f"Model saved as {model_filename_v1}.")

    # --- Second Design Iteration (incorporating feedback) ---
    cmd2 = "Create an advanced office chair with superior ergonomic design, carbon fiber construction, and featuring both lumbar support and headrest."
    print(f"\n===== SECOND DESIGN ITERATION =====")
    print(f"Processing improved command: \"{cmd2}\"")

    generated_code2 = conceivo_3d.process_3d_command(cmd2)

    script_path2 = os.path.join("generated_models", "generated_scripts", "chair_design_v2.py")
    with open(script_path2, "w") as f:
        f.write(generated_code2)
    
    print(f"\nImproved 3D modeling code generated successfully! Code saved to {script_path2}")
    print("\nGenerated code snippet (first 500 chars):")
    print("=" * 50)
    print(generated_code2[:500] + "..." if len(generated_code2) > 500 else generated_code2)
    print("=" * 50)

    # Simulate execution of the improved code
    print(f"\nSimulating execution of {script_path2}...")
    execution_result2 = conceivo_3d.execute_3d_code(script_path2)
    if execution_result2["success"]:
        print("\nExecution successful!")
        print("Script Output:\n", execution_result2["output"])
    else:
        print("\nExecution failed!")
        print("Script Error:\n", execution_result2["error"])

    # --- Retrain Periodically from Saved Data ---
    print("\nRetraining model from all saved feedback data (simulating periodic retraining)...")
    time.sleep(2) # Simulate passage of time
    avg_loss_retrain = conceivo_3d.retrain_from_saved_data()
    if avg_loss_retrain is not None:
        print(f"Retraining complete. Average loss across all data: {avg_loss_retrain:.4f}")
    else:
        print("No new data to retrain on.")

    # --- Save the Retrained Model ---
    model_filename_v2 = "conceivo_3d_feedback_model_v2.pt"
    conceivo_3d.save_model(model_filename_v2)
    print(f"Retrained model saved as {model_filename_v2}.")

    # --- Load a Specific Model Version (demonstration) ---
    print(f"\nLoading model version {model_filename_v1} for demonstration...")
    conceivo_3d.load_model(model_filename_v1)
    print(f"Model {model_filename_v1} loaded successfully.")

    # --- Analyze a Model to Demonstrate Understanding ---
    analysis_prompt = "A futuristic chair with ergonomic design using carbon fiber material, featuring adjustable height and lumbar support."
    print(f"\nAnalyzing the 3D model structure based on prompt: \"{analysis_prompt}\"")
    analysis = conceivo_3d.analyze_3d_model(analysis_prompt)
    print("\nModel Analysis Result:")
    print("=" * 50)
    print(analysis)
    print("=" * 50)

    print("\nComprehensive self-learning demonstration completed successfully!")

except Exception as e:
    print(f"\nAn unexpected error occurred: {str(e)}")
    import traceback
    traceback.print_exc() # Print full traceback for debugging
