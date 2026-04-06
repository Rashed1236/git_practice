import numpy as np
import os
from Model_Local import ModelLocal

def run_test():
    """
    This script acts as the 'User'. It sends data to Model_Local.py 
    to see if the logic works.
    """
    print("--- Starting Local Simulation ---")

    # 1. Where are we?
    # Get the directory where this script lives
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Start the Model
    # We pass the current folder so the model can find /artifacts
    try:
        my_model = ModelLocal(model_path=current_folder)
    except Exception as e:
        print(f"Stop: Could not start model. Did you put the .joblib files in the 'artifacts' folder? \nError: {e}")
        return

    # 3. Create Sample Data
    # 2 rows of Iris measurements
    sample_measurements = np.array([
        [5.1, 3.5, 1.4, 0.2], 
        [6.3, 3.3, 6.0, 2.5]
    ])
    sample_soil = ["Type_A", "Type_B"]

    # 4. Get Prediction
    print("Sending data to model...")
    output = my_model.predict_local(
        measurement_input=sample_measurements,
        soil_type_input=sample_soil
    )

    # 5. Show results
    print("\n--- TEST RESULTS ---")
    print(f"Predictions: {output['results']}")
    print(f"Status:      {output['status']}")
    print("--------------------")

if __name__ == "__main__":
    run_test()