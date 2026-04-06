import numpy as np
import pandas as pd
import os
from Model_Local import ModelLocal

def run_csv_inference():
    """
    This script acts as the 'User'. It loads data from 'target.csv',
    sends it to Model_Local.py, and saves the results.
    """
    print("--- Starting Local CSV Batch Simulation ---")

    # 1. Setup Paths
    # Get the directory where this script lives
    current_folder = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(current_folder, "target.csv")
    output_csv = os.path.join(current_folder, "target_predictions_local.csv")
    
    # 2. Start the Model
    # We pass the current folder so the model can find /artifacts
    try:
        my_model = ModelLocal(model_path=current_folder)
    except Exception as e:
        print(f"Stop: Could not start model. Error: {e}")
        return

    # 3. Load the CSV Data
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}. Please ensure target.csv is in the folder.")
        return

    try:
        df_input = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df_input)} rows from target.csv")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 4. Prepare inputs for the model
    # We extract the 4 numeric columns as a NumPy array (NDArray)
    # And the Soil_Type column as a List of strings
    try:
        measurement_cols = [
            'sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)'
        ]
        
        raw_measurements = df_input[measurement_cols].to_numpy()
        raw_soil_types = df_input['Soil_Type'].tolist()
        
    except KeyError as e:
        print(f"Error: target.csv is missing required columns: {e}")
        return

    # 5. Get Prediction from our local model engine
    print(f"Processing {len(raw_soil_types)} rows through the model...")
    try:
        output = my_model.predict_local(
            measurement_input=raw_measurements,
            soil_type_input=raw_soil_types
        )
        
        # 6. Save results back to a new CSV
        # We add the predictions to our original dataframe
        df_input['predicted_species'] = output['results']
        df_input.to_csv(output_csv, index=False)
        
        print("\n--- BATCH TEST RESULTS ---")
        print(f"Success! Predictions saved to: {output_csv}")
        print(f"First 5 predictions: {output['results'][:5]}")
        print("--------------------------")
        ####################################################
        print("\n" + "-"*50)
        print("           INFERENCE RESULTS PREVIEW")
        print("-"*50)
        
        # This will display the first 10 rows as a formatted table in the terminal
        # Using .to_string() ensures the whole width is visible
        print(df_input.head(10).to_string(index=False))
        
        print("\n" + "-"*50)
        print(f"Success! Full results saved to: {output_csv}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"X Prediction failed: {e}")

if __name__ == "__main__":
    run_csv_inference()