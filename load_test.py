import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
# few comments added here to test the commit and push functionality of git
### added another comment to test the commit and push functionality of git

def test_registry_load():
    """
    STAGE 1: VERIFY REGISTRY LOAD
    This script pulls the model from MLflow using the Run ID and runs a test.
    """
    # 1. SET YOUR RUN ID 
    # Replace this with the Run ID from your terminal or MLflow UI
    # (From your previous message: 54e99d8dfafe4291ade1a50044483c5b)
    run_id = "54e99d8dfafe4291ade1a50044483c5b" 
    model_uri = f"runs/{run_id}/iris_model"

    print(f"--- Attempting to load model from MLflow Registry ---")
    print(f"Model URI: {model_uri}")

    try:
        # 2. LOAD THE MODEL
        # This downloads the artifacts and reconstructs the MLflowWrapper class
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("✓ Model loaded successfully from Registry.")

        # 3. PREPARE TEST DATA
        # We use a DataFrame that matches the Signature we defined
        test_data = pd.DataFrame({
            'sepal length (cm)': [5.1, 6.3],
            'sepal width (cm)': [3.5, 3.3],
            'petal length (cm)': [1.4, 6.0],
            'petal width (cm)': [0.2, 2.5],
            'Soil_Type': ['Type_A', 'Type_B']
        })

        # 4. RUN PREDICTION
        print("Running prediction through loaded model...")
        predictions = loaded_model.predict(test_data)

        # 5. DISPLAY RESULTS
        print("\n" + "="*30)
        print("   REGISTRY TEST RESULTS")
        print("="*30)
        test_data['predicted_species'] = predictions
        print(test_data[['Soil_Type', 'predicted_species']])
        print("="*30)

    except Exception as e:
        print(f"X Failed to load or run model: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Ensure 'mlflow ui' is still running in another terminal.")
        print("2. Double-check that the Run ID matches your MLflow UI exactly.")
        

if __name__ == "__main__":
    test_registry_load()