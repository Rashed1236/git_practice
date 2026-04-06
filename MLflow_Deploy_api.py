import mlflow.pyfunc
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import List, Dict, Any

# --- 1. THE WRAPPER CLASS ---
class MLflowWrapper(mlflow.pyfunc.PythonModel):
    """
    This class acts as a bridge between MLflow and your ModelLocal logic.
    It encapsulates the 'Massive Task' of data reconstruction and type casting.
    """
    def load_context(self, context):
        """
        This runs when the MLflow model is loaded into memory.
        It pulls the joblib files from the MLflow artifact store.
        """
        # Load the artifacts using the internal paths MLflow managed during logging
        self.model = joblib.load(context.artifacts["model_file"])
        self.encoder = joblib.load(context.artifacts["encoder_file"])
        self.measurement_cols = [
            'sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)'
        ]

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        """
        The core prediction logic used by MLflow.
        Ensures the 'Soil_Type' is cast correctly for HistGradientBoosting.
        """
        df = model_input.copy()
        
        # Ensure correct categorical type for HGB
        if 'Soil_Type' in df.columns:
            df['Soil_Type'] = df['Soil_Type'].astype('category')
        
        # 1. Generate numeric predictions
        raw_preds = self.model.predict(df)
        
        # 2. Translate numeric codes back to species names
        species_names = self.encoder.inverse_transform(raw_preds)
        
        return pd.Series(species_names)

# --- 2. THE LOGGING LOGIC ---
def log_to_mlflow():
    """
    Packages the model artifacts and the wrapper into an MLflow Run.
    Includes a Model Signature to define the expected input/output schema.
    """
    # 1. Define paths to your local Stage 1 artifacts
    artifacts = {
        "model_file": "artifacts/best_hgb_classifier_weighted.joblib",
        "encoder_file": "artifacts/species_encoder_hgb_weighted.joblib"
    }

    # 2. Pre-flight check: Ensure artifacts exist locally
    for name, path in artifacts.items():
        if not os.path.exists(path):
            print(f"Error: Could not find '{path}'.")
            print("Please ensure your 'artifacts' folder contains the .joblib files.")
            return

    # 3. Define a Sample Input for the Signature
    # This removes the "Signature cannot be inferred" warning
    input_example = pd.DataFrame({
        'sepal length (cm)': [5.1],
        'sepal width (cm)': [3.5],
        'petal length (cm)': [1.4],
        'petal width (cm)': [0.2],
        'Soil_Type': ['Type_A']
    })
    # Predict on the example to get the output format
    # (Manual step here just to define the signature)
    output_example = pd.Series(["setosa"])
    signature = infer_signature(input_example, output_example)

    # 4. Initialize MLflow Experiment
    mlflow.set_experiment("Iris_HGB_Local_Registry")
    
    print(f"--- Registering Model to MLflow (Python {sys.version.split()[0]}) ---")
    
    with mlflow.start_run(run_name="Stage_1_Registration_With_Signature") as run:
        # Log the model using the Custom PythonModel wrapper
        mlflow.pyfunc.log_model(
            artifact_path="iris_model",
            python_model=MLflowWrapper(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "mlflow", 
                "scikit-learn==1.6.1", 
                "pandas", 
                "joblib"
            ]
        )
        
        run_id = run.info.run_id
        print(f"\n" + "="*40)
        print(f"✓ Model successfully logged to Registry.")
        print(f"✓ Run ID: {run_id}")
        print(f"✓ Experiment: Iris_HGB_Local_Registry")
        print("="*40)
        print("\nYou can now view this model in the MLflow UI at http://127.0.0.1:5000")

if __name__ == "__main__":
    log_to_mlflow()