import os
import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc

# --- 1. DEFINE THE WRAPPER CLASS ---
class IrisModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # This runs once when the model is loaded
        self.model = joblib.load(context.artifacts["classifier"])
        self.encoder = joblib.load(context.artifacts["encoder"])

    def predict(self, context, model_input) -> pd.DataFrame:
        # Handle the data cleaning inside the model!
        # We ensure only the columns the model was trained on are used
        cols_to_drop = ["species_name"]
        features = model_input.drop(columns=[c for c in cols_to_drop if c in model_input.columns])
        
        # 1. Get numeric prediction
        predictions = self.model.predict(features)
        
        # 2. Convert to species names
        return self.encoder.inverse_transform(predictions)

# --- 2. SET UP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_hgb_classifier_weighted.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "species_encoder_hgb_weighted.joblib")
DATA_PATH = os.path.join(BASE_DIR, "target.csv")
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"

mlflow.set_tracking_uri(DB_PATH)
mlflow.set_experiment("Iris_PyFunc_Project")

# --- 3. SAVE (LOG) THE MODEL ---
artifacts = {
    "classifier": MODEL_PATH,
    "encoder": ENCODER_PATH
}

with mlflow.start_run(run_name="Register_PyFunc_Model") as run:
    # Log the custom model
    mlflow.pyfunc.log_model(
        artifact_path="iris_model_package",
        python_model=IrisModelWrapper(),
        artifacts=artifacts
    )
    run_id = run.info.run_id
    print(f"Model saved successfully in run: {run_id}")

# --- 4. LOAD AND PREDICT USING PYFUNC ---
print("\nLoading model back via pyfunc for inference...")
model_uri = f"runs:/{run_id}/iris_model_package"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Load the raw data
df_test = pd.read_csv(DATA_PATH)

# Use the standardized 'predict' method
results = loaded_model.predict(df_test)

# Add results back to dataframe and show
df_test["species_name"] = results
print(df_test[["species_name", "Soil_Type"]].head())

# Save results
output_csv = os.path.join(BASE_DIR, "pyfunc_results.csv")
df_test.to_csv(output_csv, index=False)
print(f"\nInference complete. Results saved to {output_csv}")