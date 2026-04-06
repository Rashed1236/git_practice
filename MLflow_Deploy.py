import os
import joblib
import pandas as pd
import mlflow

# --- 1. SMART PATH HANDLING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_hgb_classifier_weighted.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "species_encoder_hgb_weighted.joblib")
DATA_PATH = os.path.join(BASE_DIR, "target_no_species_col.csv")
#DATA_PATH = os.path.join(BASE_DIR, "target.csv")
DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"

def run_inference():
    # --- 2. CONNECT TO MLFLOW ---
    mlflow.set_tracking_uri(DB_PATH)
    mlflow.set_experiment("Iris_Inference_Runs")

    print(f"Checking files in: {BASE_DIR}")

    # Safety check: ensure files exist
    missing_files = False
    for file_path in [MODEL_PATH, ENCODER_PATH, DATA_PATH]:
        if not os.path.exists(file_path):
            print(f"ERROR: Cannot find file at {file_path}")
            missing_files = True
    
    if missing_files:
        print("Please ensure your .joblib and .csv files are in the folder listed above.")
        return

    # --- 3. START MLFLOW RUN ---
    with mlflow.start_run(run_name="Iris_Batch_Inference"):
        print("--- Loading Model & Encoder ---")
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        
        print("--- Loading Data ---")
        df = pd.read_csv(DATA_PATH)
        print(df)
        
        # --- 4. PREPARE DATA ---
        # The error "columns are missing: {'Soil_Type'}" proves the model NEEDS Soil_Type.
        # We only drop 'species_name' because that is the empty column we want to fill.

        # cols_to_drop = ["species_name"]
        # features = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        # print(f"Input features: {list(features.columns)}")
        # print(features)
        
        # --- 5. PREDICT ---
        print(f"Predicting species for {len(df)} rows...")
        
        # The model will now see: sepal length, sepal width, petal length, petal width, AND Soil_Type
        #numeric_predictions = model.predict(features)

        numeric_predictions = model.predict(df)
        
        # Convert numbers (0, 1, 2) back to names ('setosa', etc.)
        final_species_names = label_encoder.inverse_transform(numeric_predictions)
        df["species_name"] = final_species_names

        # --- 6. LOG TO MLFLOW ---
        output_file = os.path.join(BASE_DIR, "predicted_results.csv")
        df.to_csv(output_file, index=False)
        
        mlflow.log_artifact(output_file, artifact_path="predictions")
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(ENCODER_PATH, artifact_path="encoder")


        
        
        # Log summary stats
        counts = df["species_name"].value_counts().to_dict()
        for species, count in counts.items():
            #mlflow.log_metric(f"total_{species}", count)
            mlflow.log_metric(f"total_predicted", count)
            
        print(f"SUCCESS! Results saved to: {output_file}")
        print("You can now view the results in the MLflow UI.")

if __name__ == "__main__":
    run_inference()