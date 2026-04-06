import joblib
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from numpy.typing import NDArray

class ModelLocal:
    """
    A simplified version of the model that runs on your personal computer.
    It removes all 'FF_serving' dependencies so you can see the raw logic.
    """

    def __init__(self, model_path: str):
        """
        Initialization: This is where we load the physical files (.joblib).
        'model_path' tells the code where to look for the 'artifacts' folder.
        """
        try:
            # We assume the artifacts are in a folder named 'artifacts'
            self.artifacts_dir = os.path.join(model_path, "artifacts")
            
            # These are the actual files created during your training
            model_file = os.path.join(self.artifacts_dir, "best_hgb_classifier_weighted.joblib")
            encoder_file = os.path.join(self.artifacts_dir, "species_encoder_hgb_weighted.joblib")
            
            # Load the brain (model) and the translator (encoder)
            self.model = joblib.load(model_file)
            self.encoder = joblib.load(encoder_file)
            
            # Metadata the model needs to build its internal table
            self.measurement_cols = [
                'sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'petal width (cm)'
            ]
            print(f"✓ Local Model loaded artifacts from: {self.artifacts_dir}")
            
        except Exception as e:
            print(f"ERROR during init: {e}")
            raise

    def predict_local(self, measurement_input: NDArray, soil_type_input: List[str]) -> Dict[str, Any]:
        """
        This is the core logic. It takes raw numbers and strings and returns a prediction.
        """
        # 1. DATA RECONSTRUCTION (The 'Massive Task')
        # We turn raw data into a Pandas DataFrame
        df = pd.DataFrame(measurement_input, columns=self.measurement_cols)
        df['Soil_Type'] = soil_type_input
        
        # 2. TYPE CASTING
        # HGB models require the 'category' type to work with strings
        df['Soil_Type'] = df['Soil_Type'].astype('category')
        
        # 3. PREDICTION
        # The model gives us back numbers (0, 1, 2)
        raw_numeric_preds = self.model.predict(df)
        
        # 4. TRANSLATION
        # The encoder turns (0, 1, 2) back into ('Setosa', etc.)
        species_names = self.encoder.inverse_transform(raw_numeric_preds)
        
        return {
            "results": species_names.tolist(),
            "status": "locally_tested"
        }