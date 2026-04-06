
import mlflow.pyfunc
import joblib
import pandas as pd

class IrisHGBWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_file"])
        self.encoder = joblib.load(context.artifacts["encoder_file"])
        self.measurement_cols = [
            'sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)'
        ]

    def predict(self, context, model_input: pd.DataFrame):
        df = model_input.copy()
        if 'Soil_Type' in df.columns:
            df['Soil_Type'] = df['Soil_Type'].astype('category')
        
        features = self.measurement_cols + ['Soil_Type']
        raw_numeric_preds = self.model.predict(df[features])
        species_names = self.encoder.inverse_transform(raw_numeric_preds)
        return pd.Series(species_names)
