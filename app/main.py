from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("app/model.pkl")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Mapping to align input data with model training columns
COLUMN_MAP = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol"
}

@app.post("/predict")
def predict(data: WineInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Rename columns to match the trained model format
    input_df.rename(columns=COLUMN_MAP, inplace=True)

    try:
        # Predict the quality
        prediction = model.predict(input_df)[0]
        return {"predicted_quality": prediction}
    except Exception as e:
        return {"error": str(e)}
