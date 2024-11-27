from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
model = load_model("saved_model/my_model")

# Define request schema
class PredictionRequest(BaseModel):
    features: list[float]  # Input feature vector as a list of floats

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI Model Server!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input features into a NumPy array
        input_data = np.array([request.features])
        
        # Ensure input shape matches the model
        if input_data.shape[1] != model.input_shape[1]:
            raise HTTPException(
                status_code=400, detail=f"Expected input size: {model.input_shape[1]}"
            )

        # Make prediction
        prediction = model.predict(input_data)
        predicted_label = int(np.argmax(prediction, axis=1)[0])  # Assuming classification
        
        return {"prediction": predicted_label, "confidence": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 