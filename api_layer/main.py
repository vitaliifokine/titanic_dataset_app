from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import business_layer
from business_layer.business import train_model, load_model, predict_survival

app = FastAPI()

class TrainRequest(BaseModel):
    model: str = "logistic_regression"  # Default to logistic regression
    params: dict = {}  # Parameters for the model


class PredictRequest(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    fare: float


@app.post("/train")
async def train(request: TrainRequest):
    """
    Endpoint to train a model.
    Accepts JSON with a 'model' key to specify the algorithm
    and additional parameters to be passed to the model.
    """
    algorithm = request.model
    params = request.params

    # Train the model with specified parameters
    result = train_model(algorithm, **params)
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    return result


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Endpoint to predict survival based on input features.
    The model must be trained first using the /train endpoint.
    """
    # Convert input features
    try:
        pclass = int(request.pclass)
        sex = 0 if request.sex.lower() == 'male' else 1
        age = float(request.age)
        sibsp = int(request.sibsp)
        fare = float(request.fare)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail="Invalid input data.")

    # Load the model (in case it hasn't been loaded yet)
    load_model()

    # Prepare the input data for prediction
    input_data = [pclass, sex, age, sibsp, fare]
    result = predict_survival(input_data)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    return result

# Run with: uvicorn main:app --reload
