import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import business_layer.business as business

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class TrainRequest(BaseModel):
    model: str  # Only model to use


class PredictRequest(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    fare: float
    embarked: str
    parch: int  # Add missing 'parch' field


@app.post("/train")
async def train(request: TrainRequest):
    algorithm = request.model
    result = business.train_model(algorithm)
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    return result


@app.post("/predict")
async def predict(request: PredictRequest):
    # Updated input_data with the 'parch' field
    input_data = {
        'pclass': int(request.pclass),
        'sex': request.sex,
        'age': float(request.age),
        'sibsp': int(request.sibsp),
        'fare': float(request.fare),
        'embarked': request.embarked,
        'parch': int(request.parch)  # Add 'parch' to input data
    }

    # Load model and make prediction
    business.load_model()
    result = business.predict_survival(input_data)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8040)
