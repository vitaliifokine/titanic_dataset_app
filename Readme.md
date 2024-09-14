## Titanic Survival Prediction App

This repository contains an interactive web-based application to predict the survival chances of passengers on the Titanic using machine learning models. The application is built with three layers:

1. **UI Layer**: Built with Streamlit to provide a user-friendly interface for inputting passenger data and displaying prediction results.
2. **API Layer**: Developed using FastAPI to handle HTTP requests and interact with the backend logic.
3. **Business Layer**: Implements the core machine learning logic, including data preprocessing, model training, and prediction.

### Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

### Features

- **Interactive UI**: Easy-to-use web interface to enter Titanic passenger details and get survival predictions.
- **Machine Learning Models**: Supports Logistic Regression, Decision Tree, and MLP classifiers with randomized search for hyperparameter tuning.
- **REST API**: Provides endpoints for training models and generating predictions.
- **Data Preprocessing**: Cleans and preprocesses input data to make accurate predictions.

### Requirements

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/titanic-prediction-app.git
   cd titanic-prediction-app
   ```

2. **Create and Activate a Virtual Environment**:

   For Windows:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   For Mac/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### 1. **Start the FastAPI Backend**

Run the FastAPI server to expose the API endpoints for model training and prediction.

```bash
python -m uvicorn api_layer.main:app --host 127.0.0.1 --port 8040 --reload
```

- The FastAPI server will be running on `http://127.0.0.1:8040`.

#### 2. **Start the Streamlit Frontend (UI Layer)**

Run the Streamlit UI for the Titanic prediction interface.

```bash
streamlit run titanic_prediction_app.py
```

- The Streamlit UI will be available at `http://localhost:8501`.

### Usage

1. Open the Streamlit web interface at `http://localhost:8501`.
2. Fill in the passenger details like class, age, sex, siblings/spouses aboard, parents/children aboard, fare, and port of embarkation.
3. Click on the "Generate Prediction" button.
4. The prediction result, including whether the passenger would survive and the survival probability, will be displayed.

### API Endpoints

| Method | Endpoint          | Description                                         |
|--------|-------------------|-----------------------------------------------------|
| POST   | `/train`          | Trains a machine learning model with specified parameters. |
| POST   | `/predict`        | Predicts the survival chances of a passenger using the trained model. |

#### 1. **Train Model Endpoint**

- **URL**: `/train`
- **Method**: `POST`
- **Request Body**:
  
  ```json
  {
    "model": "decision_tree"
  }
  ```

- **Response**:

  ```json
  {
    "message": "Model trained using decision_tree with best parameters found by RandomizedSearchCV.",
    "accuracy": 0.85,
    "best_params": { ... }
  }
  ```

#### 2. **Predict Endpoint**

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
  
  ```json
  {
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "sibsp": 0,
    "fare": 50,
    "embarked": "S",
    "parch": 0
  }
  ```

- **Response**:

  ```json
  {
    "survived": true,
    "survival_probability": 0.75
  }
  ```

### Project Structure

```
titanic-prediction-app/
│
├── api_layer/
│   ├── main.py              # FastAPI endpoints for training and prediction
│
├── business_layer/
│   ├── business.py          # Business logic for data preprocessing, model training, and prediction
│   └── titanic_model.pkl    # Saved model after training (generated after training)
│
├── titanic_prediction_app.py # Streamlit UI application
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

