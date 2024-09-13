# business.py

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import joblib

# Load Titanic dataset
data = sns.load_dataset('titanic')
data = data[['pclass', 'sex', 'age', 'sibsp', 'fare', 'survived']].dropna()
data['sex'] = data['sex'].map({'male': 0, 'female': 1})  # Encode 'sex' column

# Features and target
X = data[['pclass', 'sex', 'age', 'sibsp', 'fare']]
y = data['survived']

# Initialize model variable
model = None


def train_model(algorithm='logistic_regression', **kwargs):
    """
    Train a model based on the specified algorithm and parameters.
    Available algorithms: 'logistic_regression', 'decision_tree', 'mlp'.
    """
    global model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == 'logistic_regression':
        model = LogisticRegression(**kwargs)
    elif algorithm == 'decision_tree':
        model = DecisionTreeClassifier(**kwargs)
    elif algorithm == 'mlp':
        model = MLPClassifier(**kwargs)
    else:
        return {'error': 'Invalid algorithm specified.'}, 400

    model.fit(X_train, y_train)
    joblib.dump(model, 'titanic_model.pkl')  # Save the model
    return {'message': f'Model trained using {algorithm} with parameters {kwargs}.'}


def load_model():
    """Load the model from the file."""
    global model
    model = joblib.load('titanic_model.pkl')


def predict_survival(input_data):
    """
    Predict survival based on input data using the trained model.
    The model must be trained first using the train_model function.
    """
    global model
    if not model:
        return {'error': 'Model not trained. Use /train endpoint first.'}, 400

    prediction = model.predict([input_data])
    return {'survived': bool(prediction[0])}
