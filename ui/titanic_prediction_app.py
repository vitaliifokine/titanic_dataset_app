import streamlit as st
import requests

# Set the FastAPI backend URL
API_URL = "http://127.0.0.1:8040"

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

# Title of the app
st.title("Titanic Survival Prediction")

# Step 1: Select Model for Prediction
st.header("Step 1: Select Model to Train")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a model to train:",
    ("Logistic Regression", "Decision Tree", "MLP Classifier")
)

if st.button("Train Model"):
    model_map = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "MLP Classifier": "mlp"
    }

    # Show spinner while the model is training
    with st.spinner("Model is learning..."):
        # API call to train the selected model
        response = requests.post(f"{API_URL}/train", json={"model": model_map[model_choice]})

    # Display results after training is completed
    if response.status_code == 200:
        st.success("Model trained successfully!")
        st.json(response.json())
    else:
        st.error("Failed to train the model. Please check the backend logs for more details.")

# Step 2: Predict Survival with Trained Model
st.header("Step 2: Predict Survival")

# Input fields for passenger details
pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=0)
sex = st.selectbox("Sex", options=["male", "female"], index=1)
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", options=["C", "Q", "S"], index=2)

# Prediction button
if st.button("Generate Prediction"):
    # Prepare input data for the API
    input_data = {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "fare": fare,
        "embarked": embarked,
        "parch": parch
    }

    # API call to predict survival
    response = requests.post(f"{API_URL}/predict", json=input_data)

    # Display results
    if response.status_code == 200:
        result = response.json()
        st.success("Prediction generated successfully!")
        st.markdown(f"**Survived**: {'Yes' if result['survived'] else 'No'}")
        st.markdown(f"**Survival Probability**: {result['survival_probability']:.2f}")
    else:
        st.error("Failed to generate prediction. Please check the backend logs for more details.")
