import streamlit as st
import requests

# Set the FastAPI backend URL
API_URL = "http://127.0.0.1:8040"

# Title of the Streamlit app
st.title("Titanic Survival Prediction")

# Input form for Titanic passenger features
with st.form(key='prediction_form'):
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=100.0, value=30.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)

    # Submit button to generate prediction
    submit_button = st.form_submit_button(label='Generate Prediction')

# When the form is submitted
if submit_button:
    # Prepare the input data as a dictionary
    input_data = {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }

    # Send a POST request to the FastAPI backend
    try:
        response = requests.post(f"{API_URL}/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction generated successfully!")
            st.write(f"**Survived:** {'Yes' if result['survived'] else 'No'}")
            st.write(f"**Survival Probability:** {result['survival_probability']:.2f}")
        else:
            st.error(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
