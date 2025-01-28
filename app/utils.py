
import joblib
import pandas as pd

# Load the trained machine learning model
def load_model(model_path: str):
    try:
        with open(model_path, "rb") as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        raise Exception(f"Model file not found at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Preprocess the input data
def preprocess_input(data):
    # Convert input into a NumPy array or the required format
    data_frame = pd.DataFrame(data)
    return data_frame.to_numpy()

# Make prediction using the model
def make_prediction(model, features):
    # Get prediction
    prediction = model.predict(features)[0]
    # Map numerical prediction to labels (e.g., 1 -> "Good", 0 -> "Bad")
    return "Good" if prediction == 1 else "Bad"
