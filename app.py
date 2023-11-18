import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import yaml
import lightgbm 

def display_categories():
    categories_info = {
        "Artist_Reputation": ['Low_Reputation', 'High_Reputation', 'Moderate_Reputation', 'Very_High_Reputation'],
        "Material": ['Brass', 'Clay', 'Aluminium', 'Bronze', 'Wood', 'Stone', 'Marble'],
        "International": ['Yes', 'No'],
        "Express_Shipment": ['Yes', 'No'],
        "Installation_Included": ['No', 'Yes'],
        "Transport": ['Airways', 'Roadways', 'Waterways'],
        "Fragile": ['No', 'Yes'],
        "Customer_Information": ['Working Class', 'Wealthy']
    }

    st.title("Categories Information")

    # Collect user input for each category
    categorical_data = {}
    for category, values in categories_info.items():
        user_input = st.selectbox(f"Select {category} category:", values)
        categorical_data[category] = user_input

    return categorical_data


def collect_input_data():
    # Collect numerical inputs
    weight = st.number_input("Weight", min_value=0.0)
    price_of_sculpture = st.number_input("Price of Sculpture", min_value=0.0)
    base_shipping_price = st.number_input("Base Shipping Price", min_value=0.0)

    # Log-transform numerical inputs
    weight_log = np.log1p(weight + 1)
    price_of_sculpture_log = np.log1p(price_of_sculpture + 1)
    base_shipping_price_log = np.log1p(base_shipping_price + 1)

    # Collect categorical inputs
    categorical_data = display_categories()

    # Combine numerical and categorical data into a DataFrame
    input_data = pd.DataFrame({
        "Weight": [weight_log],
        "Price_Of_Sculpture": [price_of_sculpture_log],
        "Base_Shipping_Price": [base_shipping_price_log],
        **categorical_data
    })

    return input_data


def load_model(model_path):
    """
    Load a machine learning model from a joblib file.

    Parameters:
    - model_path (str): The file path to the joblib file containing the model.

    Returns:
    - model: The loaded machine learning model.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")
        return None


def make_prediction(model, input_data):
    # Replace this with the actual prediction logic based on your models
    prediction = model.predict(input_data)
    
    st.write("Predictions :")
    st.write(prediction)
    return prediction


def preprocess_data(input_data, preprocessor):
    # Apply preprocessing to the input data
    preprocessed_data = preprocessor.transform(input_data)
    return preprocessed_data


def load_preprocessor_from_file(file_path):
    # Load the preprocessor from the specified file
    preprocessor = joblib.load(file_path)
    return preprocessor

def load_yaml_parameters(file_path):
    """
    Load parameters from a YAML file.

    Parameters:
    - file_path: Path to the YAML file.

    Returns:
    - parameters: Dictionary containing parameters.
    """
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

def load_and_display_yaml(file_path):
    """
    Load parameters from a YAML file and display them in Streamlit.

    Parameters:
    - file_path: Path to the YAML file.

    Returns:
    - None
    """
    st.title("Model Parameter Display")

    # Load parameters from YAML file
    parameters = load_yaml_parameters(file_path)

    # Display parameters in Streamlit
    st.sidebar.title("Model Parameters")

    # Extract model-specific parameters
    model_name = parameters.get("Model", "Unknown")
    model_params_str = parameters.get("Model_Parameters", "")
    r2_score = parameters.get("R2_Score", "Unknown")

    # Display model name and R2 score
    st.sidebar.text(f"Model: {model_name}")
    st.sidebar.text(f"R2 Score: {r2_score}")

    # Display model-specific parameters
    if model_params_str:
        st.sidebar.text("Model Parameters:")
        try:
            # Parse the model parameters string as a dictionary
            model_params = yaml.safe_load(model_params_str.replace("''", '"'))
            for param_name, param_value in model_params.items():
                st.sidebar.text(f"{param_name}: {param_value}")
        except Exception as e:
            st.sidebar.error(f"Error parsing model parameters: {e}")
def main():
    st.title("Model Prediction App")

    # Load models from the specified folder
    Xg_boost_model = load_model(model_path="Notebook/Models/XGBoost/XGBoost_model.pkl")
    Rf_model = load_model(model_path="Notebook/Models/Random Forest/Random Forest_model.pkl")
    gbm_model = load_model(model_path="Notebook/Models/LightGBM/LightGBM_model.joblib")
    # Assuming preprocess_object is your preprocessing object
    preprocessor = load_preprocessor_from_file(file_path="Notebook/Preprocessor/one_hot_encoder.joblib")  # Implement a function to load your preprocessor object

    # User input for prediction data
    input_data = collect_input_data()

    # Display the input_data DataFrame on the Streamlit page
    st.write("Input Data:")
    st.write(input_data)

 

    if st.button("Make Predictions"):
        try:
            # Preprocess the input data
            input_data = preprocess_data(input_data, preprocessor)

            # Display the input_data DataFrame on the Streamlit page
            st.write("Preprocessed Data:")
            st.write(input_data)

            Predictions = []

            # Display predictions and parameters
            # XG Boost Model
            st.write("XG Boost Model Prediction")
            prediction_xgb = make_prediction(Xg_boost_model, input_data)
            st.write(f"Prediction: {prediction_xgb}")

            Predictions.append(prediction_xgb)
            
            load_and_display_yaml(file_path="Notebook/Models/XGBoost/XGBoost_params.yaml")

            # Random Forest
            st.write("Random Forest Model Prediction")
            prediction_rf = make_prediction(Rf_model, input_data)
            st.write(f"Prediction: {prediction_rf}")

            Predictions.append(prediction_rf)
            load_and_display_yaml(file_path="Notebook/Models/Random Forest/Random_Forest_params.yaml")


            ## Gradient Boost
            st.write("Gradient Boost Model Prediction")
            prediction_gbm = make_prediction(gbm_model, input_data)
            st.write(f"Prediction: {prediction_gbm}")
 
            Predictions.append(prediction_gbm)
            load_and_display_yaml(file_path="Notebook/Models/LightGBM/LightGBM_params.yaml")


            avg_prediction = np.mean(Predictions)

            st.write("Average of Predictions")
            st.write(avg_prediction)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()