import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np


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


def load_models(file_path):
    models = {}
    if os.path.isfile(file_path) and file_path.endswith(('.joblib', '.pkl')):
        model_name = os.path.basename(file_path).split('.')[0]
        model = joblib.load(file_path)
        st.write("Model File Path :")
        st.write(file_path)
        models[model_name] = model
    return models


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


def main():
    st.title("Model Prediction App")

    # User input for the folder containing models
    model_folder = st.text_input("Enter the path to the model folder:", "Notebook/Models/XGBoost")

    # Load models from the specified folder
    models = load_models(model_folder)

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
            st.write("Prepricessed Data:")
            st.write(input_data)

            # Make predictions for each model
            predictions = {}
            for model_name, model in models.items():
                prediction = make_prediction(model, input_data)
                predictions[model_name] = prediction[0]

            # Display predictions
            st.write("Predictions:")
            for model_name, prediction in predictions.items():
                st.write(f"- {model_name}: {prediction}")

        except ValueError:
            st.error("Invalid input data. Please enter valid values.")


if __name__ == "__main__":
    main()
