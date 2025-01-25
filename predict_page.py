import streamlit as st 
import pickle
import numpy as np
import pandas as pd
from explore_page import clean_experience, clean_education, clean_country_names

with open ('knn_model.pkl', 'rb') as file:
    data = pickle.load(file)
    knn_model = data['model']
    columns_knn = data['knn_column']

# Load the model and column names from the .pkl file
#with open('rf_model.pkl', 'rb') as file:
#   data = pickle.load(file)
 #   rf_model = data['model']
 #   columns_rf = data['X_train.columns']


# Function to encode new data and predict salary
# Function to encode new data and predict salary with dummy encoder(One-hot encoding) and Random Forest model:
# Function to encode new data and predict salary with dummyy (One-hot encoding) and KNN model:
def predict_salary(new_data):
    # Create a DataFrame from the NumPy array
    s = pd.DataFrame([new_data], columns=["Country", "EdLevel", "YearsCodePro"])

    # Clean the country names
    s = clean_country_names(s)
    
    # Apply the cleaning functions
    s["YearsCodePro"] = s["YearsCodePro"].apply(clean_experience)
    s["EdLevel"] = s["EdLevel"].apply(clean_education)
    # Display s_encoded for debugging in the app
    st.write("### DataFrame")
    st.write(s) 
    
    # Get dummies for the "Country" and "EdLevel" columns only
    s_encoded = pd.get_dummies(s, columns=["Country", "EdLevel"], drop_first=True, prefix='', prefix_sep='')
        
    # Ensure the new data has the same columns as the training data
    s_encoded = s_encoded.reindex(columns=columns_knn, fill_value=0)
       
    # Predict the salary for the new data
    return knn_model.predict(s_encoded)[0]



def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary of a software developer""", key = '0')

    # Get the input from the user
    Countries = (
        'United States', 
        'India', 
        'United Kingdom', 
        'Germany', 
        'Canada', 
        'Brazil', 
        'France', 
        'Spain', 
        'Australia', 
        'Netherlands'
    )
    Education_levels = (
        'Bachelor’s degree', 
        'Master’s degree', 
        'Less than a Bachelor’s degree', 
        'Other doctoral'
    )
    
    country = st.selectbox("Country", Countries, key = f"country")
    education = st.selectbox("EdLevel", Education_levels, key = f"education")
    experience = st.slider("YearsCodePro", 0, 50, 3, key = f"experience")
    
    # When the user clicks the 'Predict' button, make the prediction and display it
    ok = st.button("Calculate Salary")
    if ok:
        new_data = [country, education, experience]
        prediction = predict_salary(new_data)
        st.write(f"The estimated salary is ${prediction:,.2f}")


show_predict_page()





