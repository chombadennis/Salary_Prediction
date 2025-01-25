import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder

# Getting rid of all Countries with less than 400 users:
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map

# Function to convert the YearsCodePro column to numeric
def clean_experience(x):
    if x == "Less than 1 year":
        return 0.5
    if x == "More than 50 years":
        return 51
    return float(x)

# Function to clean the Education Level column
def clean_education(x):
    if "Bachelor’s degree" in x:
        return "Bachelor’s degree"
    if "Master’s degree" in x:
        return "Master’s degree"
    if "Professional degree" in x or "Other doctoral" in x:
        return "Post grad"
    return "Less than a Bachelor’s degree"


@st.cache
def load_data():
    directory = r"D:\Stack Overflow Project\stack-overflow-developer-survey-2024"
    df = pd.read_csv(directory + r"\survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
    
    df = df[df["ConvertedCompYearly"].notnull()]
    df = df.dropna()
    # Keeping datapoit where user is emloyed full-time:
    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)

    df = pd.DataFrame(df)
    country_counts = df["Country"].value_counts()

    country_map = shorten_categories(country_counts, 400)
    df["Country"] = df["Country"].map(country_map)
    df["Country"].value_counts()

    # Inspect Salary range by Visualization:
    df["ConvertedCompYearly"] = df["ConvertedCompYearly"].astype(int)
    
    # Replacing the long country names with shorter ones:
    df['Country'] = df['Country'].replace({'United States of America': 'USA', 'United Kingdom of Great Britain and Northern Ireland': 'UK'})

    
    # Filter the DataFrame to keep only rows where the salary is between 10,000 and 250,000
    df = df[(df['ConvertedCompYearly'] >= 10000) & (df['ConvertedCompYearly'] <= 250000)]
    
    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience) #Applying the function above
    df["EdLevel"] = df["EdLevel"].apply(clean_education) # Applying the function above
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df
    

df = load_data()  

def show_explore_page():
    st.title("Explore Software Developer Salaries")

    st.write(
    """
    ### Stack Overflow Developer Survey 2024
    """         
    )

    data = df["Country"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", shadow = True, startangle=90)
    ax.axis("equal")

    st.write("""#### Number of Data from Different Countries""", key = "5")

    st.pyplot(fig) 

    st.write(
        """
        #### Mean Salary Based on Country
        """
        , key = "6"
        )
    data = df.groupby("Country")["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
        #### Mean Salary Based on the Years of Professional Experience
        """
        , key = "7"
        )
    data = df.groupby("YearsCodePro")["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
    