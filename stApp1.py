import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px


# Title of the app
st.title("K-Means Clustering with PCA Analysis using Streamlit")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None: 
    df = pd.read_excel(uploaded_file)

    st.write("### Data preview")
    st.write(df.head())

    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        st.write("### Identified categorical columns")
        st.write(categorical_columns)

        # Convert categorical columns to dummies
        df = pd.get_dummies(df, columns=categorical_columns)
        st.write("### Data after converting to dummies")
        st.write(df.head())