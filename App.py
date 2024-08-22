import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px 


# App title
st.title("K-Means Clustering con Análisis PCA usando Streamlit")

# Upload Excel file
uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])

if uploaded_file is not None: 
    df = pd.read_excel(uploaded_file)

    st.write("### Vista previa de los datos")
    st.write(df.head())
    
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        st.write("### Columnas categóricas identificadas")
        st.write(categorical_columns)
        
        # Convertir columnas categóricas a dummies
        df = pd.get_dummies(df, columns=categorical_columns)
        st.write("### Datos después de la conversión a dummies")
        st.write(df.head())