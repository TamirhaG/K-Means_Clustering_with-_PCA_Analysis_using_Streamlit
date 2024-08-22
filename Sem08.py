import streamlit as st
import pandas as pd

# Título de la App
st.title("K-Means Clustering con Análisis PCA usando Streamlit")

# Subir archivo de Excel
uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])
