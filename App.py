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

