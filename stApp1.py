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

    else:
        st.write("No categorical columns found in the data.")

    # Check for missing values
    if df.isnull().values.any():
        st.write("### Missing values found")
        st.write(df.isnull().sum())
        
        # Handle missing values
        df = df.fillna(df.mean())

        st.write("### Data after handling missing values")
        st.write(df.head())
    
    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Data normalization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.dropna())

    # Apply PCA to understand the principal components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    st.write("### Variance explained by each principal component:")
    st.write(pca.explained_variance_ratio_)

    st.write("### Loadings of variables on the principal components:")
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=df.columns)
    st.write(loadings)

    # Select the number of clusters
    st.write("### Select the number of clusters")
    num_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)

    # Apply K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    # Add cluster to the original DataFrame
    df['Cluster'] = clusters

    st.write("### Data with assigned clusters")
    st.write(df.head())

    # Visualize clusters using PCA
    pca_df['Cluster'] = clusters
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='Cluster Visualization using PCA')
    st.plotly_chart(fig)