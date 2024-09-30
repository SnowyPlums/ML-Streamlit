import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import time  # For simulating progress

# Streamlit App
st.title("ML Data Preparation and EDA App")

# Sidebar for file upload and options
st.sidebar.header("Upload and Options")

# Step 1: File Upload in the Sidebar
uploaded_file = st.sidebar.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Check if it's a CSV or Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Check if the dataset has any missing values
    missing_value_counts = df.isnull().sum()
    if missing_value_counts.sum() > 0:
        # Display missing values count for each column
        st.write("Missing Values by Column:")
        st.dataframe(missing_value_counts[missing_value_counts > 0])

        # Step 2: Handle Missing Values (only if missing values exist)
        st.sidebar.subheader("Handle Missing Values")
        missing_value_options = st.sidebar.selectbox(
            "Choose how to handle missing values", 
            ["None", "Drop missing rows", "Fill with mean", "Fill with median", "Fill with mode", "KNN Imputer"]
        )

        if missing_value_options != "None":
            if missing_value_options == "Drop missing rows":
                df = df.dropna()
            elif missing_value_options == "Fill with mean":
                imputer = SimpleImputer(strategy='mean')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            elif missing_value_options == "Fill with median":
                imputer = SimpleImputer(strategy='median')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            elif missing_value_options == "Fill with mode":
                imputer = SimpleImputer(strategy='most_frequent')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            elif missing_value_options == "KNN Imputer":
                # KNN requires all values to be numeric, so we need to convert categorical columns first
                st.write("KNN Imputer may take some time for larger datasets...")

                # Progress bar
                progress_bar = st.sidebar.progress(0)
                progress_text = st.sidebar.empty()

                # Converting categorical columns to numeric for KNN
                categorical_columns = df.select_dtypes(include=['object']).columns
                df_encoded = df.copy()
                label_encoders = {}

                for col in categorical_columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    label_encoders[col] = le

                # Apply KNN Imputer
                k = st.sidebar.slider("Select the number of neighbors for KNN", min_value=1, max_value=10, value=5)
                knn_imputer = KNNImputer(n_neighbors=k)

                # Simulate progress
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                df_encoded = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df.columns)

                # Convert numeric back to categorical where necessary
                for col in categorical_columns:
                    df_encoded[col] = label_encoders[col].inverse_transform(df_encoded[col].astype(int))

                df = df_encoded

                # Complete the progress bar
                progress_bar.progress(100)
                progress_text.text("KNN Imputation completed!")

            st.write("Updated Dataset Preview (after handling missing values):")
            st.dataframe(df.head())
    else:
        st.sidebar.write("No missing values detected in the dataset.")
    
    # Step 3: Handle Categorical Variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        st.sidebar.subheader("Handle Categorical Variables")
        encoding_option = st.sidebar.selectbox(
            "Choose how to encode categorical variables", 
            ["None", "One-hot encoding", "Label encoding", "Ordinal encoding"]
        )

        if encoding_option != "None":
            if encoding_option == "One-hot encoding":
                df = pd.get_dummies(df, columns=categorical_columns)
            elif encoding_option == "Label encoding":
                label_encoders = {}
                for col in categorical_columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
            elif encoding_option == "Ordinal encoding":
                ordinal_encoder = OrdinalEncoder()
                df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

            st.write("Updated Dataset Preview (after encoding categorical variables):")
            st.dataframe(df.head())
    else:
        st.sidebar.write("No categorical variables detected in the dataset.")
    
    # Step 4: Exploratory Data Analysis (EDA)
    st.sidebar.subheader("Exploratory Data Analysis (EDA) Options")
    if st.sidebar.button("Run EDA"):
        st.subheader("Exploratory Data Analysis (EDA)")

        # Show basic statistics
        st.write("Basic Statistics:")
        st.write(df.describe())

        # Correlation Matrix
        st.write("Correlation Matrix:")
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt.gcf())
        plt.clf()

        # Histograms for numerical data
        st.write("Histograms for Numerical Columns:")
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numerical_columns:
            st.write(f"Histogram for {col}:")
            sns.histplot(df[col], kde=True)
            st.pyplot(plt.gcf())
            plt.clf()

        # Pairplot for visualizing correlations
        st.write("Pairplot for Data (can take time for large datasets):")
        sns.pairplot(df)
        st.pyplot(plt.gcf())
        plt.clf()
    
    # Step 5: Download Processed Data
    st.sidebar.subheader("Download Processed Dataset")
    st.sidebar.download_button(
        label="Download Processed Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='processed_data.csv',
        mime='text/csv',
    )
else:
    st.write("Please upload a file to get started.")
