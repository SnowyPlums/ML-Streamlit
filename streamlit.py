import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def main():
    st.title("ML Data Preparation and EDA App")

    st.sidebar.header("Upload and Options")

    uploaded_file = st.sidebar.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

    # Insert containers separated into tabs:
    edaTab, models = st.tabs(["EDA", "Models"])

    # You can also use "with" notation:
    with edaTab:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Dataset Preview:")
            st.dataframe(df.head())
            


            missing_value_counts = df.isnull().sum()
            if missing_value_counts.sum() > 0:
                st.write("Missing Values by Column:")
                st.dataframe(missing_value_counts[missing_value_counts > 0])
            else:
                st.sidebar.write("No missing values detected in the dataset.")



            df = categoricaldata(df)

            if missing_value_counts.sum() > 0:
                df = missingvalues(df)

            
            st.sidebar.header("EDA")
            EDA(df)


            st.sidebar.subheader("Download Processed Dataset")
            st.sidebar.download_button(
            label="Download Processed Data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='processed_data.csv',
            mime='text/csv',
            )

        else:
            st.write("Please upload a file to get started.")

def categoricaldata(df):
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

    return df

def missingvalues(df):
    st.sidebar.subheader("Handle Missing Values")
    missing_value_options = st.sidebar.selectbox(
        "Choose how to handle missing values", 
        ["None", "Drop missing rows", "Fill with mean", "Fill with median", "KNN Imputer"]
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
            df = df.dropna()

        st.write("Updated Dataset Preview (after handling missing values):")
        st.dataframe(df.head())

        return df
    
def EDA(df):
    # Show basic statistics
    bs = st.sidebar.checkbox("Basic Statistics")
    if bs:
        st.write("Basic Statistics:")
        st.write(df.describe())

    corrm = st.sidebar.checkbox("Correlation Matrix")
    if corrm:
        corr = df.corr()  # Compute correlation matrix
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask to display only one triangle
        
        f, ax = plt.subplots(figsize=(11, 9))  # Create a figure
        cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Define colormap

        # Create the heatmap with annotations and rotated labels
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f',  # Add annotations with 2 decimals
                    cbar_kws={"shrink": .5}, ax=ax)

        # Rotate x-axis labels by 45 degrees, right-aligned
        plt.xticks(rotation=45, ha='right')

        # Display the plot in the Streamlit app
        st.pyplot(f)

if __name__ == "__main__":
    main()
