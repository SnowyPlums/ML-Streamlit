import streamlit as st
import pandas as pd
from scipy.io import arff
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import functions from other files
from imputation import *
from visualization import *
from firstModel import *
from annGridSearch import *
from evaluation import *

def load_data(file):
    """Load data based on the file type selected."""
    if file.type == "text/csv":
        return pd.read_csv(file)
    elif file.type == "application/octet-stream":  # Common MIME type for ARFF
        data, meta = arff.loadarff('dataset/Dry_Bean_Dataset.arff')
        df = pd.DataFrame(data)

        # Convert binary string to readable format for the 'Class' column
        df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))
        return df
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or ARFF file.")
        return None

def prepare_data(df, target):
    # Encode the target variable
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    # Separate features and target variable
    X = df.drop(columns=[target])
    y = df[target]

    # Split into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature columns (use fit on train, transform on both train and test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.session_state['prepared_data'] = X_train, X_test, y_train, y_test, label_encoder
    
def prepare_model_history(path, model_filename, history_filename, X_test, y_test, label_encoder):
    model = load_model(path + model_filename)

    with open(path + history_filename, 'r') as f:
        history = json.load(f)

    y_pred = model.predict(X_test)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    report_df = classification_report(y_test, y_pred_classes, output_dict=True, target_names=label_encoder.classes_)

    return model, history, y_pred_classes, report_df
    
def display_raw_data(df):
    """Display the raw data and the count of null values."""
    st.subheader("Raw Data")
    st.write(df)
    st.subheader("Null Value Counts")
    st.write(df.isnull().sum())

def drop_columns(df):
    """Allow the user to drop columns."""
    st.subheader("Select Columns to Drop")
    columns = st.multiselect("Select columns to drop", options=df.columns)
    if columns:
        df = df.drop(columns=columns)
    return df

def handle_missing_values(df):
    """Handle missing values based on user selection."""
    impute_methods = {
        "Mean": mean_imputation,
        "Median": median_imputation,
        "Mode": mode_imputation,
        "KNN": knn_imputation,
        "Constant": constant_imputation,
        "Drop Rows with NA": drop_na
    }

    st.subheader("Missing Value Imputation")
    method = st.selectbox("Select an imputation method", options=list(impute_methods.keys()))
    
    if method == "Constant":
        constant_value = st.number_input("Enter constant value for imputation", value=0)
        imputed_df = impute_methods[method](df, constant=constant_value)
    else:
        imputed_df = impute_methods[method](df)
    
    return imputed_df

def visualize_data(df):
    """Display various visualizations based on the provided dataframe."""
    st.subheader("Data Visualization")
    
    if st.checkbox("Correlation Matrix"):
        plot_corr_matrix(df)
    
    if st.checkbox("Bean Counts"):
        plot_bean_counts(df)
    
    if st.checkbox("Pairplot"):
        plot_bean_pairplot(df)
    
    if st.checkbox("PCA Plot"):
        plot_bean_pca(df)
    
    if st.checkbox("Boxplot"):
        feature_for_boxplot = st.selectbox("Select feature for Boxplot", options=df.columns, key='boxplot')
        if feature_for_boxplot:
            plot_bean_boxplot(df, feature=feature_for_boxplot)

    if st.checkbox("Violinplot"):
        feature_for_violinplot = st.selectbox("Select feature for Violin Plot", options=df.columns, key='violinplot')
        if feature_for_violinplot:
            plot_bean_violinplot(df, feature=feature_for_violinplot)

def evaluate_model(model, history, report_df, y_test, y_pred):
    if st.checkbox("Classification Matrix"):
        plot_classification_matrix(report_df, model)

    if st.checkbox("Confusion Matrix"):
        plot_confusion_matrix(y_test, y_pred, model)
    
    if st.checkbox("Training History"):
        plot_training_history(history, model)

def main():
    st.title("ANN and XGBoost Model EDA and Training")

    # Dataset Upload
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV, ARFF)", type=['csv', 'arff'])

    uploaded_model = st.sidebar.file_uploader("Upload your model (h5)", type=['h5'])
    uploaded_history = st.sidebar.file_uploader("Upload your model history (JSON)", type=['JSON'])

    tab1, tab2, tab3, tab4 = st.tabs(["Exploratory Data Analysis (EDA)", "Model Training", "ANN Grid Search", "Model Evaluation"])

    
    if uploaded_file:
        df = load_data(uploaded_file)

        with tab1:
            data_info = {
                "Variable Name": [
                    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRatio", 
                    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity", 
                    "Roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", 
                    "ShapeFactor4", "Class"
                ],
                "Role": [
                    "Feature", "Feature", "Feature", "Feature", "Feature", 
                    "Feature", "Feature", "Feature", "Feature", "Feature", 
                    "Feature", "Feature", "Feature", "Feature", "Feature", 
                    "Feature", "Target"
                ],
                "Type": [
                    "Integer", "Continuous", "Continuous", "Continuous", "Continuous", 
                    "Continuous", "Integer", "Continuous", "Continuous", "Continuous", 
                    "Continuous", "Continuous", "Continuous", "Continuous", "Continuous", 
                    "Continuous", "Categorical"
                ],
                "Description": [
                    "Area of a bean zone, measured by the number of pixels within its boundaries",
                    "Bean circumference, representing the length of its border",
                    "Length of the longest line between two points on a bean",
                    "Longest line perpendicular to the main axis",
                    "Ratio between MajorAxisLength and MinorAxisLength",
                    "Eccentricity of an ellipse with the same moments as the bean region",
                    "Number of pixels in the smallest convex polygon encompassing the bean",
                    "Equivalent diameter of a circle with the same area as the bean",
                    "Ratio of pixels in the bounding box to the bean area",
                    "Convexity ratio: pixels in the convex shell relative to pixels in the bean",
                    "Roundness, calculated as (4πA) / P²",
                    "Roundness measurement in units of Ed/L",
                    "Shape descriptor derived from the bean geometry",
                    "Additional shape descriptor",
                    "Additional shape descriptor",
                    "Additional shape descriptor",
                    "Bean type: (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira)"
                ],
                "Units": [
                    "pixels", "", "", "", "", 
                    "", "pixels", "", "", "", 
                    "", "Ed/L", "", "", "", 
                    "", ""
                ],
                "Missing Values": [
                    "No", "No", "No", "No", "No", 
                    "No", "No", "No", "No", "No", 
                    "No", "No", "No", "No", "No", 
                    "No", "No"
                ]
            }

            # Create a DataFrame
            df_info = pd.DataFrame(data_info)

            # Display the table in Streamlit
            st.title("Dataset Column Descriptions")
            st.write("This table explains each column in the dataset along with its role, data type, description, units, and information on missing values.")
            st.table(df_info)

            display_raw_data(df)
                
            # Drop selected columns
            df = drop_columns(df)
                
            # Handle missing values
            if df.isnull().sum().sum() > 0:
                df = handle_missing_values(df)
                
            visualize_data(df)
                
            # Store the processed DataFrame
            st.session_state['processed_data'] = df
            prepare_data(df, target="Class")

        with tab2:
            if 'processed_data' in st.session_state:
                X_train, X_test, y_train, y_test, label_encoder = st.session_state['prepared_data']

                st.subheader("Simple model config")

                with st.echo():
                    model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(7, activation='softmax')  # Softmax for multi-class classification
                    ])
                    # Compile the model with categorical loss
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                # Define and Train ANN Model
                st.subheader("Train Simple ANN Model")
                if st.button("Train ANN Model"):
                    model = define_simple_model(X_train)  # Pass input shape to model definition
                    history, y_pred = train_simple_model(model, X_train, X_test, y_train, y_test)
                        
                    st.write("Model training complete.")

                    # Prompt for save path
                    save_path = st.text_input("Enter path to save model and history", "models/First")
                        
                    # Save the model and history if a save path is provided
                    if st.button("Save Model and History"):
                        save_simple_model(model, history, save_path)

        with tab3:
            st.subheader("ANN Grid Search with Custom Parameters")

            if 'processed_data' in st.session_state:
                # Prepare data before grid search
                X_train, X_test, y_train, y_test, label_encoder = st.session_state['prepared_data']

                # Define parameters for grid search with user input
                st.write("Select Grid Search Parameters:")
                
                # Single choice for num_layers to select the total amount of layers
                num_layers = st.selectbox("Number of Layers", options=[1, 2, 3, 4, 5, 6], index=0)
                
                # Multiple choices for other parameters
                learning_rates = st.multiselect("Learning Rates", options=[0.001, 0.01, 0.1], default=[0.001, 0.01])
                activations = st.multiselect("Activations", options=['relu', 'tanh', 'sigmoid'], default=['relu', 'tanh'])
                neurons = st.multiselect("Neurons per Layer", options=[32, 64, 128, 256, 512, 1024], default=[64, 128])

                # Build the parameter grid dynamically from the selected values
                param_grid = {
                    'num_layers': [num_layers],  # single choice from `selectbox`
                    'learning_rate': learning_rates,  # flattened list from `multiselect`
                    'activation': activations,  # flattened list from `multiselect`
                    'neurons': neurons  # flattened list from `multiselect`
                }

                # Create a stop flag
                stop_flag = st.button("Stop Grid Search")

                # Run Grid Search with custom parameters
                if st.button("Run Grid Search"):
                    st.write("Running Grid Search...")

                    def check_stop():
                        return stop_flag

                    grid_result = train_ann_grid(param_grid, X_train, y_train, stop_flag=check_stop)

                    if grid_result:
                        st.write("Best Parameters:", grid_result.best_params_)
                        st.write("Best Score:", grid_result.best_score_)
                        st.write("Grid Search Complete")
                    else:
                        st.write("Grid search was stopped.")
            else:
                st.warning("Please complete the EDA and Data Preparation steps in previous tabs.")


        with tab4:
            if 'processed_data' in st.session_state and uploaded_model and uploaded_history:
                X_train, X_test, y_train, y_test, label_encoder = st.session_state['prepared_data']

                file_path = st.text_input("Enter the path to your files (both files has to be in the same folder)", "models/")

                model, history, y_pred, report_df = prepare_model_history(file_path, uploaded_model.name, uploaded_history.name, X_test, y_test, label_encoder)
                evaluate_model(uploaded_model.name, history, report_df, y_test, y_pred)

if __name__ == "__main__":
    main()