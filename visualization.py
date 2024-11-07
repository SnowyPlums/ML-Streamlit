import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st  # Import Streamlit for chart embedding

def plot_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Generate correlation matrix and reset index for plotting
    correlation_matrix = df.drop(columns=['Class']).corr().reset_index().melt(id_vars="index")

    # Rename columns for Plotly compatibility
    correlation_matrix.columns = ['Feature1', 'Feature2', 'Correlation']

    # Plot heatmap
    fig = px.imshow(
        df.drop(columns=['Class']).corr(),
        color_continuous_scale='purples',
        aspect='auto',
        title="Feature Correlation Matrix",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    st.plotly_chart(fig)  # Display chart in Streamlit

def plot_bean_counts(df, bean_column='Class'):
    """Plot the count of each bean type."""
    fig = px.histogram(df, x=bean_column, color=bean_column, title="Count of Each Bean Type",
                       color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Bean Type", yaxis_title="Count")
    st.plotly_chart(fig)  # Display chart in Streamlit


def plot_bean_pairplot(df, bean_column='Class', features=None):
    """
    Plot pairwise relationships for the specified features and color by bean type.
    """
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()[:5]  # Select first 5 numerical features by default

    fig = px.scatter_matrix(df, dimensions=features, color=bean_column, title="Pair Plot of Selected Features by Bean Type",
                            color_continuous_scale=px.colors.sequential.Viridis, opacity=0.5)
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)  # Display chart in Streamlit


def plot_bean_pca(df, bean_column='Class'):
    """
    Perform PCA to reduce dimensions to 2D and plot the beans in 2D space.
    """
    features = df.select_dtypes(include='number').columns.tolist()
    X = df[features]
    y = df[bean_column]

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Convert PCA results into a DataFrame for easy plotting
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df[bean_column] = y

    fig = px.scatter(pca_df, x='PC1', y='PC2', color=bean_column, title="PCA Plot of Beans (2D)",
                     color_discrete_sequence=px.colors.sequential.Viridis, opacity=0.7)
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    st.plotly_chart(fig)  # Display chart in Streamlit


def plot_bean_boxplot(df, bean_column='Class', feature='feature_name'):
    """
    Plot a box plot for a specific feature across different bean types.
    """
    fig = px.box(df, x=bean_column, y=feature, color=bean_column,
                 title=f"Distribution of {feature} by Bean Type", color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Bean Type", yaxis_title=feature)
    st.plotly_chart(fig)  # Display chart in Streamlit


def plot_bean_violinplot(df, bean_column='Class', feature='feature_name'):
    """
    Plot a violin plot for a specific feature across different bean types.
    """
    fig = px.violin(df, x=bean_column, y=feature, color=bean_column,
                    title=f"Violin Plot of {feature} by Bean Type", box=True, points="all",
                    color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Bean Type", yaxis_title=feature)
    st.plotly_chart(fig)  # Display chart in Streamlit