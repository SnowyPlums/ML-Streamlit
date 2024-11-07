import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

def plot_classification_matrix(report_df, model_name):
    """Display classification metrics in a bar chart."""
    # Convert report dictionary to DataFrame
    report_df = pd.DataFrame(report_df).transpose()
    
    # Reset index to prepare for melt
    melted_report = report_df.reset_index().melt(id_vars='index', var_name='Metric', value_name=model_name)
    
    # Plot the classification metrics
    fig = px.bar(
        melted_report,
        x='index',
        y=model_name,
        color='Metric',
        barmode='group',
        title=f"Classification Metrics for {model_name}"
    )
    
    st.plotly_chart(fig)

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot a confusion matrix for the given model."""
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="Blues", 
                    title=f"Confusion Matrix for {model_name}", labels=dict(x="Predicted Label", y="True Label"))
    
    st.plotly_chart(fig)

def plot_training_history(history, model_name):
    """Plot training history for accuracy and loss."""
    # Plot accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=history['val_accuracy'], mode='lines', name=f'{model_name} Validation Accuracy'))
    fig_acc.update_layout(title=f'Validation Accuracy - {model_name}', xaxis_title='Epoch', yaxis_title='Accuracy')
    st.plotly_chart(fig_acc)

    # Plot loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name=f'{model_name} Validation Loss'))
    fig_loss.update_layout(title=f'Validation Loss - {model_name}', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig_loss)
