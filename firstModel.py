import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import streamlit as st
import json

class StreamlitProgressCallback(Callback):
    def __init__(self, epochs, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("accuracy", 0)
        val_accuracy = logs.get("val_accuracy", 0)
        
        # Update progress bar based on the current epoch (no need to increment beyond total epochs)
        self.progress_bar.progress((epoch + 1) / self.epochs)
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} - "
                              f"accuracy: {accuracy:.4f} - val_accuracy: {val_accuracy:.4f}")

def define_simple_model(X_train):
    # Define the ANN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(7, activation='softmax')  # Softmax for multi-class classification
    ])

    # Compile the model with categorical loss
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_simple_model(model, X_train, X_test, y_train, y_test):
    # Initialize Streamlit progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Define the Streamlit progress callback
    epochs = 50  # Set your number of epochs here
    progress_callback = StreamlitProgressCallback(epochs, progress_bar, status_text)

    # Train the model with the custom callback
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                        validation_split=0.2, verbose=0, callbacks=[progress_callback])
    
    # Clear progress elements after training completes
    progress_bar.empty()
    status_text.text("Training complete!")

    # Evaluate and predict
    y_pred = model.predict(X_test).argmax(axis=1)
    return history, y_pred

def save_simple_model(model, history, custom_path="models/First"):
    """
    Save the model and history to the specified path.

    Parameters:
    - model: Trained Keras model
    - history: Training history object
    - custom_path: Base path for saving model and history files
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(custom_path), exist_ok=True)

    # Save model
    model.save(f"{custom_path}_model.h5")

    # Save history
    with open(f"{custom_path}_history.json", "w") as f:
        json.dump(history.history, f)
    
    st.success(f"Model and history saved to {custom_path}_model.h5 and {custom_path}_history.json")