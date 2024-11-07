import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import streamlit as st
import json


class HistorySaver(Callback):
    def __init__(self):
        super().__init__()
        self.history = None

    def on_train_end(self, logs=None):
        # Store the history after each model finishes training
        self.history = self.model.history.history


def create_model(input_shape=(16,), num_layers=2, neurons=64, learning_rate=0.001, activation='relu'):
    """
    Dynamically create a model based on the number of layers and neurons.
    """
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=input_shape))
    
    # Add hidden layers
    for i in range(1, num_layers):
        model.add(Dense(max(neurons // (2 ** i), 4), activation=activation))  # Decrease neurons per layer
    
    # Output layer (assuming 7 classes in your problem)
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model




def train_ann_grid(param_grid, X_train, y_train, stop_flag):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_saver = HistorySaver()

    # Wrap the model in KerasClassifier
    model = KerasClassifier(build_fn=create_model, input_shape=(X_train.shape[1],), verbose=0)

    # Initialize GridSearchCV with StratifiedKFold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')

    fit_params = {
        'callbacks': [early_stopping, history_saver],
        'validation_split': 0.2  # Validation split for each fold
    }

    try:
        grid_result = grid.fit(X_train, y_train, **fit_params)
        if stop_flag():
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Grid search stopped by user.")
        return None
    
    return grid_result, history_saver

def save_grid_search(grid_result, history_saver, custom_path):
    model = grid_result.best_estimator_.model
    history = history_saver.history

    model.save(custom_path + "_model.h5")

    with open(custom_path + "_history.json", "w") as f:
        json.dump(history, f)

    st.success(f"Model and history saved to {custom_path}_model.h5 and {custom_path}_history.json")