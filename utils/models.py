"""
Deep Learning Models Module (LSTM & GRU)
Implements LSTM and GRU architectures, compilation, callbacks, and training pipelines.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape: tuple = (60, 1), learning_rate: float = 0.001) -> Sequential:
    """
    Builds the production LSTM architecture specified for stock price prediction.
    Architecture:
      Input (shape: input_shape)
      LSTM(64, return_sequences=True)
      Dropout(0.2)
      LSTM(32)
      Dropout(0.2)
      Dense(16, activation="relu")
      Dense(1)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, name="lstm_layer_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(32, return_sequences=False, name="lstm_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(16, activation="relu", name="dense_16"),
        Dense(1, name="output_close_price")
    ], name="LSTM_Stock_Predictor")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


def build_gru_model(input_shape: tuple = (60, 1), learning_rate: float = 0.001) -> Sequential:
    """
    Builds the production GRU architecture specified for stock price prediction.
    Architecture:
      Input (shape: input_shape)
      GRU(64, return_sequences=True)
      Dropout(0.2)
      GRU(32)
      Dropout(0.2)
      Dense(16, activation="relu")
      Dense(1)
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True, name="gru_layer_1"),
        Dropout(0.2, name="dropout_1"),
        GRU(32, return_sequences=False, name="gru_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(16, activation="relu", name="dense_16"),
        Dense(1, name="output_close_price")
    ], name="GRU_Stock_Predictor")
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


def train_model(
    model: Sequential,
    X_train,
    y_train,
    model_name: str,
    save_path: str,
    epochs: int = 60,
    batch_size: int = 32,
    validation_split: float = 0.10,
    verbose: int = 1
):
    """
    Trains the deep learning model with EarlyStopping, ReduceLROnPlateau,
    and ModelCheckpoint without chronological shuffling.
    
    Parameters:
        model (Sequential): Compiled Keras model.
        X_train: Sequence input features.
        y_train: Target values.
        model_name (str): Identifier (e.g. 'LSTM' or 'GRU').
        save_path (str): Filepath to save the best model weights.
        epochs (int): Maximum training epochs.
        batch_size (int): Mini-batch size.
        validation_split (float): Chronological validation fraction (shuffle=False).
        
    Returns:
        tuple: (trained_model, history_dict)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=verbose,
            mode="min"
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=verbose,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
            mode="min"
        )
    ]
    
    print(f"\n[Training] Starting {model_name} training (Max Epochs: {epochs}, Batch Size: {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,  # Enforce chronological validation split without data leakage
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Reload best model saved by checkpoint
    if os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)
        
    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "mae": [float(x) for x in history.history.get("mae", [])],
        "val_mae": [float(x) for x in history.history.get("val_mae", [])],
        "epochs_trained": len(history.history["loss"])
    }
    
    print(f"[Training] Completed {model_name}. Best Val Loss achieved. Saved to: {save_path}")
    return model, history_dict
