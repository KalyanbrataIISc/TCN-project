import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_dim, hidden_size):
    return models.Sequential([
        layers.Input((input_dim,)),
        layers.Dense(hidden_size, activation="relu"),
        layers.Dense(hidden_size, activation="relu"),
        layers.Dense(hidden_size, activation="relu"),
        layers.Dense(2, activation="linear")
    ])