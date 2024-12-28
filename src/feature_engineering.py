import numpy as np
import pandas as pd

# Add cyclic features for seasonality
def add_cyclic_features(data, date_column):
    data['month_sin'] = np.sin(2 * np.pi * data[date_column].dt.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data[date_column].dt.month / 12)
    return data

# Create sequences for time-series modeling
def create_sequences(data, sequence_length=7):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i + sequence_length])  # Predict the next day's summary
    return np.array(sequences), np.array(targets)
