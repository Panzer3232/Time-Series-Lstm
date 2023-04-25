import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Step 2: Load and preprocess the dataset
print("Loading and preprocessing the dataset...")
def preprocess_data(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler

def create_sequences(data, seq_length):
    inputs, targets = [], []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(inputs), np.array(targets)

# Load the dataset from CSV
data = pd.read_csv("rainfall_data.csv")
data = data.drop("Year", axis=1)

# Preprocess and split the dataset into training and testing sets
data, scaler = preprocess_data(data.values)
seq_length = 4
inputs, targets = create_sequences(data, seq_length)
train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, shuffle=False)