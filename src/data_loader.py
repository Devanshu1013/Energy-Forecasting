import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop(columns=['date', 'rv1', 'rv2'])  # drop random noise cols
    
    df = df.fillna(df.mean())
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler, df.columns.tolist()

def train_val_test_split(data, train=0.7, val=0.15):
    n = len(data)
    return (data[:int(n*train)],
            data[int(n*train):int(n*(train+val))],
            data[int(n*(train+val)):])