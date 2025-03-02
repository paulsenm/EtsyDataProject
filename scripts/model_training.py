# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:28:51 2025

@author: MichaelPaulsen
"""

import pickle
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from  keras import layers

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def train_model(input_csv, model_file, scaler_file):
    df = pd.read_csv(input_csv)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    
    # y_pred = model.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    with tf.device("/GPU:0"):
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"mean abs error: {mae: .2f}")

    model.save(model_file)


    # with open(model_file, "wb") as f:
    #     pickle.dump(model, f)
        
    
if __name__ == "__main__":
    #train_model("../data/processed_data.csv", "../models/model.pk1")
    train_model("../data/processed_data.csv", "../models/model.h5", "../models/scaler.pk1")