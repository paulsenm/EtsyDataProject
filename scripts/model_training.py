# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:28:51 2025

@author: MichaelPaulsen
"""

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(input_csv, model_file):
    df = pd.read_csv(input_csv)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
        
    
if __name__ == "__main__":
    train_model("../data/processed_data.csv", "../models/model.pk1")