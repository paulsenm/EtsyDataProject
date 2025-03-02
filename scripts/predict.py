# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:29:27 2025

@author: MichaelPaulsen
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from scripts.etsy_data_convert import preprocess_text


def load_model(model_file, vectorizer_file, scaler_file):
    model = tf.keras.models.load_model(model_file)
    # with open(model_file, "rb") as f:
    #     model = pickle.load(f)
        
    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)

    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    return model, vectorizer, scaler


def predict_num_reviews(title, model_file, vectorizer_file, scaler_file):
    model, vectorizer, scaler = load_model(model_file, vectorizer_file, scaler_file)
    
    
    cleaned_title = preprocess_text(title)
    title_vectorizer = vectorizer.transform([cleaned_title]).toarray()
    
    feature_names = [str(i) for i in range(title_vectorizer.shape[1])]
    X_input = pd.DataFrame(title_vectorizer, columns = feature_names)
    X_input_scaled = scaler.transform(X_input)
    
    #predicted_reviews = model.predict(X_input)[0]
    predicted_reviews = model.predict(X_input_scaled)[0][0]
    return predicted_reviews

if __name__ == "__main__":
    title = "silver ring"
    model_file_cpu = "../models/model.pk1"
    vectorizer_file = "../models/vectorizer.pk1"

    model_file_gpu = "../models/model.h5"
    scaler_file = "../models/scaler.pk1"

    predicted_reviews = predict_num_reviews(title, model_file_gpu, vectorizer_file, scaler_file)
    
    print(f"predicted number of reviews: {predicted_reviews}" )