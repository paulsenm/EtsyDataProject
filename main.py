# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:29 2025

@author: MichaelPaulsen
"""

from scripts.etsy_data_convert import process_data
from scripts.model_training import train_model
from scripts.predict import predict_num_reviews

if __name__ == "__main__":
    input_csv = "data/etsy.csv"
    processed_csv = "data/processed_data.csv"
    vectorizer_file = "models/vectorizer.pkl"
    model_file_cpu = "models/model.pkl"
    model_file_gpu = "models/model.h5"
    scaler_file = "models/scaler.pk1"

    # Step 1: Preprocess Data
    process_data(input_csv, processed_csv, vectorizer_file)

    # Step 2: Train Model
    train_model(processed_csv, model_file_gpu, scaler_file)

    # Step 3: Predict Example
    title = "earrings"
    num_reviews = 30
    predicted_rating = predict_num_reviews(title, model_file_gpu, vectorizer_file, scaler_file)
    print(f"Predicto-bot-5000 thinks that the title: {title} will generate {predicted_rating:.0f} reviews.")
