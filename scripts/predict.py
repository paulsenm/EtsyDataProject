# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:29:27 2025

@author: MichaelPaulsen
"""

import pickle
import numpy as np
import pandas as pd

from scripts.etsy_data_convert import preprocess_text


def load_model(model_file, vectorizer_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
        
    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_num_reviews(title, model_file, vectorizer_file):
    model, vectorizer = load_model(model_file, vectorizer_file)
    
    
    cleaned_title = preprocess_text(title)
    title_vectorizer = vectorizer.transform([cleaned_title]).toarray()
    
    feature_names = [str(i) for i in range(title_vectorizer.shape[1])]
    X_input = pd.DataFrame(title_vectorizer, columns = feature_names)
    
    predicted_reviews = model.predict(X_input)[0]
    return predicted_reviews

if __name__ == "__main__":
    title = "bla bla"
    model_file = "../models/model.pk1"
    vectorizer_file = "../models/vectorizer.pk1"
    predicted_reviews = predict_num_reviews(title, model_file, vectorizer_file)
    
    print(f"predicted number of reviews: {predicted_reviews}" )