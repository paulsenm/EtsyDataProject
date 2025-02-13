import pandas as pd
import re

from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import numpy as np

import os







def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    
    return text

def word_count(text):
    if pd.isna(text):
        return 0
    return len(text.split())


def process_data(input_csv, output_csv, vectorizer_file):
    df = pd.read_csv(input_csv)
    
    df["cleaned_name"] = df["name"].apply(preprocess_text)
    df["description_length"] = df["description"].apply(word_count)
    
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df = df[(df["average_rating"] >= 0) & (df["average_rating"] <= 5)]
    
    df["reviews_count"] = pd.to_numeric(df["reviews_count"], errors="coerce")
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df["cleaned_name"]).toarray()
    
    os.makedirs(os.path.dirname(vectorizer_file), exist_ok=True)
    
    with open(vectorizer_file, "wb") as f:
        pickle.dump(vectorizer, f)
        
    X_numerical = df[["reviews_count", "average_rating"]].fillna(0).values
    X = np.hstack((X_text, X_numerical))
    
    processed_df = pd.DataFrame(X)
    processed_df["average_rating"] = df["average_rating"]
    processed_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = "data/etsy.csv"
    output_csv = "data/processed_data.csv"
    vectorizer_file = "models/vectorizer.pkl"
    
    process_data(input_csv, output_csv, vectorizer_file)
    