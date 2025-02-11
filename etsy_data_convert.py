import pandas as pd
import re

etsy_csv_path = "etsy.csv"
etsy_df = pd.read_csv(etsy_csv_path)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().split()
    
    return text

def word_count(text):
    word_count = len(text.split())
    return word_count

etsy_df["cleaned_name"] = etsy_df["name"].apply(preprocess_text)
etsy_df["description_length"] = etsy_df["description"].apply(word_count)
print(etsy_df.head())