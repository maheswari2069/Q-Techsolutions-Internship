import pandas as pd

def load_data(filepath="data/amazon.csv"):
    return pd.read_csv(filepath)
