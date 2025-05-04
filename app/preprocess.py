import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].map({'neg': 0, 'pos': 1})  # Convert labels to binary (negative = 0, positive = 1)
    return train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
