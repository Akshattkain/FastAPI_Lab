import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    """
    Loading the Heart Disease dataset from UCI repository.
    Returns:
        X (numpy.ndarray): The features of the Heart Disease dataset.
        y (numpy.ndarray): The target values (0 = no disease, 1 = disease).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    df = pd.read_csv(url, names=columns, na_values='?')
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1-4 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test