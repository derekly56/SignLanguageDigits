import tensorflow as tf
import pandas as pd
import numpy as np
import math

def create_dataframe(file_name: str) -> pd.DataFrame:
    '''Opens up .CSV file and create a dataframe
    
    Args:
        file_name (string): Name of file
    
    Returns:
        df (pd.DataFrame): Dataframe of digit dataset
    '''
    df = pd.read_csv(file_name)
    return df

def one_hot_encode(y_labels: np.ndarray):
    '''Converts labels into one hot encoding
    
    Args:
        y_labels (np.ndarray): Label dataset
    
    Returns:
        one_hot_labels (np.ndarray): Label dataset  with one-hot-encoding
    '''
    one_hot_labels = np.zeros((y_labels.size, y_labels.max() + 1))
    one_hot_labels[np.arange(y_labels.size), y_labels] = 1

    return one_hot_labels

def process_data(df):
    '''Processes data into train, validation, and test
    
    Args:
        df (pd.DataFrame): Dataframe containing images dataseet
    
    Returns:
        X_train (np.ndarray): Training dataset
        y_train (np.ndarray): Training label dataset
        X_valid (np.ndarray): Validation dataset
        y_valid (np.ndarray): Validation label dataset
        X_test (np.ndarray): Testing dataset
        y_test (np.ndarray): Testing label dataset
    '''
    images = df['image'].values
    labels = df['label'].values

    images /= 255

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)