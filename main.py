import tensorflow as tf
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from ml_model.digits_neural_network import DigitNeuralNetwork
from utils import create_dataframe, one_hot_encode, process_data

def main():
    file_name = 'data/processed_digits.csv'
    df = create_dataframe(file_name)
    X_train, y_train, X_valid, y_valid, X_test, y_test = process_data(df)

    DigitNN = DigitNeuralNetwork(epochs=100, batch_size=32)
    DigitNN.fit(X_train, y_train, X_valid, y_valid, X_test, y_test)

if __name__ == "__main__":
    main()