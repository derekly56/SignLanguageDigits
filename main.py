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
    process_data(df)

if __name__ == "__main__":
    main()