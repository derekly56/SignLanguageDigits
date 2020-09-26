import os 
import numpy as np
import pandas as pd
from PIL import Image

def convert(dir: str) -> np.ndarray:
    '''Iterate through the given directory and convert each image
    into a numpy array

    Args:
        dir (string): Name of directory
    
    Returns:
        images_arr (np.ndarray): Images array containing converted Image files
    '''
    dir_encode = os.fsencode(dir)
    images_arr = []
    shape = (100, 100)

    for file in os.listdir(dir_encode):
        filename = os.fsdecode(file)

        if filename.endswith(".JPG"):
            img = Image.open(dir + "/" + filename).resize(shape)
            numpy_arr = np.array(img).flatten()
            images_arr.append(numpy_arr)

    return np.array(images_arr)

def save_as_csv(file_name: str, df: pd.DataFrame):
    '''Save images array into a .CSV file
    
    Args:
        file_name (string): Name of file to save to
        df (pd.DataFrame): Dataframe containing images data
    '''
    df.to_csv(file_name)

def main():
    directories = ["digits/0","digits/1","digits/2","digits/3","digits/4","digits/5","digits/6","digits/7","digits/8","digits/9"]
    images = convert(directories[0])
    file_name = 'processed_digits.csv'

    for i in range(1, len(directories)):
        converted_images = convert(directories[i])
        images = np.concatenate([images, converted_images])
    
    images_ = [image for image in images]

    df = pd.DataFrame({'image': images_}, columns=['image'])
    save_as_csv(file_name, df)


if __name__ == "__main__":
    main()
