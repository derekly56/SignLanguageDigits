import os 
import numpy as np
from PIL import Image

def convert(dir):
    dir_encode = os.fsencode(dir)
    images_arr = []

    for file in os.listdir(dir_encode):
        filename = os.fsdecode(file)
        if filename.endswith(".JPG"):
            img = Image.open(dir + "/" + filename)
            numpy_arr = np.array(img)
            images_arr.append(numpy_arr)
    return images_arr

def main():
    directories = ["0","1","2","3","4","5","6","7","8","9"]
    images = np.array(convert(directories[0]))
    for i in range(1, len(directories)):
        images = np.concatenate([images, np.array(convert(directories[i]))])
    print(images.shape)

if __name__ == "__main__":
    main()
