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
    directories = ["digits/0","digits/1","digits/2","digits/3","digits/4","digits/5","digits/6","digits/7","digits/8","digits/9"]
    images = np.array(convert(directories[0]))
    for i in range(1, len(directories)):
        images = np.concatenate([images, np.array(convert(directories[i]))])
    print(images.shape)

if __name__ == "__main__":
    main()
