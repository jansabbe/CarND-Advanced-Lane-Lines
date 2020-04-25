import cv2
import glob
from os import makedirs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import re

def number_in_filename(path):
    return int(re.sub('[^0-9]+', '', path))

def image_size(img):
    (height, width) = img.shape[:2]
    return (width, height)

def images_in_directory(directory):
    return [mpimg.imread(filename) for filename in sorted(glob.glob(f'{directory}/*.jpg'), key=number_in_filename)]

def write_images_to_directory(images, name, **kwargs):
    makedirs(f'output_images/{name}', exist_ok=True)
    for idx, image in enumerate(images):
        write_name = f'output_images/{name}/{name}_{idx+1}.jpg'
        mpimg.imsave(write_name, image, **kwargs)

RED = [255,0,0][::-1]


def show_side_by_side(images, edit_function, **kwargs):
    fig = plt.figure(figsize=(15, 30))
    for index,original_image in enumerate(images):
        fig.add_subplot(len(images), 2, (2*index)+1)
        plt.imshow(original_image)
        fig.add_subplot(len(images), 2, (2*index)+2)
        plt.imshow(edit_function(np.copy(original_image)), **kwargs)
    plt.show()
