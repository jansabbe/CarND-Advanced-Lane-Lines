import cv2
import glob
from os import makedirs

def image_size(img):
    (height, width) = img.shape[:2]
    return (width, height)

def images_in_directory(directory):
    return [cv2.imread(filename) for filename in glob.glob(f'{directory}/*.jpg')]

def write_images_to_directory(images, name):
    makedirs(f'output_images/{name}', exist_ok=True)
    for idx, image in enumerate(images):
        write_name = f'output_images/{name}/{name}_{idx+1}.jpg'
        cv2.imwrite(write_name, image)

RED = [255,0,0][::-1]