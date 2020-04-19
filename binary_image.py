from util import images_in_directory, write_images_to_directory
import numpy as np
import cv2

class BinaryImage:
    @classmethod
    def in_threshold(cls, image, min, max):
        result = np.zeros_like(image)
        result[(image > min) & (image <= max)] = 1
        return cls(result)

    def __init__(self, binary_image): 
        self.binary_image = binary_image

    def __or__(self, other):
        result = np.zeros_like(self.binary_image)
        result[(self.binary_image == 1) | (other.binary_image == 1)] = 1
        return BinaryImage(result)

    def __and__(self, other):
        result = np.zeros_like(self.binary_image)
        result[(self.binary_image == 1) & (other.binary_image == 1)] = 1
        return BinaryImage(result)

    def __eq__(self, other):
        return np.array_equal(self.binary_image, other.binary_image)

def absolute_sobel(image, orient='x', sobel_kernel=3):
    orientation = [1, 0] if orient == 'x' else [0, 1]
    derivative = cv2.Sobel(image, cv2.CV_64F, *orientation, ksize=sobel_kernel)
    absolute_derivative = np.absolute(derivative)
    return np.uint8(255 * absolute_derivative/np.max(absolute_derivative))

def filter(image):
    red_image = image[:,:,0]
    saturation_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
    sobel_image = absolute_sobel(red_image, 'x', 9)
    red = BinaryImage.in_threshold(red_image, 40, 255)
    saturation = BinaryImage.in_threshold(saturation_image, 130, 255)
    sobel = BinaryImage.in_threshold(sobel_image, 50, 255)
    return ((red & saturation) | sobel).binary_image

if __name__ == "__main__":
    binary_images = [filter(image) for image in images_in_directory('output_images/bird_view')] 
    write_images_to_directory(binary_images, 'binary_images', cmap='gray')

