from util import images_in_directory, write_images_to_directory
import numpy as np
import cv2

NUM_WINDOWS=9
MARGIN=100
MINPIX=50

class BinaryImage:
    @classmethod
    def in_threshold(cls, image, min, max):
        result = np.zeros_like(image)
        result[(image > min) & (image <= max)] = 1
        return cls(result)

    def __init__(self, binary_image): 
        self.binary_image = binary_image
        (nonzero_y, nonzero_x) = binary_image.nonzero()
        self.nonzero_xy = (nonzero_x, nonzero_y)

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

    @property
    def midpoint_x(self):
        (_, width) = self.binary_image.shape
        return width//2

    @property
    def midpoint_y(self):
        (height, _) = self.binary_image.shape
        return height//2
        
    @property
    def lower_left(self):
        return self.binary_image[self.midpoint_y:,:self.midpoint_x]

    @property
    def lower_right(self):
        return self.binary_image[self.midpoint_y:,self.midpoint_x:]

    def peak_in(self, part):
        histogram = np.sum(part, axis=0)
        indices = np.flatnonzero(histogram == np.amax(histogram))
        return np.int(np.mean(indices))

    def lane_bases(self):
        left_base = self.peak_in(self.lower_left)
        right_base = self.midpoint_x + self.peak_in(self.lower_right)
        return (left_base, right_base)

    def window_y_positions(self):
        (height, _) = self.binary_image.shape
        window_height = height//NUM_WINDOWS
        return np.array([(height - window*window_height, height-(window+1)*window_height) for window in range(NUM_WINDOWS)])

    def to_rgb_image(self):
        return np.dstack((self.binary_image, self.binary_image, self.binary_image)) * 255


class Window:
    def __init__(self, binary_image, x_midpoint, y_range, margin = MARGIN):
        self.binary_image = binary_image
        self.initial_x_midpoint = x_midpoint
        (self.x_min, self.x_max) = (x_midpoint - margin, x_midpoint + margin)
        (self.y_max, self.y_min) = y_range

    def draw_on_image(self, image, color=[255,0,0], thickness=2):
        (nonzero_x, nonzero_y) = self.nonzero_xy
        image[nonzero_y, nonzero_x] = color
        cv2.rectangle(image, (self.x_min, self.y_min), (self.x_max, self.y_max), [0,255,0], thickness)

    @property
    def nonzero_xy(self):
        (nonzero_x, nonzero_y) = self.binary_image.nonzero_xy
        indices = np.flatnonzero((nonzero_x >= self.x_min) &
             (nonzero_x < self.x_max) &
             (nonzero_y >= self.y_min) &
             (nonzero_y < self.y_max))
        return (nonzero_x[indices], nonzero_y[indices])

    @property
    def new_x_midpoint(self):
        (nonzero_x, _) = self.nonzero_xy
        if len(nonzero_x) > MINPIX:
            return np.int(np.mean(nonzero_x))
        return self.initial_x_midpoint
    
class Lane:
    def __init__(self, binary_image, initial_x_midpoint):
        self.binary_image = binary_image
        self.initial_x_midpoint = initial_x_midpoint

    @property
    def all_windows(self):
        x_midpoint = self.initial_x_midpoint
        for y_range in self.binary_image.window_y_positions():
            window = Window(self.binary_image, x_midpoint, y_range)
            yield window
            x_midpoint = window.new_x_midpoint

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
    return ((red & saturation) | sobel)

def annotate_lane(binary):
    rgb_image = binary.to_rgb_image()
    (left_x, right_x) = binary.lane_bases()
    left_lane = Lane(binary, left_x)
    right_lane = Lane(binary, right_x)
    for w in left_lane.all_windows:
        w.draw_on_image(rgb_image, [255, 0, 0])
    for w in right_lane.all_windows:
        w.draw_on_image(rgb_image, [0, 0, 255])
    return rgb_image

if __name__ == "__main__":
    binary_images = [filter(image) for image in images_in_directory('output_images/bird_view')] 
    write_images_to_directory([b.to_rgb_image() for b in binary_images], 'binary_images')
    write_images_to_directory([annotate_lane(b) for b in binary_images], 'annotated_binary_images')
