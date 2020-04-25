from util import images_in_directory, write_images_to_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2

MIDPOINT_X = 600

NUM_WINDOWS = 9
MARGIN = 100
MINPIX = 50
SMOOTH_NB_FRAMES = 10
XM_PER_PIX = 3.7 / 800
YM_PER_PIX = 27 / 720


class BinaryImage:

    @classmethod
    def in_threshold(cls, image, min, max):
        result = np.zeros_like(image)
        result[(image > min) & (image <= max)] = 1
        return cls(result)

    def __init__(self, binary_image):
        self.binary_image = binary_image
        (self.height, self.width) = binary_image.shape
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
    def midpoint_y(self):
        return self.height // 2

    def lower_left(self, midpoint_x):
        return self.binary_image[self.midpoint_y:, :midpoint_x]

    def lower_right(self, midpoint_x):
        return self.binary_image[self.midpoint_y:, midpoint_x:]

    def peak_in(self, part):
        histogram = np.sum(part, axis=0)
        indices = np.flatnonzero(histogram == np.amax(histogram))
        return np.int(np.mean(indices))

    def lane_bases(self, midpoint_x=MIDPOINT_X):
        left_base = self.peak_in(self.lower_left(midpoint_x))
        right_base = midpoint_x + self.peak_in(self.lower_right(midpoint_x))
        return left_base, right_base

    def window_y_positions(self):
        height = self.height
        window_height = height // NUM_WINDOWS
        return np.array(
            [(height - window * window_height, height - (window + 1) * window_height) for window in range(NUM_WINDOWS)])

    def to_rgb_image(self):
        return np.dstack((self.binary_image, self.binary_image, self.binary_image)) * 255


class Window:
    def __init__(self, binary_image, x_midpoint, y_range, margin=MARGIN):
        self.binary_image = binary_image
        self.initial_x_midpoint = x_midpoint
        (self.x_min, self.x_max) = (x_midpoint - margin, x_midpoint + margin)
        (self.y_max, self.y_min) = y_range

    def draw_on_image(self, image, color=[255, 0, 0], thickness=2):
        (nonzero_x, nonzero_y) = self.nonzero_xy
        image[nonzero_y, nonzero_x] = color
        cv2.rectangle(image, (self.x_min, self.y_min), (self.x_max, self.y_max), [0, 255, 0], thickness)

    @property
    def nonzero_xy(self):
        (nonzero_x, nonzero_y) = self.binary_image.nonzero_xy
        indices = np.flatnonzero((nonzero_x >= self.x_min) &
                                 (nonzero_x < self.x_max) &
                                 (nonzero_y >= self.y_min) &
                                 (nonzero_y < self.y_max))
        return nonzero_x[indices], nonzero_y[indices]

    @property
    def new_x_midpoint(self):
        (nonzero_x, _) = self.nonzero_xy
        if len(nonzero_x) > MINPIX:
            return np.int(np.mean(nonzero_x))
        return self.initial_x_midpoint


class PointFinder:
    def __init__(self, binary_image, initial_x_midpoint, previous_polynomial, margin=MARGIN):
        self.binary_image = binary_image
        self.initial_x_midpoint = initial_x_midpoint
        self.previous_polynomial = previous_polynomial
        self.margin = margin

    def find_nonzero_xy(self):
        if self.previous_polynomial is not None:
            return self.nonzero_xy_around_polynomial()
        return self.nonzero_xy_using_windows()

    def nonzero_xy_around_polynomial(self):
        (nonzero_x, nonzero_y) = self.binary_image.nonzero_xy
        (a, b, c) = self.previous_polynomial

        indices = np.flatnonzero((nonzero_x > (a * nonzero_y ** 2 + b * nonzero_y + c - self.margin))
                                 & (nonzero_x < (a * nonzero_y ** 2 + b * nonzero_y + c + self.margin)))
        return nonzero_x[indices], nonzero_y[indices]

    @property
    def all_windows(self):
        x_midpoint = self.initial_x_midpoint
        for y_range in self.binary_image.window_y_positions():
            window = Window(self.binary_image, x_midpoint, y_range)
            yield window
            x_midpoint = window.new_x_midpoint

    def nonzero_xy_using_windows(self):
        nonzero_xys = [w.nonzero_xy for w in self.all_windows]
        return (np.concatenate([nonzero_xy[0] for nonzero_xy in nonzero_xys]),
                np.concatenate([nonzero_xy[1] for nonzero_xy in nonzero_xys]))


class Lane:
    def __init__(self, binary_image, initial_x_midpoint, previous_polynomial=None):
        self.binary_image = binary_image
        self.initial_x_midpoint = initial_x_midpoint
        self.point_finder = PointFinder(binary_image, initial_x_midpoint, previous_polynomial)

    @property
    def nonzero_xy(self):
        return self.point_finder.find_nonzero_xy()

    @property
    def all_windows(self):
        return self.point_finder.all_windows

    @property
    def polynomial(self):
        (lane_x, lane_y) = self.nonzero_xy
        return np.polyfit(lane_y, lane_x, 2)

    @property
    def scaled_polynomial(self):
        (lane_x, lane_y) = self.nonzero_xy
        return np.polyfit(YM_PER_PIX * lane_y, XM_PER_PIX * lane_x, 2)

    def polynomial_xy(self):
        ploty = np.linspace(0, self.binary_image.height - 1, self.binary_image.height)
        (a, b, c) = self.polynomial
        lanex = a * ploty ** 2 + b * ploty + c
        return lanex, ploty

    def polynomial_xy_for_cv2(self):
        (plotx, ploty) = self.polynomial_xy()
        return np.int_(np.array([np.transpose(np.vstack([plotx, ploty]))]))

    def draw_line_on_image(self, image, color=[255, 255, 0], thickness=2):
        cv2.polylines(image, self.polynomial_xy_for_cv2(), False, color, thickness)

    def draw_fill_on_image(self, image, other_lane, color=[0, 255, 0]):
        first_lane_points = self.polynomial_xy_for_cv2()
        second_lane_points = np.array([np.flipud(other_lane.polynomial_xy_for_cv2()[0])])
        points = np.hstack((first_lane_points, second_lane_points))
        cv2.fillPoly(image, points, color)

    def curve_at_bottom(self):
        (a, b, _) = self.scaled_polynomial
        y = self.binary_image.height * YM_PER_PIX
        return np.power((1 + np.square(2 * a * y + b)), 3 / 2) / np.abs(2 * a)

    def deviation_from_center(self, right_lane):
        y = self.binary_image.height * YM_PER_PIX
        (a1, b1, c1) = self.scaled_polynomial
        left_lane_bottom = (a1 * y ** 2 + b1 * y + c1)
        (a2, b2, c2) = right_lane.scaled_polynomial
        right_lane_bottom = (a2 * y ** 2 + b2 * y + c2)
        actual_middle = left_lane_bottom + ((right_lane_bottom - left_lane_bottom) / 2)
        perfect_middle = MIDPOINT_X * XM_PER_PIX
        return np.abs(actual_middle - perfect_middle)

    def draw_summary(self, image, other_lane, color=[255, 255, 255]):
        curve = np.mean([self.curve_at_bottom(), other_lane.curve_at_bottom()])
        text = "Curve: {:.1f}km Off center: {:.0f}cm".format(
            curve/1000,
            self.deviation_from_center(other_lane) * 100)
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, thickness=1)
        left_x = (1280 // 2) - (width // 2)
        cv2.putText(image, text, (left_x, 700), cv2.FONT_HERSHEY_DUPLEX, 1, color)


def absolute_sobel(image, orient='x', sobel_kernel=3):
    orientation = [1, 0] if orient == 'x' else [0, 1]
    derivative = cv2.Sobel(image, cv2.CV_64F, *orientation, ksize=sobel_kernel)
    absolute_derivative = np.absolute(derivative)
    return np.uint8(255 * absolute_derivative / np.max(absolute_derivative))


def identify_lane_pixels(image):
    red_image = image[:, :, 0]
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
    left_lane.draw_line_on_image(rgb_image)
    right_lane.draw_line_on_image(rgb_image)
    left_lane.draw_summary(rgb_image, other_lane=right_lane)
    return rgb_image


if __name__ == "__main__":
    binary_images = [identify_lane_pixels(image) for image in images_in_directory('output_images/bird_view')]
    write_images_to_directory([b.to_rgb_image() for b in binary_images], 'binary_images')
    write_images_to_directory([annotate_lane(b) for b in binary_images], 'annotated_binary_images')
