import numpy as np
import cv2
from util import image_size, images_in_directory, write_images_to_directory, RED


class RegionOfInterest:
    def __init__(self, img_size):
        (self.width, self.height) = img_size
        self.bottom_y = 0.910 * self.height
        self.horizon_y = 0.630 * self.height
        self.bottom_left_x = 0.231 * self.width
        self.bottom_right_x = 0.797 * self.width
        self.horizon_left_x = 0.460 * self.width
        self.horizon_right_x = 0.545 * self.width

    @property
    def perspective(self):
        return np.float32(
            [
                [self.bottom_left_x, self.bottom_y],
                [self.horizon_left_x, self.horizon_y],
                [self.horizon_right_x, self.horizon_y],
                [self.bottom_right_x, self.bottom_y],
            ]
        )

    @property
    def rectangle(self):
        return np.float32(
            [
                [self.bottom_left_x, self.height-10],
                [self.bottom_left_x, 0],
                [self.bottom_right_x, 0],
                [self.bottom_right_x, self.height-10],
            ]
        )

    def draw_rectangle(self, image):
        cv2.polylines(image, [np.int32(self.rectangle).reshape((-1,1,2))], True, RED, thickness=2)
        return image

    def draw_perspective(self, image):
        cv2.polylines(image, [np.int32(self.perspective).reshape((-1,1,2))], True, RED, thickness=2)
        return image



class Perspective:
    def __init__(self, region_of_interest):
        self.region = region_of_interest
        self.transform = cv2.getPerspectiveTransform(
            self.region.perspective, self.region.rectangle
        )
        self.inverseTransform = cv2.getPerspectiveTransform(
            self.region.rectangle, self.region.perspective
        )

    def bird_view(self, image):
        return cv2.warpPerspective(np.copy(image), self.transform, image_size(image))

    def perspective_view(self, image):
        return cv2.warpPerspective(
            np.copy(image), self.inverseTransform, image_size(image)
        )


if __name__ == "__main__":
    region_of_interest = RegionOfInterest((1280, 720))
    perspective = Perspective(region_of_interest)
    bird_view_images = [
        perspective.bird_view(image)
        for image in images_in_directory("output_images/undistorted_test_images")
    ]
    write_images_to_directory(bird_view_images, "bird_view")
