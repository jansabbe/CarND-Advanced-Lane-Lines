from camera_calibration import Camera
from perspective import RegionOfInterest, Perspective
from util import images_in_directory, write_images_to_directory
from binary_image import Lane, identify_lane_pixels
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

camera = Camera(images_in_directory('camera_cal'))
region_of_interest = RegionOfInterest((1280, 720))
perspective = Perspective(region_of_interest)


class Pipeline:
    def __init__(self):
        self.previous_polynomial_left = None
        self.previous_polynomial_right = None

    def pipeline(self, image):
        undistorted_image = camera.undistort(image)
        bird_view_image = perspective.bird_view(undistorted_image)
        lane_pixels = identify_lane_pixels(bird_view_image)
        (left_x, right_x) = lane_pixels.lane_bases()
        left_lane = Lane(lane_pixels, left_x, self.previous_polynomial_left)
        right_lane = Lane(lane_pixels, right_x, self.previous_polynomial_right)
        bird_view_lane_image = np.zeros_like(bird_view_image).astype(np.uint8)
        left_lane.draw_fill_on_image(bird_view_lane_image, other_lane=right_lane)
        lane_image = perspective.perspective_view(bird_view_lane_image)
        self.previous_polynomial_left = left_lane.polynomial
        self.previous_polynomial_right = right_lane.polynomial
        return cv2.addWeighted(undistorted_image, 1, lane_image, 0.3, 0)

if __name__ == "__main__":
    lane_detected = [Pipeline().pipeline(image) for image in images_in_directory('test_images')] 
    write_images_to_directory(lane_detected, 'full_pipeline')

    original = VideoFileClip("project_video.mp4")
    pipeline = Pipeline()
    lanes = original.fl_image(pipeline.pipeline)
    lanes.write_videofile('output_videos/project_video.mp4', audio=False)



