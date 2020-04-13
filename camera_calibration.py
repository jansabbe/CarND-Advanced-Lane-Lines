
import numpy as np
import cv2
import glob
from os import makedirs
from os.path import basename

chessboard_size = (9,6)

def create_object_points(chessboard_size):
    (nx,ny) = chessboard_size
    object_points = np.zeros((nx*ny, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    return object_points

def image_size(img):
    (height, width) = img.shape[:2]
    return (width, height)

class Calibration:
    def __init__(self, chessboard_size, image):
        self.image = image
        self.img_size = image_size(image)
        self.chessboard_size = chessboard_size
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.found_corners, self.image_points = cv2.findChessboardCorners(grayscale, chessboard_size)
        self.object_points = create_object_points(chessboard_size)

    def draw_corners(self, output_path):
        if self.found_corners:
            output_image = np.copy(self.image)
            cv2.drawChessboardCorners(output_image, self.chessboard_size, self.image_points, self.found_corners)
            cv2.imwrite(output_path, output_image)
        

class Camera:

    @classmethod
    def from_directory(cls, directory):
        images = [cv2.imread(filename) for filename in glob.glob(f'{directory}/*.jpg')]
        return cls(images)

    def __init__(self, images) -> None:
        self.calibration_images = images
        makedirs('output_images/corners_found', exist_ok=True)
        objpoints = []
        imgpoints = []
        img_size = None
        for image in self.calibration_images:
            calibration = Calibration(chessboard_size, image)
            if calibration.found_corners:
                objpoints.append(calibration.object_points)
                imgpoints.append(calibration.image_points)
                img_size = calibration.img_size
        _, self.matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    def draw_corners(self):
        for idx, image in enumerate(self.calibration_images):
            calibration = Calibration(chessboard_size, image)
            if calibration.found_corners:
                calibration.draw_corners(f'output_images/corners_found/found_corners_{idx}.jpg')

    def undistort(self, image):
        return cv2.undistort(image, self.matrix, self.distortion_coefficients)


if __name__ == "__main__":
    camera = Camera.from_directory('camera_cal')
    camera.draw_corners()
    images = glob.glob('test_images/*.jpg')
    makedirs('output_images/undistorted', exist_ok=True)
    for filename in images:
        image = cv2.imread(filename)
        write_name = f'output_images/undistorted/{basename(filename)}'
        cv2.imwrite(write_name, camera.undistort(image))