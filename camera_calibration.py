
import numpy as np
import cv2
import glob
from os import makedirs

chessboard_size = (9,6)

def create_object_points(chessboard_size):
    (nx,ny) = chessboard_size
    object_points = np.zeros((nx*ny, 3), np.float32)
    object_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    return object_points

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

class Calibration:
    def __init__(self, chessboard_size, image):
        self.image = image
        self.img_size = image_size(image)
        self.chessboard_size = chessboard_size
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.found_corners, self.image_points = cv2.findChessboardCorners(grayscale, chessboard_size)
        self.object_points = create_object_points(chessboard_size)

    @property
    def image_with_corner(self):
        if self.found_corners:
            output_image = np.copy(self.image)
            cv2.drawChessboardCorners(output_image, self.chessboard_size, self.image_points, self.found_corners)
            return output_image
        return None
        

class Camera:
    def __init__(self, images) -> None:
        self.calibration_images = images
        self.images_with_corners = []
        objpoints = []
        imgpoints = []
        img_size = None
        for image in self.calibration_images:
            calibration = Calibration(chessboard_size, image)
            if calibration.found_corners:
                objpoints.append(calibration.object_points)
                imgpoints.append(calibration.image_points)
                self.images_with_corners.append(calibration.image_with_corner)
                img_size = calibration.img_size
        _, self.matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.matrix, self.distortion_coefficients)


if __name__ == "__main__":
    camera = Camera(images_in_directory('camera_cal'))
    write_images_to_directory(camera.images_with_corners, 'corners_found')
    undistorted_images = [camera.undistort(image) for image in images_in_directory('camera_cal')]
    write_images_to_directory(undistorted_images, 'undistorted')