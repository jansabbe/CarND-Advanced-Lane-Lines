import cv2

from camera_calibration import create_object_points, image_size


def test_create_object_points():
    points = create_object_points(chessboard_size=(2, 2))
    assert [0., 0., 0.] in points
    assert [0., 1., 0.] in points
    assert [1., 0., 0.] in points
    assert [1., 1., 0.] in points


def test_image_size():
    image = cv2.imread('camera_cal/calibration1.jpg')
    assert image_size(image) == (1280, 720)
