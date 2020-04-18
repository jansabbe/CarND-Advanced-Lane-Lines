from binary_image import BinaryImage
import numpy as np

def test_combine_images_with_and():
    a = BinaryImage(np.array([
        [1, 0],
        [1, 0]
    ]))
    b = BinaryImage(np.array([
        [1, 1],
        [0, 0]
    ]))

    assert (a & b) == BinaryImage(np.array([
        [1, 0],
        [0, 0]
    ]))

def test_combine_images_with_or():
    a = BinaryImage(np.array([
        [1, 0],
        [1, 0]
    ]))
    b = BinaryImage(np.array([
        [1, 1],
        [0, 0]
    ]))

    assert (a | b) == BinaryImage(np.array([
        [1, 1],
        [1, 0]
    ]))

def test_create_from_image_within_threshold():
    image = np.array([
        [20, 10],
        [5, 25]
    ])
    a = BinaryImage.in_threshold(image, 8, 21)
    assert a == BinaryImage(np.array([
        [1, 1],
        [0, 0]
    ]))