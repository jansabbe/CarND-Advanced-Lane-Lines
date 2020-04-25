from binary_image import BinaryImage, Window
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


def test_find_lane_bases():
    a = BinaryImage(np.array([
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 0]
    ]))

    (left_lane_base, right_lane_base) = a.lane_bases(midpoint_x=3)
    assert left_lane_base == 1
    assert right_lane_base == 5


def test_find_lane_bases_multiple_candidates():
    a = BinaryImage(np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1]
    ]))

    (left_lane_base, right_lane_base) = a.lane_bases(midpoint_x=3)
    assert left_lane_base == 1
    assert right_lane_base == 5


def test_find_lane_bases_hollow_lane():
    a = BinaryImage(np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ]))

    (left_lane_base, right_lane_base) = a.lane_bases(midpoint_x=3)
    assert left_lane_base == 1
    assert right_lane_base == 5


def test_find_y_positions():
    a = BinaryImage(np.zeros((90, 2), np.int8))
    expected_y_positions = np.array(
        [(90, 80), (80, 70), (70, 60), (60, 50), (50, 40), (40, 30), (30, 20), (20, 10), (10, 0)])
    assert np.array_equal(a.window_y_positions(), expected_y_positions)


def test_xy_positions_in_window():
    a = BinaryImage(np.array([
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ]))
    window = Window(a, 5, (6, 3), 1)
    (x, y) = window.nonzero_xy
    assert np.array_equal(x, [5, 4])
    assert np.array_equal(y, [4, 5])
