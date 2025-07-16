from typing import Sequence, Optional

import cv2
import numpy as np


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, Sequence) and not isinstance(item, str):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def adjust_for_aspect_ratio(crop: Sequence, height: int, width: int) -> tuple[int, int, int, int]:
    """Adjust the crop coordinates to maintain the aspect ratio of the given height and width.
    
    This function always reduces the crop to make it match the aspect ratio, never enlarges it.

    Args:
        crop (np.ndarray): A 1D array of shape (4,) representing the crop coordinates in the format [min_x, min_y, max_x, max_y].
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        tuple[int, int, int, int]: Adjusted crop coordinates in the format (min_x, min_y, max_x, max_y).
    """
    
    min_x, min_y, max_x, max_y = crop
    crop_width = max_x - min_x
    crop_height = max_y - min_y
    
    gt_ratio = width / height
    current_ratio = crop_width / crop_height
    
    if current_ratio > gt_ratio:
        # Current crop is too wide, reduce width to match aspect ratio
        width_adjusted = crop_height * gt_ratio
        center_x = (max_x + min_x) / 2
        min_x = center_x - width_adjusted / 2
        max_x = center_x + width_adjusted / 2
    else:
        # Current crop is too tall, reduce height to match aspect ratio
        height_adjusted = crop_width / gt_ratio
        center_y = (max_y + min_y) / 2
        min_y = center_y - height_adjusted / 2
        max_y = center_y + height_adjusted / 2
    
    return min_x, min_y, max_x, max_y

def recompute_transform_from_crop(
    crop: np.ndarray,
    transform: np.ndarray,
    height: int,
    width: int
) -> np.ndarray:
    """Recompute the transformation matrix as to place the corners of the crop at the corners of the original image.

    Args:
        crop (np.ndarray): The crop coordinates in the transformed image in the form [x_min, y_min, x_max, y_max].
        transform (np.ndarray): The original transformation matrix applied to the image.
        height (int): Height of the original image.
        width (int): Width of the original image.

    Returns:
        np.ndarray: The new transformation matrix that places the crop at the corners of the original image.
    """
    
    # Find where the crop is in the original image coordinates
    x_min, y_min, x_max, y_max = crop
    dst_corners = np.array([
        [x_min, y_min, 1],
        [x_max, y_min, 1],
        [x_max, y_max, 1],
        [x_min, y_max, 1],
    ]).reshape(4, 3)
    src_corners = dst_corners @ np.linalg.inv(transform.T)
    src_corners = src_corners[:, :2] / src_corners[:, 2:3]
    
    # Recompute the transformation matrix as to place the corners
    # of the crop at the corners of the original image
    new_dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]).reshape(4, 2)
    transform = cv2.getPerspectiveTransform(src_corners.astype(np.float32), new_dst_corners.astype(np.float32))
    return transform

def get_naive_crop(
    dst_corners: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    keep_aspect_ratio: bool = False,
) -> tuple[int, int, int, int]:
    assert dst_corners is not None or (transform is not None and height is not None and width is not None), \
        "Either dst_corners or (transform, height, width) must be provided."
    assert not dst_corners is not None and transform is not None, \
        "If dst_corners is provided, transform should not be provided."
    
    if dst_corners is None:
        assert transform.shape == (3, 3), "Transform must be a 3x3 matrix."
        # Compute dst_corners from the transform
        src_corners = np.array([
            [0, 0, 1],
            [width - 1, 0, 1],
            [width - 1, height - 1, 1],
            [0, height - 1, 1],
        ]).reshape(4, 2)
        dst_corners = src_corners @ transform.T
        dst_corners = dst_corners[:, :2] / dst_corners[:, 2:3]
    assert dst_corners.shape == (4, 2), "dst_corners must be a 4x2 array."
    
    # here
    
    min_x = np.max(dst_corners[:, 0])
    min_y = np.max(dst_corners[:, 1])
    max_x = np.min(dst_corners[:, 0])
    max_y = np.min(dst_corners[:, 1])
    
    if keep_aspect_ratio:
        min_x, min_y, max_x, max_y = adjust_for_aspect_ratio([min_x, min_y, max_x, max_y], height, width)
    
    return min_x, min_y, max_x, max_y