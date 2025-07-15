from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import minimize


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, Sequence) and not isinstance(item, str):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def adjust_for_aspect_ratio(crop: np.ndarray, height: int, width: int):
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

def optimize_crop(
    transform: np.ndarray,
    height: int,
    width: int,
    x0: np.ndarray = None,
    keep_aspect_ratio: bool = False,
    max_iter: int = 1000,
    disp: bool = True
) -> tuple[np.ndarray, float]:
    """Optimize the axis-aligned crop of an image that has undergone a transformation.
    
    Note, this problem is not strictly convex, so the result may depend on the initial guess.
    However, in practice, the it usually converges to a solution better than the naive crop, and if not, very close to it.
    You should compare the area of the naive crop and the optimal crop externally if you want to ensure the optimality of the result.

    Args:
        transform (np.ndarray): Transformation matrix applied to the image.
        height (int): Height of the image before the transformation.
        width (int): Width of the image before the transformation.
        x0 (np.ndarray): Initial guess for the crop in the form [x_min, y_min, x_max, y_max]. If None, defaults to the center crop bound by the transformation. Defaults to None. 
        keep_aspect_ratio (bool, optional): If True, the computed crop will maintain the aspect ratio of the original image. Defaults to False.
        max_iter (int, optional): Maximum number of iterations for the optimization. Defaults to 1000.
        disp (bool, optional): If True, display optimization progress. Defaults to True.
    Returns:
        tuple[np.ndarray, float]: The optimal crop coordinates in the form [x_min, y_min, x_max, y_max] and the area of the crop.
    """
    assert transform.shape == (3, 3), "Transform must be a 3x3 matrix."
    
    # Compute the corners of the transformed image
    src_corners = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]).reshape(4, 3)
    dst_corners = src_corners @ transform.T
    dst_corners = dst_corners[:, :2]  / dst_corners[:, 2:3]
    
    def objective(x):
        """Maximize the area of the crop, i.e. minimize the negative area."""
        return -(x[2] - x[0]) * (x[3] - x[1]) / scale # keep the values in a reasonable range
    
    def ineq_constraints(x):
        tl, tr, br, bl = dst_corners
            
        # Compute the four line segments of the transformed image
        # Ax + By + C = 0
        top   = (tl[1] - tr[1], tr[0] - tl[0], tl[0] * tr[1] - tr[0] * tl[1])
        bot   = (bl[1] - br[1], br[0] - bl[0], bl[0] * br[1] - br[0] * bl[1])
        left  = (bl[1] - tl[1], tl[0] - bl[0], bl[0] * tl[1] - tl[0] * bl[1])
        right = (br[1] - tr[1], tr[0] - br[0], br[0] * tr[1] - tr[0] * br[1])
        
        def validate_point(p):
            px, py = p
            ineq = [
                # The point must be inside the transformed image boundaries (>=0)
                 (top[0]   * px + top[1]   * py + top[2]),      
                 (left[0]  * px + left[1]  * py + left[2]),
                -(right[0] * px + right[1] * py + right[2]),
                -(bot[0]   * px + bot[1]   * py + bot[2]),
            ]
            return ineq
        
        # Construct the constraints
        all_ineq = []
        all_ineq.append(x[2] - x[0])  # x_max > x_min
        all_ineq.append(x[3] - x[1])  # y_max > y_min
        
        all_ineq.append(validate_point([x[0], x[1]])) # tl inside bounds
        all_ineq.append(validate_point([x[2], x[1]])) # tr inside bounds
        all_ineq.append(validate_point([x[2], x[3]])) # br inside bounds
        all_ineq.append(validate_point([x[0], x[3]])) # bl inside bounds
        
        return np.array(flatten(all_ineq))
    
    def eq_constraints(x):
        gt_ratio = width / height
        c_ratio = (x[2] - x[0]) / (x[3] - x[1] + 1e-6)  # Avoid division by zero
        return np.array([c_ratio - gt_ratio])
    
    # Starting guess
    if x0 is None:
        min_x = max(dst_corners[0, 0], dst_corners[3, 0])
        min_y = max(dst_corners[0, 1], dst_corners[1, 1])
        max_x = min(dst_corners[1, 0], dst_corners[2, 0])
        max_y = min(dst_corners[2, 1], dst_corners[3, 1])
        if keep_aspect_ratio:
            min_x, min_y, max_x, max_y = adjust_for_aspect_ratio([min_x, min_y, max_x, max_y], height, width)
        x0 = np.array([min_x, min_y, max_x, max_y])
    scale = (x0[2] - x0[0]) * (x0[3] - x0[1])  # Initial scale based on the initial guess
    
    constraints = [
        dict(type='ineq', fun=ineq_constraints),
        dict(type='eq', fun=eq_constraints) if keep_aspect_ratio else None
    ]
    constraints = [c for c in constraints if c is not None]
    
    result = minimize(
        fun=objective,
        x0=x0,
        constraints=constraints,
        method='SLSQP',
        options=dict(maxiter=max_iter, disp=disp)
    )
    if not result.success:
        if disp:
            print("Optimization failed:", result.message)
        return None
    return result.x, -result.fun * scale

def optimize_mutual_crop(
    transforms: Sequence[np.ndarray],
    height: int,
    width: int,
    x0: np.ndarray = None,
    keep_aspect_ratio: bool = False,
    max_iter: int = 1000,
    disp: bool = True
) -> tuple[np.ndarray, float]:
    """Optimize the axis-aligned crop of multiple images that have undergone different transformations.
    
     
    Note, this problem is not strictly convex, so the result may depend on the initial guess.
    However, in practice, the it usually converges to a solution better than the naive crop, and if not, very close to it.
    You should compare the area of the naive crop and the optimal crop externally if you want to ensure the optimality of the result.

    Args:
        transforms (Sequence[np.ndarray]): List of transformation matrices applied to the images.
        height (int): Height of the images before the transformations.
        width (int): Width of the images before the transformations.
        x0 (np.ndarray, optional): Initial guess for the crop in the form [x_min, y_min, x_max, y_max]. If None, defaults to the center crop encompassing all transformed images. Defaults to None.
        keep_aspect_ratio (bool, optional): If True, the computed crop will maintain the aspect ratio of the original images. Defaults to False.
        max_iter (int, optional): Maximum number of iterations for the optimization. Defaults to 1000.
        disp (bool, optional): If True, display optimization progress. Defaults to True.

    Returns:
        tuple[np.ndarray, float]: The optimal crop coordinates in the form [x_min, y_min, x_max, y_max] and the area of the crop.
    """
    assert all(t.shape == (3, 3) for t in transforms), "All transformations must be 3x3 matrices."
    n = len(transforms)
    
    # Compute the corners of the transformed images
    transforms = np.array(transforms).reshape(-1, 3, 3)
    src_corners = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]).reshape(1, 4, 3)
    src_corners = np.repeat(src_corners, len(transforms), axis=0)
    dst_corners = np.matmul(src_corners, transforms.transpose(0, 2, 1))
    dst_corners = dst_corners[:, :, :2] / dst_corners[:, :, 2:3]
    
    def objective(x):
        """Maximize the area of the crop, i.e. minimize the negative area."""
        return -(x[2] - x[0]) * (x[3] - x[1]) / scale # keep the values in a reasonable range
    
    def ineq_constraints(x):
        """Ensure the crop is within the bounds of all transformed images."""
        all_ineq = []
        all_ineq.append(x[2] - x[0])
        all_ineq.append(x[3] - x[1])
        for i in range(n):
            tl, tr, br, bl = dst_corners[i]
            
            # Compute the four line segments of the transformed image
            # Ax + By + C = 0
            top   = (tl[1] - tr[1], tr[0] - tl[0], tl[0] * tr[1] - tr[0] * tl[1])
            bot   = (bl[1] - br[1], br[0] - bl[0], bl[0] * br[1] - br[0] * bl[1])
            left  = (bl[1] - tl[1], tl[0] - bl[0], bl[0] * tl[1] - tl[0] * bl[1])
            right = (br[1] - tr[1], tr[0] - br[0], br[0] * tr[1] - tr[0] * br[1])
            
            def validate_point(p):
                px, py = p
                ineq = [
                    # The point must be inside the transformed image boundaries (>=0)
                     (top[0]   * px + top[1]   * py + top[2]),      
                     (left[0]  * px + left[1]  * py + left[2]),
                    -(right[0] * px + right[1] * py + right[2]),
                    -(bot[0]   * px + bot[1]   * py + bot[2]),
                ]
                return ineq
            
            all_ineq.append(validate_point([x[0], x[1]])) # tl inside bounds
            all_ineq.append(validate_point([x[2], x[1]])) # tr inside bounds
            all_ineq.append(validate_point([x[2], x[3]])) # br inside bounds
            all_ineq.append(validate_point([x[0], x[3]])) # bl inside bounds
        return np.array(flatten(all_ineq))
    
    def eq_constraints(x):
        """Ensure the crop maintains the aspect ratio of the original image."""
        gt_ratio = width / height
        c_ratio = (x[2] - x[0]) / (x[3] - x[1] + 1e-6)  # Avoid division by zero
        return np.array([c_ratio - gt_ratio])
    
    # Starting guess
    if x0 is None:
        min_x = min([dst_corners[i][0, 0] for i in range(n)] + [dst_corners[i][3, 0] for i in range(n)])
        min_y = min([dst_corners[i][0, 1] for i in range(n)] + [dst_corners[i][1, 1] for i in range(n)])
        max_x = max([dst_corners[i][1, 0] for i in range(n)] + [dst_corners[i][2, 0] for i in range(n)])
        max_y = max([dst_corners[i][2, 1] for i in range(n)] + [dst_corners[i][3, 1] for i in range(n)])
        x0 = np.array([min_x, min_y, max_x, max_y])
    scale = (x0[2] - x0[0]) * (x0[3] - x0[1])  # Initial scale based on the initial guess
    
    constraints = [
        dict(type='ineq', fun=ineq_constraints),
        dict(type='eq', fun=eq_constraints) if keep_aspect_ratio else None
    ]
    constraints = [c for c in constraints if c is not None]
    
    result = minimize(
        fun=objective,
        x0=x0,
        constraints=constraints,
        method='SLSQP',
        options=dict(maxiter=max_iter, disp=disp)
    )
    if not result.success:
        if disp:
            print("Optimization failed:", result.message)
        return None
    return result.x, -result.fun * scale

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    height = 1024 * 5
    width = 1536 * 5
    distortion = 0.2
    n = 7
    
    # Construct a sample transformation
    src_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]).reshape(1, 4, 2)
    src_corners = np.repeat(src_corners, n, axis=0)
    dst_corners = ((np.random.rand(n, 4, 2) - 0.5) * 2 * np.array([width, height]).reshape(1, 1, 2) * distortion) + src_corners
    
    transforms = []
    for i in range(n):
        transform = cv2.getPerspectiveTransform(src_corners[i].astype(np.float32), dst_corners[i].astype(np.float32))
        transforms.append(transform)
    
    optimal_crop, area = optimize_mutual_crop(
        transforms=transforms,
        height=height,
        width=width,
        keep_aspect_ratio=True,
        disp=True
    )
    
     # visualize
    if optimal_crop is not None:
        
        # Compute bounds
        min_x = min([dst_corners[i][0, 0] for i in range(n)] + [dst_corners[i][3, 0] for i in range(n)])
        min_y = min([dst_corners[i][0, 1] for i in range(n)] + [dst_corners[i][1, 1] for i in range(n)])
        max_x = max([dst_corners[i][1, 0] for i in range(n)] + [dst_corners[i][2, 0] for i in range(n)])
        max_y = max([dst_corners[i][2, 1] for i in range(n)] + [dst_corners[i][3, 1] for i in range(n)])
        
        # Expand bounds a bit
        min_x, max_x = min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1
        min_y, max_y = min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1
        center_y = (min_y + max_y) / 2
        center_x = (min_x + max_x) / 2
        size = max(max_x - min_x, max_y - min_y)
        min_x, max_x = center_x - size / 2, center_x + size / 2
        min_y, max_y = center_y - size / 2, center_y + size / 2

        fig, ax = plt.subplots()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        plt.gca().invert_yaxis() # flip y axis to match image coordinates
        plt.gca().set_aspect('equal') # maintain aspect ratio

        # Plot the transformed image
        for i in range(n):
            ax.add_patch(plt.Polygon(dst_corners[i], fill=False))
        
        
        # Add naive crop after resizing and cropping to correct aspect ratio
        # Naive crop
        min_x = max([dst_corners[i][0, 0] for i in range(n)] + [dst_corners[i][3, 0] for i in range(n)])
        min_y = max([dst_corners[i][0, 1] for i in range(n)] + [dst_corners[i][1, 1] for i in range(n)])
        max_x = min([dst_corners[i][1, 0] for i in range(n)] + [dst_corners[i][2, 0] for i in range(n)])
        max_y = min([dst_corners[i][2, 1] for i in range(n)] + [dst_corners[i][3, 1] for i in range(n)])
        min_x, min_y, max_x, max_y = adjust_for_aspect_ratio([min_x, min_y, max_x, max_y], height, width)
        ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, ec="b", label="Naive Crop"))
        
        
        # Add optimal crop
        ax.add_patch(plt.Rectangle((optimal_crop[0], optimal_crop[1]), optimal_crop[2] - optimal_crop[0], optimal_crop[3] - optimal_crop[1], fill=False, ec="r", label="Optimal Crop"))

        plt.legend()
        plt.savefig("crop.png")
        
        print("Naive Area:", (max_x - min_x) * (max_y - min_y))
        print("Optimal Area:", area)
        print("Naive / Optimal Ratio:", (max_x - min_x) * (max_y - min_y) / area)
        
        # print("Naive Crop Coordinates:", [min_x, min_y, max_x, max_y])
        # print("Optimal Crop Coordinates:", optimal_crop)
        
        # # Apply the crops to the image
        # corrected_transform = recompute_transform_from_crop([min_x, min_y, max_x, max_y], transform, height, width)
        # naive_crop = cv2.warpPerspective(image, corrected_transform, (width, height))
        # cv2.imwrite("naive_crop.png", naive_crop)
        
        # corrected_transform = recompute_transform_from_crop(optimal_crop, transform, height, width)
        # optimal_crop_image = cv2.warpPerspective(image, corrected_transform, (width, height))
        # cv2.imwrite("optimal_crop.png", optimal_crop_image)