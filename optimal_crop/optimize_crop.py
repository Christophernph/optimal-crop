from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import minimize

from optimal_crop.utils import adjust_for_aspect_ratio, flatten


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
        min_x = min(dst_corners[0, 0], dst_corners[3, 0])
        min_y = min(dst_corners[0, 1], dst_corners[1, 1])
        max_x = max(dst_corners[1, 0], dst_corners[2, 0])
        max_y = max(dst_corners[2, 1], dst_corners[3, 1])
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