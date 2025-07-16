import cv2
import numpy as np

from optimal_crop import optimize_crop
from optimal_crop.utils import recompute_transform_from_crop, adjust_for_aspect_ratio

# Check if matplotlib is available for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "matplotlib is not installed. Please install it to visualize the results.\n"
        "You can install it with: pip install matplotlib"
    )
    
def main():
    np.set_printoptions(precision=3, suppress=True)
    
    # Parameters
    # You can change these parameters to test with different images and transformations
    image = cv2.imread("assets/image.jpg")
    height, width = image.shape[:2]
    keep_aspect_ratio = True
    distortion = 0.2
    
    # Construct a sample transformation
    src_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    dst_corners = ((np.random.rand(4, 2) - 0.5) * 2 * np.array([width, height]).reshape(1, 2) * distortion) + src_corners
    transform = cv2.getPerspectiveTransform(src_corners.astype(np.float32), dst_corners.astype(np.float32))
    
    optimal_crop, area = optimize_crop(
        transform=transform,
        height=height,
        width=width,
        keep_aspect_ratio=keep_aspect_ratio,
        disp=True
    )
    if optimal_crop is None:
        print("No optimal crop found. Skipping visualization.")
        return
        
    # Compute plotting bounds
    min_x = np.min(dst_corners[:, 0])
    max_x = np.max(dst_corners[:, 0])
    min_y = np.min(dst_corners[:, 1])
    max_y = np.max(dst_corners[:, 1])
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_xy = max(range_x, range_y) * 1.1
    center_y = (min_y + max_y) / 2
    center_x = (min_x + max_x) / 2
    min_x, max_x = center_x - range_xy / 2, center_x + range_xy / 2
    min_y, max_y = center_y - range_xy / 2, center_y + range_xy / 2
    
    # Prepare the plot
    fig, ax = plt.subplots()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.gca().invert_yaxis() # flip y axis to match image coordinates
    plt.gca().set_aspect('equal') # maintain aspect ratio

    # Plot the transformed image
    ax.add_patch(plt.Polygon(dst_corners, fill=False))
    
    # Add naive crop after resizing and cropping to correct aspect ratio
    min_x = max(dst_corners[0, 0], dst_corners[3, 0])
    min_y = max(dst_corners[0, 1], dst_corners[1, 1])
    max_x = min(dst_corners[1, 0], dst_corners[2, 0])
    max_y = min(dst_corners[2, 1], dst_corners[3, 1])
    min_x, min_y, max_x, max_y = adjust_for_aspect_ratio([min_x, min_y, max_x, max_y], height, width) if keep_aspect_ratio else [min_x, min_y, max_x, max_y]
    ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, ec="b", label="Naive Crop"))
    
    # Add optimal crop
    ax.add_patch(plt.Rectangle((optimal_crop[0], optimal_crop[1]), optimal_crop[2] - optimal_crop[0], optimal_crop[3] - optimal_crop[1], fill=False, ec="r", label="Optimal Crop"))

    plt.legend()
    plt.savefig("plot.png")
    
    print("Naive / Optimal Ratio:", (max_x - min_x) * (max_y - min_y) / area)
    
    # Apply the crops to the image
    corrected_transform = recompute_transform_from_crop([min_x, min_y, max_x, max_y], transform, height, width)
    naive_crop = cv2.warpPerspective(image, corrected_transform, (width, height))
    cv2.imwrite("naive_crop.png", naive_crop)
    
    corrected_transform = recompute_transform_from_crop(optimal_crop, transform, height, width)
    optimal_crop_image = cv2.warpPerspective(image, corrected_transform, (width, height))
    cv2.imwrite("optimal_crop.png", optimal_crop_image)

if __name__ == "__main__":
    main()