import cv2
import numpy as np

from optimal_crop import optimize_mutual_crop
from optimal_crop.utils import adjust_for_aspect_ratio


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
    
    height = 1024
    width = 1536
    distortion = 0.2
    keep_aspect_ratio = True
    n = 7 # number of transformations to optimize over
    
    # Construct the sample transformations
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
    
    # Optimize the mutual crop
    optimal_crop, area = optimize_mutual_crop(
        transforms=transforms,
        height=height,
        width=width,
        keep_aspect_ratio=keep_aspect_ratio,
        disp=True
    )
    if optimal_crop is None:
        print("No optimal crop found. Skipping visualization.")
        return
    
        
    # Compute plotting bounds
    min_x = min([dst_corners[i][0, 0] for i in range(n)] + [dst_corners[i][3, 0] for i in range(n)])
    min_y = min([dst_corners[i][0, 1] for i in range(n)] + [dst_corners[i][1, 1] for i in range(n)])
    max_x = max([dst_corners[i][1, 0] for i in range(n)] + [dst_corners[i][2, 0] for i in range(n)])
    max_y = max([dst_corners[i][2, 1] for i in range(n)] + [dst_corners[i][3, 1] for i in range(n)])
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_xy = max(range_x, range_y) * 1.1
    center_y = (min_y + max_y) / 2
    center_x = (min_x + max_x) / 2
    min_x, max_x = center_x - range_xy / 2, center_x + range_xy / 2
    min_y, max_y = center_y - range_xy / 2, center_y + range_xy / 2

    fig, ax = plt.subplots()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.gca().invert_yaxis() # flip y axis to match image coordinates
    plt.gca().set_aspect('equal') # maintain aspect ratio

    # Plot the transformations
    for i in range(n):
        ax.add_patch(plt.Polygon(dst_corners[i], fill=False))
    
    
    # Add naive crop after resizing and cropping to correct aspect ratio
    min_x = max([dst_corners[i][0, 0] for i in range(n)] + [dst_corners[i][3, 0] for i in range(n)])
    min_y = max([dst_corners[i][0, 1] for i in range(n)] + [dst_corners[i][1, 1] for i in range(n)])
    max_x = min([dst_corners[i][1, 0] for i in range(n)] + [dst_corners[i][2, 0] for i in range(n)])
    max_y = min([dst_corners[i][2, 1] for i in range(n)] + [dst_corners[i][3, 1] for i in range(n)])
    min_x, min_y, max_x, max_y = adjust_for_aspect_ratio([min_x, min_y, max_x, max_y], height, width) if keep_aspect_ratio else [min_x, min_y, max_x, max_y]
    ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, ec="b", label="Naive Crop"))
    
    # Add optimal crop
    ax.add_patch(plt.Rectangle((optimal_crop[0], optimal_crop[1]), optimal_crop[2] - optimal_crop[0], optimal_crop[3] - optimal_crop[1], fill=False, ec="r", label="Optimal Crop"))

    plt.legend()
    plt.savefig("crop.png")
    
    print("Naive / Optimal Ratio:", (max_x - min_x) * (max_y - min_y) / area)

if __name__ == "__main__":
    main()