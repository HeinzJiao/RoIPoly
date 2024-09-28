import numpy as np
import cv2
import scipy
from scipy.optimize import linear_sum_assignment


def uniform_sampling_euclidean(gt_pts, num_corners, h, w):
    """
    Uniformly samples points along a polygon's contour, using Euclidean distance for matching.

    Args:
        gt_pts (numpy.ndarray): Ground truth polygon points, shape (N, 2).
        num_corners (int): Number of corners to sample.
        h (int): Image height.
        w (int): Image width.

    Returns:
        numpy.ndarray: Flattened array of sampled polygon points, shape (num_corners * 2).
        numpy.ndarray: Corner labels indicating which points are true corners, shape (num_corners,).
    """
    polygon = np.round(gt_pts).astype(np.int32).reshape((-1, 1, 2))
    corner_label = np.zeros((num_corners,), dtype=np.int32)

    img = np.zeros((h, w), dtype="uint8")
    img = cv2.polylines(img, [polygon], True, 255, 1)
    img = cv2.fillPoly(img, [polygon], 255)

    contour, _ = cv2.findContours(img, cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contour = contour[0].reshape((-1, 2))

    if contour.shape[0] >= num_corners:
        ind = np.linspace(0, contour.shape[0], num=num_corners, endpoint=False).round().astype(np.int32)
        cost = scipy.spatial.distance.cdist(contour[ind], polygon.reshape((-1, 2)))
        row_ind, col_ind = linear_sum_assignment(cost)
        encoded_polygon = contour[ind]
        encoded_polygon[row_ind] = polygon.reshape((-1, 2))[col_ind]
        corner_label[row_ind] = 1  # mark true corners as class 1
    else:
        encoded_polygon, corner_label = pad_polygon(contour, num_corners)

    return encoded_polygon.flatten(), corner_label


def uniform_sampling_index(gt_pts, num_corners, h, w):
    """
    Uniformly samples points, using index difference for bipartite matching.

    Args:
        gt_pts (numpy.ndarray): Ground truth polygon points, shape (N, 2).
        num_corners (int): Number of corners to sample.
        h (int): Image height.
        w (int): Image width.

    Returns:
        numpy.ndarray: Flattened array of sampled polygon points, shape (num_corners * 2).
        numpy.ndarray: Corner labels indicating which points are true corners, shape (num_corners,).
    """
    polygon = np.round(gt_pts).astype(np.int32).reshape((-1, 1, 2))
    corner_label = np.zeros((num_corners,), dtype=np.int32)

    img = np.zeros((h, w), dtype="uint8")
    img = cv2.polylines(img, [polygon], True, 255, 1)
    img = cv2.fillPoly(img, [polygon], 255)

    contour, _ = cv2.findContours(img, cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contour = contour[0].reshape((-1, 2))

    lc = contour.shape[0]
    if lc >= num_corners:
        # Find the index of each ground truth vertex in the contour point sequence
        index = []
        for j, gt_point in enumerate(polygon):  # For each annotated corner
            distances = np.linalg.norm(contour - gt_point, axis=1)  # Calculate distances to all contour points
            min_dist_id = np.argmin(distances)  # Find the index of the closest contour point
            index.append(min_dist_id)  # Add the index of the closest point
        index = np.array(index)  # shape (num_gt_corners,)

        ## Due to rasterization, some ground truth polygon corners may be missing from the dense contour point sequence.
        # To ensure no ground truth corners are missed, replace the closest contour points with their corresponding ground truth corners.
        contour[index] = polygon

        # Despite replacing the closest contour points with ground truth vertices (contour[index] = polygon),
        # some ground truth vertices may still be missed. This occurs when multiple ground truth vertices map to the same
        # contour point due to rasterization. For example, in ./train/AIcrowd/000000000005.jpg, index values like
        # [34, 32, 32, 31, 30, ...] show that the 2nd and 3rd ground truth vertices both map to contour point 32,
        # causing the 2nd vertex to be omitted.

        # Generate evenly spaced indices along the contour
        ind = np.linspace(0, lc, num=num_corners, endpoint=False).round().astype(np.int32)  # (num_corners,)

        ind = np.expand_dims(ind, axis=1)  # Expand to 2D for cost computation
        index = np.expand_dims(index, axis=0)  # Align index for matching

        # Compute the cost based on absolute index difference between sampled points and ground truth corners
        cost = np.abs(ind - index)  # shape, (num_corners, num_gt_corners)

        # Solve the assignment problem to match closest vertices
        row_ind, col_ind = linear_sum_assignment(cost)

        # Replace sampled points with corresponding true corners to ensure ground truth corners are included
        ind = ind.flatten()
        index = index.flatten()
        ind[row_ind] = index[col_ind]

        # Sort indices for ordered output
        ind.sort()
        encoded_polygon = contour[ind]

        # Mark ground truth vertices as class 1
        corner_label[np.sort(row_ind)] = 1
    else:
        encoded_polygon, corner_label = pad_polygon(contour, num_corners)

    return encoded_polygon.flatten(), corner_label


def pad_polygon(contour, num_corners):
    """
    Pads a polygon by repeating its last vertex.

    Args:
        contour (numpy.ndarray): Contour points of the polygon, shape (N, 2).
        num_corners (int): Number of corners to sample.

    Returns:
        numpy.ndarray: Padded contour points, shape (num_corners, 2).
        numpy.ndarray: Corner labels, shape (num_corners,).
    """
    lc = contour.shape[0]

    encoded_polygon = np.vstack([contour, np.tile(contour[-1], (num_corners - lc, 1))])

    corner_label = np.zeros((num_corners,), dtype=np.int32)
    corner_label[:lc] = 1  # Mark original points as corners

    return encoded_polygon, corner_label


def remove_redundant_vertices(gt_pts, epsilon=0.1):
    """
    Removes redundant vertices that are closer than 0.1 units apart.

    Args:
        gt_pts (numpy.ndarray): Ground truth polygon points, shape (N, 2).

    Returns:
        numpy.ndarray: Filtered polygon points.
    """
    gt_pts = np.array(gt_pts).reshape((-1, 2))
    return gt_pts[np.linalg.norm(np.roll(gt_pts, -1, axis=0) - gt_pts, axis=-1) > epsilon]


def approximate_polygons(polygon, tolerance=0.01):
    """
    Simplifies polygons using the specified tolerance.

    Args:
        polygon (numpy.ndarray): Polygon points, shape (N, 2).
        tolerance (float): Tolerance for polygon approximation.

    Returns:
        numpy.ndarray: Simplified polygon points.
    """
    from skimage.measure import approximate_polygon
    return approximate_polygon(polygon, tolerance)


def get_gt_bboxes(gt_pts, image_size):
    """
    Compute the bounding box around the polygon and enlarge it by 20%.
    """
    gt_pts = np.array(gt_pts).reshape((-1, 2))
    x = gt_pts[:, 0]
    y = gt_pts[:, 1]

    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()

    w, h = xmax - xmin, ymax - ymin

    # Enlarge the bounding box by 10% on all sides
    xmin = max(xmin - w * 0.1, 0.0)
    ymin = max(ymin - h * 0.1, 0.0)
    xmax = min(xmax + w * 0.1, image_size - 1e-4)
    ymax = min(ymax + h * 0.1, image_size - 1e-4)

    w = xmax - xmin
    h = ymax - ymin

    return [float(xmin), float(ymin), float(w), float(h)]


def resort_corners_and_labels(corners, labels):
    """Resort a sequence of corners so that the first corner starts
       from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:, 0]**2 + corners[:, 1]**2
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])
    labels_sorted = np.concatenate([labels[start_corner_idx:], labels[:start_corner_idx]])

    # Sort points counterclockwise (clockwise in image)
    if not is_clockwise(corners_sorted[:, :2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)
        labels_sorted[1:] = np.flip(labels_sorted[1:], 0)

    return corners_sorted.reshape(-1), labels_sorted


def resort_corners(corners):
    """Resort a sequence of corners so that the first corner starts
       from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:, 0]**2 + corners[:, 1]**2
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    # Sort points counterclockwise (clockwise in image)
    if not is_clockwise(corners_sorted[:, :2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners_sorted.reshape(-1)


def is_clockwise(points):
    """Check whether a sequence of points is clockwise ordered."""
    # points is a list of 2D points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0


