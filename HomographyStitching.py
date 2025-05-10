import cv2
import numpy as np  # 匯入 NumPy，用於數值計算

def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):   
    """ 
    
    Perform homography stitching by calculating the homography matrix.
    converting the keypoints to numpy arrays before passing them for calculating Homography Matrix.
    
    Because we are supposed to pass 2 arrays of coordinates to cv2.findHomography, as in I have these points in image-1, and I have points in image-2, so now what is the homography matrix to transform the points from image 1 to image 2

    Args:
        keypoints_train_img (list): List of keypoints from the training image.
        keypoints_query_img (list): List of keypoints from the query image.
        matches (list): List of matches between key points in the training and query images.
        reprojThresh (float): Reprojection threshold for the RANSAC algorithm.

    Returns:
        tuple: Tuple containing the matches, homography matrix, and status.
               - matches (list): List of matches between key points in the training and query images.
               - H (numpy.ndarray): Homography matrix.
               - status (numpy.ndarray): Status of inlier points.

    Note:
        The minimum number of matches required for calculating the homography is 4.
    
    
    """
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])
    
    ''' For findHomography() - I need to have an assumption of a minimum of correspondence points that are present between the 2 images. Here, I am assuming that Minimum Match Count to be 4 '''
    if len(matches) > 4:
        # construct the two sets of points
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
        
        # Calculate the homography between the sets of points
        (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)

        return (matches, H, status)
    else:
        return None