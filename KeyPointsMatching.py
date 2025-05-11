import cv2

def key_points_matching(features_train_img, features_query_img, method):
    """
    Perform key points matching between the features of a train image and a query image using a specified method.

    Parameters:
    features_train_img (np.ndarray): Features of the train image.
    features_query_img (np.ndarray): Features of the query image.
    method (str): Method to use for matching.

    Returns:
    rawMatches (list): List of raw matches between the train and query images.
    """
    bf = create_matching_object(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(features_train_img,features_query_img)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches with Brute force):", len(rawMatches))
    return rawMatches

def key_points_matching_KNN(features_train_img, features_query_img, ratio, method):
    """
    Perform key points matching using K-Nearest Neighbors (KNN) algorithm.

    Args:
        features_train_img (list): List of features from the training image.
        features_query_img (list): List of features from the query image.
        ratio (float): Ratio threshold for Lowe's ratio test.
        method (string): Method for creating the matching object.

    Returns:
        list: List of matches between key points in the training and query images.
    """
    bf = create_matching_object(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def create_matching_object(method,crossCheck):
    "Create and return a Matcher Object"
    
    # For BF matcher, first we have to create the BFMatcher object using cv2.BFMatcher(). 
    # It takes two optional params. 
    # normType - It specifies the distance measurement
    # crossCheck - which is false by default. If it is true, Matcher returns only those matches 
    # with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match 
    # and vice-versa. 
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf