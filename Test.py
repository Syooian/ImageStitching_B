import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
cv2.ocl.setUseOpenCL(False)
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Start stitching")

    feature_extraction_algo = 'sift'

    feature_to_match = 'bf'

    # Make sure that the train image is the image that will be transformed
    train_photo = cv2.imread('frame1.jpg')

    # OpenCV defines the color channel in the order BGR 
    # Hence converting to RGB for Matplotlib
    train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)

    # converting to grayscale
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

    # Do the same for the query image 
    query_photo = cv2.imread('frame0.jpg')
    query_photo = cv2.cvtColor(query_photo,cv2.COLOR_BGR2RGB)
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

    # Now view/plot the images
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
    ax1.imshow(query_photo, cmap="gray")
    ax1.set_xlabel("Query image", fontsize=14)

    ax2.imshow(train_photo, cmap="gray")
    ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)

    #==================顯示圖
    #plt.savefig("./_"+'.jpeg', bbox_inches='tight', dpi=300, format='jpeg')

    #plt.show()
    #==================顯示圖

    keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method=feature_extraction_algo)

    keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method=feature_extraction_algo)

    for keypoint in keypoints_query_img:
        x,y = keypoint.pt
        size = keypoint.size 
        orientation = keypoint.angle
        response = keypoint.response 
        octave = keypoint.octave
        class_id = keypoint.class_id

    features_query_img.shape

    #圖片合併
    #切縫合併

    # display the keypoints and features detected on both images
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)

    ax1.imshow(cv2.drawKeypoints(train_photo_gray, keypoints_train_img, None, color=(0,255,0)))

    ax1.set_xlabel("(a)", fontsize=14)

    ax2.imshow(cv2.drawKeypoints(query_photo_gray,keypoints_query_img,None,color=(0,255,0)))
    ax2.set_xlabel("(b)", fontsize=14)

    #==================顯示圖
    #plt.savefig("./output/" + feature_extraction_algo + "_features_img_"+'.jpeg', bbox_inches='tight', dpi=300,  format='jpeg')
    #plt.show()
    #==================顯示圖

    # 使用 BFMatcher 進行特徵匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(features_train_img, features_query_img)

    # 按照距離排序匹配結果
    matches = sorted(matches, key=lambda x: x.distance)

    # 呼叫 homography_stitching 函數
    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)

    if M is None:
        print("Error!")

    (matches, Homography_Matrix, status) = M

    # For the calculation of the width and height of the final horizontal panoramic images 
    # I can just add the widths of the individual images and for the height
    # I can take the max from the 2 individual images.

    width = query_photo.shape[1] + train_photo.shape[1]
    print("Total width : ", width) 
    # 2922 - Which is exactly the sum value of the width of 
    # my train.jpg and query.jpg


    height = max(query_photo.shape[0], train_photo.shape[0])
    print("Total height : ", height)


    # otherwise, apply a perspective warp to stitch the images together

    # Now just plug that "Homography_Matrix"  into cv::warpedPerspective and I shall have a warped image1 into image2 frame

    result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))

    # The warpPerspective() function returns an image or video whose size is the same as the size of the original image or video. Hence set the pixels as per my query_photo

    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo

    plt.figure(figsize=(20,10))
    plt.axis('off')
    plt.imshow(result)

    imageio.imwrite("./output/horizontal_panorama_img_"+'.jpeg', result)

    plt.show()





def select_descriptor_methods(image, method=None):    
    
    assert method is not None, "Please define a feature descriptor method. accepted Values are: 'sift', 'surf'"
    
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)

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

if __name__ == "__main__":
    main()