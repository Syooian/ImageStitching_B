import cv2  # 匯入 OpenCV，用於圖像處理
import numpy as np  # 匯入 NumPy，用於數值計算
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
import imageio  # 匯入 ImageIO，用於圖像讀取與保存
cv2.ocl.setUseOpenCL(False)  # 禁用 OpenCL 加速以避免潛在的兼容性問題
import warnings  # 匯入 warnings 模組，用於處理警告訊息
warnings.filterwarnings('ignore')  # 忽略所有警告訊息

def main():
    print("Start stitching")  # 輸出開始拼接的訊息

    feature_extraction_algo = 'sift'  # 設定特徵提取算法為 SIFT

    feature_to_match = 'bf'  # 設定特徵匹配方法為暴力匹配（BFMatcher）

    # 確保訓練圖片是將被變換的圖片
    train_photo = cv2.imread('svx2.jpg')  # 讀取訓練圖片

    # OpenCV 的顏色通道順序為 BGR，需轉換為 RGB 以便 Matplotlib 正確顯示
    train_photo = cv2.cvtColor(train_photo, cv2.COLOR_BGR2RGB)

    # 將訓練圖片轉換為灰階
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

    # 對查詢圖片執行相同操作
    query_photo = cv2.imread('svx1.jpg')  # 讀取查詢圖片
    query_photo = cv2.cvtColor(query_photo, cv2.COLOR_BGR2RGB)  # 轉換為 RGB
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)  # 轉換為灰階

    # 顯示查詢圖片和訓練圖片
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(query_photo, cmap="gray")  # 顯示查詢圖片
    ax1.set_xlabel("Query image", fontsize=14)  # 設定標籤

    ax2.imshow(train_photo, cmap="gray")  # 顯示訓練圖片
    ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)  # 設定標籤

    #==================顯示圖
    #plt.savefig("./_"+'.jpeg', bbox_inches='tight', dpi=300, format='jpeg')  # 保存圖像（目前註解掉）

    #plt.show()  # 顯示圖像（目前註解掉）
    #==================顯示圖

    # 提取訓練圖片的特徵點和描述符
    keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method=feature_extraction_algo)

    # 提取查詢圖片的特徵點和描述符
    keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method=feature_extraction_algo)

    # 遍歷查詢圖片的特徵點，提取相關屬性（僅作為範例，未使用）
    for keypoint in keypoints_query_img:
        x, y = keypoint.pt  # 特徵點的座標
        size = keypoint.size  # 特徵點的大小
        orientation = keypoint.angle  # 特徵點的方向
        response = keypoint.response  # 特徵點的響應值
        octave = keypoint.octave  # 特徵點所在的金字塔層
        class_id = keypoint.class_id  # 特徵點的類別 ID

    features_query_img.shape  # 獲取查詢圖片特徵描述符的形狀（未使用）

    #圖片合併
    #切縫合併

    # 顯示檢測到的特徵點
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(train_photo_gray, keypoints_train_img, None, color=(0, 255, 0)))  # 訓練圖片的特徵點
    ax1.set_xlabel("(a)", fontsize=14)

    ax2.imshow(cv2.drawKeypoints(query_photo_gray, keypoints_query_img, None, color=(0, 255, 0)))  # 查詢圖片的特徵點
    ax2.set_xlabel("(b)", fontsize=14)

    #==================顯示圖
    #plt.savefig("./output/" + feature_extraction_algo + "_features_img_"+'.jpeg', bbox_inches='tight', dpi=300,  format='jpeg')  # 保存特徵點圖像（目前註解掉）
    #plt.show()  # 顯示圖像（目前註解掉）
    #==================顯示圖

    # 使用 BFMatcher 進行特徵匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(features_train_img, features_query_img)

    # 按照距離排序匹配結果
    matches = sorted(matches, key=lambda x: x.distance)

    # 呼叫 homography_stitching 函數計算單應性矩陣
    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)

    if M is None:  # 如果無法計算單應性矩陣，輸出錯誤訊息
        print("Error!")

    (matches, Homography_Matrix, status) = M  # 解包返回值

    # 計算拼接後圖像的寬度和高度
    width = query_photo.shape[1] + train_photo.shape[1]
    print("Total width : ", width)  # 輸出總寬度

    height = max(query_photo.shape[0], train_photo.shape[0])
    print("Total height : ", height)  # 輸出總高度

    # 使用單應性矩陣進行透視變換，將訓練圖片對齊到查詢圖片
    result = cv2.warpPerspective(train_photo, Homography_Matrix, (width, height))

    # 將查詢圖片的像素覆蓋到結果圖像中
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo

    # 顯示拼接結果
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(result)

    # 保存拼接結果
    imageio.imwrite("./output/horizontal_panorama_img_"+'.jpeg', result)

    plt.show()  # 顯示圖像

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