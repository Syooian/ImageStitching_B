# OpenCV安裝：pip install opencv-python.import warnings  # 匯入 warnings 模組，用於處理警告訊息
from pathlib import Path
import cv2  # 匯入 OpenCV，用於圖像處理
#pip install matplotlib
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
#pip install imageio
import imageio # 匯入 ImageIO，用於圖像讀取與保存
import numpy as np 

from datetime import datetime

#因有用到Switch寫法，所以Python需使用3.10以上版本

from KeyPointsMatching import key_points_matching, key_points_matching_KNN
from SelectDescriptor import select_descriptor_methods  
from HomographyStitching import homography_stitching  # 匯入自定義的單應性拼接函數
from enum import Enum

cv2.ocl.setUseOpenCL(False)  # 禁用 OpenCL 加速以避免潛在的兼容性問題
import warnings  # 匯入 warnings 模組，用於處理警告訊息
warnings.filterwarnings('ignore')  # 忽略所有警告訊息

def main():
    StitchImageByFileName('ba2.jpg', 'ba1.jpg', True, True)

StartStitchingTime=0

def StitchImageBySource(TrainPhoto, QueryPhoto, ShowPhoto=False, SavePhoto=False, FeatureExtractionAlgo='sift', FeatureToMatch='bf'):
    print("StitchImageBySource")

    if TrainPhoto is None:
        print("TrainPhoto is None")
        return None
    if QueryPhoto is None:
        print("QueryPhoto is None")
        return None

    return __Stitching(TrainPhoto, QueryPhoto, ShowPhoto, SavePhoto, FeatureExtractionAlgo, FeatureToMatch)

def StitchImageByFileName(TrainPhoto, QueryPhoto, ShowPhoto=False, SavePhoto=False, FeatureExtractionAlgo='sift', FeatureToMatch='bf'):
    print("StitchImageByFileName "+TrainPhoto +" & "+ QueryPhoto)

    if not Path(TrainPhoto).is_file():
        print("TrainPhoto "+TrainPhoto + " is not exists")
        return None
    if not Path(QueryPhoto).is_file():
        print("QueryPhoto "+QueryPhoto + " is not exists")
        return None

    # 確保訓練圖片是將被變換的圖片
    train_photo = cv2.imread(TrainPhoto)  # 讀取訓練圖片
    # 對查詢圖片執行相同操作
    query_photo = cv2.imread(QueryPhoto)  # 讀取查詢圖片

    return __Stitching(train_photo, query_photo, ShowPhoto, SavePhoto, FeatureExtractionAlgo, FeatureToMatch)

# 私有
def __Stitching(TrainPhoto, QueryPhoto, ShowPhoto, SavePhoto, FeatureExtractionAlgo, FeatureToMatch):
    print("Start stitching FeatureExtractionAlgo : "+FeatureExtractionAlgo+", FeatureToMatch : "+FeatureToMatch)  # 輸出開始拼接的訊息

    StartStitchingTime = datetime.now()
    
    # SIFT
    # SURF
    # BRISK
    # BRIEF
    # ORB
    #feature_extraction_algo = FeatureExtractionAlgo  # 設定特徵提取算法為 SIFT

    #bf
    #knn
    #feature_to_match = FeatureToMatch  # 設定特徵匹配方法為暴力匹配（BFMatcher）


    # OpenCV 的顏色通道順序為 BGR，需轉換為 RGB 以便 Matplotlib 正確顯示
    train_photo = cv2.cvtColor(TrainPhoto, cv2.COLOR_BGR2RGB)
    # 將訓練圖片轉換為灰階
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)


    query_photo = cv2.cvtColor(QueryPhoto, cv2.COLOR_BGR2RGB)  # 轉換為 RGB
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)  # 轉換為灰階

    # 顯示查詢圖片和訓練圖片
    if ShowPhoto or SavePhoto:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
        ax1.imshow(query_photo, cmap="gray")  # 顯示查詢圖片
        ax1.set_xlabel("Query image", fontsize=14)  # 設定標籤

        ax2.imshow(train_photo, cmap="gray")  # 顯示訓練圖片
        ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)  # 設定標籤

        #存圖
        if SavePhoto:
            plt.savefig("./output/original_compare"+'.jpeg', bbox_inches='tight', dpi=300, format='jpeg')
    #==================顯示圖


    #plt.show()  # 顯示圖像（目前註解掉）
    #==================顯示圖





    # 提取訓練圖片的特徵點和描述符
    keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method=FeatureExtractionAlgo)

    # 提取查詢圖片的特徵點和描述符
    keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method=FeatureExtractionAlgo)

    # 遍歷查詢圖片的特徵點，提取相關屬性（僅作為範例，未使用）
    for keypoint in keypoints_query_img:
        x, y = keypoint.pt  # 特徵點的座標
        size = keypoint.size  # 特徵點的大小
        orientation = keypoint.angle  # 特徵點的方向
        response = keypoint.response  # 特徵點的響應值
        octave = keypoint.octave  # 特徵點所在的金字塔層
        class_id = keypoint.class_id  # 特徵點的類別 ID

    #print("keypoints_query_img : "+len(keypoints_query_img))  # 輸出查詢圖片的特徵點數量
    features_query_img.shape  # 獲取查詢圖片特徵描述符的形狀（未使用）

    #圖片合併
    #切縫合併

    # 顯示檢測到的特徵點
    if ShowPhoto or SavePhoto:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)

        ax1.imshow(cv2.drawKeypoints(train_photo_gray, keypoints_train_img, None, color=(0, 255, 0)))  # 訓練圖片的特徵點
        ax1.set_xlabel("(a)", fontsize=14)

        ax2.imshow(cv2.drawKeypoints(query_photo_gray, keypoints_query_img, None, color=(0, 255, 0)))  # 查詢圖片的特徵點
        ax2.set_xlabel("(b)", fontsize=14)

        #存圖
        if SavePhoto:
            plt.savefig("./output/" + FeatureExtractionAlgo + "_features_img_"+'.jpeg', bbox_inches='tight', dpi=300,  format='jpeg')  # 保存特徵點圖像（目前註解掉）

    #==================顯示圖
    
    #plt.show()  # 顯示圖像（目前註解掉）
    #==================顯示圖





    # 使用 BFMatcher 進行特徵匹配
    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = bf.match(features_train_img, features_query_img)
    # #matches = bf.knnMatch(features_train_img, features_query_img, k=6)  # 使用 KNN 匹配

    # # 按照距離排序匹配結果
    # matches = sorted(matches, key=lambda x: x.distance)

    #matches = None
    #mapped_features_image = None
    match FeatureToMatch:
        case 'bf':
            matches = key_points_matching(features_train_img, features_query_img, FeatureExtractionAlgo)
            mapped_features_image = cv2.drawMatches(train_photo,keypoints_train_img,query_photo,keypoints_query_img,matches[:100],
                None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        case 'knn':# Now for cross checking draw the feature-mapping lines also with KNN
            matches = key_points_matching_KNN(features_train_img, features_query_img, ratio=0.75, method=FeatureExtractionAlgo)
            mapped_features_image = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img, np.random.choice(matches,100),
                None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if ShowPhoto or SavePhoto:
        fig = plt.figure(figsize=(20, 8))

        plt.imshow(mapped_features_image)
    
    #plt.show()

    #存圖
    if SavePhoto:
        plt.savefig("./output/"+FeatureToMatch+"_matches_img_"+'.jpeg', bbox_inches='tight', dpi=300, format='jpeg')







    # 呼叫 homography_stitching 函數計算單應性矩陣
    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)

    if M is None:  # 如果無法計算單應性矩陣，輸出錯誤訊息
        print("Error!")

    try:
        (matches, Homography_Matrix, status) = M  # 解包返回值
    except Exception as e:
        print("解包返回值 EX : "+str(e))
        return None

    print("Homography_Matrix : ", Homography_Matrix)

    # 計算拼接後圖像的寬度和高度
    width = query_photo.shape[1] + train_photo.shape[1]
    print("Total width : ", width)  # 輸出總寬度

    height = max(query_photo.shape[0], train_photo.shape[0])
    print("Total height : ", height)  # 輸出總高度

    # 使用單應性矩陣進行透視變換，將訓練圖片對齊到查詢圖片
    result = cv2.warpPerspective(train_photo, Homography_Matrix, (width, height))

    # 將查詢圖片的像素覆蓋到結果圖像中
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo

    if ShowPhoto or SavePhoto:
        # 顯示拼接結果
        plt.figure(figsize=(20, 10))
        #plt.axis('off')
        plt.imshow(result)

        # 保存拼接結果
        if SavePhoto:
            imageio.imwrite("./output/horizontal_panorama_img_"+'.jpeg', result)

        #最終顯示圖像
        if ShowPhoto:
            plt.show()  

        # 釋放記憶體
        plt.close('all')  # 關閉 Matplotlib 窗口

    print("合併所耗時間："+str(datetime.now()-StartStitchingTime))

    return result  # 返回拼接結果圖像



class FeatureExtractionAlgoEnum(Enum):
    SIFT=0
    SURF=1
    BRISK=2
    BRIEF=3
    ORB=4
class FeatureToMatchEnum(Enum):
    BF = 0
    KNN=1

if __name__ == "__main__":
    main()