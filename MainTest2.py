from pickle import TRUE
import imageio
import ImageStitching
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
from ImageStitching import FeatureExtractionAlgoEnum
import cv2


def main():
    print("MainTest2.py")

    train_photo = cv2.imread('ba2.jpg')  # 讀取訓練圖片
    # print(train_photo)
    # plt.imshow(train_photo)  # 顯示訓練圖片
    # plt.show()

    query_photo = cv2.imread('ba1.jpg')  # 讀取訓練圖片
    # plt.imshow(query_photo)  # 顯示訓練圖片
    # plt.show()

    NewStitchedImage = ImageStitching.StitchImageBySource(
        train_photo, query_photo)

    plt.close('all')  # 關閉所有 Matplotlib 窗口 (因為在合併圖片中已多次使用，再次呼叫會把那些視窗都彈出來)
    plt.imshow(NewStitchedImage)  # 帶入圖片
    plt.show()  # 顯示圖片


if __name__ == "__main__":
    main()