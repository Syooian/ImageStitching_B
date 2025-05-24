import imageio
import ImageStitching
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
from ImageStitchingEnum import FeatureExtractionAlgoEnum  # 匯入自定義的特徵提取算法枚舉類

def main():
    print("MainTest1.py")

    fea = FeatureExtractionAlgoEnum.SIFT
    print(fea.SIFT.name+" : "+str(fea.SIFT.value));#SIFT : 0

    NewStitchedImage=ImageStitching.StitchImageByFileName('ba2.jpg', 'ba1.jpg')

    #寫入新檔案
    if (NewStitchedImage is not None):
        imageio.imwrite("NewStitchedImage.jpeg", NewStitchedImage)

        plt.close('all')  # 關閉所有 Matplotlib 窗口 (因為在合併圖片中已多次使用，再次呼叫會把那些視窗都彈出來)
        plt.imshow(NewStitchedImage)#帶入圖片
        plt.show()#顯示圖片
    else:
        print("NewStitchedImage is None")



if __name__ == "__main__":
    main()