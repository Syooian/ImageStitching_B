from _pytest.tmpdir import RetentionType
import cv2
import ImageStitching
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
from datetime import datetime
from ImageStitchingEnum import FeatureExtractionAlgoEnum, FeatureToMatchEnum  # 匯入自定義的特徵提取算法枚舉類

def main():
    print("WebCamTest1.py")

    WebCams=[]
    
    for a in range(2):
        print("啟動攝影機"+str(a)+"中...")

        WebCams.append(cv2.VideoCapture(a))
        if WebCams[a].isOpened():
            print("攝影機"+str(a)+"已開啟")
        else:
            print("無法開啟攝影機"+str(a))
            return

    Frames = [None] * len(WebCams)

    Sec=0

    while True:
        # if datetime.now().second != Sec:
        #     print(datetime.now())
        #     Sec=datetime.now().second

        #StartStitching=datetime.now()

        # 讀取攝影機畫面
        for a in range(len(WebCams)):
            print("讀取攝影機"+str(a))
            Ret, Frame=WebCams[a].read()
            if not Ret:
                print("無法讀取畫面"+str(a))
                break

            Frames[a]=Frame

            # 顯示攝影機畫面
            cv2.imshow("WebCam"+str(a), Frame)


        #if(cv2.waitKey(1) & 0xFF==ord('s')):#按下S鍵時合併一次，按鍵偵測一定要至少有一個cv2.imshow才會有作用
        #print("WebCamTest1 Start stitching")

        NewStitchedImage = ImageStitching.StitchImageBySource(Frames[1], Frames[0], FeatureExtractionAlgo=FeatureExtractionAlgoEnum.ORB,FeatureToMatch=FeatureToMatchEnum.KNN)
        # 如果彈出Demo畫面，ShowPhoto需直接指定False才不會跑圖出來，不知道是不是Bug
        '''
        1. 特徵提取與匹配
            •	問題: 特徵提取與匹配是整個拼接過程中最耗時的部分，尤其是使用 SIFT 或其他高精度算法。
            •	優化建議:
            •	改用更快的特徵提取算法: 如果對精度要求不高，可以考慮使用 ORB（Oriented FAST and Rotated BRIEF），它比 SIFT 快得多。
            •	減少特徵點數量: 在 select_descriptor_methods 中限制提取的特徵點數量，例如只提取前 500 個最強的特徵點。
            •	並行處理: 如果硬體支持，考慮使用多線程或 GPU 加速（如 OpenCV 的 CUDA 支持）來加速特徵提取和匹配。

        2. 特徵匹配策略
            •	問題: 使用 BFMatcher 或 KNN 進行特徵匹配可能會導致計算量過大。
            •	優化建議:
            •	調整 KNN 的 ratio: 減少匹配的比率閾值（例如從 0.75 降到 0.6），以減少匹配數量。
            •	限制匹配數量: 僅保留前 N 個最佳匹配（例如 100 個）。
            •	使用 FLANN 匹配器: FLANN（Fast Library for Approximate Nearest Neighbors）比 BFMatcher 更快，特別是在大數據集上。

        3. 單應性矩陣計算
            •	問題: 單應性矩陣的計算可能因為匹配點過多而變慢。
            •	優化建議:
            •	調整 reprojThresh: 減小重投影閾值（例如從 4 降到 3），以過濾掉更多的錯誤匹配點。
            •	RANSAC 優化: 確保 RANSAC 的參數設置合理，避免過多的迭代。

        4. 圖像處理與顯示
            •	問題: 圖像轉換和顯示可能會增加不必要的開銷。
            •	優化建議:
            •	避免不必要的顯示: 僅在需要時啟用 ShowPhoto 和 SavePhoto。
            •	減少圖像大小: 在拼接前縮小圖像尺寸（例如使用 cv2.resize），然後在需要時再放大。

        5. 其他優化
            •	問題: 整體代碼執行效率可能受限於 Python 的單線程性能。
            •	優化建議:
            •	使用 C++ 或其他高效語言重寫關鍵部分: OpenCV 支持 C++，可以將性能瓶頸部分移植到 C++ 中。
            •	啟用 OpenCV 的多線程支持: 確保 OpenCV 的多線程功能已啟用（如 TBB 或 OpenMP）。
        '''

        #plt.close('all')  # 關閉所有 Matplotlib 窗口 (因為在合併圖片中已多次使用，再次呼叫會把那些視窗都彈出來)

        if NewStitchedImage is None:
            print("合併失敗")
            #continue
        else:
            # plt.imshow(NewStitchedImage)  # 帶入圖片
            # plt.show()  # 顯示圖片
            cv2.imshow("NewStitchedImage", NewStitchedImage)

        #print("合併所耗時間："+str(datetime.now()-StartStitching))

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    for Webcam in WebCams:
        Webcam.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()