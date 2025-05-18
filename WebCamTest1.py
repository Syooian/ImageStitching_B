import cv2
import ImageStitching
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
from datetime import datetime

def main():
    print("WebCamTest1.py")

    print("啟動攝影機0中...")
    cap0 = cv2.VideoCapture(0)
    if cap0.isOpened():
        print("攝影機0已開啟")
    else:
        print("無法開啟攝影機0")
        return

    print("啟動攝影機1中...")
    cap1 = cv2.VideoCapture(1)
    if cap0.isOpened():
        print("攝影機1已開啟")
    else:
        print("無法開啟攝影機1")
        return

    Sec=0

    while True:
        if datetime.now().second != Sec:
            print(datetime.now())
            Sec=datetime.now().second


        # 讀取攝影機畫面
        ret0, frame0 = cap0.read()
        if not ret0:
            print("無法讀取畫面0")
            break
        # 顯示WebCam0畫面
        cv2.imshow('WebCam0', frame0)

        ret1, frame1 = cap1.read()
        if not ret1:
            print("無法讀取畫面1")
            break
        # 顯示WebCam1畫面
        cv2.imshow('WebCam1', frame1)

        #if(cv2.waitKey(1) & 0xFF==ord('s')):#按下S鍵時合併一次，按鍵偵測一定要至少有一個cv2.imshow才會有作用
        #print("WebCamTest1 Start stitching")

        NewStitchedImage = ImageStitching.StitchImageBySource(frame1, frame0)

        #plt.close('all')  # 關閉所有 Matplotlib 窗口 (因為在合併圖片中已多次使用，再次呼叫會把那些視窗都彈出來)

        if NewStitchedImage is None:
            print("合併失敗")
            #continue
        else:
            # plt.imshow(NewStitchedImage)  # 帶入圖片
            # plt.show()  # 顯示圖片
            cv2.imshow("NewStitchedImage", NewStitchedImage)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap0.release()
    cap1.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()