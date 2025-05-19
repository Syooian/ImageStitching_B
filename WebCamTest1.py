from _pytest.tmpdir import RetentionType
import cv2
import ImageStitching
import matplotlib.pyplot as plt  # 匯入 Matplotlib，用於圖像顯示
from datetime import datetime

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

    Sec=0

    while True:
        # if datetime.now().second != Sec:
        #     print(datetime.now())
        #     Sec=datetime.now().second

        StartStitching=datetime.now()

        # 讀取攝影機畫面
        for WebCam in WebCams:
            Ret, Frame=WebCam.read()
            if not Ret:
                print("無法讀取畫面"+str(WebCam))
                break

            # 顯示攝影機畫面
            cv2.imshow('WebCam', Frame)


        #if(cv2.waitKey(1) & 0xFF==ord('s')):#按下S鍵時合併一次，按鍵偵測一定要至少有一個cv2.imshow才會有作用
        #print("WebCamTest1 Start stitching")

        NewStitchedImage = ImageStitching.StitchImageBySource(WebCams[1], WebCams[0])

        #plt.close('all')  # 關閉所有 Matplotlib 窗口 (因為在合併圖片中已多次使用，再次呼叫會把那些視窗都彈出來)

        if NewStitchedImage is None:
            print("合併失敗")
            #continue
        else:
            # plt.imshow(NewStitchedImage)  # 帶入圖片
            # plt.show()  # 顯示圖片
            cv2.imshow("NewStitchedImage", NewStitchedImage)

        print("合併所耗時間："+str(datetime.now()-StartStitching))

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    for a in WebCams:
        WebCams[a].release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()