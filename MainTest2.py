from pickle import TRUE
import imageio
import ImageStitching
import matplotlib.pyplot as plt  # �פJ Matplotlib�A�Ω�Ϲ����
from ImageStitching import FeatureExtractionAlgoEnum
import cv2


def main():
    print("MainTest2.py")

    train_photo = cv2.imread('ba2.jpg')  # Ū���V�m�Ϥ�
    # print(train_photo)
    # plt.imshow(train_photo)  # ��ܰV�m�Ϥ�
    # plt.show()

    query_photo = cv2.imread('ba1.jpg')  # Ū���V�m�Ϥ�
    # plt.imshow(query_photo)  # ��ܰV�m�Ϥ�
    # plt.show()

    NewStitchedImage = ImageStitching.StitchImageBySource(
        train_photo, query_photo)

    plt.close('all')  # �����Ҧ� Matplotlib ���f (�]���b�X�ֹϤ����w�h���ϥΡA�A���I�s�|�⨺�ǵ������u�X��)
    plt.imshow(NewStitchedImage)  # �a�J�Ϥ�
    plt.show()  # ��ܹϤ�


if __name__ == "__main__":
    main()