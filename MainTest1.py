import imageio
from ImageStitching import ImageStitching
import matplotlib.pyplot as plt  # �פJ Matplotlib�A�Ω�Ϲ����

def main():
    print("MainTest1.py")

    NewStitchedImage=ImageStitching('ba22.jpg', 'ba1.jpg')

    #�g�J�s�ɮ�
    if (NewStitchedImage is not None):
        imageio.imwrite("NewStitchedImage.jpeg", NewStitchedImage)

        plt.close('all')  # �����Ҧ� Matplotlib ���f (�]���b�X�ֹϤ����w�h���ϥΡA�A���I�s�|�⨺�ǵ������u�X��)
        plt.imshow(NewStitchedImage)#�a�J�Ϥ�
        plt.show()#��ܹϤ�
    else:
        print("NewStitchedImage is None")



if __name__ == "__main__":
    main()