import imageio
from ImageStitching import ImageStitching
import matplotlib.pyplot as plt  # �פJ Matplotlib�A�Ω�Ϲ����

def main():
    print("MainTest1.py")

    NewStitchedImage=ImageStitching('ba2.jpg', 'ba1.jpg')

    #�g�J�s�ɮ�
    imageio.imwrite("NewStitchedImage.jpeg", NewStitchedImage)

    plt.close('all')  # �����Ҧ� Matplotlib ���f (�]���b�X�ֹϤ����w�h���ϥΡA�A���I�s�|�⨺�ǵ������u�X��)
    plt.imshow(NewStitchedImage)#�a�J�Ϥ�
    plt.show()#��ܹϤ�



if __name__ == "__main__":
    main()