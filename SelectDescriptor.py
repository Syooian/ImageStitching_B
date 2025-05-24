import cv2  # �פJ OpenCV�A�Ω�Ϲ��B�z
from ImageStitchingEnum import FeatureExtractionAlgoEnum  # 匯入自定義的特徵提取算法枚舉類

def select_descriptor_methods(image, method=None):    
    
    assert method is not None, "Please define a feature descriptor method. accepted Values are: 'sift', 'surf'"
    
    match method:
        case FeatureExtractionAlgoEnum.SIFT:
            descriptor = cv2.SIFT_create()
        case FeatureExtractionAlgoEnum.SURF:
            descriptor = cv2.SURF_create()
        case FeatureExtractionAlgoEnum.BRISK:
            descriptor = cv2.BRISK_create()
        # case 'brief':
        #     descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        case FeatureExtractionAlgoEnum.ORB:
            descriptor = cv2.ORB_create()
        
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)