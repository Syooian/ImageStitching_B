import cv2  # �פJ OpenCV�A�Ω�Ϲ��B�z

def select_descriptor_methods(image, method=None):    
    
    assert method is not None, "Please define a feature descriptor method. accepted Values are: 'sift', 'surf'"
    
    match method:
        case 'sift':
            descriptor = cv2.SIFT_create()
        case 'surf':
            descriptor = cv2.SURF_create()
        case 'brisk':
            descriptor = cv2.BRISK_create()
        # case 'brief':
        #     descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        case 'orb':
            descriptor = cv2.ORB_create()
        
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)