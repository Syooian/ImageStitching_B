from enum import Enum

class FeatureExtractionAlgoEnum(Enum):
    SIFT=0
    SURF=1
    BRISK=2
    BRIEF=3
    ORB=4
class FeatureToMatchEnum(Enum):
    BF = 0
    KNN=1