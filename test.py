import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
imgL = cv.imread(r"D:\Dev\StereoPair\img\sterio\left_cam\cam1_20240913_182029.jpg", cv.IMREAD_GRAYSCALE)
imgR = cv.imread(r"D:\Dev\StereoPair\img\sterio\right_cam\cam2_20240913_182029.jpg", cv.IMREAD_GRAYSCALE)
 
stereo = cv.StereoSGBM_create(
            minDisparity=-16,
            numDisparities=16 * 10,
            blockSize=5,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()