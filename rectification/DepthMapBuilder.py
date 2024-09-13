import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

class DepthMapBuilder:
    def __init__(self, rectification_file=r'D:\Dev\StereoPair\StereoPair\rectification\rectification_params.pkl'):
        with open(rectification_file, 'rb') as f:
            self.rectification_params = pickle.load(f)

    def build_depth_map(self, imgL, imgR):
        left_map1 = self.rectification_params['left_map1']
        left_map2 = self.rectification_params['left_map2']
        right_map1 = self.rectification_params['right_map1']
        right_map2 = self.rectification_params['right_map2']
        Q = self.rectification_params['Q']

        # Применение ремаппинга
        rectified_left = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        # Настройка и вычисление карты глубины

        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


        # Создание объекта SGBM
        stereo = cv2.StereoSGBM_create(
            minDisparity=-16,
            numDisparities=16 * 10,
            blockSize=5,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY))

        # Возвращаем карту глубины
        depth_map = cv2.reprojectImageTo3D(disparity, Q)
        return rectified_left, rectified_right, depth_map, disparity


depth_builder = DepthMapBuilder()
imgL = cv2.imread(r'D:\Dev\StereoPair\StereoPair\left.jpg')
imgR = cv2.imread(r'D:\Dev\StereoPair\StereoPair\right.jpg')
rectified_left, rectified_right, depth_map, disparity = depth_builder.build_depth_map(imgL, imgR)

disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)
# Вывод карты глубины
imS = cv2.resize(disparity_normalized, (960, 580))
# cv2.imshow('Disparity', imS)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure(figsize=(10, 7))
plt.imshow(disparity_normalized, cmap='plasma')
plt.colorbar()  # Отображает цветовую шкалу
plt.title('Depth Map')
plt.show()

# imSL = cv2.resize(rectified_left, (960, 540))
# cv2.imshow('Rectified Left', imSL)
# imSR = cv2.resize(rectified_right, (960, 540))
# cv2.imshow('Rectified Right', imSR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
