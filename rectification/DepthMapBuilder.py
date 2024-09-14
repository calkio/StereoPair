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


    def triangulate_points(self, points_file_left, points_file_right):
        # Загрузить точки из файлов
        points_left = np.loadtxt(points_file_left, delimiter=' ')
        points_right = np.loadtxt(points_file_right, delimiter=' ')

        # Параметры камеры и ректификации
        with open(r'D:\Dev\StereoPair\StereoPair\rectification\calibration_params.pkl', 'rb') as f:
            calib_params = pickle.load(f)
        
        mtxL = calib_params['mtxL']
        distL = calib_params['distL']
        mtxR = calib_params['mtxR']
        distR = calib_params['distR']
        R = calib_params['R']
        T = calib_params['T']

        # Преобразование точек в формат, который ожидает функция cv2.undistortPoints
        points_left = np.array([list(map(float, p.split())) for p in open(points_file_left)])
        points_right = np.array([list(map(float, p.split())) for p in open(points_file_right)])

        # Трансформация точек в нормализованные координаты
        points_left_normalized = cv2.undistortPoints(np.expand_dims(points_left, axis=1), mtxL, distL)
        points_right_normalized = cv2.undistortPoints(np.expand_dims(points_right, axis=1), mtxR, distR)

        # Убедитесь, что точки нормализованы
        points_left_normalized = np.squeeze(points_left_normalized)
        points_right_normalized = np.squeeze(points_right_normalized)

        # Получение матриц проекции
        P1 = np.hstack((mtxL, np.zeros((3, 1))))
        P2 = np.hstack((np.dot(mtxR, R), T.reshape(-1, 1)))

        # Проверьте размерность P1 и P2
        if P1.shape != (3, 4) or P2.shape != (3, 4):
            raise ValueError(f"Matrix dimensions are incorrect: P1.shape = {P1.shape}, P2.shape = {P2.shape}")

        # Триангуляция точек
        points_4D_homogeneous = cv2.triangulatePoints(P1, P2, points_left_normalized.T, points_right_normalized.T)
        points_3D = points_4D_homogeneous[:3] / points_4D_homogeneous[3]

        points_3D -= points_3D.mean(axis=1, keepdims=True)
        max_range = np.max(np.abs(points_3D))
        points_3D /= max_range

        # Визуализация точек
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Отображение точек
        ax.scatter(points_3D[0], points_3D[1], points_3D[2], c='r', marker='o')
        
        # Установка меток осей
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        a = 1
        # Установка масштабов осей
        ax.set_xlim([-a, a])
        ax.set_ylim([-a, a])
        ax.set_zlim([-a, a])

        plt.show()

depth_builder = DepthMapBuilder()
depth_builder.triangulate_points(r'D:\Dev\StereoPair\StereoPair\GenerateData\NodesLeft.txt', r'D:\Dev\StereoPair\StereoPair\GenerateData\NodesRight.txt')
# imgL = cv2.imread(r'D:\Dev\StereoPair\StereoPair\left.png')
# imgR = cv2.imread(r'D:\Dev\StereoPair\StereoPair\right.png')
# rectified_left, rectified_right, depth_map, disparity = depth_builder.build_depth_map(imgL, imgR)

# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# disparity_normalized = np.uint8(disparity_normalized)
# Вывод карты глубины
# imS = cv2.resize(disparity_normalized, (960, 580))
# cv2.imshow('Disparity', imS)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.figure(figsize=(10, 7))
# plt.imshow(disparity_normalized, cmap='plasma')
# plt.colorbar()  # Отображает цветовую шкалу
# plt.title('Depth Map')
# plt.show()

# imSL = cv2.resize(rectified_left, (960, 540))
# cv2.imshow('Rectified Left', imSL)
# imSR = cv2.resize(rectified_right, (960, 540))
# cv2.imshow('Rectified Right', imSR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
