import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StereoDepthMap:
    def __init__(self, calibration_file):
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None
        self.T = None
        self.load_calibration(calibration_file)

    def load_calibration(self, filepath):
        with open(filepath, 'rb') as f:
            calibration_data = pickle.load(f)
            self.camera_matrix_left = calibration_data['camera_matrix_left']
            self.dist_coeffs_left = calibration_data['dist_coeffs_left']
            self.camera_matrix_right = calibration_data['camera_matrix_right']
            self.dist_coeffs_right = calibration_data['dist_coeffs_right']
            self.R = calibration_data['R']
            self.T = calibration_data['T']

    def compute_depth_map(self, left_img_path, right_img_path):
        # Чтение изображений
        img_left = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Параметры SGBM (можно изменять для улучшения результата)
        min_disparity = 0
        num_disparities = 16 * 6  # Должно быть кратно 16
        block_size = 5  # Размер блока для расчета диспаратности
        uniqueness_ratio = 10
        speckle_window_size = 100
        speckle_range = 32
        disp12_max_diff = 1

        # Создание объекта SGBM
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            disp12MaxDiff=disp12_max_diff
        )

        # Построение карты диспаратности
        disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

        # Нормализация карты глубины для отображения
        depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)

        return depth_map


    def generate_3d_plot(self, disparity_map, left_img_path, step=30):
        # Чтение исходного изображения для получения размеров
        img_left = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img_left.shape

        # Создание сетки координат (x, y)
        Q = cv2.reprojectImageTo3D(disparity_map, self._get_Q_matrix())
        
        # Извлечение точек с валидными диспаратностями
        mask = disparity_map > disparity_map.min()
        points_3d = Q[mask]
        colors = img_left[mask]  # Получение значений яркости точек для цвета

        points_3d = points_3d[::step]
        colors = colors[::step]

        # Построение 3D-графика
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Разделение точек на координаты
        xs = points_3d[:, 0]
        ys = points_3d[:, 1]
        zs = points_3d[:, 2]
        ax.scatter(xs, ys, zs, c=colors, cmap='gray', s=1)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.show()

    def _get_Q_matrix(self):
        # Параметры камеры
        focal_length = self.camera_matrix_left[0, 0]  # Предполагаем, что fx для левой камеры используется как фокусное расстояние
        cx = self.camera_matrix_left[0, 2]
        cy = self.camera_matrix_left[1, 2]
        baseline = np.linalg.norm(self.T)  # Длина базиса (расстояние между камерами)

        # Матрица Q для преобразования диспаратности в 3D
        Q = np.float32([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, focal_length],
            [0, 0, -1/baseline, 0]
        ])
        return Q

    def display_depth_map(self, depth_map):
        imS = cv2.resize(depth_map, (960, 540))
        cv2.imshow('Depth Map', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def generate_3d_plot_from_points(self, disparity_map, points):
        """
        Строит 3D карту по коллекции точек и карте глубины.
        
        :param disparity_map: Карта глубины (диспаратности).
        :param points: Коллекция точек с координатами (x, y) на изображении.
        :param step: Шаг для прореживания точек, чтобы не отображать слишком много данных.
        """
        # Подготовка 3D точек для заданных координат
        Q = cv2.reprojectImageTo3D(disparity_map, self._get_Q_matrix())

        # Создание пустых массивов для хранения 3D координат и цвета
        points_3d = []
        colors = []

        for (x, y) in points:
            # Преобразуем x и y в целые числа
            x, y = int(x), int(y)
            
            # Извлечение 3D координат для каждой точки
            point_3d = Q[y, x]  # Q имеет форму (h, w, 3), поэтому обращаемся как к (y, x)
            if np.all(point_3d != 0):  # Пропускаем невалидные точки
                points_3d.append(point_3d)

                # Добавляем яркость или цвет для точки
                color = disparity_map[y, x]
                colors.append(color)

        # Преобразуем списки в массивы NumPy для удобства работы
        points_3d = np.array(points_3d)
        colors = np.array(colors)

        # Построение 3D графика
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Разделение точек на координаты
        xs = points_3d[:, 0]
        ys = points_3d[:, 1]
        zs = points_3d[:, 2]
        ax.scatter(xs, ys, zs, c=colors, cmap='gray', s=1)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.show()



    def find_chessboard_corners(self, image_path, pattern_size=(11, 11)):
        """
        Находит координаты узлов шахматной доски на изображении.
        
        :param image_path: Путь к изображению с шахматной доской.
        :param pattern_size: Размерность шахматной доски (число внутренних углов по ширине и высоте).
        :return: Коллекция координат узлов (углов) шахматной доски.
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Уточнение углов (опционально, можно удалить, если не нужно)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Возврат координат углов
            return corners.reshape(-1, 2)  # Преобразование в массив координат (x, y)
        else:
            raise ValueError("Шахматная доска не найдена на изображении.")



depth_map = StereoDepthMap("calibration_data.pkl")
depth = depth_map.compute_depth_map("D:\Dev\StereoPair\StereoPair\left.jpg", "D:\Dev\StereoPair\StereoPair\\right.jpg")

depth_map.display_depth_map(depth)
# depth_map.generate_3d_plot(depth, 'D:\Dev\StereoPair\StereoPair\left.jpg')

points = depth_map.find_chessboard_corners('D:\Dev\StereoPair\StereoPair\left.jpg')
depth_map.generate_3d_plot_from_points(depth, points)
