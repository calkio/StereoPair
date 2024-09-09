import pickle
import cv2
import numpy as np

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


    def display_depth_map(self, depth_map):
        imS = cv2.resize(depth_map, (960, 540))
        cv2.imshow('Depth Map', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




depth_map = StereoDepthMap("calibration_data.pkl")
depth = depth_map.compute_depth_map("D:\Dev\StereoPair\StereoPair\left.jpg", "D:\Dev\StereoPair\StereoPair\\right.jpg")
depth_map.display_depth_map(depth)
