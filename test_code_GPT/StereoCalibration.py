import cv2
import numpy as np
import os
import pickle

class StereoCalibration:
    def __init__(self, left_images_folder, right_images_folder, stereo_left_folder, stereo_right_folder):
        self.left_images_folder = left_images_folder
        self.right_images_folder = right_images_folder
        self.stereo_left_folder = stereo_left_folder
        self.stereo_right_folder = stereo_right_folder

        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None  # Вращение
        self.T = None  # Трансляция

    def find_corners(self, images_folder, chessboard_size, square_size):
        obj_points = []
        img_points = []

        # 3D точки в реальном пространстве, масштабированные по square_size
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Масштабируем объектные точки в зависимости от размера клетки

        for img_name in os.listdir(images_folder):
            img_path = os.path.join(images_folder, img_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                img_points.append(corners)
                obj_points.append(objp)

        return obj_points, img_points


    def calibrate(self, chessboard_size, square_size):
        # Калибровка левой камеры
        obj_points_left, img_points_left = self.find_corners(self.left_images_folder, chessboard_size, square_size)
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(obj_points_left, img_points_left, (3264, 2448), None, None)

        # Калибровка правой камеры
        obj_points_right, img_points_right = self.find_corners(self.right_images_folder, chessboard_size, square_size)
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(obj_points_right, img_points_right, (3264, 2448), None, None)

        # Стереокалибровка для парных изображений
        stereo_obj_points, stereo_img_points_left = self.find_corners(self.stereo_left_folder, chessboard_size, square_size)
        _, stereo_img_points_right = self.find_corners(self.stereo_right_folder, chessboard_size, square_size)

        if len(stereo_img_points_left) != len(stereo_img_points_right):
            raise ValueError(f"Количество валидных парных изображений не совпадает: левых — {len(stereo_img_points_left)}, правых — {len(stereo_img_points_right)}")

        # Стереокалибровка
        _, self.camera_matrix_left, self.dist_coeffs_left, self.camera_matrix_right, self.dist_coeffs_right, self.R, self.T, _, _ = cv2.stereoCalibrate(
            stereo_obj_points, stereo_img_points_left, stereo_img_points_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            (3264, 2448), criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-6)
        )


    def save_calibration(self, filepath):
        calibration_data = {
            'camera_matrix_left': self.camera_matrix_left,
            'dist_coeffs_left': self.dist_coeffs_left,
            'camera_matrix_right': self.camera_matrix_right,
            'dist_coeffs_right': self.dist_coeffs_right,
            'R': self.R,
            'T': self.T
        }

        with open(filepath, 'wb') as f:
            pickle.dump(calibration_data, f)




stereo_calib = StereoCalibration("D:\Dev\StereoPair\img\onli_left_cam", "D:\Dev\StereoPair\img\onli_right_cam", "D:\Dev\StereoPair\img\sterio\left_cam", "D:\Dev\StereoPair\img\sterio\\right_cam")
stereo_calib.calibrate((11, 11), 10.0)  # Размер шахматной доски и размер клетки
stereo_calib.save_calibration("calibration_data.pkl")
