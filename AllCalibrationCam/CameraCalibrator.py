import cv2
import numpy as np
import os
import glob
import yaml

class CameraCalibrator:
    def __init__(self, image_dir, chessboard_size=(11, 11), square_size=10.0):
        """
        :param image_dir: Путь к папке с изображениями шахматной доски
        :param chessboard_size: Размеры шахматной доски (внутренние углы)
        :param square_size: Размер квадрата шахматной доски (в реальных единицах, например, миллиметрах)
        """
        self.image_dir = image_dir
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Массивы для хранения объектов и изображений точек
        self.obj_points = []  # 3D точки в реальном пространстве
        self.img_points = []  # 2D точки в плоскости изображения

        # Настройка координат углов шахматной доски
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # Масштабируем под реальный размер квадрата

    def find_corners(self):
        # Получаем список изображений из папки
        images = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        for image_path in images:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Поиск углов шахматной доски
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.obj_points.append(self.objp)
                self.img_points.append(corners)

                # Отрисовка углов на изображении (для проверки)
                cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Corners', 1920, 1080)  # указываем размеры окна
                cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret)
                cv2.imshow('Corners', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

    def calibrate_camera(self):
        # Загрузить изображения и найти углы шахматной доски
        self.find_corners()

        # Калибровка камеры
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, 
                                                                            (3264, 2448), None, None)

        # Сохранение параметров в файл YAML
        metrics = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "rotation_vectors": [rvec.tolist() for rvec in rvecs],
            "translation_vectors": [tvec.tolist() for tvec in tvecs]
        }
        with open('AllCalibrationCam/metrics.yml', 'w') as file:
            yaml.dump(metrics, file)

        print("Калибровка завершена. Параметры сохранены в 'metrics.yml'")

# Пример использования
calibrator = CameraCalibrator(r'D:\Dev\StereoPair\img\onli_left_cam', chessboard_size=(11, 11), square_size=10.0)
calibrator.calibrate_camera()