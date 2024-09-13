import cv2
import numpy as np
import glob
import yaml

class AxisDrawer:
    def __init__(self, calibration_file, image_pattern, chessboard_size=(11, 11)):
        """
        :param calibration_file: Путь к файлу с параметрами калибровки камеры (например, 'B.npz')
        :param image_pattern: Шаблон для поиска изображений (например, 'left*.jpg')
        :param chessboard_size: Размеры шахматной доски (внутренние углы)
        """
        self.chessboard_size = chessboard_size
        self.image_pattern = image_pattern
        

        # Загрузка параметров калибровки камеры из YAML-файла
        with open(calibration_file, 'r') as file:
            calibration_data = yaml.safe_load(file)
            self.mtx = np.array(calibration_data['camera_matrix'])
            self.dist = np.array(calibration_data['dist_coeffs'])


        # Критерий для уточнения углов шахматной доски
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 3D координаты углов шахматной доски
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # Оси для отображения (длина осей — 3 единицы)
        self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    def draw_axes(self, img, corners, imgpts):
        """
        Отрисовка осей координат на изображении
        :param img: Изображение
        :param corners: Углы шахматной доски
        :param imgpts: Проекционные точки для осей
        """
        corner = tuple(map(int, corners[0].ravel()))
        img = cv2.line(img, corner, tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 5)
        return img

    
    def process_images(self):
        """
        Обработка изображений и отрисовка осей координат на них
        """
        images = glob.glob(self.image_pattern)
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Поиск углов шахматной доски
            ret, corners = cv2.findChessboardCornersSB(gray, self.chessboard_size, None)
            
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (28, 25), (-1, -1), self.criteria)
                
                # Определение векторов вращения и трансляции
                result = cv2.solvePnPRansac(self.objp, corners2, self.mtx, self.dist)
                if len(result) == 3:
                    rvecs, tvecs = result[:2]  # Используем только rvecs и tvecs
                else:
                    _, rvecs, tvecs, _ = result  # Для совместимости с версиями, возвращающими 3 значения

                # Проекция 3D точек на плоскость изображения
                imgpts, _ = cv2.projectPoints(self.axis, rvecs, tvecs, self.mtx, self.dist)
                
                # Отрисовка осей на изображении
                img = self.draw_axes(img, corners2, imgpts)

                # Уменьшение изображения для удобства просмотра
                scale_factor = 0.25  # Коэффициент уменьшения изображения (50% от оригинального размера)
                img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                
                cv2.imshow('img', img_resized)
                k = cv2.waitKey(0) & 0xFF
                
                # Сохранение изображения, если нажата клавиша 's'
                if k == ord('s'):
                    cv2.imwrite(fname[:6] + '_axes.png', img)
    
    cv2.destroyAllWindows()


# Пример использования
axis_drawer = AxisDrawer(r'D:\Dev\StereoPair\StereoPair\AllCalibrationCam\metrics.yml', r'D:\Dev\StereoPair\img\sterio\right_cam\*.jpg', chessboard_size=(28, 25))
axis_drawer.process_images()
