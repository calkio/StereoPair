import cv2
import numpy as np
import glob
import pickle

count_vertical_point = 6
count_horizontal_point = 9


def find_nodes(path, name_file):
    calib_images = glob.glob(path)

    # Подготовка объектов 3D координат шахматной доски
    objpoints = []
    imgpoints = []

    objp = np.zeros((count_horizontal_point * count_vertical_point, 3), np.float32)
    objp[:, :2] = np.mgrid[0:count_vertical_point, 0:count_horizontal_point].T.reshape(-1, 2)

    count_valid = 0
    image_size = None
    # Находим углы шахматной доски на изображениях и добавляем в списки
    for img in calib_images:
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Не удалось загрузить изображения: {gray}")
            continue
        if image_size is None:
            image_size = gray.shape[::-1]
            
        ret, corners = cv2.findChessboardCorners(gray, (count_horizontal_point, count_vertical_point), None)
        
        if ret:
            count_valid += 1
            imgpoints.append(corners)
            objpoints.append(objp)
    print(count_valid, "images found")

    ret_cam, mtx_cam, dist_cam, rvecs_cam, tvecs_cam = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    # Сохранение параметров калибровки в файл
    calibration_data = {
        'ret': ret_cam,
        'mtx': mtx_cam,
        'dist': dist_cam,
        'rvecs': rvecs_cam,
        'tvecs': tvecs_cam
    }

    with open(f'{name_file}.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)

    print(f"Калибровка завершена и данные сохранены в '{name_file}.pkl'")

def stereo_calibrate(frames_folder1, frames_folder2, output_file):
    # Загрузите данные калибровки из файлов
    with open(r"D:\Dev\StereoPair\StereoPair\first_camera.pkl", 'rb') as f:
        calibration_data1 = pickle.load(f)
    with open(r"D:\Dev\StereoPair\StereoPair\second_camera.pkl", 'rb') as f:
        calibration_data2 = pickle.load(f)

    mtx1 = calibration_data1['mtx']
    dist1 = calibration_data1['dist']
    mtx2 = calibration_data2['mtx']
    dist2 = calibration_data2['dist']

    # Читайте синхронизированные кадры
    images_names1 = glob.glob(frames_folder1)
    images_names1 = sorted(images_names1)
    images_names2 = glob.glob(frames_folder2)
    images_names2 = sorted(images_names2)

    c1_images_names = images_names1
    c2_images_names = images_names2

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)

        _im = cv2.imread(im2, 1)
        c2_images.append(_im)

    # Измените это, если стереокалибровка не хорошая.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    world_scaling = 0.019  # Измените это на реальный размер квадрата в мире. Или нет.

    # Координаты квадратов в пространстве шахматной доски
    objp = np.zeros((count_horizontal_point * count_vertical_point, 3), np.float32)
    objp[:, :2] = np.mgrid[0:count_horizontal_point, 0:count_vertical_point].T.reshape(-1, 2)
    objp = world_scaling * objp

    # Размеры кадра. Кадры должны быть одинакового размера.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Пиксельные координаты шахматной доски
    imgpoints_left = []  # 2D точки в плоскости изображения.
    imgpoints_right = []
    # Координаты шахматной доски в пространстве шахматной доски.
    objpoints = []  # 3D точки в реальном мире
    count = 0
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (count_horizontal_point, count_vertical_point), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (count_horizontal_point, count_vertical_point), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            count += 1
    print(count, "images found")

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags)

    print(ret)

    # Запишите полученные значения в файл
    with open(output_file, 'wb') as f:
        pickle.dump({'ret': ret, 'CM1': CM1, 'dist1': dist1, 'CM2': CM2, 'dist2': dist2, 'R': R, 'T': T, 'E': E, 'F': F}, f)

    return R, T


# find_nodes('onli_first_cam/*.jpg', 'first_camera')
# find_nodes('onli_second_cam/*.jpg', 'second_camera')

stereo_calibrate(r'D:\Dev\StereoPair\img\sterio\left_cam\*.jpg', r'D:\Dev\StereoPair\img\sterio\right_cam\*.jpg', 'calibration_pair.pckl')


# # Путь к изображениям для калибровки левой и правой камеры
# calib_images_left = glob.glob('first_camera/*.jpg')
# calib_images_right = glob.glob('second_camera/*.jpg')

# # Подготовка объектов 3D координат шахматной доски
# objpoints = []
# imgpoints_left = []
# imgpoints_right = []

# objp = np.zeros((count_horizontal_point * count_vertical_point, 3), np.float32)
# objp[:, :2] = np.mgrid[0:count_vertical_point, 0:count_horizontal_point].T.reshape(-1, 2)

# count_valid = 0
# # Находим углы шахматной доски на изображениях и добавляем в списки
# for img_left, img_right in zip(calib_images_left, calib_images_right):
#     gray_left = cv2.imread(img_left, cv2.IMREAD_GRAYSCALE)
#     gray_right = cv2.imread(img_right, cv2.IMREAD_GRAYSCALE)
    
#     if gray_left is None or gray_right is None:
#         print(f"Не удалось загрузить изображения: {img_left} или {img_right}")
#         continue
    
#     ret_left, corners_left = cv2.findChessboardCorners(gray_left, (count_horizontal_point, count_vertical_point), None)
#     ret_right, corners_right = cv2.findChessboardCorners(gray_right, (count_horizontal_point, count_vertical_point), None)
    
#     if ret_left and ret_right:
#         count_valid += 1
#         imgpoints_left.append(corners_left)
#         imgpoints_right.append(corners_right)
#         objpoints.append(objp)

# print(count_valid)

# # Калибровка отдельных камер
# ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
# ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# # Калибровка стереопары (находим внешние параметры)
# flags = cv2.CALIB_FIX_INTRINSIC
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
# ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
#     objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], criteria=criteria, flags=flags)

# # Сохранение параметров калибровки в файл
# calibration_data = {
#     'mtx_left': mtx_left,
#     'dist_left': dist_left,
#     'mtx_right': mtx_right,
#     'dist_right': dist_right,
#     'R': R,
#     'T': T
# }

# with open('calibration_data.pkl', 'wb') as f:
#     pickle.dump(calibration_data, f)

# print("Калибровка завершена и данные сохранены в 'calibration_data.pkl'")

