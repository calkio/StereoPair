import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_points(points):
    """
    Функция для визуализации коллекции трёхмерных точек.

    :param points: Список или массив numpy с точками вида [(x1, y1, z1), (x2, y2, z2), ...].
    """
    # Преобразуем коллекцию точек в numpy массив для удобства
    points = np.array(points)

    # Создаем фигуру и оси 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Разделяем точки на X, Y, Z для простоты
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    # Визуализируем точки в 3D пространстве
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Настраиваем оси
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Устанавливаем название графика
    ax.set_title('3D Plot of Points')

    # Отображаем график
    plt.show()

# # Пример использования:
# points = [
#     (-596.15796, 1483.5068, 12528.521),
#     (100.0, 200.0, 300.0),
#     (-200.0, -100.0, 500.0),
#     # Добавьте сюда другие точки
# ]

# plot_3d_points(points)


def triangulate(stereo_params_file, cam1_calib_file, cam2_calib_file, x1, y1, x2, y2):
    # Загрузка данных стереопары
    with open(stereo_params_file, 'rb') as f:
        stereo_params = pickle.load(f)
        R = stereo_params['R']
        T = stereo_params['T']

    # Загрузка калибровочных данных первой камеры
    with open(cam1_calib_file, 'rb') as f:
        cam1_data = pickle.load(f)
        mtx1 = cam1_data['mtx']
        dist1 = cam1_data['dist']

    # Загрузка калибровочных данных второй камеры
    with open(cam2_calib_file, 'rb') as f:
        cam2_data = pickle.load(f)
        mtx2 = cam2_data['mtx']
        dist2 = cam2_data['dist']

    # Установим координаты точек для обеих камер
    uv1 = np.array([x1, y1], dtype=np.float32).reshape(-1, 1, 2)
    uv2 = np.array([x2, y2], dtype=np.float32).reshape(-1, 1, 2)

    # Композиция матриц проекции для двух камер
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Матрица проекции первой камеры
    P1 = np.dot(mtx1, RT1)

    RT2 = np.hstack((R, T))  # Матрица проекции второй камеры
    P2 = np.dot(mtx2, RT2)

    # Триангуляция
    point_4d_hom = cv.triangulatePoints(P1, P2, uv1, uv2)

    # Преобразование из однородных координат в декартовы 3D координаты
    point_3d = point_4d_hom[:3] / point_4d_hom[3]

    return point_3d.flatten()


def read_coordinates(file_path1, file_path2):
    points_2d = []
    with open(file_path1, 'r') as f1, open(file_path2, 'r') as f2:
        for line1, line2 in zip(f1, f2):
            # x1, y1 = map(float, line1.split())
            # x2, y2 = map(float, line2.split())
            x1, y1 = map(float, line2.split())
            x2, y2 = map(float, line1.split())
            points_2d.append((int(x1), int(y1), int(x2), int(y2)))
    return points_2d


def build_depth_map(self, imgL, imgR):
    left_map1 = self.rectification_params['left_map1']
    left_map2 = self.rectification_params['left_map2']
    right_map1 = self.rectification_params['right_map1']
    right_map2 = self.rectification_params['right_map2']
    Q = self.rectification_params['Q']

    # Применение ремаппинга
    rectified_left = cv.remap(imgL, left_map1, left_map2, cv.INTER_LINEAR)
    rectified_right = cv.remap(imgR, right_map1, right_map2, cv.INTER_LINEAR)

    # Настройка и вычисление карты глубины

    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


    # Создание объекта SGBM
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
    disparity = stereo.compute(cv.cvtColor(rectified_left, cv.COLOR_BGR2GRAY),
                                cv.cvtColor(rectified_right, cv.COLOR_BGR2GRAY))

    # Возвращаем карту глубины
    depth_map = cv.reprojectImageTo3D(disparity, Q)
    return rectified_left, rectified_right, depth_map, disparity


# Пример использования
stereo_params_file = r'D:\Dev\StereoPair\StereoPair\calibration_pair.pckl'
cam1_calib_file = r'D:\Dev\StereoPair\StereoPair\first_camera.pkl'
cam2_calib_file = r'D:\Dev\StereoPair\StereoPair\second_camera.pkl'

# points_2d = [
#     # (938, 1425, 784, 1391), 
#     (1333, 1447, 1131, 1411), 
#     (1445, 608, 1244, 575),
#     (1818, 642, 1616, 613),
#     (1692, 1503, 1490, 1476)
#     ]

points_2d = read_coordinates(r'D:\Dev\StereoPair\StereoPair\GenerateData\NodesLeft.txt', r'D:\Dev\StereoPair\StereoPair\GenerateData\NodesRight.txt')
points_3d = []
for point in points_2d:
    x1, y1, x2, y2 = point
    # вызов метода для преобразования координат в трехмерную точку
    point_3d = triangulate(stereo_params_file, cam1_calib_file, cam2_calib_file, x1, y1, x2, y2)
    print(point_3d)
    points_3d.append(point_3d)

plot_3d_points(points_3d)

# x1, y1, x2, y2 = (938, 1425, 784, 1391)
# point_3d = triangulate(stereo_params_file, cam1_calib_file, cam2_calib_file, x1, y1, x2, y2)
# print("Трёхмерная координата точки:", point_3d)