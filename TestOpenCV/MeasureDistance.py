import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import yaml

def load_calibration_parameters(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    
    # Функция для чтения матрицы из файла
    def read_matrix(key):
        return fs.getNode(key).mat()

    # Чтение параметров калибровки
    cameraMatrix1 = read_matrix("M1")
    distCoeffs1 = read_matrix("D1")
    cameraMatrix2 = read_matrix("M2")
    distCoeffs2 = read_matrix("D2")
    R = read_matrix("R")
    T = read_matrix("T")
    E = read_matrix("E")
    F = read_matrix("F")

    fs.release()

    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

def load_rectify_parameters(yaml_file):
    fs = cv.FileStorage(yaml_file, cv.FILE_STORAGE_READ)

    def read_matrix(key):
        return fs.getNode(key).mat()

    R1 = read_matrix("R1")
    R2 = read_matrix("R2")
    P1 = read_matrix("P1")
    P2 = read_matrix("P2")
    Q = read_matrix("Q")

    fs.release()

    return R1, R2, P1, P2, Q

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

def measure_distance(depth_map, point1, point2):
    """
    Measure the 3D distance between two points in the depth map.
    :param depth_map: The depth map image.
    :param point1: The coordinates (x, y) of the first point.
    :param point2: The coordinates (x, y) of the second point.
    :return: The 3D distance between the two points.
    """
    # Get depth values at the given points
    depth1 = depth_map[point1[1], point1[0]]
    depth2 = depth_map[point2[1], point2[0]]
    
    # Convert depth values to meters if necessary (depends on your depth map scale)
    depth1 = depth1 / 1000.0
    depth2 = depth2 / 1000.0
    
    # Calculate Euclidean distance between points in 3D
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = depth2 - depth1
    
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return distance

def triangulate(R, T, mtx1, mtx2, x1, y1, x2, y2):
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


def calculate_camera_centers(R, T, mtx1, mtx2):
    # Оптический центр первой камеры
    center1 = np.array([0, 0, 0], dtype=np.float32)

    # Для второй камеры, вычисляем её оптический центр
    # Сначала создадим матрицу проекции второй камеры
    RT2 = np.hstack((R, T.reshape(-1, 1)))  # Размерности (3x4)
    P2 = np.dot(mtx2, RT2)  # Размерности (3x4)

    # Обратная матрица проекции P2 (все векторы в однородных координатах)
    # Решаем уравнение P2 * X = 0, где X - гомогенные координаты оптического центра
    # Поскольку P2 имеет размер 3x4, нам нужно добавить еще одну строку (0,0,0,1) для вычисления
    P2_ext = np.vstack((P2, [0, 0, 0, 1]))  # Размерности (4x4)

    # Теперь P2_ext * X = 0, где X - это (x, y, z, 1) для оптического центра
    _, _, VT = np.linalg.svd(P2_ext)
    center2_hom = VT[-1]
    center2 = center2_hom[:3] / center2_hom[3]  # Нормализуем

    return center1, center2
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

def plot_3d_points_with_cameras(points, camera_centers):
    """
    Функция для визуализации коллекции трёхмерных точек и оптических центров камер.

    :param points: Список или массив numpy с точками вида [(x1, y1, z1), (x2, y2, z2), ...].
    :param camera_centers: Список или массив numpy с координатами оптических центров камер вида [(x1, y1, z1), (x2, y2, z2)].
    """
    # Преобразуем коллекцию точек в numpy массив для удобства
    points = np.array(points)
    camera_centers = np.array(camera_centers)

    # Создаем фигуру и оси 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Разделяем точки на X, Y, Z для простоты
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    # Визуализируем точки в 3D пространстве
    ax.scatter(xs, ys, zs, c='r', marker='o', label='3D Points')

    # Разделяем координаты оптических центров
    cx = camera_centers[:, 0]
    cy = camera_centers[:, 1]
    cz = camera_centers[:, 2]

    # Визуализируем оптические центры камер
    ax.scatter(cx, cy, cz, c='b', marker='^', s=100, label='Camera Centers')

    # Настраиваем оси
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Устанавливаем название графика
    ax.set_title('3D Plot of Points with Camera Centers')

    # Добавляем легенду
    ax.legend()

    # Отображаем график
    plt.show()


# Load calibration parameters
cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = load_calibration_parameters(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\stereo_cam.yml")

# Load rectification parameters
R1, R2, P1, P2, Q = load_rectify_parameters(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\rectify.yml")

# Load stereo images
img_left = cv.imread(r"C:\Users\david\Downloads\WIN_20240914_13_00_55_Pro.jpg", cv.IMREAD_GRAYSCALE)
img_right = cv.imread(r"C:\Users\david\Downloads\WIN_20240914_13_00_45_Pro.jpg", cv.IMREAD_GRAYSCALE)

# Load point coordinates
points = read_coordinates(r'D:\Dev\StereoPair\StereoPair\GenerateData\NodesLeft.txt', r'D:\Dev\StereoPair\StereoPair\GenerateData\NodesRight.txt')

points_3d = []
for point in points:
    x1, y1, x2, y2 = point
    # вызов метода для преобразования координат в трехмерную точку
    point_3d = triangulate(R, T, cameraMatrix1, cameraMatrix2, x1, y1, x2, y2)
    print(point_3d)
    points_3d.append(point_3d)


centers_3D = [] 
c1, c2 = calculate_camera_centers(R, T, cameraMatrix1, cameraMatrix2)
centers_3D.append(c1)
centers_3D.append(c2)
plot_3d_points_with_cameras(points_3d, centers_3D)

# Compute disparity map
# stereo = cv.StereoSGBM_create(
#             minDisparity=-16,
#             numDisparities=16 * 10,
#             blockSize=5,
#             disp12MaxDiff=1,
#             uniquenessRatio=15,
#             speckleWindowSize=0,
#             speckleRange=2,
#             preFilterCap=63,
#             mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
#         )
# disparity = stereo.compute(img_left, img_right)

# disparity_normalized = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
# disparity_normalized = np.uint8(disparity_normalized)
# plt.figure(figsize=(10, 7))
# plt.imshow(disparity_normalized, cmap='plasma')
# plt.colorbar()  # Отображает цветовую шкалу
# plt.title('Depth Map')
# plt.show()

# # Convert disparity to depth map
# focal_length = cameraMatrix1[0, 0]  # Use focal length from cameraMatrix1
# baseline = 0.7  # Example baseline (in meters)
# depth_map = (focal_length * baseline) / (disparity.astype(np.float32) + 1e-6)  # Avoid division by zero


# # Measure distance between the first two points
# distances = []
# if len(points) >= 2:
#     for point in points:
#         x1, y1, x2, y2 = point
#         # вызов метода для преобразования координат в трехмерную точку
#         point1, point2 = (x1, y1), (x2, y2)
#         distance = measure_distance(depth_map, point1, point2)
#         distances.append(distance)
# else:
#     print("Not enough points to measure distance.")

# for item in distances:
#     print(item)

# # Преобразование координат и расстояний в массивы для построения графика
# x_coords = [point[0] for point in points]
# y_coords = [point[1] for point in points]
# distances = np.array(distances)

# # Построение графика
# plt.figure(figsize=(12, 6))

# # Рисуем координаты точек
# plt.scatter(x_coords, y_coords, color='red', label='Points')

# # Соединяем точки линиями, чтобы визуализировать расстояния
# for i, (point1, point2) in enumerate([(points[i], points[i+1]) for i in range(len(points)-1)]):
#     x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
#     plt.plot([x1, x2], [y1, y2], color='blue', linestyle='--', linewidth=1)

# # Настройка графика
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('2D Projection of 3D Points and Distances')
# plt.legend()
# plt.grid(True)
# plt.show()
