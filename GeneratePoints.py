import cv2
import numpy as np

count_vertical_point = 25
count_horizontal_point = 28

def find_chessboard_corners(img_path1, img_path2, output_file1, output_file2):
    # Читаем изображения
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Преобразуем изображения в grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Находим углы шахматной доски
    ret1, corners1 = cv2.findChessboardCorners(gray1, (count_horizontal_point, count_vertical_point), None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, (count_horizontal_point, count_vertical_point), None)

    # Если углы найдены, записываем их в файлы
    if ret1:
        np.savetxt(output_file1, corners1.reshape(-1, 2))
    if ret2:
        np.savetxt(output_file2, corners2.reshape(-1, 2))

# Пример использования
find_chessboard_corners(r'D:\Dev\StereoPair\first_camera\test_left.jpg', r'D:\Dev\StereoPair\second_camera\test_right.jpg', r'D:\Dev\StereoPair\GenerateData\NodesLeft.txt', r'D:\Dev\StereoPair\GenerateData\NodesRight.txt')