import cv2 as cv
import numpy as np
import yaml


def load_calibration_parameters(yaml_file):
    fs = cv.FileStorage(yaml_file, cv.FILE_STORAGE_READ)
    
    # Функция для чтения матрицы из файла
    def read_matrix(key):
        return fs.getNode(key).mat()

    # Чтение параметров калибровки
    cameraMatrix1 = read_matrix("M1")
    distCoeffs1 = read_matrix("D1")
    cameraMatrix2 = read_matrix("M2")
    distCoeffs2 = read_matrix("D2")

    fs.release()

    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2


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

def showRectifiedImages(imageListLeft, imageListRight, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R1, R2, P1, P2, imageSize):
    # Создаем карты для ремаппинга
    map1x, map1y = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv.CV_32FC1)

    for imgL_path, imgR_path in zip(imageListLeft, imageListRight):
        imgL = cv.imread(imgL_path)
        imgR = cv.imread(imgR_path)

        if imgL is None or imgR is None:
            print(f"Cannot load images {imgL_path} and {imgR_path}")
            continue

        # Выпрямляем изображения
        rectifiedL = cv.remap(imgL, map1x, map1y, cv.INTER_LINEAR)
        rectifiedR = cv.remap(imgR, map2x, map2y, cv.INTER_LINEAR)

        # Соединяем изображения для наглядности
        concat_img = cv.hconcat([rectifiedL, rectifiedR])

        # Рисуем горизонтальные линии для проверки выравнивания
        for y in range(0, concat_img.shape[0], 50):
            cv.line(concat_img, (0, y), (concat_img.shape[1], y), (0, 255, 0), 1)


        # Задайте новый размер
        width = 1500
        height = int((width / concat_img.shape[1]) * concat_img.shape[0])
        new_size = (width, height)

        # Измените размер изображения
        resized_img = cv.resize(concat_img, new_size)
        cv.imshow('Rectified Pair', resized_img)
        if cv.waitKey(0) & 0xFF == 27:  # Нажмите ESC для выхода
            break

    cv.destroyAllWindows()

# Пример вызова функции
imageListLeft = [r"D:\Dev\StereoPair\img\onli_left_cam\cam1_20240913_183148.jpg", r"D:\Dev\StereoPair\img\onli_left_cam\cam1_20240913_183152.jpg", r"D:\Dev\StereoPair\img\onli_left_cam\cam1_20240913_183201.jpg"]  # Ваши изображения

imageListRight = [r"D:\Dev\StereoPair\img\onli_right_cam\cam2_20240913_183148.jpg", r"D:\Dev\StereoPair\img\onli_right_cam\cam2_20240913_183152.jpg", r"D:\Dev\StereoPair\img\onli_right_cam\cam2_20240913_183201.jpg"]

imageSize = (3264, 2448)  # Укажите правильный размер изображения

cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = load_calibration_parameters(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\stereo_cam.yml")
R1, R2, P1, P2, Q = load_rectify_parameters(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\rectify.yml")

# Здесь используйте R1, R2, P1, P2 из стереокалибровки
showRectifiedImages(imageListLeft, imageListRight, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R1, R2, P1, P2, imageSize)