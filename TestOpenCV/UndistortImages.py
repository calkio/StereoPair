import cv2 as cv
import numpy as np

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

def undistortImages(imageList, cameraMatrix, distCoeffs):
    for imgPath in imageList:
        img = cv.imread(imgPath)
        if img is None:
            print(f"Cannot load image {imgPath}")
            continue
        
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
        undistortedImg = cv.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)

        # Обрезаем изображение, если нужно (на основе ROI)
        x, y, w, h = roi
        undistortedImg = undistortedImg[y:y+h, x:x+w]

        # Задайте новый размер
        width = 800
        height = int((width / undistortedImg.shape[1]) * undistortedImg.shape[0])
        new_size = (width, height)

        # Измените размер изображения
        resized_img = cv.resize(undistortedImg, new_size)

        cv.imshow('Undistorted Image', resized_img)
        cv.waitKey(0)

    cv.destroyAllWindows()

cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = load_calibration_parameters(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\stereo_cam.yml")

# Проверяем калибровку для левой камеры
imageListLeft = [r"D:\Dev\StereoPair\img\onli_left_cam\cam1_20240913_183148.jpg", r"D:\Dev\StereoPair\img\onli_left_cam\cam1_20240913_183152.jpg", r"D:\Dev\StereoPair\img\onli_left_cam\cam1_20240913_183201.jpg"]
undistortImages(imageListLeft, cameraMatrix1, distCoeffs1)

# Проверяем калибровку для правой камеры
imageListRight = [r"D:\Dev\StereoPair\img\onli_right_cam\cam2_20240913_183148.jpg", r"D:\Dev\StereoPair\img\onli_right_cam\cam2_20240913_183152.jpg", r"D:\Dev\StereoPair\img\onli_right_cam\cam2_20240913_183201.jpg"]
undistortImages(imageListRight, cameraMatrix2, distCoeffs2)
