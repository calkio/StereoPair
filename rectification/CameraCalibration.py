import cv2
import numpy as np
import glob
import pickle

class CameraCalibration:
    def __init__(self, chessboard_size=(9, 6), square_size=25):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.objpoints = []  # 3D точки в мировом пространстве
        self.imgpoints_left = []  # 2D точки в изображении с левой камеры
        self.imgpoints_right = []  # 2D точки в изображении с правой камеры
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

    def find_corners(self, images_left, images_right):
        for img_left, img_right in zip(images_left, images_right):
            imgL = cv2.imread(img_left)
            imgR = cv2.imread(img_right)
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            retL, cornersL = cv2.findChessboardCornersSB(grayL, self.chessboard_size, None, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
            # if not retL:
            #     retL, cornersL = cv2.findChessboardCornersSB(grayL, (25, 28), None, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)

            retR, cornersR = cv2.findChessboardCornersSB(grayR, self.chessboard_size, None, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
            # if not retR:
            #     retR, cornersR = cv2.findChessboardCornersSB(grayL, (25, 28), None, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)

            if retL and retR:
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(cornersL)
                self.imgpoints_right.append(cornersR)

    def calibrate(self, image_shape):
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(self.objpoints, self.imgpoints_left, image_shape, None, None)
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(self.objpoints, self.imgpoints_right, image_shape, None, None)

        flags = cv2.CALIB_FIX_INTRINSIC
        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(self.objpoints, self.imgpoints_left, self.imgpoints_right,
                                                                        mtxL, distL, mtxR, distR, image_shape, criteria=None, flags=flags)
        # Сохранение параметров в файл
        data = {'mtxL': mtxL, 'distL': distL, 'mtxR': mtxR, 'distR': distR, 'R': R, 'T': T}
        with open('calibration_params.pkl', 'wb') as f:
            pickle.dump(data, f)

        return mtxL, distL, mtxR, distR, R, T



chessboard_size=(9, 6)
square_size=19
calib = CameraCalibration(chessboard_size, square_size)
images_left = glob.glob(r'D:\Dev\StereoPair\img\sterio\left_cam\*.jpg')
images_right = glob.glob(r'D:\Dev\StereoPair\img\sterio\right_cam\*.jpg')
calib.find_corners(images_left, images_right)
calib.calibrate(cv2.imread(images_left[0]).shape[:2])