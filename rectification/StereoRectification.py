import pickle
import cv2
import glob

class StereoRectification:
    def __init__(self, calibration_file=r'D:\Dev\StereoPair\StereoPair\rectification\calibration_params.pkl'):
        with open(calibration_file, 'rb') as f:
            self.calib_params = pickle.load(f)

    def rectify(self, image_shape):
        mtxL = self.calib_params['mtxL']
        distL = self.calib_params['distL']
        mtxR = self.calib_params['mtxR']
        distR = self.calib_params['distR']
        R = self.calib_params['R']
        T = self.calib_params['T']

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_shape, R, T)

        left_map1, left_map2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_shape, cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_shape, cv2.CV_16SC2)

        # Сохраняем карты ректификации и параметры
        rectification_data = {'left_map1': left_map1, 'left_map2': left_map2, 'right_map1': right_map1, 'right_map2': right_map2, 'Q': Q}
        with open('rectification_params.pkl', 'wb') as f:
            pickle.dump(rectification_data, f)

        return left_map1, left_map2, right_map1, right_map2, Q
    


images_left = glob.glob(r'D:\Dev\StereoPair\img\sterio\left_cam\*.jpg')
images_right = glob.glob(r'D:\Dev\StereoPair\img\sterio\right_cam\*.jpg')
rect = StereoRectification()
rect.rectify(cv2.imread(images_left[0]).shape[:2])

