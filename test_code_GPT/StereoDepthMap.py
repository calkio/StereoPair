import pickle
import cv2
import numpy as np

class StereoDepthMap:
    def __init__(self, calibration_file):
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None
        self.T = None
        self.load_calibration(calibration_file)

    def load_calibration(self, filepath):
        with open(filepath, 'rb') as f:
            calibration_data = pickle.load(f)
            self.camera_matrix_left = calibration_data['camera_matrix_left']
            self.dist_coeffs_left = calibration_data['dist_coeffs_left']
            self.camera_matrix_right = calibration_data['camera_matrix_right']
            self.dist_coeffs_right = calibration_data['dist_coeffs_right']
            self.R = calibration_data['R']
            self.T = calibration_data['T']

    def compute_depth_map(self, left_image_path, right_image_path):
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Настройка SGBM для построения карты глубины
        stereo = cv2.StereoSGBM_create(minDisparity=0,
                                       numDisparities=16 * 4,  # Чем больше значение, тем точнее карта
                                       blockSize=5,
                                       P1=8 * 3 * 5**2,
                                       P2=32 * 3 * 5**2,
                                       disp12MaxDiff=1,
                                       uniquenessRatio=15,
                                       speckleWindowSize=0,
                                       speckleRange=2,
                                       preFilterCap=63)

        disparity = stereo.compute(gray_left, gray_right)
        depth_map = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)

        return depth_map

    def display_depth_map(self, depth_map):
        cv2.imshow('Depth Map', depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




depth_map = StereoDepthMap("calibration_data.pkl")
depth = depth_map.compute_depth_map("D:\Dev\StereoPair\left.jpg", "D:\Dev\StereoPair\right.jpg")
depth_map.display_depth_map(depth)
