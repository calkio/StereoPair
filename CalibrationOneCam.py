import cv2
import numpy as np
import pickle
import os

class StereoCalibration:
    def __init__(self, left_images_path, right_images_path, board_size=(9, 6), square_size=0.019):
        self.left_images_path = left_images_path
        self.right_images_path = right_images_path
        self.board_size = board_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
    def _find_corners(self, images):
        obj_points = []
        img_points = []
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2) * self.square_size
        
        count = 0
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                img_points.append(corners2)
                count += 1
        
        print(count, "images found")
        return obj_points, img_points
    
    def calibrate(self):
        left_images = [cv2.imread(os.path.join(self.left_images_path, f)) for f in os.listdir(self.left_images_path) if f.endswith('.jpg')]
        right_images = [cv2.imread(os.path.join(self.right_images_path, f)) for f in os.listdir(self.right_images_path) if f.endswith('.jpg')]
        
        left_obj_points, left_img_points = self._find_corners(left_images)
        right_obj_points, right_img_points = self._find_corners(right_images)
        
        if not left_img_points or not right_img_points:
            raise ValueError("No chessboard corners found in images.")
        
        # Assuming all images are the same size, use the first image
        img_size = left_images[0].shape[1::-1]
        
        # Calibrate left camera
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(left_obj_points, left_img_points, img_size, None, None)
        # Save calibration data for the left camera
        with open('first_camera.pkl', 'wb') as f:
            pickle.dump({'mtx': mtx1, 'dist': dist1, 'rvecs': rvecs1, 'tvecs': tvecs1}, f)
        
        print(ret1, "ret1")

        # Calibrate right camera
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(right_obj_points, right_img_points, img_size, None, None)
        # Save calibration data for the right camera
        with open('second_camera.pkl', 'wb') as f:
            pickle.dump({'mtx': mtx2, 'dist': dist2, 'rvecs': rvecs2, 'tvecs': tvecs2}, f)

        print(ret2, "ret2")

# Usage example:
calibrator = StereoCalibration(r'D:\Dev\StereoPair\img\onli_right_cam', r'D:\Dev\StereoPair\img\onli_left_cam')
calibrator.calibrate()
