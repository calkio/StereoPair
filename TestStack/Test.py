import cv2 as cv
import numpy as np
import glob
import pickle

############### FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,6) # Other chessboard sizes used - (5,3) OR (9,6)

# Paths to the captured frames (should be in synch) (stereoLeft and stereoRight)
CALIBRATION_IMAGES_PATH_LEFT = r'D:\Dev\StereoPair\img\sterio\left_cam\*.jpg'
CALIBRATION_IMAGES_PATH_RIGHT = r'D:\Dev\StereoPair\img\sterio\right_cam\*.jpg'
frameSize = (3264, 2448)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32) # creates 9*6 list of (0.,0.,0.)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) # formats list with (column no., row no., 0.) where max column no. = 8, and max row no. = 5

size_of_chessboard_squares_mm = 19
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imagesLeft = sorted(glob.glob(CALIBRATION_IMAGES_PATH_LEFT))
imagesRight = sorted(glob.glob(CALIBRATION_IMAGES_PATH_RIGHT))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
            
    # Get the corners of the chess board
    retL, cornersL = cv.findChessboardCornersSB(grayL, chessboardSize, None, flags=cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY)
    retR, cornersR = cv.findChessboardCornersSB(grayR, chessboardSize, None, flags=cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY)

    # Add object points and image points if chess board corners are found        
    if retL and retR == True:

        objpoints.append(objp) 
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        #Draw corners for user feedback
        # cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        # imS = cv.resize(imgL, (960, 580))
        # cv.imshow('img left', imS)

        # cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        # imS = cv.resize(imgR, (960, 580))
        # cv.imshow('img right', imS)
        # cv.waitKey()


cv.destroyAllWindows()

############# CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

######### Stereo Vision Calibration #############################################
## stereoCalibrate Output: retStereo is RSME, newCameraMatrixL and newCameraMatrixR are the intrinsic matrices for both
                ## cameras, distL and distR are the distortion coeffecients for both cameras, rot is the rotation matrix,
                ## trans is the translation matrix, and essentialMatrix and fundamentalMatrix are self descriptive
                
# R and T are taken from stereoCalibrate to use in triangulation
header = ['Rotation','Translation', 'ProjectionLeft', 'ProjectionRight'] # for the csv file

flags = 0
flags = cv.CALIB_FIX_INTRINSIC

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

ret, mtxL, distL, mtxR, distR, R, T, E, F = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)
print(ret)
print('stereo vision done')

# Сохранение параметров в файл
data = {'mtxL': mtxL, 'distL': distL, 'mtxR': mtxR, 'distR': distR, 'R': R, 'T': T}
with open('calibration_params.pkl', 'wb') as f:
    pickle.dump(data, f)