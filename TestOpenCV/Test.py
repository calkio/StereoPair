import cv2 as cv
import numpy as np
import os

def StereoCalib(imagelist, boardSize, useCalibrated=True, showRectified=True):
    if len(imagelist) % 2 != 0:
        print("Error: the image list contains an odd number of elements")
        return

    displayCorners = False
    maxScale = 2
    squareSize = 19.0  # Set this to your actual square size
    imagePoints = [[], []]
    objectPoints = []
    imageSize = None
    goodImageList = []
    
    for i in range(0, len(imagelist), 2):
        imgL = cv.imread(imagelist[i], 0)
        imgR = cv.imread(imagelist[i+1], 0)

        if imgL is None or imgR is None:
            print(f"Cannot load images: {imagelist[i]} and {imagelist[i+1]}")
            continue

        if imageSize is None:
            imageSize = imgL.shape[::-1]
        else:
            if imgL.shape[::-1] != imageSize or imgR.shape[::-1] != imageSize:
                print(f"Image sizes are inconsistent: {imagelist[i]} and {imagelist[i+1]}")
                continue
        
        foundL, cornersL = cv.findChessboardCorners(imgL, boardSize, 
                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        foundR, cornersR = cv.findChessboardCorners(imgR, boardSize, 
                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

        if foundL and foundR:
            cornersL = cv.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), 
                        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            cornersR = cv.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), 
                        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01))

            imagePoints[0].append(cornersL)
            imagePoints[1].append(cornersR)
            goodImageList.append(imagelist[i])
            goodImageList.append(imagelist[i+1])

            objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
            objp *= squareSize
            objectPoints.append(objp)
        else:
            print(f"Chessboard not found in images: {imagelist[i]} and {imagelist[i+1]}")
    
    print(f"{len(imagePoints[0])} pairs have been successfully detected.")
    if len(imagePoints[0]) < 2:
        print("Error: too few pairs to run the calibration")
        return

    # Stereo calibration
    print("Running stereo calibration ...")
    cameraMatrix1 = np.eye(3)
    cameraMatrix2 = np.eye(3)
    distCoeffs1 = np.zeros((5, 1))
    distCoeffs2 = np.zeros((5, 1))

    flags = 0
    flags |= cv.CALIB_FIX_ASPECT_RATIO
    flags |= cv.CALIB_ZERO_TANGENT_DIST
    flags |= cv.CALIB_SAME_FOCAL_LENGTH
    flags |= cv.CALIB_RATIONAL_MODEL
    flags |= cv.CALIB_FIX_K3
    flags |= cv.CALIB_FIX_K4
    flags |= cv.CALIB_FIX_K5

    # Задаем критерии остановки
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # Калибруем каждую камеру отдельно
    ret1, cameraMatrix1, distCoeffs1, rvecs1, tvecs1 = cv.calibrateCamera(
        objectPoints, imagePoints[0], imageSize, cameraMatrix1, distCoeffs1, flags=flags)
    ret2, cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = cv.calibrateCamera(
        objectPoints, imagePoints[1], imageSize, cameraMatrix2, distCoeffs2, flags=flags)


    print("Running stereo calibration ...")
    ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
        objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        imageSize, criteria=criteria, flags=flags)

    print(f"Calibration done with RMS error = {ret}")
    print("Saving calibration results...")


    # ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
    #                                                                                                     cameraMatrix1, distCoeffs1,
    #                                                                                                     cameraMatrix2, distCoeffs2,
    #                                                                                                     imageSize, None, None, None, None,
    #                                                                                                     cv.CALIB_FIX_INTRINSIC)

    print(f"Calibration done. RMS error: {R}")
    print("Saving calibration results...")

    fs = cv.FileStorage(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\stereo_cam.yml", cv.FILE_STORAGE_WRITE)
    fs.write("M1", cameraMatrix1)
    fs.write("D1", distCoeffs1)
    fs.write("M2", cameraMatrix2)
    fs.write("D2", distCoeffs2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.release()

    # Rectification
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrix1, distCoeffs1,
                                               cameraMatrix2, distCoeffs2,
                                               imageSize, R, T,
                                               flags=0, alpha=0)

    re = cv.FileStorage(r"D:\Dev\StereoPair\StereoPair\TestOpenCV\rectify.yml", cv.FILE_STORAGE_WRITE)
    re.write("R1", R1)
    re.write("R2", R2)
    re.write("P1", P1)
    re.write("P2", P2)
    re.write("Q", Q)
    re.release()

    if showRectified:
        for i in range(len(goodImageList) // 2):
            imgL = cv.imread(goodImageList[2 * i], 0)
            imgR = cv.imread(goodImageList[2 * i + 1], 0)

            map1L, map2L = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv.CV_16SC2)
            map1R, map2R = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv.CV_16SC2)

            imgL_rectified = cv.remap(imgL, map1L, map2L, cv.INTER_LINEAR)
            imgR_rectified = cv.remap(imgR, map1R, map2R, cv.INTER_LINEAR)

            cv.imshow('Left Rectified', cv.resize(imgL_rectified, None, fx=0.3, fy=0.3))
            cv.imshow('Right Rectified', cv.resize(imgR_rectified, None, fx=0.3, fy=0.3))

            cv.waitKey(500)

def readImageList(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == "__main__":
    # Parameters (adjust these as necessary)
    boardSize = (9, 6)  # Number of inner corners per a chessboard row and column
    imagelist_file = r"D:\Dev\StereoPair\StereoPair\GenerateData\stereo_images.txt"  # Text file with image pairs
    showRectified = True  # Whether to display rectified images

    # Load the image list
    imagelist = readImageList(imagelist_file)

    # Run stereo calibration
    StereoCalib(imagelist, boardSize, useCalibrated=True, showRectified=showRectified)
