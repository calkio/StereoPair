from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse


input_image = r"D:\Dev\Calibrovka\10_06_2024\Nodes\0\10.06.2024142739.jpg"
src = cv.imread(cv.samples.findFile(input_image))
if src is None:
    print('Could not open or find the image:', input_image)
    exit(0)

def load_points(file_name):
    with open(r"D:\Dev\Calibrovka\10_06_2024\Nodes\0\Nodes.txt", 'r') as file:
        data = file.read()

    data = data.replace('[', '').replace(']', '').replace(' ', '').replace(',', '.')
    points = []
    points = np.array([list(map(float, pair.split(';'))) for pair in data.split('\n') if pair])
    points = points.astype(np.float32)
    return points

srcTri = load_points(r"D:\Dev\Calibrovka\10_06_2024\Nodes\0\Nodes.txt")
dstTri = load_points(r"D:\Dev\Calibrovka\NodesOmsk\100\Nodes.txt")
 
affine_mat, inliers = cv.estimateAffinePartial2D(srcTri, dstTri)

print(affine_mat)

image = cv.imread(input_image)
warped_image = cv.warpAffine(image, affine_mat, (image.shape[1], image.shape[0]))

# Отображаем результат
cv.imshow("Warped Image", cv.resize(warped_image, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA))
cv.waitKey(0)
cv.destroyAllWindows()