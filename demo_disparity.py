import os.path

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
rootpath = 'D:\\Data\\1cm\\test7'
imgL = cv.imread(os.path.join(rootpath, 'IMG_0018.jpg'), cv.IMREAD_GRAYSCALE)
imgR = cv.imread(os.path.join(rootpath, 'IMG_0018.jpg'), cv.IMREAD_GRAYSCALE)
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity/10.0,'gray')
plt.show()