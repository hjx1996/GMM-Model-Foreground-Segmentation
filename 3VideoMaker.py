# !/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
import cv2 as cv

origin_path = r'.\WavingTrees\b00*.bmp'
after_path = r'.\output\*.bmp'

left_side = glob(origin_path)
right_side = glob(after_path)
file_index = 0
video_writer = cv.VideoWriter('result.avi', cv.VideoWriter_fourcc(*'MJPG'), 25, (320, 120))

if __name__ == '__main__':
    for i in range(len(left_side)):
        print('Left side: ' + left_side[i])
        print('Right side: ' + right_side[i])
        result = np.concatenate((cv.imread(left_side[i]), cv.imread(right_side[i])), axis=1)
        # cv.imwrite(r'./video/' + '%05d' % file_index + '.bmp', result)
        video_writer.write(result)
        file_index += 1
