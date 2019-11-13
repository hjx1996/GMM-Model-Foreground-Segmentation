# !/usr/bin/python3
# -*- coding: utf-8 -*-

from glob import glob
from PublicDef import Gaussian
from PublicDef import check_bkg
import os

import numpy as np
import cv2 as cv

G_mat = np.load('.\G_mat.npy')
G_weight = np.load('.\G_weight.npy')


def infer(img, mat, weight):  # 推断图片的背景，如果像素为背景则rgb都设为0，如果不是背景则不进行处理
	result = np.array(img)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			gaussian_pixel = mat[i][j]
			for g in range(4):
				if check_bkg(img[i][j], gaussian_pixel[g]) and weight[i][j][g] > 0.25:
					# 阈值，将符合任意一个权重较大的高斯分布的像素点变为黑色
					result[i][j] = [0, 0, 0]
				continue
	return result


if __name__ == '__main__':
	if os.path.exists('.\Output') is False:
		os.mkdir('.\Output')
	file_list = glob(r'.\WavingTrees\b00*.bmp')
	file_index = 0
	for file in file_list:
		print('Processing: {}'.format(file))
		img = cv.imread(file)
		img_infer = infer(img, G_mat, G_weight)
		cv.imwrite(r'.\Output\\' + '%05d' % file_index + '.bmp', img_infer)
		file_index += 1
