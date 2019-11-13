# !/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


# 定义一个高斯分布类
class Gaussian:
	def __init__(self, u, sigma):
		self.u = u
		self.sigma = sigma


# 检测像素点是否符合高斯分布，即是否少于2.5个标准差
def check_bkg(pixel, gaussian):
	u = np.mat(gaussian.u).T
	x = np.mat(np.reshape(pixel, (3, 1)))
	sigma = np.mat(gaussian.sigma)
	d = np.sqrt((x - u).T * sigma.I * (x - u))
	if d < 2.5:
		return True
	else:
		return False
