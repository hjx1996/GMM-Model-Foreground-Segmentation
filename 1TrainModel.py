# !/usr/bin/python3
# -*- coding: utf-8 -*-

from PublicDef import check_bkg
from PublicDef import Gaussian
import numpy as np
import cv2 as cv
import os

# 初始化GMM模型参数
INIT_SIGMA = 225 * np.eye(3)
INIT_ALPHA = 0.01
INIT_U = None
EPSILON = 0.00000001

train_epoch = 15  # 确定训练轮数
sample_num = 240  # 训练样本量
data_dir = '.\WavingTrees'

# =================
# 以上为基本变量定义
# =================


class GaussianMat:
	def __init__(self, shape, k):
		self.shape = shape
		self.k = k
		# 初始化高斯分布
		g = [Gaussian(INIT_U, INIT_SIGMA) for i in range(k)]
		# 建模为多维高斯混合分布
		self.mat = [[[Gaussian(INIT_U, INIT_SIGMA) for i in range(k)] for j in range(shape[1])] for l in range(shape[0])]
		# 初始化权重数值
		# self.weight = [[[1, 0, 0, 0] for j in range(shape[1])] for l in range(shape[0])]
		self.weight = [[[0.7, 0.1, 0.1, 0.1] for j in range(shape[1])] for l in range(shape[0])]


class GMM:
	def __init__(self, arg1, arg2, alpha=INIT_ALPHA):
		self.data_dir = arg1
		self.sample_num = arg2
		self.alpha = alpha
		self.g = None
		self.K = None

	def train(self, K=4):
		self.K = K
		file_list = []
		for i in range(sample_num):
			file_name = os.path.join(self.data_dir, 'b%05d' % i + '.bmp')
			file_list.append(file_name)
		img_init = cv.imread(file_list[0])
		img_shape = img_init.shape
		self.g = GaussianMat(img_shape, self.K)

		# 根据图片的长，宽和图像通道数建立多个高斯
		for i in range(img_shape[0]):
			for j in range(img_shape[1]):
				for k in range(K):
					self.g.mat[i][j][k].u = np.array(img_init[i][j]).reshape(1, 3)
		# 迭代训练，获得具有普适性的高斯模型
		for n in range(train_epoch):
			for file in file_list:
				print('GMM Model is training, now processing: {}'.format(file) + ', epoch finished: %d' % n)
				img = cv.imread(file)
				for i in range(img.shape[0]):
					for j in range(img.shape[1]):
						flag = 0  # 使用flag检测是否有高斯匹配
						for k in range(K):
							if check_bkg(img[i][j], self.g.mat[i][j][k]):
								flag = 1
								m = 1
								self.g.weight[i][j][k] = self.g.weight[i][j][k] + self.alpha * (m - self.g.weight[i][j][k])
								# 若检测到该像素与第k个高斯匹配，则增大其权重
								u = self.g.mat[i][j][k].u
								sigma = self.g.mat[i][j][k].sigma
								x = img[i][j].astype(np.float)
								delta = x - u
								# 检测像素是否与第k个高斯匹配，若匹配，改变该高斯分布均值，接近x
								self.g.mat[i][j][k].u = u + m * (self.alpha / (self.g.weight[i][j][k] + EPSILON)) * delta
								self.g.mat[i][j][k].sigma = sigma + m * (self.alpha / (self.g.weight[i][j][k] + EPSILON))\
															* (np.matmul(delta, delta.T) - sigma)
							else:
								m = 0
								self.g.weight[i][j][k] = self.g.weight[i][j][k] + self.alpha * (m - self.g.weight[i][j][k])
							# 如果找不到匹配的高斯就重新初始化
						if flag == 0:
							w_list = [self.g.weight[i][j][k] for k in range(K)]
							index = w_list.index(min(w_list))
							self.g.mat[i][j][index].u = np.array(img[i][j]).reshape(1, 3)
							self.g.mat[i][j][index].sigma = np.array(INIT_SIGMA)
						# 对权重进行归一化处理
						s = sum([self.g.weight[i][j][temp_k] for temp_k in range(K)])
						for temp_k in range(K):
							self.g.weight[i][j][temp_k] /= s
				print('img:', format(img[10][10]))
				print('weight', format(self.g.weight[10][10]))
				for i in range(self.K):
					print('u:{}'.format(self.g.mat[10][10][i].u))
		np.save('.\g_mat.npy', self.g.mat)
		np.save('.\g_weight.npy', self.g.weight)


if __name__ == '__main__':
	GMM = GMM(data_dir, sample_num)
	GMM.train()
	print("Train finished!")
