# -*- coding: utf-8 -*-
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time

# faster than the original one
def get_dark_channel(img,size = 15):
	r, g, b = cv2.split(img)
	min_img = cv2.min(r, cv2.min(g, b))
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
	dc_img = cv2.erode(min_img,kernel)
	return dc_img

def get_atmosphere(img, dark_channel, percent = 0.001):
	m,n = img.shape[:2]
	n_pixels = m*n
	n_search_pixels = np.floor(n_pixels * 0.001).astype('int')
	dark_vec = np.reshape(dark_channel,[n_pixels,1])
	image_vec = np.reshape(img,[n_pixels,3])
	indices = np.argsort(-dark_vec,axis =0)
	accumulator = np.zeros((1,3),dtype = float)
	for k in range(0,n_search_pixels):
		accumulator[0,:] = accumulator[0,:] + image_vec[indices[k],:]
	A = np.zeros((1,1,3),dtype = float)
	for k in range(0,3):
		A[0,0,k] = accumulator[:,k]/n_search_pixels
	return A

def dehaze_low_light(path,output = None):
	im = cv2.imread(path)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im = im.astype('float')/255
	H,W = im.shape[:2]
	# resize the image 
	# resize_ratio = 0.5
	# im = cv2.resize(im,(H*resize_ratio,W*resize_ratio),interpolation=cv2.INTER_CUBIC)
	I = 1 - im
	dark_channel = get_dark_channel(I,15)
	A = get_atmosphere(I,dark_channel)
	r, g, b = cv2.split(I)
	Y = 0.299 * r + 0.587 * g +0.114 * b
	t = cv2.max(1 - 0.98 * Y, 0.01)
	# you can apply Median filter on t for noise reduction
	J = np.empty_like(im)
	for i in range(3):
		J[:,:,i] = (I[:,:,i]-A[0,0,i])/t[:,:] + A[0,0,i]
	J = 1 - J
	J = J * (J >= 0.0) + 0 * (~(J >=0.0))
	return J

start_time = time.time()
J = dehaze_low_light("IMG_7028.JPG")
print("dehaze_low_light time: %f"%(time.time()-start_time))
try:
	plt.imshow(J)
	plt.show()
except Exception as e:
	print("error" + str(e))


