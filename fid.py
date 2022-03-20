# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import cv2
 
# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
# define two collections of activations
#act1 = random(10*2048)
#act1 = act1.reshape((10,2048))
#act2 = random(10*2048)
#act2 = act2.reshape((10,2048))
#print(act2)
#img_test = cv2.imread("./testGenImg0.jpg", cv2.IMREAD_GRAYSCALE)
#img_in = cv2.imread("./testInputImg0.jpg", cv2.IMREAD_GRAYSCALE)
#plt.figure()
#plt.imshow(img_in)
#plt.show()
# fid between act1 and act1
#fid = calculate_fid(img_in, img_in)
#print('FID (same): %.3f' % fid)
# fid between act1 and act2
#fid = calculate_fid(img_in, img_test)
#print('FID (different): %.3f' % fid)