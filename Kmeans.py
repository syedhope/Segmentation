# -*- coding: utf-8 -*-
import math
from scipy import signal
from PIL import Image
import numpy
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
import random
from scipy.spatial import distance


km = 0 # Number of times the algorithm is iterated....
#..... this value will be updated later as go to the next iteration
nc = 2  # Number of clusters to be used.... 
# .....this is basically the k value as per the formula used in the algorithm
I = array(Image.open('input1.jpg').convert('L')) # read the input image

cp = [] # Array containing the pixel intensity of all k clusters
cp1 = range(nc) # initializing a new array for comparision
check = range(1,nc+1) # initializing an array to store the difference
'''
for i in range(0,nc):
    cp.append(randint(0,256)) # choose random centroids
'''
cp = [25,210]
while count_nonzero(check) > 0 : # choose number of iterations
    # these nested arrays are the collection of indices and intensities for all the clusters
    kx = [[] for _ in xrange(nc)] # to store the x index
    ky = [[] for _ in xrange(nc)] # to store the y index
    kp = [[] for _ in xrange(nc)] # to store the pixels
    # calculating the distance between each pixel and centroids
    for i in range(0,len(I[:,0])):
        for j in range(0,len(I[0,:])):
            tcomp = [] # temp array for comparision
            for k in range(0,nc):
                tcomp.append(abs(I[i,j]-cp[k])) #(cx[k],cy[k])
                c = argmin(tcomp) # The centroid to which the pixel belongs
                kx[c].append(i) # note x index
                ky[c].append(j) # note y index
                kp[c].append(I[i,j]) # note the pixel intensity
	
    #== Mean Calculation for each cluster =======#
    for i in range(nc):
        cp1[i] = (mean(kp[i])) # Mean for each cluster
        check[i] = (abs(cp[i]-cp1[i])) # Compare the new set with the previous set
    cp2 = cp
    cp = cp1 # updating the mean values
	
    km+=1 # Keeping a coutn if the number of iterations
print ('Number of iterations = %d\n')%km
#==========Output the cluster==============#
cluster = []
for i in range(0,nc):
     cluster.append(len(kp[i])) # choosing the cluster with max pixels
ind = cluster.index(min(cluster))
zipk = zip(kx[ind],ky[ind],kp[ind]) # retriving the indices of the chosen pixels
I = 255 * numpy.ones(shape = shape(I))
for k,l,m in zipk:
    I[k,l] = 0
figure()
plt.imshow(I,cmap=cm.gray)
title('Output Image of  K-means Algorithm')
show()
#------------F score calculation--------------
tp,tn,fp,fn = (0.0,0.0,0.0,0.0) # initializing the true/false positives/negatives
J = array(Image.open('out1.jpg').convert('L'))
for i in range(len(I[:,0])):    
    for j in range(len(I[0,:])):    
	if ((I[i,j]>200) & (J[i,j]>200)): # implementing true negative
	  tn+=1
	elif((I[i,j]==0) & (J[i,j]==0)): # implementing true positive
	  tp+=1
	elif((I[i,j]>200) & (J[i,j]==0)): # implementing false negative
	  fn+=1
	elif((I[i,j]==0) & (J[i,j]>200)): # implementing false positive
	  fp+=1
sen = tp/(tp+fn) # Sensitivity
fot = fp/(fp+tn) # Fall Out 
fsc = (2*tp)/((2*tp)+fp+fn) # F-score
print "Finished quantitative evaluation \nSensitivity = %f\nFall-out = %f\nF-score = %f\n"%(sen,fot,fsc)
