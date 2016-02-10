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
import matplotlib.image as mpimg

def rgb2gray(rgb): # function to convert a color image stored in numpy array to gray scale array
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144]) # basically using the formula ---> out = 0.299 R + 0.587 G + 0.114 B 

img = cv2.imread('input2.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
hist,bins = np.histogram(img.ravel(),256,[0,256])

km = 0 # Number of times the algorithm is iterated....
#..... this value will be updated later as go to the next iteration
th = 9250
nc = sum(hist>th)  # Number of clusters to be used.... 
# .....this is basically the k value as per the formula used in the algorithm
I = array(Image.open('input2.jpg')) # read the input image

cp =[[] for _ in xrange(nc)] # Array containing the pixel intensity of all k clusters
cp1 = [[0,0,0] for _ in xrange(nc)] # initializing a new array for comparision
check = 1 # initialization to keep the while loop running
'''
for i in range(0,nc):
    for j in range(0,3): # because we have 3 intensities i.e, R,G,B
        cp[i].append(randint(1,254)) # choose random centroids
'''
cp = [[6, 205, 83], [211, 206, 191], [213, 175, 33], [147, 229, 71]]


while check!=0 and check!=nan : # choose number of iterations
#while not(check ==nan or check == 0 or km >= 10) : # choose number of iterations
    # these nested arrays are the collection of indices and intensities for all the clusters
    kx = [[] for _ in xrange(nc)] # to store the x index
    ky = [[] for _ in xrange(nc)] # to store the y index
    kp = [[[],[],[]] for _ in xrange(nc)] # to store all three intensity of pixels
    # calculating the distance between each pixel and centroids
    for i in range(0,len(I[:,0])):
        for j in range(0,len(I[0,:])):
            tcomp = [] # temp array for comparision
            for k in range(0,nc):
                tcomp.append(sqrt((cp[k][0]-I[i,j,0])**2+(cp[k][1]-I[i,j,1])**2+(cp[k][2]-I[i,j,2])**2)) #(cx[k],cy[k])
                c = argmin(tcomp) # The centroid to which the pixel belongs
                kx[c].append(i) # note x index
                ky[c].append(j) # note y index
                for l in range(0,3): # for R,G,B
                    kp[c][l].append(I[i,j,l]) # note the pixel intensity
   	
    #== Mean Calculation for each cluster =======#
    check = 0 # initialize a variable to check if the means are moving
    for i in range(nc):
        for l in range(0,3): # DO each color from RGB at once
            cp1[i][l] = mean(kp[i][l]) # Mean for each cluster
            check = check + (cp1[i][l]-cp[i][l])**2
            try:
                check = int(sqrt(check))
            except ValueError:
                check = 0
            cp2 = cp
            cp = cp1 # updating the mean values
    km+=1 # Keeping a coutn if the number of iterations
print ('Number of iterations = %d\n')%km

#==========Output the cluster==============#
cluster = []
out = 102*ones([I.shape[0],I.shape[1]])#nan*numpy.ones(shape = (I.shape[0],I.shape[1]))
out_intensity = [190,170,152,40]
for i in range(0,nc):
    cluster.append(len(kp[i][0])) # choosing the cluster
for i in range(0,nc-1):
    ind = cluster.index(sorted(list(set(cluster)))[i])
    zipk = zip(kx[ind],ky[ind]) # retriving the indices of the chosen pixels
    for k,l, in zipk:
        out[k,l] = out_intensity[i]

figure()
plt.imshow(out,cmap=cm.gray,vmin=0,vmax=255)
figure()
plt.hist(img.ravel(),256,[0,256]); 
title ('Histogram of the input image')
show()

#------------F score calculation--------------
tp,tn,fp,fn = (0.0,0.0,0.0,0.0) # initializing the true/false positives/negatives
J = array(Image.open('out2.jpg').convert('L'))
for i in range(len(out[:,0])):    
    for j in range(len(out[0,:])):    
        if ((out[i,j]==102) & (J[i,j]==102)): # implementing true negative
	  tn+=1
	elif((out[i,j]>150) & (J[i,j]>150)): # implementing true positive
	  tp+=1
	elif((out[i,j]==102) & (J[i,j]>150)): # implementing false negative
	  fn+=1
	elif((out[i,j]>150) & (J[i,j]<=102)): # implementing false positive
	  fp+=1
sen = tp/(tp+fn) # Sensitivity
fot = fp/(fp+tn) # Fall Out 
fsc = (2*tp)/((2*tp)+fp+fn) # F-score
print "Finished quantitative evaluation\nSensitivity = %f\nFall-out = %f\nF-score = %f\n"%(sen,fot,fsc)
