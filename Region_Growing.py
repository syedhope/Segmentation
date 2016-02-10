import math
from scipy import signal
from PIL import Image
import numpy
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
from scipy.spatial import distance

#====================================Region Growing segmentation==========================
def reg_grow (in_img, # input image
t # The desired threshold
):
    #-----------Selecting the seed attributes------------#
    I = array(Image.open(in_img).convert('L')) # read the input image
    plt.figure()
    title('Click the pixel that is to be seeded')
    xlabel('The pixel you choose will be given as the input seed to the program')
    plt.imshow(I,cmap=cm.gray) #Display the image for seed selection
    xy = ginput(1) # allow just one seed
    x = int(xy[0][0]) # make a note of the the x index of the seed 
    y = int(xy[0][1]) # make a note of the the y index of the seed
    show()
    plt.close() # close the window and proceed...
    #----------------------Initialization------------------------#
    visited = I*0 # a matrix to keep a check of which pixels have been accessed
    # 'visited' matrix helps us avoid the repetition of seeds
    out = 0 * ones(shape=shape(I)) # initializing the output binary image
    collection = [[I[x,y]],[x],[y]] # stores the attributes of seeds that are to be used for further iterations
    centroid = [I[x,y],x,y] # stores the attributes of the seed that is being used for the current iteration
    # the seed for the next iteration is chosen from the 'seed' matrix
    '''
    specification of 'seed' matrix is as follows:
    -
    | Row1 ---> pixel intensity................n      |
    -                                                 |
    | Row2 ---> x index of the pixel...........n      | where(n = number of pixels)
    -                                                 |
    | Row3 ---> y index of the pixel...........n      |
    -
    '''
    #Now we run the region growing algorithm until we have accessed all the pixels from the image 
    #------------- Region Growth ----------------#
    for index in range(0,len(I[:,0])*len(I[0,:])):
        n_x = [x-1,x,x+1,x-1,x+1,x-1,x,x+1] # x indices of the neighbours for the current seed
        n_y = [y-1,y-1,y-1,y,y,y+1,y+1,y+1] # y indices of the neighbours for the current seed
        #---making sure that the indices lie inside the image---
        n_x = filter(lambda a: a <321, n_x) # if any of these indices go beyond the number of rows, we drop that index
        n_y = filter(lambda a: a <481, n_y) # if any of these indices go beyond the number of columns, we drop that index
        n_xy = zip(n_x,n_y)# to choose x and y index at a time
        for i,j in n_xy: # checking the condition for each neighbour
            if visited[i,j] == 0: # if the pixel has not been already accessed...
                if (distance.euclidean(centroid[0],I[i,j])) <= t: # check if the difference between the seed and neighbour is within the desired threshold
                    out[i,j] = 255 # make the pixel visible
    		collection[0].append(I[i,j]) # note the next pixel that is to be seeded
    		collection[1].append(i) # note the x index
    		collection[2].append(j) # note the y index
            visited[i,j] = 1 # mark the pixel as visited
        centroid = [collection[0][index+1],collection[1][index+1],collection[2][index+1]] # choose the next seed
        x = collection[1][index+1] # update the x index of the seed
        y = collection[2][index+1] # update the x index of the seed
    # Display the binary output image
    plt.imshow(out,cmap=cm.gray)
    show()
        
    #------------F score calculation--------------
    tp,tn,fp,fn = (0.0,0.0,0.0,0.0) # initializing the true/false positives/negatives
    J = array(Image.open('out1.jpg').convert('L'))
    for i in range(len(out[:,0])):    
        for j in range(len(out[0,:])):    
            if ((out[i,j]>200) & (J[i,j]>200)): # implementing true negative
                tn+=1
            elif((out[i,j]<100) & (J[i,j]<100)): # implementing true positive
                tp+=1
            elif((out[i,j]>200) & (J[i,j]<100)): # implementing false negative
                fn+=1
            elif((out[i,j]<100) & (J[i,j]>200)): # implementing false positive
                fp+=1
    sen = tp/(tp+fn) # Sensitivity
    fot = fp/(fp+tn) # Fall Out 
    fsc = (2*tp)/((2*tp)+fp+fn) # F-score
    print "Finished quantitative evaluation\nSensitivity = %f\nFall-out = %f\nF-score = %f\n"%(sen,fot,fsc)
    
reg_grow('input1.jpg',7) # choose the input image and threshold