K-means Clustering
------------------

K-means algorithm is an unsupervised clustering algorithm that classifies the input data points into multiple classes based on their inherent distance from each other. The algorithm assumes that the data features form a vector space and tries to find natural clustering in them.

1.Segmentation via K-means Clustering
-------------------------------------

Implements conventional k-means clustering algorithm for gray-scale image segmentation with appropriate k value found automatically Sample output segmentation file (out1.jpg) is used for evaluating accuracy of the segmentation.


------------------------
2. Clustering
------------------------
Implements k-means clustering algorithm for color images, the algorithm finds the parameter k automatically from the histogram. Sample output segmentation file (out2.jpg) is used to evaluate accuracy of the segmentation.

PSEUDO CODE:
Data: Input image
Result: Output image segmented via k-means
Choose target number k of clusters (either manually or automatically);
Initialize the centroids with k random intensities;
for all other data points do
		Cluster the points based on distance of their intensities from the centroid intensities;
end
for all data points do
		Compute the new centroid for each of the clusters;
end

------------------------------
3.Region Growing Segmentation
------------------------------

Implements region growing algorithm for gray-scale images where absolute intensity differences is used for
region definition. Algorithm accepts one seed input from users via clicking a point in the image and
immediately returns the segmentation results. 8-connectivity of pixels to do segmentation has been used.
Finally the accuracy has been calculated.