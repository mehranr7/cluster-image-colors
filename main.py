import cv2
import sklearn.cluster as sk_cluster
import numpy as np

# read image and define related info
img = cv2.imread("man1.jpg")
cv2.imshow('Main', img)
img_width = len(img[0])
img_height = len(img)

# reshape h*w dimensional array with 3 feature to 1 dimensional array with 3 feature
img_data = img.reshape(-1, 3)

# clusters number
n_clusters = 3

# fit data using kmeans model
model = sk_cluster.KMeans(n_clusters)
clustered_image = model.fit(img_data)

# reshape 1 dimensional array to h*w dimensional array
img_labels = clustered_image.labels_.reshape(img_height, img_width, 1)

# create an empty array to fill image
blank_image = np.zeros((img_height, img_width, 3), np.uint8)

# set the pixel from cluster centers
for i in range(img_height):
    for j in range(img_width):
        color = clustered_image.cluster_centers_[img_labels[i,j]]
        blank_image[i, j] = color

# display the clustered image
cv2.imshow('Clustered', blank_image)
cv2.waitKey(0)