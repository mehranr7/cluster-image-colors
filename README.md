# Image Color Clustering (K-Means)

A Python-based computer vision tool that uses Unsupervised Machine Learning to perform **Color Quantization** (reducing the number of distinct colors in an image) and **Image Segmentation**. It identifies the most dominant colors in any given image and reconstructs the image using only those specific color palettes.

## 📌 Overview

In digital image processing, reducing the color palette while preserving the overall visual structure is crucial for tasks like image compression, artistic rendering, and feature extraction. 

This project reads an image, flattens its pixel data into a 2D array of RGB values, and applies the **K-Means Clustering** algorithm to group similar pixels together. The center of each cluster becomes the "dominant color," and every pixel in the original image is then reassigned to its nearest cluster center, generating a simplified, clustered version of the image.

## ✨ Key Features

- **Unsupervised Learning:** Automatically discovers the underlying color structures without any labeled data.
- **Customizable Clusters:** The number of dominant colors (`n_clusters`) can be easily adjusted to see different levels of color quantization (e.g., extracting exactly 3, 5, or 10 dominant colors).
- **Matrix Manipulation:** Utilizes NumPy for efficient reshaping of multi-dimensional image arrays (Width x Height x 3 channels) to 1D feature arrays and back again.
- **Side-by-Side Visualization:** Uses OpenCV to display the original image and the newly generated, clustered image for direct comparison.

## 🚀 How It Works

1. **Image Loading:** The script reads an image (e.g., `man1.jpg`) using OpenCV.
2. **Data Reshaping:** The 3D image matrix (Height x Width x RGB) is flattened into a 2D array representing individual pixels.
3. **K-Means Fitting:** The `scikit-learn` K-Means model processes the pixels and finds `K` optimal cluster centers (the dominant colors).
4. **Image Reconstruction:** An empty NumPy array is created, and each pixel is filled with the RGB value of its assigned cluster center.
5. **Output:** The simplified image is displayed on the screen.

## 💻 Code Example

```python
# reshape h*w dimensional array with 3 feature to 1 dimensional array with 3 feature
img_data = img.reshape(-1, 3)

# fit data using kmeans model
model = sk_cluster.KMeans(n_clusters=3)
clustered_image = model.fit(img_data)

# Reconstruct image based on cluster centers...
