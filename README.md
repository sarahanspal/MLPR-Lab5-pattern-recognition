# Lab 5:

**Sara Hanspal** (U20240111)  
MLPR - Plaksha University  
February 15, 2026

---

## Overview

This lab implements face detection and clustering based on color features. The project uses OpenCV's Haar Cascade classifier to detect faces in a group photograph, extracts Hue and Saturation values from the HSV color space, and applies K-Means clustering to group similar faces. A template image (Dr. Shashi Tharoor) is then classified into one of the existing clusters.

---

## Objectives

1. Detect faces in the group photograph using Haar Cascade classifier
2. Extract Hue and Saturation features from detected faces in HSV color space
3. Apply K-Means clustering (K=3) to group faces based on color similarity
4. Create visualizations including scatter plots with face thumbnails and centroids
5. Classify the template image (Dr. Shashi Tharoor) into the appropriate cluster

---

## Files Used

- `Plaksha_Faculty.jpg` - the main group photo
- `Dr_Shashi_Tharoor.jpg` - template image to classify

---

## Methodology

### Face Detection
The image was loaded and converted to grayscale as required by the Haar Cascade classifier. Using `haarcascade_frontalface_default.xml` with `scaleFactor=1.1` and `minNeighbors=5`, all faces in the image were detected. Rectangles were drawn around each detected face for visualization.

### Feature Extraction
The image was converted from BGR to HSV color space since Hue and Saturation features are more suitable for skin tone analysis than RGB values. For each detected face region:
- Mean Hue value was calculated from channel 0
- Mean Saturation value was calculated from channel 1
- Face images were stored for later visualization

The extracted features formed a 2D feature space of (Hue, Saturation) coordinates.

### K-Means Clustering
K-Means clustering was applied with K=3 clusters using scikit-learn's implementation. The algorithm was initialized with `random_state=42` for reproducibility. This produced cluster labels for each face and computed centroids representing the average Hue-Saturation values for each cluster.

### Visualization
Multiple visualizations were created:
1. **Face thumbnails plot**: Individual face images plotted at their (Hue, Saturation) coordinates using matplotlib's `OffsetImage` and `AnnotationBbox`
2. **Scatter plot with clusters**: Data points colored by cluster assignment (green for Cluster 0, indigo for Cluster 1)
3. **Centroids**: Marked as hotpink stars on the scatter plot
4. All plots include proper axis labels, titles, legends, and grid lines

### Template Classification
The template image was processed following the same pipeline:
1. Face detection using Haar Cascade
2. HSV conversion and Hue-Saturation feature extraction
3. Cluster prediction using the trained K-Means model
4. Visualization as a salmon-colored point on the final scatter plot

---

## Code Snippets

**Face Detection:**
```python
img = cv2.imread('/content/Plaksha_Faculty.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
```

**Feature Extraction:**
```python
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for (x, y, w, h) in faces_rect:
    face = img_hsv[y:y + h, x:x + w]
    hue = np.mean(face[:, :, 0])
    saturation = np.mean(face[:, :, 1])
    hue_saturation.append((hue, saturation))
    face_images.append(face)
```

**Clustering:**
```python
kmeans = KMeans(n_clusters=3, random_state=42).fit(hue_saturation)
```

**Plotting:**
```python
plt.scatter(cluster_0_points[:, 0], cluster_0_points[:, 1], c='green', label='cluster no. 0')
plt.scatter(cluster_1_points[:, 0], cluster_1_points[:, 1], c='indigo', label='cluster no. 1')

centroid_0 = kmeans.cluster_centers_[0]
centroid_1 = kmeans.cluster_centers_[1]
plt.scatter(centroid_0[0], centroid_0[1], c='hotpink', marker='*', s=200, label='centroid no. 0')
plt.scatter(centroid_1[0], centroid_1[1], c='hotpink', marker='*', s=200, label='centroid no. 1')

plt.plot(template_hue, template_saturation, marker='o', c='salmon', markersize=10, label='dr. shashi')
```

---

## Results

### Visualizations
[Insert screenshots here]

### Key Findings
The clustering successfully separated faces based on their Hue-Saturation values. K=3 clusters grouped faces with similar skin tones together. The centroids represent the average color characteristics of each cluster. The template image (Dr. Shashi Tharoor) was classified and its position in the feature space can be seen on the final scatter plot.

---

## Challenges

### Haar Cascade XML File Download

The main challenge was downloading the Haar Cascade classifier file. The download code used:

```python
import os
if not os.path.exists('haarcascade_frontalface_default.xml'):
    !wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

There were issues with file access and network connectivity in the Colab environment. This was resolved by verifying the file was successfully downloaded before attempting to load the classifier.

---

## Dependencies

### Required Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
```

**Platform:** Google Colab (all libraries pre-installed)

---

## References

- Lecture 7-8: Distance-based Features Classification, MLPR Course, Plaksha University
- OpenCV Documentation: Haar Cascade Face Detection
- Scikit-learn: K-Means Clustering Algorithm
- Haar Cascade XML Files: https://github.com/opencv/opencv/tree/master/data/haarcascades


