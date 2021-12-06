#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
# PYTHON PROGRAM TO COMPARE THE FEATURES BETWEEN 2 FINGERPRINTS IMAGES
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
# 1 =>
# Python libraries use to create fingerprint analysis program
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 2 =>
# import the location of images and the image file in the system
dataset_path =('path or folder of the images')
img_fingerprint = cv2.imread(os.path.join(dataset_path, 'image file location'))

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 3 =>
# Convert image colours from cv's default color order to RGB.
# (colour conversion for easier identification)
img_fingerprint = cv2.cvtColor(img_fingerprint, cv2.COLOR_BGR2RGB)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 4 =>
# ORB detector from opencv is used to detect the keypoints and description of the image.
# ORB = Oriented FAST and Rotated BRIEF
orb = cv2.ORB_create()
key_points, description = orb.detectAndCompute(img_fingerprint, None)
img_fingerprint_keypoints = cv2.drawKeypoints(img_fingerprint,
                                              key_points,
                                              img_fingerprint,
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 5 =>
# Draw circles plot using matplot library
plt.figure(figsize=(16, 16))
plt.title('-- ORB Interest Points --')
plt.imshow(img_fingerprint_keypoints);
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 6 =>
# ****** FEATURE MATCHING ******

# SIFT = Scale-Invariant-Feature-Transform | DES = description | KP = keypoints.
# function to check the matching parts.
def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(dataset_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des    

# function to detect the matching parts.
def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    """Draw ORB feature matches of the given two images."""
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)
    # Sort matches by distance.  Best come first.
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2)
    # Show top 10 matches
    plt.figure(figsize=(16, 16))
    plt.title(type(detector))
    plt.imshow(img_matches);
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 7 =>
# Matching the key points. Using ORB, now we match the similar parts.
# SIFT algorithm used to detect the different keypoints & features.
# Plot the figure of the fingerprint.
orb = cv2.ORB_create()
draw_image_matches(orb, 'image file 1', 'image file 2')

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img_fingerprint, None)
img_kp = cv2.drawKeypoints(img_fingerprint, kp, img_fingerprint)

plt.figure(figsize=(15, 15))
plt.imshow(img_kp);
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------
# 8 =>
# Detect both the images,and then its high level features,
# using its keypoints and description.
# Plot the figure.
img1, kp1, des1 = image_detect_and_compute(sift, '1.jpg')
img2, kp2, des2 = image_detect_and_compute(sift, '2.jpg')

# FLANN = Fast library for approximate Nearest Neighbors.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0]
for i in range(len(matches))]
# ratio test.
for i, (m, n) in enumerate(matches):
    if m.distance < 0.55*n.distance:
        matchesMask[i] = [1, 0]

draw_parameters = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

# plot graph for the high level features.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_parameters)
plt.figure(figsize=(18, 18))
plt.imshow(img3);
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------

# using this analysis method, we identify the features through the diagram.
# Hence, comparing the 2 fingerprint images we understand they belong to 2 different people,
# since there were more keypoints in the second diagram.

