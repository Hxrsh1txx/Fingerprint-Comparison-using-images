# Fingerprint-Comparison-using-images
Python program to Compare 2 fingerprints using images and its features.

== PROJECT PROGRAM DESCRIPTION ==
With 2 images of fingerprints, we are extracting the features and matching them to see if they are similar or not.
For our project we used libraries like NumPy, pandas, matplotlib and OpenCV.
There are various algorithms for the different description and detection of features.

== Algorithm ==

ORB: Oriented fast & Rotated Brief. It is a part of OpenCV and is mostly used for computer vision related programs.

(Feature Based Algorithms):

1.	SIFT: Scale Invariant Feature Transform. It is used in detection of features like key points and description in computer vision.
2.	BF matcher: Brute Force matcher. It is used to identify simple features from the dataset.
3.	FLANN: Fast library for approximate nearest neighbors. It is used for matching high-level features from the given datasets. 

== The Process ==

1.	Constructing a scale space so that the features are scale independent.
2.	Key point localization to identify the key points.
3.	Ensuring that key points are rotation invariant.
4.	Description for each key point.

This is the diagram for the algorithm used for fingerprint comparison.



![image](https://user-images.githubusercontent.com/91388375/144934558-2b81d0fb-9203-42ee-b50c-7a813d839179.png)




== CODE DESCRIPTION == 

1.	Importing the libraries used.
2.	Importing the image location and image file.
3.	Converting the color of the image for easier detection.
4.	ORB detector used to apply on image to detect the key points and description. 
5.	Matplotlib used to plot circles and the images on a graph figure.
6.	Feature matching check: identifying and matching key points and descriptions with the help of SIFT algorithm. Plotting the Figure with all the details.
7.	Plotting the image figure and drawing the key points from the SIFT algorithm.
8.	Using the key points, description, images and FLANN algorithm we detect the high-level features. After detection, the features are compared with the 2 fingerprint images.

== CONCLUSION ==
Comparing the fingerprint images we understand if they are different or same person.
