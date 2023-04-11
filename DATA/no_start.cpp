// MNIST_ML.cpp : This file contains the 'main' function. Program execution begins and ends there.






/*
Prepare the data: Divide the dataset into a training set (e.g., 60,000 images) and a test set (e.g., 10,000 images). The training set is used to find the neighbors, and the test set is used to evaluate the model's performance.

Preprocess the data: Flatten each 28x28 image into a 1D array of 784 values (pixels), representing each image as a point in a 784-dimensional space. You may also normalize the pixel values to a range between 0 and 1 for better performance.

Choose the number of neighbors (K): Select a positive integer, K, representing the number of nearest neighbors to consider when making predictions. You may need to experiment with different K values or use techniques like cross-validation to find the optimal K value.

Calculate distances: For each test image, calculate the distance between the test image and all training images. You can use a distance metric like Euclidean distance to measure similarity.

Identify K-nearest neighbors: Find the K training images that are closest to the test image.

Decision-making: Take a majority vote among the class labels (digits) of the K-nearest neighbors. Assign the most frequent digit as the predicted label for the test image.

Evaluate the model: Repeat steps 4-6 for all test images, and calculate the model's accuracy by comparing the predicted labels with the true labels of the test images
*/