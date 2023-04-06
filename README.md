# MachineLearning

   1. Load each image as a data object, with a feature vector of size 784 (28x28), where each element is a byte value between 0-255 representing image pixels.
   2. Read the feature labels and assign them to each corresponding data object (assuming they are in the correct order).
   3. Split the data using the Fisher-Yates algorithm by shuffling the indices of the data objects, and then dividing the data into training, test, and validation sets based on a constant percentage.
   4. Count the unique classes (in this case, there are only numbers 0-9 in the file) and create a class_map, which maps each label to a unique integer.
   
   ## Short Desc for KNN
   
   Calculate the distance between all training images and a single validation image, then select the k training points with the smallest distances. Determine   the most frequent label among these neighbors and check if it matches the label of the validation image.
   
