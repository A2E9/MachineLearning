# MachineLearning -> MNIST and IRIS.data

  ## Brief data_handler (Extract-Transform-Load) description
  
Load each image as a data object, with a feature vector of size 784 (28x28), where each element is a byte value between 0-255 representing image pixels.
Read the feature labels and assign them to each corresponding data object (assuming they are in the correct order).
Split the data using the Fisher-Yates algorithm by shuffling the indices of the data objects, and then dividing the data into training, test, and validation sets based on a constant percentage.
Count the unique classes (in this case, there are only numbers 0-9 in the file) and create a class_map, which maps each label to a unique integer.
   
   ## Brief KNN (K-nearest neighbors) description
   
First we loop through each validation object which goes through the find_knearest function to find the smallest distance in all training points. then we look at the k value get k objects with the smallest distance and add them to the neighbors vector. In the for loop we try to predict the current validation object with the neighbors list.we loop through all elements to see which is the most frequent. Lastly we check if that element is correct with the real validation object in the loop.
   
   ## Brief Kmeans Clustering description
First we initialize clusters based on k times the number needed. we create random training item lists k times. then we train the model by calculating Euclidean distances between cluster centroids and training data. We find the smallest distance, add the item to that cluster and update the centroid. During validation we find the minimal distance and check if the most frequent point in the cluster matches the validation set
   
  ## Brief ANN (Artificial Neural Network) description
  
For the first we are setting how many iterations should the model be trained on.
next we use forward propagation to make initial predictions for the labels.
following this we use back propagation to calculate the error gradients and to adjust the 
weights which helps improve predictions in subsequent epochs to predict the labels.
after each epochs we display the (mse) Mean square error to track the perf.
and with each epochs the model becomes better which reduces the mse due to learning the underlying "patterns"


### Needs exploration...
> bias term shifts activation

> sigmoid and its derivative (gradients)

> backpropagation is just a bit hard to memorize

#### Tutorial link
> https://www.youtube.com/watch?v=E1K9SZCm0fQ&list=PL79n_WS-sPHKklEvOLiM1K94oJBsGnz71
