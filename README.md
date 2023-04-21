# MachineLearning

  ## Brief data_handler (Extract-Transform-Load) description
  
Load each image as a data object, with a feature vector of size 784 (28x28), where each element is a byte value between 0-255 representing image pixels.
Read the feature labels and assign them to each corresponding data object (assuming they are in the correct order).
Split the data using the Fisher-Yates algorithm by shuffling the indices of the data objects, and then dividing the data into training, test, and validation sets based on a constant percentage.
Count the unique classes (in this case, there are only numbers 0-9 in the file) and create a class_map, which maps each label to a unique integer.
   
   ## Brief KNN (K-nearest neighbors) description
   
   Calculate the distance between all training images and a single validation image, then select the k training points with the smallest distances. Determine   the most frequent label among these neighbors and check if it matches the label of the validation image.
   
  ## Brief ANN (Artificial Neural Network) description
  
  So for the first we are setting how many iterations should the model be trained on.
next we use forward propagation to make initial predictions for the labels.
following this we use back propagation to calculate the error gradients and to adjust the 
weights which helps improve predictions in subsequent epochs to predict the labels.
after each epochs we display the (mse) Mean square error to track the perf.
and with each epochs the model becomes better which reduces the mse due to learning the underlying "patterns"


### Needs exploration...
> bias term shifts activation

> sigmoid and its derivative (gradients)

> backpropagation is just a bit hard to memorize
