#include "../include/network.hpp"


network::network()
{
	layers = {};
	learning_rate = 0.0f;
}


network::network(std::vector<int> neurons_per_layer, size_t extracted_data_size, int classes_num, float learning_rate)
{
	this->learning_rate = learning_rate;

	for (int i = 0; i < neurons_per_layer.size(); i++) // { 11, 12 }.size()
	{
		int neurons_num = neurons_per_layer[i];

		if (i == 0)
			layers.emplace_back(new layer(extracted_data_size, neurons_num)); // MNIST first hidden layer: layer(784, 11) -> 11 neurons -> each (784 + 1 bias) rand weight values

		else
			layers.emplace_back(new layer(layers.back()->neurons.size(), neurons_num));	// Hidden layer i: layer(11, 12) -> 12 neurons -> (11 + 1 bias) rand weight values

	}
	layers.emplace_back(new layer(layers.back()->neurons.size(), classes_num)); // Output layer: layer(12, 10) -> 10 neurons -> (10 + 1 bias) rand weight values
}


/// <summary>
/// Activation and sigmoid functions used for forward and backward propagation
/// 
/// activation = sumup all weights multipld by all inputs + bias; (random_weights * extracted_data values)
/// </summary>
float network::activate(std::vector<float> weights, std::vector<float> inputs)
{
	float activation = weights.back(); // bias term

	for (size_t i = 0; i < weights.size() - 1; i++)
	{
		activation += weights[i] * inputs[i];
	}
	return activation;
}
/// <summary>
/// add non-linearity to the output of a neuron
/// </summary>
/// <param name="activation"></param>
/// <returns></returns>
float network::sigmoid(float activation)
{
	return 1.0f / (1.0f + exp(-activation)); // logistic (sigmoid) function | squashes to the range 0-1
}


/// <summary> Needs further exploration...
/// in mathematics the derivative is a measure of how a function changes as its input changes
/// 
/// The derivative of the activation function is used to calculate the gradient of the error
/// with respect to the weights, which is then used to update the weights in the direction that minimizes the error
/// </summary>
/// <param name="sigmoid(x)"></param>
/// <returns>
/// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
/// </returns>
float network::transfer_dervative(float output)
{
	return output * (1.0f - output);
}


/// <summary>
/// (forward propagation): 
/// Calcs all layers neurons weights together
/// </summary>
/// <returns>
/// network's prediction vector
/// </returns>
std::vector<float> network::fprop(data* d)
{
	std::vector<float> inputs = *d->get_float_extracted_data();
	for (size_t i = 0; i < layers.size(); i++)
	{
		std::vector<float> new_inputs;
		for (auto& n : layers.at(i)->neurons)
		{									//   weights_list is extracted_data_size with rand_vals
			float activation = this->activate(n->weights_list, inputs); // weighted sum of inputs
			n->output = this->sigmoid(activation); // squash the activation to (0-1) graph
			new_inputs.emplace_back(n->output); // add the squashed output values of the neurons in new_inputs
		}
		inputs = new_inputs; // used for further layers to calculate
	}
	return inputs; // outputs (size of last layer)
}


/// <summary>...needs futher exploration...
/// (backward propagation): 
/// Calculates the error gradients for each neuron in the network
/// starting from the output layer and moving back through the hidden layers
/// </summary>
/// <param name="data"></param>
void network::bprop(data* data)
{
	for (int i = (int)layers.size() - 1; i >= 0; i--) // starting from last layer
	{
		layer* layer = layers.at(i);
		std::vector<float> errors;
		if (i != layers.size() - 1) // check if !output-layer 
		{
			for (int j = 0; j < layer->neurons.size(); j++) // Calc errors for each neuron in the layer
			{
				float error = 0.0;
				for (auto& n : layers.at(i + 1)->neurons) // ~next~[penultimate] layer each neuron
				{
					error += (n->weights_list.at(j) * n->error_delta); // each weight in penultimate layer * error_delta
				}
				errors.emplace_back(error); 
			}
		}
		else // first exec
		{
			for (int j = 0; j < layer->neurons.size(); j++) // all neurons of layer
			{
				neuron* n = layer->neurons.at(j);										 
				errors.emplace_back((float)data->get_class_vector()->at(j) - n->output); // check for error [actual - activated (0-1)]
			}
		}
		for (size_t j = 0; j < layer->neurons.size(); j++) // all neurons of layer
		{
			auto* n = layer->neurons.at(j);
			n->error_delta = errors.at(j) * this->transfer_dervative(n->output);// gradient ...needs futher exploration... | measure how much the error changes
		}
	}
}


/// <summary>
/// A method that updates the weights_list of each neuron in the network based on the calculated gradients and the learning rate.
/// </summary>
/// <param name="data"></param>
void network::update_weights(data* data)
{
	std::vector<float> inputs = *data->get_float_extracted_data(); // also weights size
	for (size_t i = 0; i < layers.size(); i++)
	{
		if (i != 0)
		{
			for (auto& n : layers.at(i - 1)->neurons)
			{
				inputs.emplace_back(n->output);
			}
		}
		for (auto& n : layers.at(i)->neurons)
		{
			for (size_t j = 0; j < inputs.size(); j++) // going through all weights
			{
				n->weights_list.at(j) += this->learning_rate * n->error_delta * inputs.at(j); // Σ rate * error_gradient * feature_data (eg mnist pixel bytes)
			}
			n->weights_list.back() += this->learning_rate * n->error_delta; // bias term used for atcivation shifting "more flexibility to learn complex patterns"
		}
		inputs.clear();
	}
}


/// <summary>
///  A method that trains the network using the training dataset for a specified number of epochs
/// </summary>
/// <param name="num_epochs"></param>
void network::train(size_t num_epochs)
{
	for (int i = 1; i <= num_epochs; i++)
	{
		float sum_error = 0.0f;
		size_t num_samples = 0;
		for (auto& data : *this->training_data) // giving data for prediction and expected
		{
			std::vector<float> outputs = fprop(data); // forward propagation to get the output
			std::vector<int>* expected = data->get_class_vector(); // what is expected
			float temp_err_sum = 0.0f;
			for (size_t j = 0; j < outputs.size(); j++) // measure the average squared difference MSE = (1/n) * Σ(actual – forecast)²
			{
				float error = expected->at(j) - outputs.at(j); //(actual – forecast)
				temp_err_sum += error * error; // ²
			}
			sum_error += temp_err_sum; // Σ
			num_samples++;
			bprop(data);
			update_weights(data);
		}
		float mse = sum_error / num_samples; // * (1/n)
		std::cout << "Iteration: " << i << "\t Error = " << mse << " MSE" << std::endl;
	}
}




/// <summary>
/// A method that generates a prediction for a given input data instance by selecting the class with the highest output value from the output layer
/// 
/// Prediction runs after all processes were done so it is taking calcing with last calcd layer
/// </summary>
/// <param name="data"></param>
/// <returns>
/// distance from first element till position of the biggest element
/// </returns>
int network::predict(data* data)
{
	std::vector<float> outputs = fprop(data);
	return (int)std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

const float network::evaluate_performance(const std::vector<data*> &data_set)
{
	float num_correct = 0.0;
	float count = 0.0;
	for (auto& data : data_set)
	{
		int index = predict(data); // index of the largest output 
		
		std::cout << "Class Vector: " << data->get_class_vector()->at(index) << "\n";

		if (data->get_class_vector()->at(index) == 1)
			num_correct++;
		count++;
	}
	return (num_correct / count) * 100;
}
const void network::validate() 
{
	printf("\nValidation Performance: %.4f %%\n", evaluate_performance(*this->validation_data));
}
const void network::test() 
{
	printf("Test Performance: %.4f %%\n", evaluate_performance(*this->test_data));
}


void network::run_neural(data_handler* dh)
{
	if (!dh)
	{
		dh = new data_handler();
		dh->read_csv("../iris_data/iris.data", ",");
		dh->split_data();
	}

	size_t iterations = 0;
	std::cout << "How many Iterations: ";
	std::cin >> iterations;
	std::cout << "\n";

	std::vector<int> neurons_per_layer = { 11 }; 

	auto lambda = [&]()
	{
		network* net = new network(neurons_per_layer,
								   dh->get_training_data()->at(0)->get_float_extracted_data()->size(),	// MNIST 784 <<>> IRIS 4
								   dh->get_class_counts(),												// MNIST 10 <<>> IRIS 3
								   0.25f);

		net->set_training_data(dh->get_training_data());
		net->set_test_data(dh->get_test_data());
		net->set_validation_data(dh->get_validation_data());

		net->train(iterations);
		net->validate();
		net->test();
	};
	lambda();

}

/*		IRIS.DATA

Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class:
	  -- Iris Setosa
	  -- Iris Versicolour
	  -- Iris Virginica


Class Distribution: 33.3% for each of 3 classes.

*/

