#include "../include/network.hpp"
//#include "../include/layer.hpp"

#include <numeric>

network::network()
{
	layers = {};
	learning_rate = 0.0f;
	test_performance = 0.0f;
}

//network::~network(){}


network::network(std::vector<int> spec, size_t input_size, int num_classes, float learning_rate)
{
	test_performance = NULL;

	for (int i = 0; i < spec.size(); i++)
	{
		if (i == 0)
			layers.emplace_back(new layer(input_size, spec.at(i)));
		else
			layers.emplace_back(new layer(layers.at(i - 1)->neurons.size(), spec.at(i)));

	}
	layers.emplace_back(new layer(layers.at(layers.size() - 1)->neurons.size(), num_classes));
	this->learning_rate = learning_rate;
}


/// <summary>
/// activate, transfer, and transfer_derivative: Activation and transfer functions (and their derivatives) used for forward and backward propagation in the network
/// </summary>
float network::activate(std::vector<float> weights, std::vector<float> input)
{
	float activation = weights.back(); // bias term

	for (size_t i = 0; i < weights.size() - 1; i++)
	{
		activation += weights[i] * input[i];
	}
	return activation;
}

float network::transfer(float activation)
{
	return 1.0f / (1.0f + exp(-activation));
}

float network::transfer_dervative(float output)
{
	return output * (1 - output);
}

/// <summary>
///  (forward propagation): A method that calculates the output of each neuron in the network given an input data instance.
/// </summary>
/// <param name="d"></param>
/// <returns></returns>
std::vector<float> network::fprop(data* d)
{
	std::vector<float> inputs = *d->get_normalized_feature_vector();
	for (size_t i = 0; i < layers.size(); i++)
	{
		layer* layer = layers.at(i);
		std::vector<float> new_inputs;
		for (auto& n : layer->neurons)
		{
			float activation = this->activate(n->weights, inputs);
			n->output = this->transfer(activation);
			new_inputs.emplace_back(n->output);
		}
		inputs = new_inputs;
	}
	return inputs; // output layer ouput layers
}


/// <summary>
/// (backward propagation): A method that calculates the error gradients for each neuron in the network, starting from the output layer and moving back through the hidden layers
/// </summary>
/// <param name="data"></param>
void network::bprop(data* data)
{
	for (int i = (int)layers.size() - 1; i >= 0; i--)
	{
		layer* layer = layers.at(i);
		std::vector<float> errors;
		if (i != layers.size() - 1)
		{
			for (int j = 0; j < layer->neurons.size(); j++)
			{
				float error = 0.0;
				for (auto& n : layers.at(i + 1)->neurons)
				{
					error += (n->weights.at(j) * n->delta);
				}
				errors.emplace_back(error);
			}
		}
		else
		{
			for (int j = 0; j < layer->neurons.size(); j++)
			{
				neuron* n = layer->neurons.at(j);
				errors.emplace_back((float)data->get_class_vector()->at(j) - n->output); //expected - actual
			}
		}
		for (size_t j = 0; j < layer->neurons.size(); j++)
		{
			auto* n = layer->neurons.at(j);
			n->delta = errors.at(j) * this->transfer_dervative(n->output);// gradient / derivative part of backprop
		}
	}
}


/// <summary>
/// A method that updates the weights of each neuron in the network based on the calculated gradients and the learning rate.
/// </summary>
/// <param name="data"></param>
void network::update_weights(data* data)
{
	std::vector<float> inputs = *data->get_normalized_feature_vector();
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
			for (size_t j = 0; j < inputs.size(); j++)
			{
				n->weights.at(j) += this->learning_rate * n->delta * inputs.at(j);
			}
			n->weights.back() += this->learning_rate * n->delta;
		}
		inputs.clear();
	}
}


/// <summary>
/// A method that generates a prediction for a given input data instance by selecting the class with the highest output value from the output layer.
/// </summary>
/// <param name="data"></param>
/// <returns></returns>
int network::predict(data* data)
{
	std::vector<float> outputs = fprop(data);
	return (int)std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}


/// <summary>
///  A method that trains the network using the training dataset for a specified number of epochs.
/// </summary>
/// <param name="num_epochs"></param>
void network::train(int num_epochs)
{
	for (int i = 0; i < num_epochs; i++)
	{
		float sum_error = 0.0f;
		for (data* d : *this->training_data)
		{
			std::vector<float> outputs = fprop(d);
			std::vector<int>* expected = d->get_class_vector();
			float temp_err_sum = 0.0f;
			for (size_t j = 0; j < outputs.size(); j++)
			{
				//temp_err_sum += (float)pow((float)expected->at(j) - (float)outputs.at(j), 2); slow
				float error = expected->at(j) - outputs.at(j);
				temp_err_sum += error * error;
			}
			sum_error += temp_err_sum;
			bprop(d);
			update_weights(d);
		}
		std::cout << "Iteration: " << i << "\t Error = " << sum_error << std::endl;
	}
}

float network::test()
{
	float num_correct = 0.0;
	float count = 0.0;
	for (auto& data : *this->test_data)
	{
		count++;
		int index = predict(data);

		if (data->get_class_vector()->at(index) == 1)
			num_correct++;
	}
	test_performance = (num_correct / count);
	return test_performance;
}

void network::validate()
{
	float num_correct = 0.0;
	float count = 0.0;
	for (auto& data : *this->validation_data)
	{
		count++;
		int index = predict(data);

		if (data->get_class_vector()->at(index) == 1)
			num_correct++;
	}

	printf("Validation Performance: %.4f\n", num_correct / count);
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

	std::vector<int> hidden_layers = { 10 };
	auto lambda = [&]()
	{
		network* net = new network(hidden_layers,
								   dh->get_training_data()->at(0)->get_normalized_feature_vector()->size(),
								   dh->get_class_counts(),
								   0.25f);
		net->set_training_data(dh->get_training_data());
		net->set_test_data(dh->get_test_data());
		net->set_validation_data(dh->get_validation_data());

		net->train(iterations);
		net->validate();
		printf("Test Performance: %.3f\n", net->test());
	};
	lambda();

}