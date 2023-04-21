#include "../include/layer.hpp"


layer::layer(size_t weights_num, int neurons_num)
{
	for (size_t i = 0; i < neurons_num; i++)
	{
		neurons.emplace_back(new neuron(weights_num, neurons_num));
	}
}
