#include "../include/layer.hpp"

layer::layer(size_t prev_layer_size, int crr_layer_size)
{
	for (size_t i = 0; i < crr_layer_size; i++)
	{
		neurons.emplace_back(new neuron(prev_layer_size, crr_layer_size));
	}
	this->crr_layer_size = crr_layer_size;
}