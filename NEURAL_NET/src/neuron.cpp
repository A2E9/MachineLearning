#include "../include/neuron.hpp"
#include <random>

float gen_rand_num(float min, float max)
{
	/*float random = (float)rand() / (RAND_MAX);
	return min + random * (max - min);*/
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	return dis(gen);
}

neuron::neuron(size_t prev_layer_size, int crr_layer_size)
{
	initialize_weights(prev_layer_size);
}
neuron::~neuron() {}

void neuron::initialize_weights(size_t prev_layer_size)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, 1.0);
	for (int i = 0; i < prev_layer_size+1; i++)
	{
		weights.emplace_back(gen_rand_num(-1.0, 1.0));
	}
}