#include "../include/neuron.hpp"


float gen_rand_num(float min, float max)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(min, max);
	return dis(gen);
}


neuron::neuron(size_t weights_num, int neurons_num)
{
	error_delta = 0.0f, output = 0.0f;
	float sqrt_neurons = static_cast<float>(std::sqrt(neurons_num));
	//Initialize weights_list
	for (int i = 0; i < weights_num + 1; i++)
		weights_list.emplace_back(gen_rand_num(-1.0f / sqrt_neurons,
											    1.0f / sqrt_neurons));
}
