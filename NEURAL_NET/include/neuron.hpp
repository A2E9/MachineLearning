#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>


class neuron
{
public:

	float output;
	float delta;
	std::vector<float> weights;

	neuron(size_t, int);
	~neuron();

	void initialize_weights(size_t prev_layer_size);
};
#endif // !__NEURON_HPP
