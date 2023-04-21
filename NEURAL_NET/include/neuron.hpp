#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>
#include <random>


class neuron
{
public:

	float output;
	float error_delta;
	std::vector<float> weights_list;

	neuron(size_t, int );

};
#endif // !__NEURON_HPP
