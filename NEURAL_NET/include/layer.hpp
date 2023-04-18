#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"
#include <vector>

static int layerId = 0;

struct layer
{
	int crr_layer_size;
	std::vector<neuron*> neurons;
	std::vector<float> layer_outputs;

	layer(size_t prev, int crr);
	~layer(); 

	std::vector<float> get_layer_outputs;
	int get_size();

	

};

#endif // !__LAYER_H
