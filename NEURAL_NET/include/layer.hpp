#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"
#include <vector>


struct layer
{
	std::vector<neuron*> neurons;

	layer(size_t prev, int crr);
};

#endif // !__LAYER_H
