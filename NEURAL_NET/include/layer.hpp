#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"
#include <vector>


struct Layer
{
	int crr_layer_size;
	std::vector<Neuron*> neurons;
	std::vector<float> layer_output;

	Layer(int, int);
	~Layer(); 

	std::vector<float> get_layer_outputs;
	int get_size();

	

};

#endif // !__LAYER_H
