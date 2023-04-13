#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <iostream>

#include "../../DATA/include/data.hpp"
#include "../../DATA/include/common.hpp"

#include "layer.hpp"
#include "neuron.hpp"

#include "input_layer.hpp"
#include "output_layer.hpp"
#include "hidden_layer.hpp"

class Network : public common_data
{
public:
	Network() { start_neural(); }

	Network(std::vector<int> hidden_layer_spec, int, int);
	~Network() {}

	void test();
	void train();
	void validate();
	void update_weights();
	void b_prop(data* data);
	void f_prop(data* data);


	void start_neural()
	{
		std::cout << "Hello World!\n";
	}


private:
	InputLayer* input_layer;
	OutputLayer* output_layer;

	std::vector<HiddenLayer*> hidden_layers;
	float eta; //learning rate
};


#endif // !__NETWORK_HPP
