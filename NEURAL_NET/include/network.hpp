#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <iostream>

#include "../../DATA/include/data.hpp"
#include "../../DATA/include/common.hpp"
#include "../../DATA/include/data_handler.hpp"

#include "layer.hpp"
#include "neuron.hpp"


class network : public common_data
{
public:
	std::vector<layer*> layers;
	float learning_rate;
	float test_performance;



	//network(/*just to run*/);
	network();

	network(std::vector<int> spec, size_t, int, float);

	//network(std::vector<int> hidden_layer_spec, int, int);
	//~network();
	std::vector<float> fprop(data* data);
	float activate(std::vector<float>, std::vector<float>); // dot product
	float transfer(float);
	float transfer_dervative(float); // for backprop
	void bprop(data* data);
	void update_weights(data* data);
	int predict(data* data); // return the index of the maximm value in output array
	void train(int);
	float test();
	void validate();




	void run_neural(data_handler* dh = nullptr);

};


#endif // !__NETWORK_HPP
