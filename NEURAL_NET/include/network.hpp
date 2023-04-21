#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include <iostream>
#include <numeric>

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

	network();
	network(std::vector<int> spec, size_t, int, float);

	std::vector<float> fprop(data* data);
	float activate(std::vector<float>, std::vector<float>); 
	float sigmoid(float);
	float transfer_dervative(float);
	void bprop(data* data);
	void update_weights(data* data);
	int predict(data* data);
	void train(size_t);


	const float evaluate_performance(const std::vector<data*>&);
	const void test();
	const void validate() ;




	void run_neural(data_handler* dh = nullptr); // MNIST or Iris therefore nullptr

};


#endif // !__NETWORK_HPP
