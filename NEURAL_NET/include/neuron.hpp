#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>


class Neuron
{
public:
	Neuron(int, int){}

	~Neuron() {}

	void setError(float);
	void setWeight(float, int);
	void initialize_wights(int prev_layer_size, int crr_layer_size);
	
	float relu();
	float sigmoid();
	float activate();
	float get_error();
	float get_output();
	float leaky_relu();
	float inverse_sqrt_relu();
	float get_output_derivative();
	float calc_output_derivative();
	float calc_pre_activation(std::vector<float>);

	std::vector<float> get_weights();



private:
	std::vector<float> weights;
	
	float alpha;
	float error;
	float pre_activation;
	float activated_ouput;
	float output_derivative;
};


#endif // !__NEURON_HPP
