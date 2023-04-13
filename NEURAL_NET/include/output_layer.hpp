#ifndef __OUTPUT_LAYER_HPP
#define __OUTPUT_LAYER_HPP

#include "layer.hpp"
#include "../../DATA/include/data.hpp"

class OutputLayer : public Layer
{
public:
	OutputLayer(int prev, int crr) : Layer(prev, crr)
	{}

	void feed_forward(Layer);
	void back_prop(data* data);
	void update_weights(float, Layer*);

};


#endif // !__OUTPUT_LAYER_HPP
