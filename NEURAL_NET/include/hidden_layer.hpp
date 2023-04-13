#ifndef __HIDDEN_LAYER_HPP
#define __HIDDEN_LAYER_HPP

#include "layer.hpp"


class HiddenLayer : public Layer
{
public:
	HiddenLayer(int prev, int crr) : Layer(prev, crr) {}

	void feed_forward(Layer prev);
	void back_prop(Layer next);
	void update_weights(float, Layer*);

};




#endif // !__HIDDEN_LAYER_HPP
