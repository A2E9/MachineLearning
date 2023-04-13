#ifndef __INPUT_LAYER_HPP
#define __INPUT_LAYER_HPP

#include "layer.hpp"
#include "../../DATA/include/data.hpp"

class InputLayer : public Layer
{
public:
	InputLayer (int prev, int crr) : Layer(prev, crr){}
	void set_layer_outputs(data *d);


private:

};


#endif // !__INPUT_LAYER_HPP
