#define _ITERATOR_DEBUG_LEVEL 0
#include "../include/data.hpp"


data::data()
{
	feature_vector = new std::vector<uint8_t>;

	label = NULL;
	distance = NULL;
	enum_label = NULL;
	class_vector = NULL;
	float_feature_vector = NULL;
}
data::~data()
{

}

void data::set_feature_vector(std::vector<uint8_t>* vect)
{
	feature_vector = vect;
}
void data::append_to_feature_vector(uint8_t val)
{
	feature_vector->emplace_back(val);
}
void data::set_label(uint8_t val)
{
	label = val;
}
void data::set_enumerated_label(int val)
{
	enum_label = val;
}
void data::set_distance(double val)
{
	distance = val;
}

double data::get_distance()
{
	return distance;
}


size_t data::get_feature_vector_size()
{
	return feature_vector->size();
}
uint8_t data::get_label()
{
	return label;
}
uint8_t data::get_enumerated_label()
{
	return enum_label;
}

std::vector<uint8_t>* data::get_feature_vector()
{
	return feature_vector;
}


//Deep
void data::set_class_vector(int count)
{
	class_vector = new std::vector<int>();
	for (int i = 0; i < count; i++)
	{
		// ??? class_vector->at(i) = 1; 
		if (i == label)
		{
			class_vector->emplace_back(1); 
		}
		else
		{
			class_vector->emplace_back(0);
		}
	}
}
void data::append_to_feature_vector(float val)
{
	float_feature_vector->emplace_back(val);
}
void data::set_float_feature_vector(std::vector<float>* vect)
{
	float_feature_vector = vect;
}

std::vector<int>* data::get_class_vector()
{
	return class_vector;
}
std::vector<float>* data::get_float_feature_vector()
{
	return float_feature_vector;
}
//Learn