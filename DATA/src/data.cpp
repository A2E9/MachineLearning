#define _ITERATOR_DEBUG_LEVEL 0
#include "../include/data.hpp"
#include <iostream>


data::data()
{
	extracted_data = new std::vector<uint8_t>;

	label = NULL;
	distance = NULL;
	enum_label = NULL;
	class_vector = NULL;
	extracted_float_data = NULL;
}
data::~data()
{

}

void data::set_extracted_data(std::vector<uint8_t>* vect)
{
	extracted_data = vect;
}
void data::append_to_extracted_data(uint8_t val)
{
	extracted_data->emplace_back(val);
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

size_t data::get_extracted_data_size()
{
	return extracted_data->size();
}
uint8_t data::get_label()
{
	return label;
}
uint8_t data::get_enumerated_label()
{
	return enum_label;
}

std::vector<uint8_t>* data::get_extracted_data()
{
	return extracted_data;
}


//Deep
std::vector<int>* data::get_class_vector()
{
	return class_vector;
}
std::vector<float>* data::get_float_extracted_data()
{
	return extracted_float_data;
}

void data::set_class_vector(int count)
{
	class_vector = new std::vector<int>();
	for (int i = 0; i < count; i++)
	{
		if (i == label)
			class_vector->emplace_back(1);
		else
			class_vector->emplace_back(0);
	}
}
void data::set_float_extracted_data(std::vector<float>* vect)
{
	extracted_float_data = vect;
}
void data::append_to_float_extracted_data(float val)
{
	extracted_float_data->emplace_back(val);
}

void data::print_normalized_vector()
{
	printf("[ ");
	for (auto val : *extracted_float_data)
	{
		printf("%.2f ", val);
	}
	printf("]\n");

}
void data::print_vector()
{
	printf("[ ");
	for (uint8_t val : *extracted_data)
	{
		printf("%u ", val);
	}
	printf("]\n");
}
//Learn