#define _ITERATOR_DEBUG_LEVEL 0
#include "../include/common.hpp"


void common_data::set_training_data(std::vector<data*>* vect)
{
	training_data = vect;
}
void common_data::set_test_data(std::vector<data*>* vect)
{
	test_data = vect;
}
void common_data::set_validation_data(std::vector<data*>* vect)
{
	validation_data = vect;
}




