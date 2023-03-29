#define _ITERATOR_DEBUG_LEVEL 0
#include <iostream>
#include "data_handler.hpp"
#include <KNN.hpp>





int main()
{
	data_handler* dh = new data_handler();
	std::string train_images = "train-images-idx3-ubyte";
	std::string train_labels = "train-labels-idx1-ubyte";

	dh->read_feature_vector(train_images);
	dh->read_feature_labels(train_labels);

	dh->split_data();
	dh->count_classes();

	KNN* knearest = new KNN();
	
	knearest->set_training_data(dh->get_training_data());
	knearest->set_test_data(dh->get_test_data());
	knearest->set_validation_data(dh->get_validation_data());
	double performance = 0;
	double best_performance = 0;
	int best_k = 1;
	for (int i = 1; i < 4; i++)
	{
		if (i == 1)
		{
			knearest->set_k(i);
			performance = knearest->validate_performance();
			best_performance = performance;

		} else
		{
			knearest->set_k(i);
			performance = knearest->validate_performance();
			if (performance > best_performance)
			{
				best_performance = performance;
				best_k = i;
			}
		}
	}


	knearest->set_k(best_k);
	knearest->test_performance();


	getchar();

	delete dh;
}
