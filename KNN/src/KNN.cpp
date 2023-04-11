#define _ITERATOR_DEBUG_LEVEL 0
#include <cmath>
#include<limits>
#include <map>
#include "stdint.h"
#include <queue>

#include "KNN.hpp"
#include "DATA/include/data_handler.hpp"

//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <iostream>


KNN::KNN(int val)
{
	k = val;
}
KNN::KNN()
{}
KNN::~KNN() { /*clear mem*/ }


// O(N^2) if K ~ N
// if k = 2 then O(~N)
//O(NlogN)

void KNN::set_k(int val)
{
	k = val;
}


/// <summary>
/// Predicts the class label for the given query point
/// based on the k-nearest neighbors
/// </summary>
/// <returns>
/// most frequent label
/// </returns>
uint8_t KNN::predict()
{
	std::map<uint8_t, int> class_freq;
	for (size_t i = 0; i < neighbors->size(); i++)
	{
		if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
		{
			class_freq[neighbors->at(i)->get_label()] = 1;
		}
		else
		{
			class_freq[neighbors->at(i)->get_label()]++;
		}
	}

	uint8_t best = 0;
	int max = 0;


	for (auto& kv : class_freq)
	{
		if (kv.second > max)// frequency 
		{
			best = kv.first; // label
			max = kv.second;

		}
	}

	delete neighbors;
	return best;
}
/// <summary>
///  It maintains a priority queue to store the distances of
///  training points from the query point and gets the top k neighbors
/// </summary>
/// <param name="query_point"></param>
void KNN::find_knearest(data* query_point)
{
	neighbors = new std::vector<data*>; // new vector of data
	std::priority_queue<std::pair<double, data*>> nearest_queue; // queue for distance and tranining instance

	for (data* train_point : *training_data)
	{
		double distance = calculate_distance(query_point, train_point);
		nearest_queue.push(std::make_pair(-distance, train_point)); // negative distance between query_point and train_point | archieve ascending
	}

	for (size_t i = 0; i < k; i++)
	{
		neighbors->push_back(nearest_queue.top().second);
		nearest_queue.pop();
	}
}


double KNN::calculate_distance(data* from_point, data* to_point)
{
	double distance = 0.0;
	const size_t from_size = from_point->get_feature_vector_size(); // 784

	if (from_size != to_point->get_feature_vector_size()) // valid size
	{
		printf("\nError Vector Size Mismatch.\n");
		exit(1);
	}

#ifndef EUCLID
	// distance = sqrt( pow(a1-b1) + pow(a2-b2) )
	for (size_t i = 0; i < from_size; i++)
	{
		// (a1-b1)² + (a2-b2)² + ...
		distance += pow(from_point->get_feature_vector()->at(i) - to_point->get_feature_vector()->at(i), 2);
	}
	distance = sqrt(distance);

#elif defined MANHATTAN
	//MANHATTAN IMPL
#endif // !EUCLID
	return distance;
}



/// <summary>
/// Calculate the distance between all training images and a single validation image, 
/// then select the k training points with the smallest distances. Determine the most 
/// frequent label among these neighbors and check if it matches the label of the validation image.
/// </summary>
/// <returns></returns>
double KNN::validate_performance()
{
	double crr_performance = 0;
	int checked = 0;
	int count = 0;
	int data_index = 0;
	for (auto& validation_point : *validation_data) // Validation Vector [0.05 - 3000]
	{
		find_knearest(validation_point); // creating neighbors to the foto
		uint8_t prediction = predict(); // most frequent label

		auto check = prediction == validation_point->get_label();
		printf("\n%d  <---------[%d -> %d]--------->  %s\n", checked++, prediction, validation_point->get_label(), check ? "same" : "[(FALSE)]");

		//cv::Mat image = validation_point->get_image(); 
		//cv::imshow("Image", image);
		//cv::waitKey(500);

		if (prediction == validation_point->get_label())
		{
			count++;
		}
		data_index++;
		printf("Performance = %.3f %%\n", ((double)count * 100.0) / ((double)data_index));
	}
	crr_performance = (double)count * 100.0 / (double)validation_data->size();
	printf("Validation Performance for K = %d: %.3f %%\n", k, crr_performance);

	return crr_performance;

}
double KNN::test_performance()
{
	double crr_performance = 0;
	int count = 0;
	for (auto& query_point : *test_data)
	{
		find_knearest(query_point);
		int prediction = predict();
		if (prediction == query_point->get_label())
		{
			count++;
		}
	}

	crr_performance = (double)count * 100.0 / (double)test_data->size();
	printf("Tested Performance %.3f %%\n", crr_performance);
	return crr_performance;
}

void KNN::run_knn(data_handler* dh)
{
	KNN* knearest = new KNN();

	knearest->set_training_data(dh->get_training_data());
	knearest->set_test_data(dh->get_test_data());
	knearest->set_validation_data(dh->get_validation_data());

	double performance = 0;
	double best_performance = 0;
	int best_k = 1;

	//Perofrming...
	for (int i = 1; i < 4; i++)
	{
		if (i == 1)
		{
			printf("First run\n");
			knearest->set_k(i);
			performance = knearest->validate_performance();
			best_performance = performance;

		}
		else
		{
			printf("Next run\n");
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
}
