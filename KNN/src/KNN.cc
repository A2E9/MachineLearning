#define _ITERATOR_DEBUG_LEVEL 0
#include <cmath>
#include<limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"
#include "KNN.hpp"
#include <queue>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


KNN::KNN(int val)
{
	k = val;
}
KNN::KNN()
{}
KNN::~KNN() {}


// O(N^2) if K ~ N
// if k = 2 then O(~N)
//O(NlogN)


/// <summary>
///  It maintains a priority queue to store the distances of
///  training points from the query point and gets the top k neighbors
/// </summary>
/// <param name="query_point"></param>
void KNN::find_knearest(data* query_point)
{
	neighbors = new std::vector<data*>;
	std::priority_queue<std::pair<double, data*>> nearest_queue;

	for (data* train_point : *training_data)
	{
		double distance = calculate_distance(query_point, train_point);
		nearest_queue.push(std::make_pair(-distance, train_point));
	}

	for (size_t i = 0; i < k; i++)
	{
		neighbors->push_back(nearest_queue.top().second);
		nearest_queue.pop();
	}
}
//void KNN::find_knearest(data* query_point)
//{
//	neighbors = new std::vector<data*>;
//	double min = std::numeric_limits<double>::max();
//	double previous_min = min;
//	int index = 0;
//
//	for (size_t i = 0; i < k; i++)
//	{
//		if (i == 0)
//		{
//			for (size_t j = 0; j < training_data->size(); j++)
//			{
//				double distance = calculate_distance(query_point, training_data->at(j));
//				training_data->at(j)->set_distance(distance);
//				if (distance < min)
//				{
//					min = distance;
//					index = j;
//				}
//			}
//			neighbors->push_back(training_data->at(index));
//			previous_min = min;
//			min = std::numeric_limits<double>::max();
//		} else
//		{
//			for (size_t j = 0; j < training_data->size(); j++)
//			{
//				double distance = training_data->at(j)->get_distance();
//				if (distance > previous_min && distance < min)
//				{
//					min = distance;
//					index = j;
//				}
//			}
//			neighbors->push_back(training_data->at(index));
//			previous_min = min;
//			min = std::numeric_limits<double>::max();
//		}
//	}
//}
void KNN::set_training_data(std::vector<data*>* vect)
{
	training_data = vect;
}
void KNN::set_test_data(std::vector<data*>* vect)
{
	test_data = vect;
}
void KNN::set_validation_data(std::vector<data*>* vect)
{
	validation_data = vect;
}
void KNN::set_k(int val)
{
	k = val;
}


/// <summary>
/// Predicts the class label for the given query point
/// based on the k-nearest neighbors
/// </summary>
/// <returns></returns>
int KNN::predict()
{
	std::map<uint8_t, int> class_freq;
	for (size_t i = 0; i < neighbors->size(); i++)
	{
		if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
		{
			class_freq[neighbors->at(i)->get_label()] = 1;
		} else
		{
			class_freq[neighbors->at(i)->get_label()]++;
		}
	}

	int best = 0;
	int max = 0;

	for (auto& kv : class_freq)
	{
		if (kv.second > max)
		{
			max = kv.second;
			best = kv.first;
		}
	}

	delete neighbors;
	return best;
}


double KNN::calculate_distance(data* query_point, data* input)
{
	double distance = 0.0;
	if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
	{
		printf("\nError Vector Size Mismatch.\n");
		exit(1);
	}
#ifndef EUCLID
	for (size_t i = 0; i < query_point->get_feature_vector_size(); i++)
	{
		distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
		//printf("Distance %f \n", distance);
	}
	distance = sqrt(distance);

#elif defined MANHATTAN
	//MANHATTAN IMPL
#endif // !EUCLID
	return distance;
}
double KNN::validate_performance()
{
	double crr_performance = 0;
	int checked = 0;
	int count = 0;
	int data_index = 0;
	for (auto& query_point : *validation_data)
	{
		find_knearest(query_point);
		int prediction = predict();

		auto check = prediction == query_point->get_label();
		printf("%d -> %d  -------------------->  %s\n", prediction, query_point->get_label(), check ? "same":"[(FALSE)]");
		printf("Checked: %d\n", checked++);

		//cv::Mat image = query_point->get_image(); // Assuming `get_image()` returns a cv::Mat object
		//cv::imshow("Image", image);
		//cv::waitKey(500);

		if (prediction == query_point->get_label())
		{
			count++;
		}
		data_index++;
		printf("Current Performance = %.3f %%\n", ((double)count * 100.0) / ((double)data_index));
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