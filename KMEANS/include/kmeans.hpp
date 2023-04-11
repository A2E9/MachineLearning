#pragma once

#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "../../DATA/include/common.hpp"
#include "../../DATA/include/data_handler.hpp"

typedef struct cluster
{
	std::vector<double>* centroid;
	std::vector<data*>* cluster_points;
	std::map<int, int> class_count;

	int most_freq_class;

	cluster(data* initial_point)
	{
		centroid = new std::vector<double>;
		cluster_points = new std::vector<data*>;

		for (auto& value : *(initial_point->get_feature_vector()))
		{
			centroid->push_back(value);// pushing feature vector elements(bytes) of init_point
		}
		cluster_points->push_back(initial_point); // pushing point into cluster
		class_count[initial_point->get_label()] = 1; // creating element at label = 1
		most_freq_class = initial_point->get_label(); // setting initial point label
	}

	void add_to_cluster(data* point)
	{
		size_t pcluster_size = cluster_points->size();
		cluster_points->push_back(point); // pushing point into cluster

		for (size_t i = 0; i < centroid->size() - 1; i++)
		{
			double value = centroid->at(i);				 // [=] i feature of the data point
			value *= pcluster_size;						 // [*] prev cluster size (weight the contribution of the previous centroid)
			value += point->get_feature_vector()->at(i); // [+] new point byte at i (new feature value adding to weighted sum of feature values)
			value /= (double)cluster_points->size();	 // [/] new cluster size (to calculate the new centroid value)
			centroid->at(i) = value;					 // [=] to i centroid calcd_value
		}

		if (class_count.find(point->get_label()) == class_count.end())
		{
			class_count[point->get_label()] = 1;
		}
		else
		{
			class_count[point->get_label()]++;
		}
		set_most_frequent_class();
	}


	void set_most_frequent_class()
	{
		int best_class = 0;
		int freq = 0;

		for (auto& kv : class_count)
		{
			if (kv.second > freq)
			{
				freq = kv.second;
				best_class = kv.first;
			}
		}
		most_freq_class = best_class;
	}

} cluster_t;

class kmeans : public common_data
{
	int num_clusters;
	std::vector<cluster_t*>* clusters;
	std::unordered_set<int>* used_indexes;
public:
	kmeans();
	kmeans(int k);

	void init_clusters();
	void init_clusters_foreach_class();
	void train();
	
	double euclidean_distance(std::vector<double>*, data*);
	double validate();
	double test();

	void run_kmean(data_handler*);

};