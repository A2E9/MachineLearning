#pragma once

#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "../../DATA/include/common.hpp"
#include "../../DATA/include/data_handler.hpp"

//struct has public memb opposite to class
typedef struct cluster
{
	int most_freq_class;

	std::vector<float>* centroid;
	std::vector<data*>* cluster_points;
	std::map<int, int> class_count;



	/// <summary>
	/// assigning the extracted_data from initial to the centroid
	/// adding to cluster_points
	/// initializes a class count map with the initial points label 
	/// </summary>
	/// <param name="initial_point">
	/// used to create the new cluster
	/// </param>
	cluster(data* initial_point)
	{
		centroid = new std::vector<float>;
		cluster_points = new std::vector<data*>;

		const auto& init_extracted_data = *(initial_point->get_extracted_data());

		centroid->assign(init_extracted_data.begin(), init_extracted_data.end()); // uint8_t to float
		cluster_points->emplace_back(initial_point);

		class_count[initial_point->get_label()] = 1;
		most_freq_class = initial_point->get_label();
	}


	/// <summary>
	/// calc new centroid 
	/// </summary>
	/// <param name="point"></param>
	void add_to_cluster(data* point)
	{
		size_t pcluster_size = cluster_points->size();
		cluster_points->emplace_back(point); // pushing point into cluster

		for (size_t i = 0; i < centroid->size() - 1; i++)
		{
			float value = centroid->at(i);				 // [=] i feature of the data point
			value *= pcluster_size;						 // [*] prev cluster size (weight the contribution of the previous centroid)
			value += point->get_extracted_data()->at(i); // [+] new point byte at i (new feature value adding to weighted sum of extracted_data)
			value /= (float)cluster_points->size();	     // [/] new cluster size (to calculate the new centroid value)
			centroid->at(i) = value;					 // [=] to i centroid calcd_value
		}


		//frequency of labels in class_counts
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
private:
	int num_clusters;
	std::vector<cluster_t*>* clusters;
	std::unordered_set<int>* used_indexes;

public:
	kmeans();
	kmeans(int k);
	~kmeans();

	void init_clusters();
	void init_clusters_foreach_class();
	void train();

	float validate();
	float test();
	float euclidean_distance(const std::vector<float>* centroid, const std::vector<uint8_t>* feat_vect);

	void run_kmean(data_handler*);
};