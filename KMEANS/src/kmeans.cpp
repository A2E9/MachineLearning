#include "../include/kmeans.hpp"

#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>

kmeans::kmeans()
{
	num_clusters = NULL;
	clusters = NULL;
	used_indexes = NULL;
}
kmeans::kmeans(int k)
{
	num_clusters = k;
	clusters = new std::vector<cluster_t*>;
	used_indexes = new std::unordered_set<int>;
}
kmeans::~kmeans()
{
	if (clusters != nullptr)
	{
		for (auto& cluster : *clusters)
		{
			delete cluster;
		}
		delete clusters;
	}
	if (used_indexes != nullptr)
	{
		delete used_indexes;
	}
}

/// <summary>
/// adds shuffled (cluster)data to clusters
/// </summary>
void kmeans::init_clusters()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	std::vector<size_t> indexes(training_data->size());
	std::iota(indexes.begin(), indexes.end(), 0);
	std::shuffle(indexes.begin(), indexes.end(), gen);

	for (size_t i = 0; i < num_clusters; i++)
	{
		clusters->emplace_back(new cluster(training_data->at(indexes[i]))); //placing shuffeled indexes or random cluster inst
	}
}


/// <summary>
/// calcs the distance between training point and all centroids
/// then with smalles distance centroid for one training point 
/// adding at that cluster the training data point (x45000)
/// which calcs new average centroid
/// </summary>
void kmeans::train()
{
	//Creating shuffeled indexes with size of training data
	std::random_device rd;
	std::mt19937 gen(rd());

	std::vector<size_t> indexes(training_data->size());
	std::iota(indexes.begin(), indexes.end(), 0);
	std::shuffle(indexes.begin(), indexes.end(), gen);


	for (size_t i = 0; i < indexes.size(); i++)
	{
		size_t best_cluster = 0;
		float min_dist = std::numeric_limits<float>::max();
		for (size_t j = 0; j < clusters->size(); j++)
		{
			float current_dist = euclidean_distance(clusters->at(j)->centroid, training_data->at(indexes[i])->get_feature_vector());

			if (current_dist < min_dist) // selecting smallest distance
			{
				min_dist = current_dist;
				best_cluster = j;
			}
		}
		clusters->at(best_cluster)->add_to_cluster(training_data->at(indexes[i])); // calcs new centroid value
	}
}


/// <summary>
/// calcing the distance between centroid and validationData
/// then with minimal distance of all clusters check whether 
/// the prediction is right or wrong by comparing the most_freq
/// </summary>
/// <returns>
/// the accuracy
/// </returns>
float kmeans::validate()
{
	float num_correct = 0.0;
	for (auto& query_point : *validation_data)
	{
		float min_dist = std::numeric_limits<float>::max();
		size_t best_cluster = 0;
		for (size_t i = 0; i < clusters->size(); i++)
		{
			float crr_dist = euclidean_distance(clusters->at(i)->centroid, query_point->get_feature_vector());
			if (crr_dist < min_dist)//minimal distance = best cluster
			{
				min_dist = crr_dist;
				best_cluster = i;
			}
		}
		if (clusters->at(best_cluster)->most_freq_class == query_point->get_label()) num_correct++; // check the predicted class
	}
	return (100.0f * (num_correct / (float)validation_data->size()));
}


float kmeans::test()
{

	float num_correct = 0.0;
	for (auto& query_point : *test_data)
	{
		float min_dist = std::numeric_limits<float>::max();
		size_t best_cluster = 0;
		for (size_t i = 0; i < clusters->size(); i++)
		{
			float crr_dist = euclidean_distance(clusters->at(i)->centroid, query_point->get_feature_vector());
			if (crr_dist < min_dist)
			{
				min_dist = crr_dist;
				best_cluster = i;
			}
		}
		if (clusters->at(best_cluster)->most_freq_class == query_point->get_label()) num_correct++;
	}
	return 100.0f * (num_correct / (float)test_data->size());
}


float kmeans::euclidean_distance(const std::vector<float>* centroid, const std::vector<uint8_t>* feat_vect)
{
	float dist = 0.0;
	for (size_t i = 0; i < centroid->size(); i++)
	{
		dist += (float)pow(centroid->at(i) - feat_vect->at(i), 2); // (a1-b1)² + (a2-b2)² + ...
	}
	return sqrt(dist);
}


//why
void kmeans::init_clusters_foreach_class()
{
	std::unordered_set<int> classes_used;
	for (size_t i = 0; i < training_data->size(); i++)
	{
		if (classes_used.find(training_data->at(i)->get_label()) == classes_used.end())
		{
			clusters->emplace_back(new cluster_t(training_data->at(i)));
			classes_used.insert(training_data->at(i)->get_label());
			used_indexes->insert((int)i);
		}
	}
}


//Runnn
void kmeans::run_kmean(data_handler* dh)
{
	int best_k = 1;
	float performance = 0.0;
	float best_performance = 0.0;
	for (int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.1; k++) // 10 < 4500
	{
		kmeans* km = new kmeans(k); // k = 10->4500

		km->set_validation_data(dh->get_validation_data());
		km->set_training_data(dh->get_training_data());
		km->set_test_data(dh->get_test_data());

		km->init_clusters();
		km->train(); // creates clusters

		performance = km->validate(); // checks distance with clusters
		printf("Current Performance @ K = %d: %.2f\n", k, performance);

		if (performance > best_performance) // selectng best k-num
		{
			best_performance = performance;
			best_k = k;
		}
	}

	//Test best_k
	kmeans* km = new kmeans(best_k);

	km->set_training_data(dh->get_training_data());
	km->set_test_data(dh->get_test_data());
	km->set_validation_data(dh->get_validation_data());

	km->init_clusters();
	performance = km->test();

	printf("Test Performance @ K = %d: %.2f\n", best_k, performance);
}