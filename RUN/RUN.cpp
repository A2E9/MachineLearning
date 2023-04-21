// RUN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define _ITERATOR_DEBUG_LEVEL 0


#include "../DATA/include/data_handler.hpp"
#include "../KNN/include/KNN.hpp"
#include "../KMEANS/include/kmeans.hpp"
#include <../NEURAL_NET/include/network.hpp>


int main()
{
	//Preparing data...
	data_handler* mnist = new data_handler();
	std::string train_images = "../mnist_data/train-images-idx3-ubyte";
	std::string train_labels = "../mnist_data/train-labels-idx1-ubyte";

	mnist->extract_feature_data(train_images);
	mnist->extract_feature_labels(train_labels);

	mnist->split_data();
	mnist->count_classes();

	while (true)
	{
		int choice;
		std::cout << "\nChoose algorithm to run: " << std::endl;
		std::cout << "1. K-means clustering" << std::endl;
		std::cout << "2. K-nearest neighbors classification" << std::endl;
		std::cout << "3. Deep Learning Neural Network" << std::endl;
		std::cout << "4. Quit" << std::endl;
		std::cout << "Your choice >> ";
		std::cin >> choice;
		std::cout << "\n";

		if (choice == 1)
		{
			kmeans* km = new kmeans();
			km->run_kmean(mnist);
		}
		else if (choice == 2)
		{
			KNN* knn = new KNN();
			knn->run_knn(mnist);

		}
		else if (choice == 3)
		{
			network* net = new network();

			std::cout << "[1] Iris.data or [2] MNIST >> ";
			std::cin >> choice;
			std::cout << "\n";

			if (choice == 1) net->run_neural();

			else if (choice == 2) net->run_neural(mnist);
		}
		else
		{
			std::cout << "Exiting program." << std::endl;
			delete mnist;
			return 0;
		}
	}
}
