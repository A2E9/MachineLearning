#ifndef __KNN_H
#define __KNN_H
#include <vector>
#include "data.hpp"


class KNN
{
private:
	int k;
	std::vector<data*>* neighbors;
	std::vector<data*>* training_data;
	std::vector<data*>* test_data;
	std::vector<data*>* validation_data;

public:
	KNN(int);
	KNN();
	~KNN();

	void find_knearest(data* query_point);
	void set_training_data(std::vector<data*>* vect);
	void set_test_data(std::vector<data*>* vect);
	void set_validation_data(std::vector<data*>* vect);
	void set_k(int val);

	int predict();
	double calculate_distance(data* query_point, data* input);
	double validate_performance();
	double test_performance();

};

#endif // !__KNN_H
