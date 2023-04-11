#ifndef __KNN_H
#define __KNN_H
#include "../../DATA/include/common.hpp"
#include "../../DATA/include/data_handler.hpp"


class KNN : public common_data
{
private:
	int k;
	std::vector<data*>* neighbors;


public:
	KNN(int);
	KNN();
	~KNN();

	uint8_t predict();
	void find_knearest(data* query_point);
	double calculate_distance(data* query_point, data* input);

	double validate_performance();
	double test_performance();

	void set_k(int val);

	void run_knn(data_handler* dh);

};

#endif // !__KNN_H
