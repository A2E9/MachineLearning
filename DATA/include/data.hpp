#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"


/*#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>*/
/*
Container to hold each line item
*/

class data
{
private:
	uint8_t label;
	int enum_label; //A -> 1, B -> 2
	double distance;
	std::vector<uint8_t>* extracted_data; // byte(0-255) of data size: 784

	//Deep
	std::vector<int>* class_vector;
	std::vector<float>* extracted_float_data;
	//Learn

public:
	data(/* args */);
	~data();

	void set_extracted_data(std::vector<uint8_t>*);
	void append_to_extracted_data(uint8_t); // data_handler -> extract_feature_data
	void set_label(uint8_t); // data_handler -> extract_feature_labels
	void set_enumerated_label(int); // data_handler -> count_classes
	void set_distance(double val);

	double get_distance();

	uint8_t get_label();
	uint8_t get_enumerated_label();
	size_t get_extracted_data_size();
	std::vector<uint8_t>* get_extracted_data();




	// *** Deep *** 
	void append_to_float_extracted_data(float);

	void set_class_vector(int);
	void set_float_extracted_data(std::vector<float>*);

	void print_vector();
	void print_normalized_vector();

	std::vector<int>* get_class_vector();
	std::vector<float>* get_float_extracted_data();
	// *** Learn *** 









	/*cv::Mat get_image()
	{
		int image_width = (int)sqrt(get_extracted_data_size());
		int image_height = (int)get_extracted_data_size() / image_width;

		cv::Mat image(image_height, image_width, CV_8UC1);

		for (int i = 0; i < image_height; i++)
		{
			for (int j = 0; j < image_width; j++)
			{
				image.at<uchar>(i, j) = get_extracted_data()->at(i * image_width + j);
			}
		}

		return image;
	}*/
};



#endif