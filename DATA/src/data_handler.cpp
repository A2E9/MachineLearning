#define _ITERATOR_DEBUG_LEVEL 0
#include "../include/data_handler.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <memory>
#include "../../KMEANS/include/kmeans.hpp"


//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>


data_handler::data_handler()
{
	num_classes = 0;
	feature_vector_size = 0;
	data_array = new std::vector<data*> ;
	training_data = new std::vector<data*>;
	test_data = new std::vector<data*>;
	validation_data = new std::vector<data*>;
}
data_handler::~data_handler()
{
	// free dyn aloc
}



void data_handler::read_feature_vector(std::string& path)
{
	uint32_t header[4]; // magic|num images|Rowsize|Colsize

	FILE* file = fopen(path.c_str(), "rb");
	if (!file)
	{
		printf("Failed to open file %s\n", path.c_str());
		return;
	}

	// Read the header
	if (fread(header, sizeof(uint32_t), 4, file) != 4)
	{
		printf("Error reading file header.\n");
		exit(1);
	}

	// Convert the header to little-endian format inside the loop
	for (int i = 0; i < 4; i++)
	{
		header[i] = convert_to_little_endian(header[i]);
		std::cout << "Read: " << header[i] << '\n';
	}

	printf("Done getting Input File header.\n");

	size_t image_size = static_cast<size_t>(header[2]) * header[3];
	for (size_t i = 0; i < header[1]; i++) // images
	{
		// printf("\n count: %lu", i);
		auto d = new data();
		uint8_t element[1];
		for (int j = 0; j < image_size; j++) // image size "Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)"
		{
			if (fread(element, sizeof(element), 1, file)) // For each pixel, one byte of data is read and stored in element
			{
				d->append_to_feature_vector(element[0]); // appending element (pixel) which contains a byte(0-255) of data to feature_vector 
			}
			else
			{
				printf("Error reading file.\n");
				exit(1);
			}
		}
		data_array->emplace_back(d); // contains pointers to all of the data objects representing the images    784Size  -> 60 000 images
	}
	normalize();
	printf("Success read and stored %lu feature vectors. \n", (int)data_array->size());
	fclose(file);
}



void data_handler::read_feature_labels(std::string path)
{
	uint32_t header[2]; // magic|num images
	FILE* f = fopen(path.c_str(), "rb");
	if (f)
	{
		if (fread(header, sizeof(uint32_t), 2, f) != 2)
		{
			printf("Error reading file header.\n");
			exit(1);
		}

		for (int i = 0; i < 2; i++)
		{
			header[i] = convert_to_little_endian(header[i]);
		}

		printf("Done getting Label File header.\n");
		for (size_t i = 0; i < header[1]; i++) // images
		{
			uint8_t element;

			if (fread(&element, sizeof(element), 1, f)) //labels, one byte of data is read and stored
			{
				data_array->at(i)->set_label(element);
			}
			else
			{
				printf("Error reading file.\n");
				exit(1);
			}
		}
		printf("Success read and stored label.\n");
	}
	else
	{
		printf("File not found!");
	}
	f ? fclose(f) : NULL; // could be 0
}



void data_handler::split_data()
{
	srand((unsigned int)time(0));
	// vector of indices from 0 to data_array->size() - 1
	std::vector<int> indices(data_array->size());
	std::iota(indices.begin(), indices.end(), 0); // sequence of incrementing values 

	// fisher-yates algorithm
	for (size_t i = indices.size() - 1; i > 0; --i)
	{
		size_t j = rand() % (i + 1);
		std::swap(indices[i], indices[j]); // swaping last one i with random 
	}

	int train_size = static_cast<int>(data_array->size() * TRAIN_SET_PERCENT);
	int test_size = static_cast<int>(data_array->size() * TEST_SET_PERCENT);

	// spliting...
	for (size_t i = 0; i < indices.size(); ++i)
	{
		if (i < train_size) // 45000
		{
			training_data->emplace_back(data_array->at(indices[i]));   // 0.75
		}
		else if (i < train_size + test_size) // 12000
		{
			test_data->emplace_back(data_array->at(indices[i]));       // 0.20
		}
		else  // 3000
		{
			validation_data->emplace_back(data_array->at(indices[i])); // 0.05
		}
	}

	printf("Training Data Size: %lu.\n", (int)training_data->size());
	printf("Test Data Size: %lu.\n", (int)test_data->size());
	printf("Validation Data Size: %lu.\n", (int)validation_data->size());
}

void data_handler::count_classes()
{
	int count = 0;

	for (unsigned i = 0; i < data_array->size(); i++)
	{
		//returns an iterator to found key-value          position after the last element
		if (class_map.find(data_array->at(i)->get_label()) == class_map.end()) // if not found find returns == end()
		{
			class_map[data_array->at(i)->get_label()] = count; //mapping each unique label to an integer
			data_array->at(i)->set_enumerated_label(count);
			count++;
		}
	}
	num_classes = count;
	for (const auto& data : *data_array)
		data->set_class_vector(num_classes);

	printf("Successfully extracted %d unique Classes.\n", num_classes);
}

uint32_t data_handler::convert_to_little_endian(uint32_t num) const
{
	return ((num >> 24) & 0xff) |
		((num << 8) & 0xff0000) |
		((num >> 8) & 0xff00) |
		((num << 24) & 0xff000000);
}

std::vector<data*>* data_handler::get_training_data() const
{
	return training_data;
}
std::vector<data*>* data_handler::get_test_data() const
{
	return test_data;
}
std::vector<data*>* data_handler::get_validation_data() const
{
	return validation_data;
}
int data_handler::get_class_counts() const
{
	return num_classes;
}

//Deep
void data_handler::read_csv(std::string path, std::string delimiter)
{
	num_classes = 0;
	std::ifstream data_file;
	data_file.open(path.c_str());
	std::string line;
	if (!data_file)
	{
		std::cout << "File2 not found\n";
		std::cout << path.c_str();

		return;
	}

	while (std::getline(data_file, line))
	{
		if (line.length() == 0) continue;
		data* d = new data();
		d->set_normalized_feature_vector(new std::vector<float>());
		size_t position = 0;
		std::string token; // value in between delimiter
		while ((position = line.find(delimiter)) != std::string::npos)
		{
			token = line.substr(0, position);
			d->append_to_feature_vector((uint8_t)std::stof(token));
			line.erase(0, position + delimiter.length());
		}
		if (class_s_map.find(line) != class_s_map.end())
		{
			d->set_label(class_s_map[line]);
		}
		else
		{
			class_s_map[line] = num_classes;
			d->set_label(class_s_map[line]);
			num_classes++;
		}
		data_array->emplace_back(d);
	}
	for (data* d : *data_array)
		d->set_class_vector(num_classes);
	//normalize();
	feature_vector_size = data_array->at(0)->get_normalized_feature_vector()->size();
}


void data_handler::normalize()
{
	std::vector<float> min, max; //fill lists

	data* d = data_array->at(0);
	for (const auto& val : *d->get_feature_vector())
	{
		min.emplace_back(val);
		max.emplace_back(val);
	}


	for (size_t i = 0; i < data_array->size(); i++)
	{
		d = data_array->at(i);
		for (size_t j = 0; j < d->get_feature_vector_size(); j++)
		{
			float value = (float)d->get_feature_vector()->at(j);
			if (value < min.at(j)) min[j] = value;
			if (value > max.at(j)) min[j] = value;
		}
	}
	//normailize data_array

	for (size_t i = 0; i < data_array->size(); i++)
	{
		const auto &data_vector = data_array->at(i); //??????????????????????????????????????????????

		data_vector->set_normalized_feature_vector(new std::vector<float>());
		data_vector->set_class_vector(num_classes);	

		for (size_t j = 0; j < data_vector->get_feature_vector_size(); j++)
		{
			if (max[j] - min[j] == 0)
				data_vector->append_to_float_feature_vector(0.0f);
			else
				data_vector->append_to_float_feature_vector((float)(data_vector->get_feature_vector()->at(j) - min[j]) / (max[j] - min[j]));
		}
	}
}

//Learn


//void data_handler::display_image(int index)
//{
//    if (index >= data_array->size())
//    {
//        std::cout << "Invalid index." << std::endl;
//        return;
//    }
//
//    data* image_data = data_array->at(index);
//    int image_width = sqrt(image_data->get_feature_vector_size());
//    int image_height = image_data->get_feature_vector_size() / image_width;
//
//    cv::Mat image(image_height, image_width, CV_8UC1);
//
//    for (int i = 0; i < image_height; i++)
//    {
//        for (int j = 0; j < image_width; j++)
//        {
//            image.at<uchar>(i, j) = image_data->get_feature_vector()->at(i * image_width + j);
//        }
//    }
//
//    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Image", image);
//    cv::waitKey(0);
//}
