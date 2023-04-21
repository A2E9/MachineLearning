#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include <cstdint>
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <iostream>

/*
Implement Logic needed to read-in, split the data
count the number of unique classes
convert to little-endian
pass around the trained, death, validation data
...
*/

class data_handler
{
private:
    std::vector<data *> *data_array; // all data pre-split
    std::vector<data *> *training_data;
    std::vector<data *> *test_data;
    std::vector<data *> *validation_data;

    int num_classes;
    size_t extracted_data_size;
    std::map<uint8_t, int> class_map;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_PERCENT = 0.05;
    /*const double TRAIN_SET_PERCENT = 0.1;
    const double TEST_SET_PERCENT = 0.075;
    const double VALIDATION_PERCENT = 0.005;*/
    //Deep
    std::map<std::string, int> class_s_map;


    //Learn


public:
    data_handler(/* args */);
    ~data_handler();

    void extract_feature_data(std::string& path);
    void extract_feature_labels(std::string path);
    void split_data();
    void count_classes();

    //void display_image(int index);
    int get_class_counts() const;
    uint32_t convert_to_little_endian(uint32_t num) const;

    std::vector<data *> *get_training_data() const;
    std::vector<data *> *get_test_data() const;
    std::vector<data *> *get_validation_data() const;



    //Deep
    void read_csv(std::string path, std::string delimiter);
    void normalize();

    //Learn

};

#endif