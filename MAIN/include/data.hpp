#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
/*
Container to hold each line item
*/

class data
{
private:
    std::vector<uint8_t> * feature_vector; 
    uint8_t label;
    int enum_label; //A -> 1, B -> 2
    double distance;
public:
    data(/* args */);
    ~data();

    void set_feature_vector(std::vector<uint8_t>*);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enumerated_label(int);
    void set_distance(double val);

    double get_distance();
    size_t get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumerated_label();

    std::vector<uint8_t> * get_feature_vector();


    cv::Mat get_image()
    {
        int image_width = (int)sqrt(get_feature_vector_size());
        int image_height = (int)get_feature_vector_size() / image_width;

        cv::Mat image(image_height, image_width, CV_8UC1);

        for (int i = 0; i < image_height; i++)
        {
            for (int j = 0; j < image_width; j++)
            {
                image.at<uchar>(i, j) = get_feature_vector()->at(i * image_width + j);
            }
        }

        return image;
    }
};



#endif