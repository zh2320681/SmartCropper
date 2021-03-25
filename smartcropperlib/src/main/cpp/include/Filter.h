//
// Created by 张维 on 3/24/21.
//

#ifndef SMARTCROPPER_FILTER_H
#define SMARTCROPPER_FILTER_H

#include <opencv2/opencv.hpp>

class Filter {
    public:
//        virtual ~filter();
        void colorEnhance(cv::Mat& inpImg, cv::Mat& outImg);

        void sauvolaWithSigmoid(cv::Mat& inpImg, cv::Mat& outImg, int window, float k);

        float getScale(int height, int width, int max_size );

        void coloring(cv::Mat& inpImg, cv::Mat& blackWhite, cv::Mat& outImg);

        void brighten(cv:: Mat& inpImg, cv::Mat& outImg);
};


#endif //SMARTCROPPER_FILTER_H
