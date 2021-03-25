//
// Created by 张维 on 3/24/21.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <jni.h>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include "android/log.h"
#include "include/Filter.h"

using namespace std;


float sigmoid(float x)
{
    /*非线性sigmodi函数，范围0-1*/
    return (1 / (1 + exp(-x)));
}


void arraySigmoid(cv::Mat& inpImg, cv::Mat& outImg) {
    /*对矩阵进行sigmoid函数*/
    inpImg.copyTo(outImg);

    for (int y = 0; y < inpImg.rows; y++)
    {
        for (int x = 0; x < inpImg.cols; x++)
        {
            outImg.at<float>(y, x) = sigmoid(inpImg.at<float>(y, x));

        }
    }

}


void Filter::sauvolaWithSigmoid(cv::Mat& inpImg, cv::Mat& outImg, int window, float k)
{
    /*
     黑白功能
     c++与python版本一致
    */
    inpImg.copyTo(outImg);
    cv::Mat integral, integral2;

    int nOffSet = window / 2;
    int N = window*window;

    int height = outImg.rows;
    int width = outImg.cols;

    //给图像padding一圈，用于后续窗口滑动
    int padHeight = height + 2 * nOffSet;
    int padWidth = width + 2 * nOffSet;
    cv::Mat image_padded(padHeight, padWidth, CV_32F, cv::Scalar(0)); //Initialize the larger Mat to 0
    cv::copyMakeBorder(outImg, image_padded, nOffSet, nOffSet, nOffSet, nOffSet, cv::BORDER_REFLECT, cv::Scalar(0));
    //计算积分图像， opencv中interal和intergral2类型不一致，将类型转为一致，便于矩阵运算
    cv::integral(image_padded, integral, integral2);
    integral.convertTo(integral, CV_32F);
    integral2.convertTo(integral2, CV_32F);

    //通过积分图计算均值
    cv::Mat A0 = integral(cv::Rect(window, window, width,height));
    cv::Mat A1 = integral(cv::Rect(0, 0, width, height));
    cv::Mat A2 = integral(cv::Rect(window, 0, width, height));
    cv::Mat A3 = integral(cv::Rect(0, window, width, height));
    cv::Mat AMean = (A0 + A1 - A2 - A3)/ N;
    //通过积分图计算方差
    cv::Mat B0 = integral2(cv::Rect(window, window, width, height));
    cv::Mat B1 = integral2(cv::Rect(0, 0, width, height));
    cv::Mat B2 = integral2(cv::Rect(window, 0, width, height));
    cv::Mat B3 = integral2(cv::Rect(0, window, width, height));
    cv::Mat B = (B0 + B1 - B2 - B3)/ N;
    cv::Mat variance = B - AMean.mul(AMean);
    variance.setTo(0, variance < 0);  //将小于0的赋为0，便于后续开方
    cv::Mat v;
    sqrt(variance, v);

    //求局部阈值的公式
    cv::Mat One = cv::Mat::ones(cv::Size(width, height), CV_32F);
    cv::Mat thresh = AMean.mul(One + k*(v / 128 - One));
    outImg.convertTo(outImg, CV_32F);

    //阈值附近使用非线性变换，sigmoid函数（0-1）的映射
    outImg = (outImg - thresh) / 5;

    arraySigmoid(outImg, outImg);

    //映射到0-255区间
    cv::Mat constant1(height, width, CV_32F, cv::Scalar(0.1));
    outImg = (outImg *5 /4 - constant1) * 255;
    outImg.setTo(0, outImg < 0);
    outImg.setTo(255, outImg > 255);

    //返回输出
    outImg.convertTo(outImg, CV_8U);

}



float Filter::getScale(int height, int width, int max_size=1024 ) {
    /*
    得到要归一化的尺度,小图不做处理，大图缩放
    */
    float scale;
    int max_edge = max(height, width);
    if (max_edge < max_size) {
        scale = 1.0;
    }

    else if  (max_edge == height) {
        scale = float(max_size) / height;
    }
    else
    {
        scale = float(max_size) / width;
    }

    return scale;

}

void saturationEnhance(cv::Mat& inpImg, cv::Mat& outImg) {
    //空间转换
    cv::Mat hlsImg;
    inpImg.copyTo(outImg);
    outImg.convertTo(outImg, CV_32F);
    outImg = outImg / 255.0;
    cv::cvtColor(outImg, hlsImg, cv::COLOR_BGR2HLS);

    // 通道分离, 调高跑合度
    vector<cv::Mat> temp;
    cv::split(hlsImg, temp);
    cv::Mat S;
    S = 1.5* temp[2];

    S.setTo(1, S > 1);
    temp[2] = S;

    //合并通道，空间转换，类型转换
    cv::merge(temp, hlsImg);
    cv::cvtColor(hlsImg, outImg, cv::COLOR_HLS2BGR);
    outImg = outImg * 255;
    outImg.convertTo(outImg, CV_8U);
}

void Filter::coloring(cv::Mat& inpImg, cv::Mat& blackWhite, cv::Mat& outImg) {
    /*
    上色功能： 将黑白图像黑色区域用原图对应的颜色填充,并使色彩更鲜艳
    */

    inpImg.copyTo(outImg);
    outImg.setTo(cv::Vec3b(255,255,255), blackWhite > 150);
    saturationEnhance(outImg, outImg);

}

//filter::~filter() {
//}

void Filter::colorEnhance(cv::Mat& inpImg, cv::Mat& outImg) {
    /*
    增强功能：使用锐化+双边滤波
    */
    //inpImg.copyTo(outImg);
    cv::Mat temp;
    cv::Mat kern = (cv::Mat_<char>(3, 3) << 0, -1, 0,
            -1, 5, -1,
            0, -1, 0);
    filter2D(inpImg, temp, inpImg.depth(), kern);
//    outImg = temp.clone();
    cv::bilateralFilter(temp, outImg, 5, 150, 150);

}



float getLightValue(cv::Mat& inpImg) {
    /*
    对rgb通道的均值进行加权，根据范围活动要调节的亮度值
    */
    vector<cv::Mat> bgr;
    cv::split(inpImg, bgr);
    cv::Scalar tempValB = cv::mean(bgr[0]);
    cv::Scalar tempValG = cv::mean(bgr[1]);
    cv::Scalar tempValR = cv::mean(bgr[2]);

    float meanB = tempValB.val[0];
    float meanG = tempValG.val[0];
    float meanR = tempValR.val[0];

    float thresh = sqrt(0.241 * pow(meanR, 2) + 0.691 * pow(meanG, 2) + 0.068 * pow(meanB, 2));

    float lightValue = 0;
    if (thresh < 120) {
        lightValue = 10;
    }
    else if (thresh < 150) {
        lightValue = 8;
    }
    else if (thresh < 180) {
        lightValue = 6;
    }
    else if (thresh < 210) {
        lightValue = 4;
    }
    else if (thresh < 240) {
        lightValue = 2;
    }
    else if (thresh < 250) {
        lightValue = 0;
    }
    else {
        lightValue = -10;
    }

    return lightValue;

}


void Filter::brighten(cv:: Mat& inpImg, cv::Mat& outImg) {
    /*
    增亮功能：根据rgb通道均值进行加权计算一个值，根据值的范围确定要调节的亮度值
    c++与python版本一致
    */
    float lightValue = getLightValue(inpImg);

    inpImg.copyTo(outImg);
    //空间转换
    cv:: Mat temp;
    outImg.convertTo(outImg, CV_32F);
    outImg = outImg / 255.0;
    cv::cvtColor(outImg, temp, cv::COLOR_BGR2HLS);

    //通道分离，调节亮度通道
    vector<cv::Mat> hls;
    split(temp, hls);
    cv::Mat l;
    l = (lightValue + 20) / 20 * hls[1];

    l.setTo(1, l > 1);
    hls[1] = l;

    //合并通道，空间转换，类型转换
    cv::merge(hls, temp);
    cv::cvtColor(temp, outImg, cv::COLOR_HLS2BGR);
    outImg = outImg * 255;
    outImg.convertTo(outImg, CV_8U);

}


//int main() {
//    cv::Mat img_src = cv::imread("data/3.jpg");
//    cv::Mat img;
//    cout << img_src.channels() << endl;
//    if (img_src.channels() == 1) {
//        cv::cvtColor(img_src, img, cv::COLOR_GRAY2BGR);
//    }
//    else if (img_src.channels() == 4) {
//        cv::cvtColor(img_src, img, cv::COLOR_BGRA2BGR);
//    }
//    else {
//        img_src.copyTo(img);
//    }
//
//    //DWORD star_time = GetTickCount();
//
//    cv::Mat gray, blackWhite,colorImg, enhanceImg, brightImg;
//
//    int height = img.rows;
//    int width = img.cols;
//    float fScale = getScale(height, width, 2000);  //1024值可以设定，表示长边尺寸不得超过1024，按照长边归一化
//
//    int window_size = (int)(width*fScale * 20 / 256);
//    window_size = window_size % 2 ? window_size : window_size + 1;
//
//    if (fScale == 1.0) {
//
//        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
//        SauvolaWithSigmoid(gray, blackWhite, window_size, 0.1); //黑白功能
//        coloring(img, blackWhite, colorImg);//上色功能
//        colorEnhance(colorImg, enhanceImg);//上色锐化功能
//        brighten(img, brightImg);//增亮功能
//
//
//    }
//    else
//    {
//        //将图像缩小
//        cv::Mat imgReisize, blackWhiteResize, colorImgResize, enhanceImgResize, brightImgResize;
//        cv::resize(img, imgReisize, cv::Size(), fScale, fScale, cv::INTER_CUBIC);
//
//        //对缩小后的图像进行处理
//        cv::cvtColor(imgReisize, gray, cv::COLOR_BGR2GRAY);
//        SauvolaWithSigmoid(gray, blackWhiteResize, window_size, 0.1);  //黑白功能
//        coloring(imgReisize, blackWhiteResize, colorImgResize); //上色功能
//        colorEnhance(colorImgResize, enhanceImgResize); //上色锐化功能
//        brighten(imgReisize, brightImgResize); //增亮功能
//
//        //对处理后的图片放大到原有尺寸
//        cv::resize(blackWhiteResize, blackWhite, cv::Size(width, height), cv::INTER_CUBIC);
//        cv::resize(colorImgResize, colorImg, cv::Size(width, height), cv::INTER_CUBIC);
//        cv::resize(enhanceImgResize, enhanceImg, cv::Size(width, height), cv::INTER_CUBIC);
//        cv::resize(brightImgResize, brightImg, cv::Size(width, height), cv::INTER_CUBIC);
//
//    }
//    // 	DWORD end_time1 = GetTickCount();
//    // 	cout << "time：" << (end_time1 - star_time) << "ms." << endl;
//
//    cv::imwrite("data/test_result0.png", blackWhite);
//    cv::imwrite("data/test_result1.png", colorImg);
//    cv::imwrite("data/test_result2.png", enhanceImg);
//    cv::imwrite("data/test_result3.png", brightImg);
//
//
//    return 0;
//}


