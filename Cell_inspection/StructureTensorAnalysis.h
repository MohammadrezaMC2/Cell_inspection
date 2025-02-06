#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "spline.h"
class StructureTensorAnalysis
{
private:
    bool checkExistence(const std::string& filename)
    {
        std::ifstream f;
        f.open(filename);

        return f.is_open();
    }
public:
    enum class GRADIENT_METHOD {
        CUBIC_SPLINE,
        FINITE_DIFFERENCE,
        FOURIER,
        RIESZ,
        GAUSSIAN,
        HESSIAN
    };

    cv::Mat read_image(const std::string& Path);

    std::tuple<cv::Mat, cv::Mat> computeGradients(const cv::Mat& grayImage,
        GRADIENT_METHOD gradientMethod,
        int windowSize = 2);

    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeStructuralTensor(const cv::Mat& gradX,
        const cv::Mat& gradY,
        int windowSize);

    cv::Mat computeEnery(const cv::Mat& Ixx, const cv::Mat& Iyy);

    cv::Mat computeOrientation(const cv::Mat& Ixx, 
        const cv::Mat& Iyy, 
        const cv::Mat Ixy);

    cv::Mat computeCoherency(const cv::Mat& Ixx,
        const cv::Mat& Iyy,
        const cv::Mat Ixy);

    cv::Mat computeColorSurvay(const cv::Mat& Image,
        int windowSize = 2);

};

