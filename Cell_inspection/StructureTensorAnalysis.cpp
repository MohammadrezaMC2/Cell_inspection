#include "StructureTensorAnalysis.h"


cv::Mat StructureTensorAnalysis::read_image(const std::string& Path)
{

    cv::Mat img = cv::imread(Path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
    }

    return img;
}

std::tuple<cv::Mat, cv::Mat> StructureTensorAnalysis::computeGradients(const cv::Mat& grayImage,
    StructureTensorAnalysis::GRADIENT_METHOD gradientMethod,
    int windowSize)
    
{
    cv::Mat gradX, gradY;
    if (gradientMethod == StructureTensorAnalysis::GRADIENT_METHOD::FINITE_DIFFERENCE)
    {
        cv::Mat kernelX = (cv::Mat_<float>(1, 3) << -1, 0, 1);
        cv::Mat kernelY = (cv::Mat_<float>(3, 1) << -1, 0, 1);

        
        cv::filter2D(grayImage, gradX, CV_32F, kernelX);
        cv::filter2D(grayImage, gradX, CV_32F, kernelY);

        return std::make_tuple(gradX, gradY);
    }

    else if (gradientMethod == StructureTensorAnalysis::GRADIENT_METHOD::GAUSSIAN)
    {
        cv::Mat smoothed;
        int kernel_size = 5;
        double sigma = 2.0;
        cv::GaussianBlur(grayImage, smoothed, cv::Size(kernel_size, kernel_size), sigma, sigma);
        cv::Sobel(smoothed, gradX, CV_32F, 1, 0, 3);
        cv::Sobel(smoothed, gradY, CV_32F, 0, 1, 3);
        return std::make_tuple(gradX, gradY);
    }

    else if (gradientMethod == StructureTensorAnalysis::GRADIENT_METHOD::CUBIC_SPLINE)
    {
        cv::Mat x = cv::Mat(1, grayImage.cols, CV_32S);
        cv::Mat y = cv::Mat(grayImage.rows, 1, CV_32S);

        for (size_t i = 0; i < grayImage.cols; i++)
            x.at<int>(0, i) = i;

        for (size_t i = 0; i < grayImage.rows; i++)
            y.at<int>(i, 0) = i;

        gradX = cv::Mat::zeros(grayImage.size(), CV_32F);
        gradY = cv::Mat::zeros(grayImage.size(), CV_32F);

        std::vector<float> cs;
        for (size_t i = 0; i < grayImage.rows; i++)
        {
            tk::spline S(x, gradX.row(i));
        }
        
    }

    else if (gradientMethod == StructureTensorAnalysis::GRADIENT_METHOD::FOURIER)
    {
        
    }
}