#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random> 
#include <boost/random.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
class MathOperation
{
public:

	static void cubicSplineInterpolation(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY );

};

