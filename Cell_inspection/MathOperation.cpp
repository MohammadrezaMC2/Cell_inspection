#include "MathOperation.h"

void MathOperation::cubicSplineInterpolation(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY)
{
	int rows = grayImage.rows;
	int cols = grayImage.cols;

	gradX = cv::Mat::zeros(rows, cols, CV_64F);
	gradY = cv::Mat::zeros(rows, cols, CV_64F);

	for (size_t i = 0; i < rows; i++)
	{
		std::vector<double> x(cols), values(cols);
		for (size_t j = 0; j < cols; j++)
		{
			x[j] = j;
			values[j] = static_cast<double>(grayImage.at<uchar>(i, j));
		}

		auto spline = boost::math::interpolators::cardinal_cubic_b_spline<double>(values.begin(), values.end(), 0.0, 1.0);


		for (size_t j = 0; j < cols; j++)
		{
			gradX.at<double>(i, j) = spline.prime(j);
		}
	}

	for (size_t j = 0; j < cols; j++)
	{
		std::vector<double> y(rows), values(rows);
		for (size_t i = 0; i < rows; i++)
		{
			y[i] = i;
			values[i] = static_cast<double>(grayImage.at<uchar>(i, j));
		}

		auto spline = boost::math::interpolators::cardinal_cubic_b_spline<double>(values.begin(), values.end(), 0.0, 1.0);
		for (size_t i = 0; i < rows; i++)
		{
			gradY.at<double>(i, j) = spline.prime(i);
		}
	}
}

