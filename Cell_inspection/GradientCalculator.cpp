#include "GradientCalculator.h"

/**
 * @brief Computes image gradients using the finite difference method.
 *
 * This method uses simple convolution with [-1, 0, 1] kernels to approximate
 * the first-order derivative of the image in both x and y directions.
 *
 * @param grayImage Input grayscale image.
 * @param gradX Output gradient in the X direction.
 * @param gradY Output gradient in the Y direction.
 */
void GradientCalculator::computeFiniteDifferenceGradient(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY)
{
	cv::Mat kernelX = (cv::Mat_<float>(1, 3) << -1, 0, 1);
	cv::Mat kernelY = (cv::Mat_<float>(3, 1) << -1, 0, 1);

	cv::filter2D(grayImage, gradX, CV_32F, kernelX);
	cv::filter2D(grayImage, gradY, CV_32F, kernelY);
}

/**
 * @brief Computes image gradients using Gaussian smoothing followed by Sobel operators.
 *
 * This method applies a Gaussian blur to reduce noise before computing gradients using the Sobel operator.
 *
 * @param grayImage Input grayscale image.
 * @param gradX Output gradient in the X direction.
 * @param gradY Output gradient in the Y direction.
 */
void GradientCalculator::computeGaussianGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY)
{
	cv::Mat smoothed;
	int kernel_size = 5;
	double sigma = 2.0;
	cv::GaussianBlur(grayImage, smoothed, cv::Size(kernel_size, kernel_size), sigma, sigma);
	cv::Sobel(smoothed, gradX, CV_32F, 1, 0, 3);
	cv::Sobel(smoothed, gradY, CV_32F, 0, 1, 3);
}

/**
 * @brief Computes image gradients using cubic spline interpolation.
 *
 * This method fits cubic splines along rows and columns to estimate the gradients.
 *
 * @param grayImage Input grayscale image.
 * @param gradX Output gradient in the X direction.
 * @param gradY Output gradient in the Y direction.
 */
void GradientCalculator::cubicSplineInterpolation(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY)
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

/**
 * @brief Computes the frequency grid required for Fourier-based gradient computation.
 *
 * @param size Size of the grid (assumed to be the image dimension).
 * @return A frequency grid matrix.
 */
cv::Mat GradientCalculator::computeFrequencyGrid(int size)
{
	cv::Mat freq(size, 1, CV_32F);
	for (size_t i = 0; i < size; i++)
	{
		freq.at<float>(i, 0) = (i < size / 2) ? float(i) / size : float(i - size) / size;
	}
	return freq;
}

/**
 * @brief Computes image gradients using Fourier Transform techniques.
 *
 * This method computes gradients in the frequency domain using the Discrete Fourier Transform (DFT).
 *
 * @param grayImage Input grayscale image.
 * @param gradX Output gradient in the X direction.
 * @param gradY Output gradient in the Y direction.
 */
void GradientCalculator::computeFourierGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY)
{
	// Convert input image to floating point
	cv::Mat floatImage;
	grayImage.convertTo(floatImage, CV_32F);

	// Perform DFT
	cv::Mat planes[] = { floatImage, cv::Mat::zeros(floatImage.size(), CV_32F) };
	cv::Mat complexImage;
	cv::merge(planes, 2, complexImage);
	cv::dft(complexImage, complexImage, cv::DFT_COMPLEX_OUTPUT);

	// Compute frequency grid
	cv::Mat freqX = computeFrequencyGrid(grayImage.cols);
	cv::Mat freqY = computeFrequencyGrid(grayImage.rows);

	// Compute gradient in X direction
	cv::Mat planesX[2];
	cv::split(complexImage, planesX);
	cv::Mat realPartX = planesX[0];
	cv::Mat imagPartX = planesX[1];

	cv::Mat freqXexpanded;
	cv::repeat(freqX.t(), grayImage.rows, 1, freqXexpanded);

	cv::Mat realGradX = -2 * CV_PI * freqXexpanded.mul(imagPartX);
	cv::Mat imagGradX = 2 * CV_PI * freqXexpanded.mul(realPartX);

	cv::Mat complexGradX;
	cv::merge(std::vector<cv::Mat>{realGradX, imagGradX}, complexGradX);
	cv::idft(complexGradX, gradX, cv::DFT_REAL_OUTPUT);

	// Compute gradient in Y direction
	cv::Mat planesY[2];
	cv::split(complexImage, planesY);
	cv::Mat realPartY = planesY[0];
	cv::Mat imagPartY = planesY[1];

	cv::Mat freqYexpanded;
	cv::repeat(freqY, 1, grayImage.cols, freqYexpanded);

	cv::Mat realGradY = -2 * CV_PI * freqYexpanded.mul(imagPartY);
	cv::Mat imagGradY = 2 * CV_PI * freqYexpanded.mul(realPartY);

	cv::Mat complexGradY;
	cv::merge(std::vector<cv::Mat>{realGradY, imagGradY}, complexGradY);
	cv::idft(complexGradY, gradY, cv::DFT_REAL_OUTPUT);
}

/**
 * @brief Computes second-order derivatives using Hessian-based methods.
 *
 * This method applies Gaussian smoothing followed by second-order derivatives using Sobel operators.
 *
 * @param grayImage Input grayscale image.
 * @param windowSize Size of the Gaussian kernel for smoothing.
 * @param gradX Output second-order derivative in the X direction.
 * @param gradY Output second-order derivative in the Y direction.
 */
void GradientCalculator::computeSecondOrderDerivatives(const cv::Mat& grayImage, int windowSize, cv::Mat& gradX, cv::Mat& gradY)
{
	cv::Mat grayFloat;
	grayImage.convertTo(grayFloat, CV_32F);

	cv::Mat blurred;
	cv::GaussianBlur(grayFloat, blurred, cv::Size(0, 0), windowSize);

	cv::Sobel(blurred, gradX, CV_32F, 0, 2, 3);
	cv::Sobel(blurred, gradY, CV_32F, 2, 0, 3);
}

void GradientCalculator::computeRieszGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY)
{
	cv::Mat floatImage;
	grayImage.convertTo(floatImage, CV_32F);

	cv::Mat planes[] = { floatImage, cv::Mat::zeros(floatImage.size(), CV_32F) };
	cv::Mat complexImage;

	cv::merge(planes, 2, complexImage);

	cv::dft(complexImage, complexImage, cv::DFT_COMPLEX_OUTPUT);

	int rows = grayImage.rows;
	int cols = grayImage.cols;

	cv::Mat freqX(rows, cols, CV_32F);
	cv::Mat freqY(rows, cols, CV_32F);

	for (size_t i = 0; i < rows; i++)
	{
		float fy = (i < rows / 2) ? float(i) / rows : float(i - rows) / rows;
		for (size_t j = 0; j < cols; j++)
		{
			float fx = (j < cols / 2) ? float(j) / cols : float(j - cols) / cols;
			freqX.at<float>(i, j) = fx;
			freqY.at<float>(i, j) = fy;
		}
	}

	cv::Mat denominator;
	cv::sqrt(freqX.mul(freqX) + freqY.mul(freqY) + 1e-5, denominator);

	cv::Mat rieszXreal = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Mat rieszXimag = freqX / denominator;
	cv::Mat rieszYreal = cv::Mat::zeros(rows, cols, CV_32F);
	cv::Mat rieszYimag = freqY / denominator;

	cv::Mat planesDFT[2];
	cv::split(complexImage, planesDFT);
	cv::Mat realPart = planesDFT[0];
	cv::Mat imagPart = planesDFT[1];

	cv::Mat realGradX = -rieszXimag.mul(imagPart);
	cv::Mat imagGradX = rieszXimag.mul(realPart);
	cv::Mat realGradY = -rieszYimag.mul(imagPart);
	cv::Mat imagGradY = rieszYimag.mul(realPart);

	cv::Mat complexGradX, complexGradY;
	cv::merge(std::vector<cv::Mat>{realGradX, imagGradX}, complexGradX);
	cv::merge(std::vector<cv::Mat>{realGradY, imagGradY}, complexGradY);

	cv::idft(complexGradX, gradX, cv::DFT_REAL_OUTPUT);
	cv::idft(complexGradY, gradY, cv::DFT_REAL_OUTPUT);







}
