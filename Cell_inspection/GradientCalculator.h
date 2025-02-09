#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random> 
#include <boost/random.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

/**
 * @class GradientCalculator
 * @brief Provides various methods for computing image gradients using different techniques.
 *
 * This class implements multiple approaches to compute image gradients, including:
 * - Finite Difference
 * - Gaussian-based smoothing
 * - Cubic Spline Interpolation
 * - Fourier Transform
 * - Riesz Transform
 * - Hessian-based Second Order Derivatives
 *
 * The methods take a grayscale image as input and output two gradient matrices (X and Y directions).
 */
class GradientCalculator
{
public:

    /**
     * @brief Computes image gradients using the finite difference method.
     * @param grayImage Input grayscale image.
     * @param gradX Output gradient in the X direction.
     * @param gradY Output gradient in the Y direction.
     */
    static void computeFiniteDifferenceGradient(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY);

    /**
     * @brief Computes image gradients using Gaussian smoothing followed by Sobel operators.
     * @param grayImage Input grayscale image.
     * @param gradX Output gradient in the X direction.
     * @param gradY Output gradient in the Y direction.
     */
    static void computeGaussianGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY);

    /**
     * @brief Computes image gradients using cubic spline interpolation.
     * @param grayImage Input grayscale image.
     * @param gradX Output gradient in the X direction.
     * @param gradY Output gradient in the Y direction.
     */
    static void cubicSplineInterpolation(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY);

    /**
     * @brief Computes image gradients using Fourier Transform techniques.
     * @param grayImage Input grayscale image.
     * @param gradX Output gradient in the X direction.
     * @param gradY Output gradient in the Y direction.
     */
    static void computeFourierGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY);

    /**
     * @brief Computes image gradients using the Riesz Transform.
     * @param grayImage Input grayscale image.
     * @param gradX Output gradient in the X direction.
     * @param gradY Output gradient in the Y direction.
     */
    static void computeRieszGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY);

    /**
     * @brief Computes second-order derivatives using Hessian-based methods.
     * @param grayImage Input grayscale image.
     * @param windowSize Size of the local window used for computation.
     * @param gradX Output second-order derivative in the X direction.
     * @param gradY Output second-order derivative in the Y direction.
     */
    static void computeSecondOrderDerivatives(const cv::Mat& grayImage, int windowSize, cv::Mat& gradX, cv::Mat& gradY);

private:

    /**
     * @brief Computes a frequency grid matrix used for Fourier-based gradient computation.
     * @param size Size of the grid (assumed to be the image dimension).
     * @return A frequency grid matrix.
     */
    static cv::Mat computeFrequencyGrid(int size);
};
