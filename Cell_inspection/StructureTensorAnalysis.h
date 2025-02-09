#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "spline.h"
#include "GradientCalculator.h"

// Class for performing structure tensor analysis on images
class StructureTensorAnalysis
{

public:
    // Enumeration of different gradient computation methods
    enum class GRADIENT_METHOD {
        CUBIC_SPLINE,
        FINITE_DIFFERENCE,
        FOURIER,
        RIESZ,
        GAUSSIAN,
        HESSIAN
    };

    // Constructors
    StructureTensorAnalysis() {};
    StructureTensorAnalysis(cv::Mat Image, GRADIENT_METHOD GradientMethod, int WindowSize = 2);

    // Function to read an image from a given file path
    cv::Mat read_image(const std::string& Path);

    // Set the gradient computation method and window size
    void setGradientandWindowSize(GRADIENT_METHOD GradientMethod, int WindowSize = 2);

    // Getter functions for gradient, energy, orientation, and coherency matrices
    cv::Mat getGradX() const { return gradX; }
    cv::Mat getGradY() const { return gradY; }
    cv::Mat getEnegry() const { return Energy; }
    cv::Mat getOrientation() const { return Orientation; } 
    cv::Mat getCoherency() const { return Coherency; }

private:
    cv::Mat image; // Input image

    // Matrices for gradient and structure tensor components
    cv::Mat gradX;
    cv::Mat gradY;
    cv::Mat Energy;
    cv::Mat Orientation;
    cv::Mat Coherency;

    cv::Mat Ixx;
    cv::Mat Iyy;
    cv::Mat Ixy;

    GRADIENT_METHOD gradientMethod; // Selected gradient computation method
    int windowSize; // Window size for tensor computation

    // Helper function to check if a file exists
    bool checkExistence(const std::string& filename)
    {
        std::ifstream f;
        f.open(filename);
        return f.is_open();
    }

    // Compute image gradients based on the selected method
    void computeGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY,
        GRADIENT_METHOD gradientMethod, int windowSize = 2);

    // Compute structure tensor components from gradients
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeStructuralTensor(const cv::Mat& gradX,
        const cv::Mat& gradY, int windowSize);

    // Compute energy matrix from structure tensor components
    cv::Mat computeEnergy(const cv::Mat& Ixx, const cv::Mat& Iyy);

    // Compute orientation matrix from structure tensor components
    cv::Mat computeOrientation(const cv::Mat& Ixx,
        const cv::Mat& Iyy, const cv::Mat Ixy);

    // Compute coherency matrix from structure tensor components
    cv::Mat computeCoherency(const cv::Mat& Ixx,
        const cv::Mat& Iyy, const cv::Mat Ixy);

    // Compute color survey visualization of the image
    cv::Mat computeColorSurvay(const cv::Mat& Image, int windowSize = 2);

    // Alternative function to compute structure tensor components
    void computeStructuralTensor(const cv::Mat& gradX, const cv::Mat& gradY, int windowSize,
        cv::Mat& Ixx, cv::Mat& Iyy, cv::Mat& Ixy);

    // Compute all relevant parameters for analysis
    void computeParameters();
};
