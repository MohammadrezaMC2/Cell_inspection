#include "StructureTensorAnalysis.h"

// Constructor: Initializes the object with an image, gradient method, and window size, then computes parameters
StructureTensorAnalysis::StructureTensorAnalysis(cv::Mat Image, GRADIENT_METHOD GradientMethod, int WindowSize) :
	image{ Image }, gradientMethod{ GradientMethod }, windowSize{ WindowSize }
{
	computeParameters();
}

// Reads an image from the given path and converts it to grayscale
cv::Mat StructureTensorAnalysis::read_image(const std::string& Path)
{
	cv::Mat img = cv::imread(Path, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Error: Could not open or find the image!" << std::endl;
	}
	return img;
}

// Computes image gradients based on the selected gradient method
void StructureTensorAnalysis::computeGradients(const cv::Mat& grayImage, cv::Mat& gradX, cv::Mat& gradY,
	StructureTensorAnalysis::GRADIENT_METHOD gradientMethod,
	int windowSize)
{
	gradX = cv::Mat::zeros(grayImage.size(), CV_32F);
	gradY = cv::Mat::zeros(grayImage.size(), CV_32F);
	switch (gradientMethod)
	{
	case StructureTensorAnalysis::GRADIENT_METHOD::FINITE_DIFFERENCE:
		GradientCalculator::computeFiniteDifferenceGradient(grayImage, gradX, gradY);
		break;

	case StructureTensorAnalysis::GRADIENT_METHOD::GAUSSIAN:
		GradientCalculator::computeGaussianGradients(grayImage, gradX, gradY);
		break;

	case StructureTensorAnalysis::GRADIENT_METHOD::CUBIC_SPLINE:
		GradientCalculator::cubicSplineInterpolation(grayImage, gradX, gradY);
		break;

	case StructureTensorAnalysis::GRADIENT_METHOD::FOURIER:
		GradientCalculator::computeFourierGradients(grayImage, gradX, gradY);
		break;

	case StructureTensorAnalysis::GRADIENT_METHOD::RIESZ:
		GradientCalculator::computeRieszGradients(grayImage, gradX, gradY);
		break;

	case StructureTensorAnalysis::GRADIENT_METHOD::HESSIAN:
		GradientCalculator::computeSecondOrderDerivatives(grayImage, windowSize, gradX, gradY);
		break;

	default:
		throw std::runtime_error("Gradient method is not implemented.");
	}

	// Error handling in case the method is not implemented

}

// Computes the structural tensor components from the gradients
void StructureTensorAnalysis::computeStructuralTensor(const cv::Mat& gradX, const cv::Mat& gradY, int windowSize,
	cv::Mat& Ixx, cv::Mat& Iyy, cv::Mat& Ixy)
{
	cv::Mat gradXSquare, gradYSquare, gradXYSquare;

	cv::multiply(gradX, gradX, gradXSquare);
	cv::multiply(gradY, gradY, gradYSquare);
	cv::multiply(gradX, gradY, gradXYSquare);


	cv::GaussianBlur(gradXSquare, Ixx, cv::Size(0, 0), windowSize, windowSize);
	cv::GaussianBlur(gradXSquare, Iyy, cv::Size(0, 0), windowSize, windowSize);
	cv::GaussianBlur(gradXYSquare, Ixy, cv::Size(0, 0), windowSize, windowSize);
}

// Computes the energy from structure tensor components
cv::Mat StructureTensorAnalysis::computeEnergy(const cv::Mat& Ixx, const cv::Mat& Iyy)
{
	return Ixx + Iyy;
}

// Computes the orientation of the structure tensor
cv::Mat StructureTensorAnalysis::computeOrientation(const cv::Mat& Ixx, const cv::Mat& Iyy, const cv::Mat Ixy)
{
	cv::Mat theta;
	cv::phase(2 * Ixy, Ixx - Iyy, theta, false);
	return 0.5 * theta;
}

// Computes the coherency measure of the structure tensor
cv::Mat StructureTensorAnalysis::computeCoherency(const cv::Mat& Ixx,
	const cv::Mat& Iyy,
	const cv::Mat Ixy)
{
	cv::Mat coherency;
	cv::Mat lambda1, lambda2;
	cv::Mat square1, square2;
	cv::Mat root1, root2;

	cv::multiply(Ixx - Iyy, Ixx - Iyy, square1);
	cv::multiply(Ixy, Ixy, square2);

	cv::sqrt(square1 + 4 * square2, root1);
	cv::sqrt(square1 + 4 * square2, root2);

	lambda1 = (Ixx + Iyy + root1);
	lambda2 = (Ixx + Iyy - root2);

	cv::divide(lambda1 - lambda2, lambda1 + lambda2 + 1e-5, coherency);

	return coherency;
}

// Computes all necessary parameters for the structure tensor analysis
void StructureTensorAnalysis::computeParameters()
{
	computeGradients(image, gradX, gradY, gradientMethod, windowSize);
	computeStructuralTensor(gradX, gradY, windowSize, Ixx, Iyy, Ixy);
	Energy = computeEnergy(Ixx, Iyy);
	Orientation = computeOrientation(Ixx, Iyy, Ixy);
	Coherency = computeCoherency(Ixx, Iyy, Ixy);
}

// Sets the gradient computation method and window size, then recalculates parameters
void StructureTensorAnalysis::setGradientandWindowSize(GRADIENT_METHOD GradientMethod, int WindowSize)
{
	gradientMethod = GradientMethod;
	windowSize = WindowSize;
	computeParameters();
}
