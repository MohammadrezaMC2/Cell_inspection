#include <cstdio>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "StructureTensorAnalysis.h"

int main(int, char**) {

	std::string Path = "C:\\Users\\Sepehr\\Desktop\\Maryam_malekpour.jpg";

	cv::Mat img = cv::imread(Path, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cerr << "Error: Could not open or find the image!" << std::endl;
	}

	std::shared_ptr<StructureTensorAnalysis> structureTensorAnalysis = std::make_shared<StructureTensorAnalysis>(img, StructureTensorAnalysis::GRADIENT_METHOD::FOURIER, 2);

	cv::Mat Energy = structureTensorAnalysis->getEnegry();


	cv::imshow("Energy", Energy);

	// Wait for a key press indefinitely or for a specified delay (0 means infinite wait)
	cv::waitKey(0);

	// Destroy the window (optional)
	cv::destroyAllWindows();

	

}