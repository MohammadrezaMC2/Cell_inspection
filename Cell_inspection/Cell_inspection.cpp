
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
using namespace cv;

int main() {
    std::string path = "F:/Maryam_malekpour.jpg";

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;  // Exit the program to prevent a crash
    }
    namedWindow("my first program", WINDOW_AUTOSIZE);
    cv::imshow("my first program", img);
    cv::waitKey(0);
    return 0;
}