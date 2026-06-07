#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <iostream>

int main() {
    int count = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA Devices: " << count << std::endl;
    if (count > 0) {
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    }
    return 0;
}
