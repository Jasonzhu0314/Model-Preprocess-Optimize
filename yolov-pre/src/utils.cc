#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils.h"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return( (double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}


// 函数用于筛选图像文件（这里只考虑常见的图像文件扩展名）
bool isImageFile(const std::string& filename) {
    std::vector<std::string> imageExtensions = { ".jpg", ".jpeg", ".png", ".gif", ".bmp" };
    for (const auto& ext : imageExtensions) {
        if (filename.size() >= ext.size() &&
            std::equal(ext.rbegin(), ext.rend(), filename.rbegin())) {
            return true;
        }
    }
    return false;
}

// 递归遍历文件夹，获取图像文件的完整路径
void getAllImageFilesInFolder(const fs::path& folderPath, std::vector<std::string>& imagePaths) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (fs::is_directory(entry)) {
            // 如果是子文件夹，递归处理
            getAllImageFilesInFolder(entry.path(), imagePaths);
        } else if (fs::is_regular_file(entry) && isImageFile(entry.path().extension().string())) {
            // 如果是图像文件，将其完整路径添加到vector中
            imagePaths.push_back(entry.path().string());
        }
    }
}

void cmp_mat(cv::Mat& cpu_img, cv::Mat &gpu_img, int width, int height) {
    int max_inter = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b cpu_pixel = cpu_img.at<cv::Vec3b>(i, j);
            cv::Vec3b gpu_pixel = gpu_img.at<cv::Vec3b>(i, j);
            for (int c = 0; c < 3; c++) {
                if (cpu_pixel[c] != gpu_pixel[c]) {
                    // printf("[%d][%d][%d]:cpu[%d] != gpu[%d]\n", i, j, c, cpu_pixel[c], gpu_pixel[c]);
                    max_inter = std::max(int(abs(cpu_pixel[c] - gpu_pixel[c])), max_inter);
                }
            }
        }
    }

    printf("max inter:%d\n", max_inter);
}


void cmp_vector(std::vector<float> &a, std::vector<float> &b, int width, int height) {
    float max_inter = 0.;
    
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float* a_ptr = a.data() + c * height * width + i * width;
                float* b_ptr = b.data() + c * height * width + i * width;
                if (a_ptr[j] - b_ptr[j] > 1e-5) {
                    // printf("[%d][%d][%d]:cpu[%f] != gpu[%f]\n", i, j, c, a_ptr[j], b_ptr[j]);
                    max_inter = std::max(abs(a_ptr[j] - b_ptr[j]), max_inter);
                }
            }
        }
    }
    printf("max inter:%f\n", max_inter);
}
