#pragma once
#include <time.h>
#include <sys/time.h>
#include <boost/filesystem.hpp>
#include <algorithm>

namespace fs = boost::filesystem;

double cpuSecond();

void getAllImageFilesInFolder(const fs::path& folderPath, std::vector<std::string>& imagePaths);

bool isImageFile(const std::string& filename);

int cmp_mat(cv::Mat& cpu_img, cv::Mat &gpu_img, int width, int height);

float cmp_vector(std::vector<float> &a, std::vector<float> &b, int width, int height, float epsilon=1e-5);
