#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"

void letterbox(const cv::Mat& image, 
                cv::Mat& outImage,
                const cv::Size& newShape = cv::Size(640, 640),
                const cv::Scalar& color = cv::Scalar(114, 114, 114)
                );

void resize_op(const cv::Mat& image, cv::Mat& outImage, const cv::Size& newShape);

void normalize_op(cv::Mat img, float* out);

void copymakeborder_op(const cv::Mat& image, cv::Mat& outImage);