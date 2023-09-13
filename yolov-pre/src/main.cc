#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include "pre_opencv.h"
#include "pre_cuda.h"
#include "utils.h"


void cpu_letterbox_normalize(const cv::Mat& img, float* out) {
    cv::Mat opencv_img;
    double cpu_start = cpuSecond();
    letterbox(img, opencv_img);

    normalize_op(opencv_img, out);

    double cpu_end = cpuSecond();
    printf("cpu opencv letterbox time: %.2fms \n", (cpu_end - cpu_start) * 1000);

    std::string opencv_save_path = "../imgs/opencv_image.jpg";
    cv::imwrite(opencv_save_path, opencv_img);
}

void convertChwToHwc(cv::Mat& in, float* out, int height, int width) {
    std::vector<cv::Mat> channelsArray;
    cv::split(in, channelsArray);
    int chanelLength = height * width;
    for (int c = 0; c < 3; ++c) {
        memcpy(out, channelsArray[c].data, height * width * sizeof(float));
        out += height * width;
    }
}

void cv_resize(const cv::Mat& img, cv::Mat& resize_img, cv::Mat& res, std::vector<float>& chw) {
    int width = 640;
    int height = 640;
    int channel = 3;

    cv::Mat float_res;
    double cpu_start = cpuSecond();
    resize_op(img, resize_img, cv::Size(640, 640));
    copymakeborder_op(resize_img, res);
    // res.convertTo(float_res, CV_32FC3, 1.0 / 255.0);
    // float* float_res_ptr = float_res.ptr<float>();
    // printf("res.data[0]:%u\n", res.data[0]);
    // printf("float_res.data[0]:%f\n", float_res_ptr[0]);
    // res.convertTo(float_res, CV_32FC3);
    // cv::Scalar scale = cv::Scalar(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
    // cv::multiply(float_res, scale, float_res);
    // convertChwToHwc(float_res, chw.data(), height, width);
    double cpu_end = cpuSecond();

    printf("chw.data[0]:%f\n", chw[0]);
    printf("opencv letterbox time: %.2fms \n", (cpu_end - cpu_start) * 1000);
    printf("resize_img w: %d, h: %d\n", resize_img.cols, resize_img.rows);

    std::string opencv_save_path = "../imgs/opencv_image.jpg";
    cv::imwrite(opencv_save_path, res);
}


void custom_resize(const cv::Mat& img, cv::Mat& out, cv::Mat& res) {
    uint32_t out_width = out.cols;
    uint32_t out_height = out.rows;
    double cpu_start = cpuSecond();
    
    cudapre::cpu_resize(img.data, out.data, img.cols, img.rows, out_width, out_height);

    double cpu_end = cpuSecond();
    printf("custom resize time: %.2fms \n", (cpu_end - cpu_start) * 1000);

    cudapre::cpu_copymakeborder(out.data, res.data, out.cols, out.rows, res.cols, res.rows, 114);

    std::string opencv_save_path = "../imgs/custom_image.jpg";
    cv::imwrite(opencv_save_path, res);
}


void gpu_letterbox(cv::Mat &resize_out, cv::Mat& res, 
                    std::vector<float>& gpu_out,
                    std::vector<std::string>& image_paths,
                    int out_h, int out_w
                    ) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint32_t max_width = 1920;
    uint32_t max_height = 1080;

    uint32_t channel = 3;
    uint32_t nums = max_width * max_height * channel;

    uint8_t* data_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&data_dev, nums));

    // float* out_data;
    // CHECK_CUDA_ERROR(cudaMalloc(&out_data, out_width * out_height * 3 * sizeof(float)));

    uint8_t* resize_data;
    uint32_t out_nums = out_h * out_w * 3;
    CHECK_CUDA_ERROR(cudaMalloc(&resize_data, out_nums));

    uint8_t* res_data;
    uint32_t res_nums = res.rows * res.cols * 3;
    CHECK_CUDA_ERROR(cudaMalloc(&res_data, res_nums));

    for (const auto& img_path : image_paths) {
        cv::Mat img = cv::imread(img_path);
        cv::Size newShape(out_h, out_w);
        cv::Size shape = img.size();
        float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);

        float ratio[2] {r, r};
        int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};
        resize_out = cv::Mat(newUnpad[1], newUnpad[0], CV_8UC3);
        uint32_t out_width = resize_out.cols;
        uint32_t out_height = resize_out.rows;
        printf("gpu resize h:%d, gpu resize w: %d\n", out_height, out_width);
        
        double cpu_start = cpuSecond();
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(data_dev, img.data, nums, cudaMemcpyHostToDevice, stream));
            cudapre::gpu_resize(data_dev, resize_data, stream, img.cols, img.rows, out_width, out_height);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(resize_out.data, resize_data, out_width * out_height * 3, cudaMemcpyDeviceToHost, stream));

            CHECK_CUDA_ERROR(cudaMemcpyAsync(res_data, res.data, res_nums, cudaMemcpyHostToDevice, stream));
            cudapre::copymakeborder(resize_data, res_data, stream, resize_out.cols, resize_out.rows, res.cols, res.rows);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(res.data, res_data, res_nums, cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }
        double cpu_end = cpuSecond();
        printf("gpu resize time: %.2fms \n", (cpu_end - cpu_start) * 1000);
        break;
    }

    std::string opencv_save_path = "../imgs/gpu_image.jpg";
    CHECK_CUDA_ERROR(cudaFree(resize_data));
    CHECK_CUDA_ERROR(cudaFree(res_data));

    cudaStreamDestroy(stream);
    // cv::imwrite(opencv_save_path, out);
    cv::imwrite(opencv_save_path, res);
}

void gpu_letter_box_fusion(cv::Mat &out, 
                         cv::Mat& res, std::vector<std::string> image_paths) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint32_t max_width = 1920;
    uint32_t max_height = 1080;

    uint32_t channel = 3;
    uint32_t nums = max_width * max_height * channel;

    uint8_t* data_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&data_dev, nums));

    // float* out_data;
    // CHECK_CUDA_ERROR(cudaMalloc(&out_data, out_width * out_height * 3 * sizeof(float)));
    uint32_t out_width = out.cols;
    uint32_t out_height = out.rows;
    uint8_t* resize_data;
    uint32_t out_nums = out_width * out_height * 3;
    CHECK_CUDA_ERROR(cudaMalloc(&resize_data, out_nums));

    uint8_t* res_data;
    uint32_t res_nums = res.rows * res.cols * 3;
    CHECK_CUDA_ERROR(cudaMalloc(&res_data, res_nums));

    for (const auto& img_path : image_paths) {
        cv::Mat img = cv::imread(img_path);
        double cpu_start = cpuSecond();
        CHECK_CUDA_ERROR(cudaMemcpyAsync(data_dev, img.data, nums, cudaMemcpyHostToDevice, stream));
        cudapre::gpu_resize(data_dev, resize_data, stream, img.cols, img.rows, out_width, out_height);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(out.data, resize_data, out_nums, cudaMemcpyDeviceToHost, stream));
        
        cudaStreamSynchronize(stream);
        double cpu_end = cpuSecond();
        printf("gpu resize time: %.2fms \n", (cpu_end - cpu_start) * 1000);
        break;
    }

    std::string opencv_save_path = "../imgs/gpu_image.jpg";
    CHECK_CUDA_ERROR(cudaFree(resize_data));
    CHECK_CUDA_ERROR(cudaFree(res_data));

    cudaStreamDestroy(stream);
    // cv::imwrite(opencv_save_path, out);
    cv::imwrite(opencv_save_path, res);

}



int main() {
    int desiredDeviceId = 1;
    cudaSetDevice(desiredDeviceId);

    std::vector<std::string> image_paths;
    fs::path folderPath("../imgs/image2"); // 替换成您的文件夹路径
    getAllImageFilesInFolder(folderPath, image_paths);

    // 打印所有图像文件的完整路径

    std::string img_path = "../imgs/image1.jpg";
    cv::Mat img = cv::imread(img_path);
    uint32_t w = 640;
    uint32_t h = 640;

    cv::Mat opencv_resize_res;
    cv::Mat opencv_res = cv::Mat(640, 640, CV_8UC3);
    std::vector<float> chw(h * w * 3);
    for (int i = 0; i < 1000; i++) {
        for (const auto& img_path : image_paths) {
            std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            cv::Mat opencv_res = cv::Mat(640, 640, CV_8UC3);
            cv_resize(img, opencv_resize_res, opencv_res, chw);
            break;
        }
        break;
    }

    // printf("w: %d, h:%d\n", w, h);
    // cv::Mat custom_img(h, w, CV_8UC3);
    cv::Scalar color(114, 114, 114);
    // cv::Mat custom_res(640, 640, CV_8UC3, color);
    // custom_resize(img, custom_img, custom_res);
    // cmp(opencv_res, custom_res, w, h);

    cv::Mat gpu_res(640, 640, CV_8UC3, color);
    std::vector<float> gpu_out(h * w * 3);
    // for (int i = 0; i < 10; i++) {
    cv::Mat gpu_resize_res;
    gpu_letterbox(gpu_resize_res, gpu_res, gpu_out, image_paths, h, w);
        // cmp(opencv_img, gpu_img, w, h);
    printf("gpu_resize cols:%d, rows:%d\n", gpu_resize_res.cols, gpu_resize_res.rows);
    cmp_mat(opencv_resize_res, gpu_resize_res, gpu_resize_res.cols, gpu_resize_res.rows);

    return 0;
}