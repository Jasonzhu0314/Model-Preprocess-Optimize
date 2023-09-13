#include <iostream>
#include <pre_opencv.h>

void letterbox(const cv::Mat& image, cv::Mat& outImage,
                      const cv::Size& newShape,
                      const cv::Scalar& color
                      )
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);

    float ratio[2] {r, r};
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        // cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    // copy make Border
    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f)); // 0.5 防止越界
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void resize_op(const cv::Mat& image, cv::Mat& outImage, const cv::Size& newShape) {
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);

    float ratio[2] {r, r};
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]), 0, 0, cv::INTER_LINEAR);
    // cv::resize(image, outImage, cv::Size(), r, r, cv::INPAINT_TELEA);

}

void normalize_op(cv::Mat img, float* out){
	// img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				// 通道在前, BGR -> RGB
				// float pix = img.ptr<uchar>(i)[j * 3 + c];
				out[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}


void copymakeborder_op(
    const cv::Mat& image, cv::Mat& outImage
) {
    cv::Size newShape = cv::Size(outImage.cols, outImage.rows);
    cv::Size newUnpad = cv::Size(image.cols, image.rows);
    cv::Scalar color = cv::Scalar(114, 114, 114);
    
    auto dw = (float)(newShape.width - newUnpad.width);
    auto dh = (float)(newShape.height - newUnpad.height);

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f)); // 0.5 防止越界
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(image, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

