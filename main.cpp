#include <iostream>
#include <string>
#include <string.h>
#include <opencv2/opencv.hpp>

#include "yolo_infer.h"

using namespace std;


#ifdef __cplusplus
extern "C" {
#endif


YoloDetecter* YoloDetecter_new(char* trtFile, int gpuId)
{
    // std::string trtFile = "./resources/model.plan";
    // int gpuId = 0;
    return new YoloDetecter(std::string(trtFile), gpuId);
}

float* inference_one(YoloDetecter* instance, const uchar* srcImgData, const int srcH, const int srcW)
{
    cv::Mat srcImg(srcH, srcW, CV_8UC3);
    memcpy(srcImg.data, srcImgData, srcH * srcW * 3 * sizeof(uchar));
    return instance->inference(srcImg);
}

void destroy(YoloDetecter* instance)
{
    delete instance;
}


#ifdef __cplusplus
}
#endif
