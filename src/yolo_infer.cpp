#include "yolo_infer.h"
#include "common.h"
#include "preprocess.h"

using namespace nvinfer1;


const int     INPUT_H = Yolo::INPUT_H;
const int     INPUT_W = Yolo::INPUT_W;
const int     OUTPUT_SIZE = 1 + Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float);  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1;
const float   NMS_THRESH = 0.4;
const float   CONF_THRESH = 0.5;


YoloDetecter::YoloDetecter(const std::string trtFile, const int gpuId): trtFile_(trtFile)
{
    gLogger = new YoloLogger(ILogger::Severity::kERROR);
    cudaSetDevice(gpuId);

    // load engine
    runtime = createInferRuntime(*gLogger);
    engine = getEngine();
    context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {1, 3, INPUT_H, INPUT_W}});

    // bytes of input and output
    vTensorSize.resize(2, 0);
    vTensorSize[0] = 3 * INPUT_H * INPUT_W * sizeof(float);
    vTensorSize[1] = OUTPUT_SIZE * sizeof(float);

    // prepare input data and output data ---------------------------
    inputData = new float[3 * INPUT_H * INPUT_W];
    outputData = new float[OUTPUT_SIZE];

    // prepare input and output space on device
    vBufferD.resize(2, nullptr);
    for (int i = 0; i < 2; i++)
    {
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }
}


YoloDetecter::~YoloDetecter()
{
    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    // context->destroy();
    // engine->destroy();
    // runtime->destroy();
    delete context;
    delete engine;
    delete runtime;

    delete [] inputData;
    delete [] outputData;

    delete gLogger;
}


ICudaEngine* YoloDetecter::getEngine()
{
    ICudaEngine* engine = nullptr;
    if (access(trtFile_.c_str(), F_OK) != 0)
    {
        std::cout << "ERROR: TensorRT engine plan file not found." << std::endl;
        return engine;
    }

    std::ifstream engineFile(trtFile_, std::ios::binary);
    long int fsize = 0;

    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineString(fsize);
    engineFile.read(engineString.data(), fsize);
    if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return nullptr; }
    std::cout << "Succeeded getting serialized engine!" << std::endl;

    // IRuntime* runtime {createInferRuntime(gLogger)};
    engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
    if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return nullptr; }
    std::cout << "Succeeded loading engine!" << std::endl;

    return engine;
}


void YoloDetecter::inference()
{
    CHECK(cudaMemcpy(vBufferD[0], (void *)inputData, vTensorSize[0], cudaMemcpyHostToDevice));

    context->executeV2(vBufferD.data());

    CHECK(cudaMemcpy((void *)outputData, vBufferD[1], vTensorSize[1], cudaMemcpyDeviceToHost));
}

/*
std::vector<DetectResult> YoloDetecter::inference(cv::Mat& img)
{
    preprocess(img, inputData, INPUT_H, INPUT_W);  // put image data on inputData

    inference();

    std::vector<Yolo::Detection> res;
    nms(res, outputData, CONF_THRESH, NMS_THRESH);

    std::vector<DetectResult> final_res;
    for (size_t j = 0; j < res.size(); j++)
    {
        cv::Rect r = get_rect(img, res[j].bbox);
        DetectResult single_res {r, res[j].conf, (int)res[j].class_id};
        final_res.push_back(single_res);
    }

    return final_res;
}
*/

float* YoloDetecter::inference(cv::Mat& img)
{
    preprocess(img, inputData, INPUT_H, INPUT_W);  // put image data on inputData

    inference();

    return outputData;
}