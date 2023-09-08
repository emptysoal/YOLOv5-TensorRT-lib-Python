cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)

# ============= cuda ============
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# =========== tensorrt ==========
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# ============ opencv ===========
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# ====== yolo infer shared lib ======
cuda_add_library(yolo_infer SHARED 
    ${PROJECT_SOURCE_DIR}/src/preprocess.cu 
    ${PROJECT_SOURCE_DIR}/src/yololayer.cu 
    ${PROJECT_SOURCE_DIR}/src/yolo_infer.cpp
    ${PROJECT_SOURCE_DIR}/main.cpp
)
target_link_libraries(yolo_infer nvinfer cudart ${OpenCV_LIBS})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
