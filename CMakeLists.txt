cmake_minimum_required(VERSION 4.1)
project(CudaPractice3_2D_SobelEdgeDetection CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(CudaPractice3_2D_SobelEdgeDetection main.cu)

set_target_properties(CudaPractice3_2D_SobelEdgeDetection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
