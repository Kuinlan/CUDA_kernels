cmake_minimum_required(VERSION 3.22)

project(copy_if LANGUAGES CUDA CXX)

find_package(CUDA 12.4 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

add_executable(copy_if copyif.cu)

