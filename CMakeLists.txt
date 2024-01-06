cmake_minimum_required(VERSION 3.12)
project(OGL_PointCloud)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_NVCC_FLAGS
        ${CMAKE_NVCC_FLAGS};
        -Xcompiler
        -fPIC
        -use_fast_math
        -std=c++11
        -gencode arch=compute_86,code=sm_86)
include_directories(${CUDA_INCLUDE_DIRS})

# Eigen
include_directories(/usr/local/include/eigen3)

# glm
find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OpenMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# glad
add_library(glad 3rd/glad/src/glad.c)
include_directories(3rd/glad/include)

# glfw
add_subdirectory(3rd/glfw)
include_directories(3rd/glfw/include)

# imgui
add_library(imgui 3rd/imgui/imgui.cpp
        3rd/imgui/imgui_demo.cpp
        3rd/imgui/imgui_draw.cpp
        3rd/imgui/imgui_tables.cpp
        3rd/imgui/imgui_widgets.cpp
        3rd/imgui/backends/imgui_impl_glfw.cpp
        3rd/imgui/backends/imgui_impl_opengl3.cpp)
include_directories(3rd/imgui 3rd/imgui/backends)

set(EXTRA_LIBS
        ${CUDA_LIBRARIES}
        glm::glm
        ${OPENGL_LIBRARIES}
        ${OpenCV_LIBS}
        glad
        glfw
        imgui)

include_directories(include)

add_executable(OGL_PointCloud main.cpp include/stb_image.h include/stb_image.cpp include/popl.hpp src/PointCloud.cu)
target_link_libraries(OGL_PointCloud PUBLIC ${EXTRA_LIBS})
target_compile_options(OGL_PointCloud PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_NVCC_FLAGS}>)
