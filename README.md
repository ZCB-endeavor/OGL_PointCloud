# OpenGL Point Cloud Viewer

## Introduction
This project realizes the reading of color depth images and concatenation into point cloud display, supporting user interaction using mouse and keyboard.\
本项目实现了读取彩色深度图像并拼接为点云显示，支持用户使用鼠标和键盘进行交互。

## Demonstration
<div align="center"> 
  <img width="80%" src="assets/demo.gif"/>
</div>

## Dependency
To run this project, the following dependencies need to be installed.\
若要运行此项目，需要安装以下依赖项。
- OpenCV
- OpenGL
- CUDA
- glm
- Eigen
- glad
- glfw
- imgui

## Compile and Run
```shell
cd [source_directory]
mkdir build
cd build
cmake ..
make
./OGL_PointCloud
```

## Options
```shell
./OGL_PointCloud --help
Allowed Options:
  -h, --help                       help message
  -i, --input arg (=../datasets)   input image and param file path
  -o, --output arg (=../datasets)  output point cloud file path
  -n, --image_num arg (=4)         image number
  --width arg (=1920)              image width
  --height arg (=1080)             image height
  --save arg (=0)                  save point cloud
```

## Controls
- Keyboard W ===== forward
- Keyboard A ===== left
- Keyboard S ===== backward
- Keyboard D ===== right
- left mouse button ===== rotation
