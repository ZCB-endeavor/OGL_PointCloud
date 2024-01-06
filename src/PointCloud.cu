//
// Created by zcb on 2023/6/14.
//

#include "PointCloud.cuh"

__global__ void cuUV2WorldKernel(uchar3 *rgb, ushort1 *depth, int width, int height, int camNum,
                                 float *intrinsic, float *extrinsic, float3 *point, float3 *color) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        //! 确定位于第几张图像
        for (int i = 0; i < camNum; i++) {
            float fx = intrinsic[i * 9 + 0];
            float fy = intrinsic[i * 9 + 4];
            float cx = intrinsic[i * 9 + 2];
            float cy = intrinsic[i * 9 + 5];
            float d = (float) depth[i * width * height + y * width + x].x / 1000.f;
            if (d != 0) {
                float pcx = d * ((float) x - cx) / fx;
                float pcy = d * ((float) y - cy) / fy;
                float pcz = d;
                //! 这里的外参是相机到世界坐标的矩阵
                float pwx = extrinsic[i * 16 + 0] * pcx + extrinsic[i * 16 + 1] * pcy + extrinsic[i * 16 + 2] * pcz +
                            extrinsic[i * 16 + 3];
                float pwy = extrinsic[i * 16 + 4] * pcx + extrinsic[i * 16 + 5] * pcy + extrinsic[i * 16 + 6] * pcz +
                            extrinsic[i * 16 + 7];
                float pwz = extrinsic[i * 16 + 8] * pcx + extrinsic[i * 16 + 9] * pcy + extrinsic[i * 16 + 10] * pcz +
                            extrinsic[i * 16 + 11];

                float r = rgb[i * width * height + y * width + x].z;
                float g = rgb[i * width * height + y * width + x].y;
                float b = rgb[i * width * height + y * width + x].x;
                //! 世界点与颜色存入对应的数据结构中
                float3 pw = make_float3(pwx, pwy, pwz);
                float3 pixel_color = make_float3(r, g, b);
                point[i * width * height + y * width + x] = pw;
                color[i * width * height + y * width + x] = pixel_color;
            }
        }
    }
}

extern "C" void cuUV2World(uchar3 *rgb, ushort1 *depth, int width, int height, int camNum,
                           float *intrinsic, float *extrinsic, float3 *point, float3 *color) {
    dim3 threads(32, 32);
    unsigned int blockX = (threads.x + width - 1) / threads.x;
    unsigned int blockY = (threads.y + height - 1) / threads.y;
    dim3 blocks(blockX, blockY);
    cuUV2WorldKernel<<<blocks, threads>>>(rgb, depth, width, height, camNum, intrinsic, extrinsic, point, color);
    cudaDeviceSynchronize();
}
