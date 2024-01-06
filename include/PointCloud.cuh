//
// Created by zcb on 2023/6/14.
//

#ifndef OGLSHOW_POINTCLOUD_CUH
#define OGLSHOW_POINTCLOUD_CUH

extern "C" void cuUV2World(uchar3 *rgb, ushort1 *depth, int width, int height, int camNum,
                           float *intrinsic, float *extrinsic, float3 *point, float3 *color);

#endif //OGLSHOW_POINTCLOUD_CUH
