#pragma once
#include "parse_scene.cuh"

struct Ray;

struct Camera {
    Vector3 lookfrom;
    Vector3 lookat;
    Vector3 up;
    Real vfov;
};

//we initialize a camera only on host
Camera from_parsed_camera(const ParsedCamera &cam);

struct CameraRayData {
    Vector3 origin;
    Vector3 top_left_corner;
    Vector3 horizontal;
    Vector3 vertical;
};

__host__ __device__ CameraRayData compute_camera_ray_data(const Camera &cam, int width, int height);

__host__ __device__ Ray generate_primary_ray(const CameraRayData &cam_ray_data, Real u, Real v);
