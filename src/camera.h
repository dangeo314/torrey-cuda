#pragma once

#include "parse_scene.h"

struct Ray;

struct Camera {
    Vector3 lookfrom;
    Vector3 lookat;
    Vector3 up;
    Real vfov;
};

Camera from_parsed_camera(const ParsedCamera &cam);

struct CameraRayData {
    Vector3 origin;
    Vector3 top_left_corner;
    Vector3 horizontal;
    Vector3 vertical;
};

CameraRayData compute_camera_ray_data(const Camera &cam, int width, int height);

Ray generate_primary_ray(const CameraRayData &cam_ray_data, Real u, Real v);
