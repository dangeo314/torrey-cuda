#pragma once

#include "torrey.cuh"
#include "vector.cuh"

struct Intersection {
    Vector3 position;
    Vector3 geometric_normal;
    Vector3 shading_normal;
    Real distance = infinity<Real>();
    Vector2 uv;
    int material_id;
    int area_light_id;
};
