#pragma once

#include "torrey.h"
#include "vector.h"

struct Intersection {
    Vector3 position;
    Vector3 geometric_normal;
    Vector3 shading_normal;
    Real distance;
    Vector2 uv;
    int material_id;
    int area_light_id;
};
