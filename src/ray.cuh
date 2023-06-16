#pragma once

#include "torrey.cuh"
#include "vector.cuh"

struct Ray {
    Vector3 org, dir;
    Real tnear, tfar;
};
