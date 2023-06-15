#pragma once

#include "torrey.h"
#include "vector.cuh"

struct Ray {
    Vector3 org, dir;
    Real tnear, tfar;
};
