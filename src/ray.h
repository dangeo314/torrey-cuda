#pragma once

#include "torrey.h"
#include "vector.h"

struct Ray {
    Vector3 org, dir;
    Real tnear, tfar;
};
