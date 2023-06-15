#pragma once

#include "torrey.h"
#include "texture.cuh"
#include "vector.cuh"
#include <variant>

struct Diffuse {
    Texture reflectance;
};

struct Mirror {
    Texture reflectance;
};

struct Plastic {
    Real eta; // IOR
    Texture reflectance;
};

using Material = std::variant<Diffuse, Mirror, Plastic>;
