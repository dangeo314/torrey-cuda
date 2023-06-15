#pragma once

#include "torrey.h"
#include "vector.cuh"
#include "image.cuh"
#include <variant>
#include "perlin.cuh"

struct ConstantTexture {
    Vector3 color;
};

struct ImageTexture {
    Image3 image;
    Real uscale = 1, vscale = 1;
    Real uoffset = 0, voffset = 0;
};

/*
struct PerlinTexture {
    perlin noise;
};
*/

using Texture = std::variant<ConstantTexture, ImageTexture>;

Vector3 eval(const Texture &texture, const Vector2 &uv);
