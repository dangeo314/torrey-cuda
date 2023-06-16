#pragma once

#include "torrey.cuh"
#include "vector.cuh"
#include "image.cuh"
// #include <variant>
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

enum TextureType {
    CONSTANT,
    IMAGE
    // PERLIN
};

struct Texture {
    TextureType type;
    union TextureUnion{
        __host__ __device__ TextureUnion(){};
        ConstantTexture constant;
        ImageTexture image;
        __host__ __device__ ~TextureUnion(){};
        // PerlinTexture perlin;
    } data;

    __host__ __device__ Texture() {
        type = CONSTANT;
        data.constant = ConstantTexture();
    }

    __host__ __device__ Texture(const Texture& other) {
        type = other.type;
        switch(type) {
            case CONSTANT:
                data.constant = ConstantTexture(other.data.constant);
                break;
            case IMAGE:
                data.image = ImageTexture(other.data.image);
                break;
        }
    }

    __host__ __device__ ~Texture() {
        switch(type) {
            case CONSTANT:
                delete &data.constant;
                break;
            case IMAGE:
                delete &data.image;
                break;
        }
    }
    
    __host__ __device__ Texture& operator=(const Texture& other) {
        if (this != &other) { // protect against invalid self-assignment
            this->~Texture(); // destruct current object state

            // allocate and initialize new state to the object
            type = other.type;
            switch(type) {
                case CONSTANT:
                    data.constant = other.data.constant;
                    break;
                case IMAGE:
                    data.image = other.data.image;
                    break;
                // case PERLIN:
                //     data.perlin = other.data.perlin;
                //     break;
            }
        }

    // return the existing object so we can chain this operator
    return *this;
    };
};

// using Texture = std::variant<ConstantTexture, ImageTexture>;

__host__ __device__ Vector3 eval(const Texture &texture, const Vector2 &uv);
