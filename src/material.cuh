#pragma once

#include "torrey.cuh"
#include "texture.cuh"
#include "vector.cuh"
// #include <variant>

enum MaterialType {
    DIFFUSE,
    MIRROR,
    PLASTIC
};

struct Diffuse {
    Texture reflectance;
    /*Diffuse(){};
    Diffuse(const Diffuse& other){
        reflectance = Texture(other.reflectance);
    };
    ~Diffuse(){};*/
};

struct Mirror {
    Texture reflectance;
};

struct Plastic {
    Real eta; // IOR
    Texture reflectance;
};

struct Material {
    MaterialType type;
    union MaterialUnion{
        Diffuse diffuse;
        Mirror mirror;
        Plastic plastic;
        __host__ __device__ MaterialUnion(){};
        __host__ __device__ ~MaterialUnion(){};
        // PerlinTexture perlin;
    } data;

    __host__ __device__ Material() {
        type = DIFFUSE;
        data.diffuse = Diffuse();
    }

    __host__ __device__ Material(const Material& other) {
        type = other.type;
        switch(type) {
            case DIFFUSE:
                data.diffuse = Diffuse(other.data.diffuse);
                break;
            case MIRROR:
                data.mirror = Mirror(other.data.mirror);
                break;
            case PLASTIC:
                data.plastic = Plastic(other.data.plastic);
                break;
        }
    }

    __host__ __device__ ~Material() {
        switch(type) {
            case DIFFUSE:
                delete &data.diffuse;
                break;
            case MIRROR:
                delete &data.mirror;
                break;
            case PLASTIC:
                delete &data.plastic;
                break;
        }
    }
    
    __host__ __device__ Material& operator=(const Material& other) {
        if (this != &other) { // protect against invalid self-assignment
            this->~Material(); // destruct current object state

            // allocate and initialize new state to the object
            type = other.type;
            switch(type) {
                case DIFFUSE:
                    data.diffuse = other.data.diffuse;
                    break;
                case MIRROR:
                    data.mirror = other.data.mirror;
                    break;
                case PLASTIC:
                    data.plastic = other.data.plastic;
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


// using Material = std::variant<Diffuse, Mirror, Plastic>;
