#pragma once

#include "torrey.cuh"
#include "vector.cuh"
// #include <variant>

struct PointLight {
    Vector3 intensity;
    Vector3 position;    
};

struct DiffuseAreaLight {
    int shape_id;
    Vector3 radiance;
};

enum LightType {
    POINT,
    AREA
};

struct Light {
    LightType type;
    union LightUnion{
        __host__ __device__ LightUnion(){};
        PointLight point;
        DiffuseAreaLight area;
        __host__ __device__ ~LightUnion(){};
        // PerlinTexture perlin;
    } data;

    __host__ __device__ Light() {
        type = POINT;
        data.point = PointLight();
    }

    __host__ __device__ Light(const Light& other) {
        type = other.type;
        switch(type) {
            case POINT:
                data.point = PointLight(other.data.point);
                break;
            case AREA:
                data.area = DiffuseAreaLight(other.data.area);
                break;
        }
    }

   __host__ __device__ ~Light() {
        switch(type) {
            case POINT:
                delete &data.point;
                break;
            case AREA:
                delete &data.area;
                break;
        }
    }
};


 // using Light = std::variant<PointLight, DiffuseAreaLight>;
