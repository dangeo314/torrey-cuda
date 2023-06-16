#pragma once

#include "torrey.cuh"
#include "vector.cuh"
#include "intersection.cuh"
#include <optional>
#include <variant>
#include <vector>
#include "pcg.cuh"
#include <thrust/universal_vector.h>

struct Ray;

struct ShapeBase {
    int material_id = -1;
    int area_light_id = -1;
};

struct Sphere : public ShapeBase {
    Vector3 center;
    Real radius;
};

struct TriangleMesh : public ShapeBase {
    thrust::universal_vector<Vector3> positions;
    thrust::universal_vector<Vector3i> indices;
    thrust::universal_vector<Vector3> normals;
    thrust::universal_vector<Vector2> uvs;
    
    Vector3* positions_ptr;
    Vector3i* indices_ptr;
    Vector3* normals_ptr;
    Vector2* uvs_ptr;

    
    int num_positions;
    int num_indices;
    int num_normals;
    int num_uvs;
};

struct Triangle {
    int face_index;
    const TriangleMesh *mesh;
};

//using Shape = std::variant<Sphere, TriangleMesh, Triangle>;

enum ShapeType{
    SPHERE,
    TRIANGLE_MESH,
    TRIANGLE
};

struct Shape{
    ShapeType type;
    union ShapeUnion{
        Sphere sphere;
        TriangleMesh tri_mesh;
        Triangle tri;
        __host__ __device__ ShapeUnion(){};
        __host__ __device__ ~ShapeUnion(){};
        
    } data;

    __host__ __device__ Shape() {
        type = SPHERE;
        data.sphere = Sphere{};
    }
    __host__ __device__ Shape(const Shape& other) {
        type = other.type;
        switch(type) {
            case SPHERE:
                data.sphere = Sphere(other.data.sphere);
                break;
            case TRIANGLE:
                data.tri = Triangle(other.data.tri);
                break;
            case TRIANGLE_MESH:
                data.tri_mesh = TriangleMesh(other.data.tri_mesh);
                break;
        }
    }

    __host__ __device__ ~Shape() {
        switch(type) {
            case SPHERE:
                delete &data.sphere;
                break;
            case TRIANGLE:
                delete &data.tri;
                break;
            case TRIANGLE_MESH:
                delete &data.tri_mesh;
                break;
        }
    }
    __host__ __device__ Shape& operator=(const Shape& other) {
        if (this != &other) { // protect against invalid self-assignment
            this->~Shape(); // destruct current object state

            // allocate and initialize new state to the object
            type = other.type;
            switch(type) {
                case SPHERE:
                    data.sphere = other.data.sphere;
                    break;
                case TRIANGLE:
                    data.tri = other.data.tri;
                    break;
                case TRIANGLE_MESH:
                    data.tri_mesh = other.data.tri_mesh;
                    break;
            }
        }

        // return the existing object so we can chain this operator
        return *this;
    };
};

__host__ __device__ Intersection intersect(const Shape &shape, const Ray &ray);
__host__ __device__ bool occluded(const Shape &shape, const Ray &ray);

__host__ __device__ Vector3
    intersect_triangle(const Ray &ray,
                       const Vector3 &p0,
                       const Vector3 &p1,
                       const Vector3 &p2);

struct PointAndNormal {
    Vector3 point;
    Vector3 normal;
};

__host__ __device__ PointAndNormal sample_on_shape(const Shape &shape,
                               const Vector2 &u);

Real pdf_sample_on_shape(const Shape &shape,
                         const Vector3 &p, pcg32_state rng);
