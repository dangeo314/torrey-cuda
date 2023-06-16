#pragma once

#include "torrey.cuh"
#include "bvh.cuh"
#include "camera.cuh"
#include "light.cuh"
#include "intersection.cuh"
#include "material.cuh"
#include "parse_scene.cuh"
#include "shape.cuh"
#include <optional>
#include <thrust/universal_vector.h>

struct Scene {
    Scene(const ParsedScene &scene);

    Camera camera;
    int width, height;
    

    thrust::universal_vector<Shape> shapes;
    thrust::universal_vector<Material> materials;
    thrust::universal_vector<Light> lights;
    Vector3 background_color;

    int samples_per_pixel;
    // For the Triangle in the shapes to reference to.
    thrust::universal_vector<TriangleMesh> meshes;

    thrust::universal_vector<BVHNode> bvh_nodes;
    int bvh_root_id;

    //alternatives for the vector on device, initialized on host in scene construction
    Shape* shapes_ptr; int num_shapes;
    Material* materials_ptr; int num_materials;
    Light* lights_ptr; int num_lights;
    BVHNode* bvh_nodes_ptr; int num_nodes;
    TriangleMesh* meshes_ptr; int num_meshes;

};

__host__ __device__ Intersection intersect(const Scene &scene, Ray ray);
__host__ __device__ bool occluded(const Scene &scene, const Ray &ray);
