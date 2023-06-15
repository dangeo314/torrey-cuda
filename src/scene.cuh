#pragma once

#include "torrey.h"
#include "bvh.cuh"
#include "camera.cuh"
#include "light.cuh"
#include "intersection.cuh"
#include "material.cuh"
#include "parse_scene.cuh"
#include "shape.cuh"
#include <optional>

struct Scene {
    Scene(const ParsedScene &scene);

    Camera camera;
    int width, height;
    std::vector<Shape> shapes;
    std::vector<Material> materials;
    std::vector<Light> lights;
    Vector3 background_color;

    int samples_per_pixel;
    // For the Triangle in the shapes to reference to.
    std::vector<TriangleMesh> meshes;

    std::vector<BVHNode> bvh_nodes;
    int bvh_root_id;
};

std::optional<Intersection> intersect(const Scene &scene, Ray ray);
bool occluded(const Scene &scene, const Ray &ray);
