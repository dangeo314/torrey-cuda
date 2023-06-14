#pragma once

#include "torrey.h"
#include "bvh.h"
#include "camera.h"
#include "light.h"
#include "intersection.h"
#include "material.h"
#include "parse_scene.h"
#include "shape.h"
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
