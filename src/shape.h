#pragma once

#include "torrey.h"
#include "vector.h"
#include "intersection.h"
#include <optional>
#include <variant>
#include <vector>
#include "pcg.h"

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
    std::vector<Vector3> positions;
    std::vector<Vector3i> indices;
    std::vector<Vector3> normals;
    std::vector<Vector2> uvs;
};

struct Triangle {
    int face_index;
    const TriangleMesh *mesh;
};

using Shape = std::variant<Sphere, TriangleMesh, Triangle>;

std::optional<Intersection> intersect(const Shape &shape, const Ray &ray);
bool occluded(const Shape &shape, const Ray &ray);

std::optional<Vector3>
    intersect_triangle(const Ray &ray,
                       const Vector3 &p0,
                       const Vector3 &p1,
                       const Vector3 &p2);

struct PointAndNormal {
    Vector3 point;
    Vector3 normal;
};

PointAndNormal sample_on_shape(const Shape &shape,
                               const Vector2 &u);

Real pdf_sample_on_shape(const Shape &shape,
                         const Vector3 &p, pcg32_state rng);
