#pragma once

#include "torrey.h"
#include "vector.h"
#include "ray.h"

struct BBox {
    Vector3 p_min = Vector3{infinity<Real>(),
                            infinity<Real>(),
                            infinity<Real>()};
    Vector3 p_max = Vector3{-infinity<Real>(),
                            -infinity<Real>(),
                            -infinity<Real>()};
};

inline bool intersect(const BBox &bbox, Ray ray) {
	// https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/anoptimizedaabbhitmethod
    for (int i = 0; i < 3; i++) {
        Real inv_dir = Real(1) / ray.dir[i];
        Real t0 = (bbox.p_min[i] - ray.org[i]) * inv_dir;
        Real t1 = (bbox.p_max[i] - ray.org[i]) * inv_dir;
        if (inv_dir < 0) {
            std::swap(t0, t1);
        }
        ray.tnear = t0 > ray.tnear ? t0 : ray.tnear;
        ray.tfar = t1 < ray.tfar ? t1 : ray.tfar;
        if (ray.tfar < ray.tnear) {
            return false;
        }
    }
    return true;
}

inline int largest_axis(const BBox &box) {
    Vector3 extent = box.p_max - box.p_min;
    if (extent.x > extent.y && extent.x > extent.z) {
        return 0;
    } else if (extent.y > extent.x && extent.y > extent.z) {
        return 1;
    } else { // z is the largest
        return 2;
    }
}

inline BBox merge(const BBox &box1, const BBox &box2) {
    Vector3 p_min = Vector3{
        std::min(box1.p_min.x, box2.p_min.x),
        std::min(box1.p_min.y, box2.p_min.y),
        std::min(box1.p_min.z, box2.p_min.z)};
    Vector3 p_max = Vector3{
        std::max(box1.p_max.x, box2.p_max.x),
        std::max(box1.p_max.y, box2.p_max.y),
        std::max(box1.p_max.z, box2.p_max.z)};
    return BBox{p_min, p_max};
}

