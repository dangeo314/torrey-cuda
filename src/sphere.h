#pragma once

#include "torrey.h"
#include "hw1_scenes.h"
#include "intersection.h"
#include "ray.h"
#include "vector.h"
#include <optional>

/// Numerically stable quadratic equation solver at^2 + bt + c = 0
/// See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
/// returns false when it can't find solutions.
inline bool solve_quadratic(Real a, Real b, Real c, Real *t0, Real *t1) {
    // Degenerated case
    if (a == 0) {
        if (b == 0) {
            return false;
        }
        *t0 = *t1 = -c / b;
        return true;
    }

    Real discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    }
    Real root_discriminant = sqrt(discriminant);
    if (b >= 0) {
        *t0 = (- b - root_discriminant) / (2 * a);
        *t1 = 2 * c / (- b - root_discriminant);
    } else {
        *t0 = 2 * c / (- b + root_discriminant);
        *t1 = (- b + root_discriminant) / (2 * a);
    }
    return true;
}

inline std::optional<Intersection> intersect(const hw1::Sphere &sph, const Ray &ray) {
    // Our sphere is ||p - x||^2 = r^2
    // substitute x = o + d * t, we want to solve for t
    // ||p - (o + d * t)||^2 = r^2
    // (p.x - (o.x + d.x * t))^2 + (p.y - (o.y + d.y * t))^2 + (p.z - (o.z + d.z * t))^2 - r^2 = 0
    // (d.x^2 + d.y^2 + d.z^2) t^2 + 2 * (d.x * (o.x - p.x) + d.y * (o.y - p.y) + d.z * (o.z - p.z)) t + 
    // ((p.x-o.x)^2 + (p.y-o.y)^2 + (p.z-o.z)^2  - r^2) = 0
    // A t^2 + B t + C
    Vector3 v = ray.org - sph.center;
    Real A = dot(ray.dir, ray.dir);
    Real B = 2 * dot(ray.dir, v);
    Real C = dot(v, v) - sph.radius * sph.radius;
    Real t0, t1;
    if (!solve_quadratic(A, B, C, &t0, &t1)) {
        // No intersection
        return {};
    }
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    Real t = t0;
    if (t0 >= ray.tnear && t0 < ray.tfar) {
        t = t0;
    }
    if (t1 >= ray.tnear && t1 < ray.tfar && t < ray.tnear) {
        t = t1;
    }

    if (t >= ray.tnear && t < ray.tfar) {
        // Record the intersection
        Vector3 p = ray.org + t * ray.dir;
        Vector3 n = normalize(p - sph.center);
        Real theta = acos(-n.y);
        Real phi = atan2(-n.z, n.x) + c_PI;
        Vector2 uv{phi / (2 * c_PI), theta / c_PI};
        return Intersection{p, n, n, t, uv, sph.material_id};
    }
    return {};
}

inline bool occluded(const hw1::Sphere &sph, const Ray &ray) {
    return bool(intersect(sph, ray));
}

inline bool occluded(const hw1::Scene &scene, const Ray &ray) {
    for (const hw1::Sphere &sph : scene.shapes) {
        if (occluded(sph, ray)) {
            return true;
        }
    }
    return false;
}
