#include "shape.cuh"
#include "ray.cuh"
#include "pcg.cuh"

/// Numerically stable quadratic equation solver at^2 + bt + c = 0
/// See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
/// returns false when it can't find solutions.
__host__ __device__ inline bool solve_quadratic(Real a, Real b, Real c, Real *t0, Real *t1) {
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

__host__ __device__ inline Intersection intersect(const Sphere &sph, const Ray &ray) {
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
        Real theta = acos(n.y);
        Real phi = atan2(-n.z, n.x) + c_PI;
        Vector2 uv{phi / (2 * c_PI), theta / c_PI};
        return Intersection{p, // position
                            n, // geometric normal
                            n, // shading normal
                            t, // distance
                            uv, // UV
                            sph.material_id,
                            sph.area_light_id};
    }
    return {};
}

__host__ __device__ Vector3
    intersect_triangle(const Ray &ray,
                       const Vector3 &p0,
                       const Vector3 &p1,
                       const Vector3 &p2) {
    Vector3 e1 = p1 - p0;
    Vector3 e2 = p2 - p0;
    Vector3 s1 = cross(ray.dir, e2);
    Real divisor = dot(s1, e1);
    if (divisor == 0) {
        return {};
    }
    Real inv_divisor = 1 / divisor;
    Vector3 s = ray.org - p0;
    Real u = dot(s, s1) * inv_divisor;
    Vector3 s2 = cross(s, e1);
    Real v = dot(ray.dir, s2) * inv_divisor;
    Real t = dot(e2, s2) * inv_divisor;

    if (t > ray.tnear && t < ray.tfar && u >= 0 && v >= 0 && u + v <= 1) {
        return Vector3{u, v, t};
    }

    return {};
}

__host__ __device__ inline Intersection intersect(const Triangle &tri, const Ray &ray) {
	const TriangleMesh &mesh = *tri.mesh;
	Vector3i index = mesh.indices_ptr[tri.face_index];
	Vector3 p0 = mesh.positions_ptr[index[0]];
	Vector3 p1 = mesh.positions_ptr[index[1]];
	Vector3 p2 = mesh.positions_ptr[index[2]];
    auto uvt_ = intersect_triangle(ray, p0, p1, p2);
	if (uvt_[2] < infinity<Real>()) {
		Vector2 b = Vector2{uvt_.x, uvt_.y};
		Real t = uvt_.z;
		Vector3 p = (1 - b[0] - b[1]) * p0 + b[0] * p1 + b[1] * p2;
		Vector3 geometric_normal = normalize(cross(p1 - p0, p2 - p0));
        Vector2 uv = b;
        if (mesh.uvs.size() > 0) {
            Vector2 uv0 = mesh.uvs_ptr[index[0]];
            Vector2 uv1 = mesh.uvs_ptr[index[1]];
            Vector2 uv2 = mesh.uvs_ptr[index[2]];
            uv = (1 - b[0] - b[1]) * uv0 + b[0] * uv1 + b[1] * uv2;
        }
        Vector3 shading_normal = geometric_normal;
        if (mesh.normals.size() > 0) {
            Vector3 n0 = mesh.normals_ptr[index[0]];
            Vector3 n1 = mesh.normals_ptr[index[1]];
            Vector3 n2 = mesh.normals_ptr[index[2]];
            shading_normal = normalize((1 - b[0] - b[1]) * n0 + b[0] * n1 + b[1] * n2);
        }
		return Intersection{p, // position
                            geometric_normal,
                            shading_normal,
                            t, // distance
                            uv,
                            mesh.material_id,
                            mesh.area_light_id};
	}
	return {};
}

__host__ __device__ Intersection intersect(const Shape &shape, const Ray &ray) {
	if (shape.type==SPHERE) {
        const Sphere* sph = &shape.data.sphere;
		return intersect(*sph, ray);
	} else if (shape.type==TRIANGLE) {
        const Triangle* tri = &shape.data.tri;
		return intersect(*tri, ray);
	} else {
		assert(false);
		return {};
	}
}

__host__ __device__ bool occluded(const Shape &shape, const Ray &ray) {
    return bool(intersect(shape, ray).distance < infinity<Real>());
}

__host__ __device__ PointAndNormal sample_on_shape(const Shape &shape,
                               const Vector2 &u) {

    if (shape.type==SPHERE) {
        const Sphere* sph = &shape.data.sphere;
        Real theta = acos(std::clamp(1 - 2 * u[0], Real(-1), Real(1)));
        Real phi = 2 * c_PI * u[1];
        Vector3 n{
            cos(phi) * sin(theta),
            sin(phi) * sin(theta),
            cos(theta)
        };
        Vector3 p = sph->radius * n + sph->center;
        return {p, n};
    } else if (shape.type==TRIANGLE) {
        const Triangle* tri = &shape.data.tri;
        Real b1 = 1 - sqrt(max(u[0], Real(0)));
        Real b2 = u[1] * sqrt(max(u[0], Real(0)));
        const TriangleMesh *mesh = tri->mesh;
        Vector3i index = mesh->indices_ptr[tri->face_index];
        Vector3 p0 = mesh->positions_ptr[index[0]];
        Vector3 p1 = mesh->positions_ptr[index[1]];
        Vector3 p2 = mesh->positions_ptr[index[2]];
        Vector3 p = (1 - b1 - b2) * p0 + b1 * p1 + b2 * p2;
        Vector3 geometric_normal = normalize(cross(p1 - p0, p2 - p0));
        Vector3 shading_normal = geometric_normal;
        if (mesh->normals.size() > 0) {
            Vector3 n0 = mesh->normals_ptr[index[0]];
            Vector3 n1 = mesh->normals_ptr[index[1]];
            Vector3 n2 = mesh->normals_ptr[index[2]];
            shading_normal = normalize((1 - b1 - b2) * n0 + b1 * n1 + b2 * n2);
        }
        if (dot(geometric_normal, shading_normal) < 0) {
            geometric_normal = -geometric_normal;
        }
        return {p, geometric_normal};
    } else {
        assert(false);
        return {};
    }
}

__host__ __device__ Real pdf_sample_on_shape(const Shape &shape,
                         const Vector3 &p, pcg32_state rng) {
    if (shape.type==SPHERE) {
        const Sphere* sph = &shape.data.sphere;
        // cone angle = solid angle = sqrt(1 - (r^2)/(lensquared(center - point)))
        /*
        std::cout << "Sphere Center: " << sph->center << std::endl;
        std::cout << "Point origin / ray start: " << p << std::endl;
        auto max_cos = sqrt(1 - (pow(sph->radius, 2.0))/(length_squared(sph->center - p)));
        

        // Compute the PDF (1 / solid angle) based on the sampled direction and the cone angle
        auto result = Real(1) / (2 * c_PI * (1 - max_cos));
        //std::cout << result << std::endl;
        return(result);
        // Don't sample for other side of sphere

        // Old code without cone sampling:
        
        */
        auto result = Real(1) / (4 * c_PI * square(sph->radius));
        return result;
        

    } else if (shape.type==TRIANGLE) {
        const Triangle* tri = &shape.data.tri;
        const TriangleMesh *mesh = tri->mesh;
        Vector3i index = mesh->indices_ptr[tri->face_index];
        Vector3 p0 = mesh->positions_ptr[index[0]];
        Vector3 p1 = mesh->positions_ptr[index[1]];
        Vector3 p2 = mesh->positions_ptr[index[2]];
        return Real(1) / (Real(0.5) * length(cross(p1 - p0, p2 - p0)));
    } else {
        assert(false);
        return Real(0);
    }
}
