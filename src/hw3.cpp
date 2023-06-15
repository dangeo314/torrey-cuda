#include "hw3.cuh"
#include "parse_scene.cuh"
#include "bbox.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "parallel.cuh"
#include "parse_scene.cuh"
#include "pcg.cuh"
#include "print_scene.cuh"
#include "progressreporter.cuh"
#include "ray.cuh"
#include "scene.cuh"
#include "timer.cuh"
#include <optional>

Vector3 schlick_fresnel(const Vector3 &F0, Real cos_theta) {
    return F0 + (1 - F0) * pow(1 - cos_theta, 5);
}

static Vector3 radiance_perlin(const Scene &scene, Ray ray, pcg32_state &rng) {
    if (auto hit_isect = intersect(scene, ray)) {
        Vector3 p = hit_isect->position;
        Vector3 n = hit_isect->shading_normal;
        Vector3 L = Vector3{0, 0, 0};
        if (hit_isect->area_light_id != -1) {
            const DiffuseAreaLight &light =
                std::get<DiffuseAreaLight>(scene.lights[hit_isect->area_light_id]);
            if (dot(-ray.dir, n) > 0) {
                L += light.radiance;
            }
        }
        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }
        const Material &mat = scene.materials[hit_isect->material_id];

        // Direct lighting
        // Currently we only evaluate the diffuse part.
        perlin noise;
        Vector3 refl = Vector3{0, 0, 0};
        if (auto *diffuse = std::get_if<Diffuse>(&mat)) {
            refl =  noise.noise(p) * eval(diffuse->reflectance, hit_isect->uv);
        } else if (auto *plastic = std::get_if<Plastic>(&mat)) {
            Real F0s = square((plastic->eta - 1) / (plastic->eta + 1));
            Vector3 F0{F0s, F0s, F0s};
            Vector3 F = schlick_fresnel(F0, dot(n, -ray.dir));
            refl = (1 - F) * noise.noise(p) * eval(plastic->reflectance, hit_isect->uv);
        } else {
            refl = Vector3{0, 0, 0};
        }

        if (max(refl) > Real(0)) {
            // Loop over the lights
            for (const Light &light : scene.lights) {
                if (auto *point_light = std::get_if<PointLight>(&light)) {
                    Vector3 l = point_light->position - p;
                    Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                    if (!occluded(scene, shadow_ray)) {
                        L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                             (point_light->intensity / length_squared(l)) *
                             refl;
                    }
                } else if (auto *area_light = std::get_if<DiffuseAreaLight>(&light)) {
                    const Shape &shape = scene.shapes[area_light->shape_id];

                    // Added stratified sampling
                    int N_sqrt = 2;
                    int N = pow(N_sqrt, 2.0);
                    std::vector<Vector2> uList;
                    for (int a = 0; a < N_sqrt; a++) {
                        for (int b = 0; b < N_sqrt; b++) {
                            Real firstU = (a + next_pcg32_real<Real>(rng)) / N_sqrt;
                            Real secondU = (b + next_pcg32_real<Real>(rng)) / N_sqrt;

                            uList.push_back(Vector2{firstU, secondU});
                        }
                    }

                    // Vector2 u{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
                    for (auto u : uList) {
                        PointAndNormal pn = sample_on_shape(shape, u);
                        Vector3 l = pn.point - p;
                        Real cos_l = -dot(pn.normal, normalize(l));
                        Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                        if (cos_l > 0 && !occluded(scene, shadow_ray)) {
                            Real pdf = pdf_sample_on_shape(shape, pn.point, rng);
                            L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                                (area_light->radiance / length_squared(l)) *
                                cos_l * refl / (N * pdf); // POSSIBLY CHANGE TO * N
                        }
                    }
                }
            }
        }

        // Scattering
        if (auto *mirror = std::get_if<Mirror>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            Vector3 F0 = eval(mirror->reflectance, hit_isect->uv) * noise.noise(p);
            Vector3 F = schlick_fresnel(F0, dot(n, refl_ray.dir));
            L += F * radiance_perlin(scene, refl_ray, rng);
        } else if (auto *plastic = std::get_if<Plastic>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            Real F0s = square((plastic->eta - 1) / (plastic->eta + 1));
            Vector3 F0{F0s, F0s, F0s};
            Vector3 F = schlick_fresnel(F0, dot(n, -ray.dir));
            L += F * radiance_perlin(scene, refl_ray, rng);
        }

        return L;
    }
    return scene.background_color;
}

static Vector3 radiance(const Scene &scene, Ray ray, bool use_shading_normal) {
    Vector3 color = scene.background_color;
    if (auto hit_isect = intersect(scene, ray)) {
        Vector3 p = hit_isect->position;
        Vector3 n = use_shading_normal ?
            hit_isect->shading_normal : hit_isect->geometric_normal;
        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }
        const Material &mat = scene.materials[hit_isect->material_id];
        if (auto *diffuse = std::get_if<Diffuse>(&mat)) {
            Vector3 L = Vector3{0, 0, 0};
            // Loop over the lights
            for (const Light &light : scene.lights) {
                // Assume point lights for now.
                const PointLight &point_light = std::get<PointLight>(light);
                Vector3 l = point_light.position - p;
                Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                if (!occluded(scene, shadow_ray)) {
                    L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                         (point_light.intensity / length_squared(l)) *
                         eval(diffuse->reflectance, hit_isect->uv);
                }
            }
            color = L;
        } else if (auto *mirror = std::get_if<Mirror>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            return eval(mirror->reflectance, hit_isect->uv) *
                radiance(scene, refl_ray, use_shading_normal);
        }
    }
    return color;
}

Image3 hw_3_1(const std::vector<std::string> &params) {
    // Homework 3.1: image textures
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    Timer timer;
    tick(timer);
    ParsedScene parsed_scene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;

    tick(timer);
    Scene scene(parsed_scene);
    std::cout << "Scene construction done. Took " << tick(timer) << " seconds." << std::endl;
    int spp = scene.samples_per_pixel;

    Image3 img(scene.width, scene.height);

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(
        scene.camera, w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < spp; s++) {
                    Real u, v;
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    img(x, y) += radiance(scene, ray,
                        false /*use_shading_normal*/) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_3_2(const std::vector<std::string> &params) {
    // Homework 3.2: shading normals
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    Timer timer;
    tick(timer);
    ParsedScene parsed_scene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;

    tick(timer);
    Scene scene(parsed_scene);
    std::cout << "Scene construction done. Took " << tick(timer) << " seconds." << std::endl;
    int spp = scene.samples_per_pixel;

    Image3 img(scene.width, scene.height);

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(
        scene.camera, w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < spp; s++) {
                    Real u, v;
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    img(x, y) += radiance(scene, ray,
                        true /*use_shading_normal*/) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

static Vector3 radiance_new(const Scene &scene, Ray ray, pcg32_state &rng) {
    if (auto hit_isect = intersect(scene, ray)) {
        Vector3 p = hit_isect->position;
        Vector3 n = hit_isect->shading_normal;
        Vector3 L = Vector3{0, 0, 0};
        if (hit_isect->area_light_id != -1) {
            const DiffuseAreaLight &light =
                std::get<DiffuseAreaLight>(scene.lights[hit_isect->area_light_id]);
            if (dot(-ray.dir, n) > 0) {
                L += light.radiance;
            }
        }
        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }
        const Material &mat = scene.materials[hit_isect->material_id];

        // Direct lighting
        // Currently we only evaluate the diffuse part.
        Vector3 refl = Vector3{0, 0, 0};
        if (auto *diffuse = std::get_if<Diffuse>(&mat)) {
            refl = eval(diffuse->reflectance, hit_isect->uv);
        } else if (auto *plastic = std::get_if<Plastic>(&mat)) {
            Real F0s = square((plastic->eta - 1) / (plastic->eta + 1));
            Vector3 F0{F0s, F0s, F0s};
            Vector3 F = schlick_fresnel(F0, dot(n, -ray.dir));
            refl = (1 - F) * eval(plastic->reflectance, hit_isect->uv);
        } else {
            refl = Vector3{0, 0, 0};
        }

        if (max(refl) > Real(0)) {
            // Loop over the lights
            for (const Light &light : scene.lights) {
                if (auto *point_light = std::get_if<PointLight>(&light)) {
                    Vector3 l = point_light->position - p;
                    Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                    if (!occluded(scene, shadow_ray)) {
                        L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                             (point_light->intensity / length_squared(l)) *
                             refl;
                    }
                } else if (auto *area_light = std::get_if<DiffuseAreaLight>(&light)) {
                    const Shape &shape = scene.shapes[area_light->shape_id];
                    Vector2 u{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
                    PointAndNormal pn = sample_on_shape(shape, u);
                    Vector3 l = pn.point - p;
                    Real cos_l = -dot(pn.normal, normalize(l));
                    Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                    if (cos_l > 0 && !occluded(scene, shadow_ray)) {
                        Real pdf = pdf_sample_on_shape(shape, pn.point, rng);
                        L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                             (area_light->radiance / length_squared(l)) *
                             cos_l * refl / pdf;
                    }
                }
            }
        }

        // Scattering
        if (auto *mirror = std::get_if<Mirror>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            Vector3 F0 = eval(mirror->reflectance, hit_isect->uv);
            Vector3 F = schlick_fresnel(F0, dot(n, refl_ray.dir));
            L += F * radiance_new(scene, refl_ray, rng);
        } else if (auto *plastic = std::get_if<Plastic>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            Real F0s = square((plastic->eta - 1) / (plastic->eta + 1));
            Vector3 F0{F0s, F0s, F0s};
            Vector3 F = schlick_fresnel(F0, dot(n, -ray.dir));
            L += F * radiance_new(scene, refl_ray, rng);
        }

        return L;
    }
    return scene.background_color;
}

Image3 hw_3_3(const std::vector<std::string> &params) {
    // Homework 3.3: Fresnel
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    Timer timer;
    tick(timer);
    ParsedScene parsed_scene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;

    tick(timer);
    Scene scene(parsed_scene);
    std::cout << "Scene construction done. Took " << tick(timer) << " seconds." << std::endl;
    int spp = scene.samples_per_pixel;

    Image3 img(scene.width, scene.height);

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(
        scene.camera, w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < spp; s++) {
                    Real u, v;
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    img(x, y) += radiance_new(scene, ray, rng) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_3_4(const std::vector<std::string> &params) {
    // Homework 3.4: area lights
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    Timer timer;
    tick(timer);
    ParsedScene parsed_scene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;

    tick(timer);
    Scene scene(parsed_scene);
    std::cout << "Scene construction done. Took " << tick(timer) << " seconds." << std::endl;
    int spp = scene.samples_per_pixel;

    Image3 img(scene.width, scene.height);

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(
        scene.camera, w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < spp; s++) {
                    Real u, v;
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    img(x, y) += radiance_new(scene, ray, rng) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_3_5(const std::vector<std::string> &params) {
    // Homework 3 Bonus: Perlin Noise
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    ParsedScene parsed_scene = parse_scene(params[0]);
    Scene scene(parsed_scene);
    int spp = scene.samples_per_pixel;

    Image3 img(scene.width, scene.height);

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(
        scene.camera, w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < spp; s++) {
                    Real u, v;
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    img(x, y) += radiance_perlin(scene, ray, rng) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();

    return img;
}
