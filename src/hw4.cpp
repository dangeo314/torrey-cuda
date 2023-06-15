#include "hw4.cuh"
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

static Vector3 schlick_fresnel(const Vector3 &F0, Real cos_theta) {
    return F0 + (1 - F0) * pow(1 - cos_theta, 5);
}

static Real schlick_fresnel(const Real F0, Real cos_theta) {
    return F0 + (1 - F0) * pow(1 - cos_theta, 5);
}

Vector2 sample_wnaught(Vector2 u1u2) {
    auto u1 = u1u2[0];
    auto u2 = u1u2[1];
    return{0.5 * acos(1.0 - 2.0*u1), 2.0*c_PI*u2};
}

Real pdf_wnaught(Vector2 phi_and_theta) {
    return (cos(phi_and_theta[0])/c_PI);
}

Vector3 wnaught_from_angles(Vector2 phi_and_theta, Vector3 normal) {
    auto n = normalize(normal);
    Vector3 a;

    if(fabs(n.x) > 0.9) {
        a = {0.0, 1.0, 0.0};
    } else {
        a = {1.0, 0.0, 0.0};
    }
    Vector3 t = normalize(cross(n, a));
    Vector3 s = cross(n,t);

    return normalize (s*cos(phi_and_theta[1])*sin(phi_and_theta[0]) + 
                    t*sin(phi_and_theta[1])*sin(phi_and_theta[0]) + 
                    n*cos(phi_and_theta[0]));
}

static Vector3 radiance(const Scene &scene, Ray ray, pcg32_state rng, int depth) {
    // if (depth <= 0) { return (Vector3{0,0,0}); }
    if (depth <= 0) { return Vector3{0.0, 0.0, 0.0}; }
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

            // This is for uniform square sampling
            Vector2 u1u2{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

            auto phi_and_theta = sample_wnaught(u1u2);

            auto pdf = pdf_wnaught(phi_and_theta);
            Vector3 wnaught = wnaught_from_angles(phi_and_theta, n);
            Ray scattered_ray {p, wnaught, Real(1e-5), infinity<Real>()};
            // std::cout << "Depth: " << depth << std::endl;
            L += refl * max(dot(n, wnaught), 0.0) / pdf / c_PI 
                  * radiance(scene, scattered_ray, rng, depth-1);

        } else if (auto *mirror = std::get_if<Mirror>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            Vector3 F0 = eval(mirror->reflectance, hit_isect->uv);
            Vector3 F = schlick_fresnel(F0, dot(n, refl_ray.dir));
            L += F * radiance(scene, refl_ray, rng, depth-1);

        } else if (auto *plastic = std::get_if<Plastic>(&mat)) {
            Real F0 = square((plastic->eta - 1) / (plastic->eta + 1));
            Real F = schlick_fresnel(F0, dot(n, -ray.dir));
            Real RandomVar = next_pcg32_real<Real>(rng);
            // Send ray on specular case
            if(RandomVar <= F) {
                Ray refl_ray{p, ray.dir - 2*dot(ray.dir, n)*n, Real(1e-4), infinity<Real>()};
                L += radiance(scene, refl_ray, rng, depth-1);
            // Send ray on diffusive case
            } else {
                refl = eval(plastic->reflectance, hit_isect->uv);
                Vector2 u1u2{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

                auto phi_and_theta = sample_wnaught(u1u2);
                auto pdf = pdf_wnaught(phi_and_theta);
                Vector3 wnaught = wnaught_from_angles(phi_and_theta, n);
                Ray scattered {p, wnaught, Real(1e-5), infinity<Real>()};
                L += refl * max(dot(n, wnaught), 0.0) / pdf / c_PI * radiance(scene, scattered, rng, depth-1);
            }
            refl = Vector3{0, 0, 0};
        }
        return L;
    }
    return scene.background_color; //scene.background_color;
}

static Vector3 radiance_roulette(const Scene &scene, Ray ray, pcg32_state rng, int depth) {
    // if (depth <= 0) { return (Vector3{0,0,0}); }
    auto roulette = next_pcg32_real<Real>(rng);
    if (roulette <= 0.1) {
        return Vector3{0.0, 0.0, 0.0};
    }
    if (depth <= 0) { return Vector3{0.0, 0.0, 0.0}; }
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

            // This is for uniform square sampling
            Vector2 u1u2{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

            auto phi_and_theta = sample_wnaught(u1u2);

            auto pdf = pdf_wnaught(phi_and_theta);
            Vector3 wnaught = wnaught_from_angles(phi_and_theta, n);
            Ray scattered_ray {p, wnaught, Real(1e-5), infinity<Real>()};
            // std::cout << "Depth: " << depth << std::endl;
            L += refl * max(dot(n, wnaught), 0.0) / pdf / c_PI 
                  * radiance_roulette(scene, scattered_ray, rng, depth-1);

        } else if (auto *mirror = std::get_if<Mirror>(&mat)) {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            Vector3 F0 = eval(mirror->reflectance, hit_isect->uv);
            Vector3 F = schlick_fresnel(F0, dot(n, refl_ray.dir));
            L += F * radiance_roulette(scene, refl_ray, rng, depth-1);

        } else if (auto *plastic = std::get_if<Plastic>(&mat)) {
            Real F0 = square((plastic->eta - 1) / (plastic->eta + 1));
            Real F = schlick_fresnel(F0, dot(n, -ray.dir));
            Real RandomVar = next_pcg32_real<Real>(rng);
            // Send ray on specular case
            if(RandomVar <= F) {
                Ray refl_ray{p, ray.dir - 2*dot(ray.dir, n)*n, Real(1e-4), infinity<Real>()};
                L += radiance_roulette(scene, refl_ray, rng, depth-1);
            // Send ray on diffusive case
            } else {
                refl = eval(plastic->reflectance, hit_isect->uv);
                Vector2 u1u2{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

                auto phi_and_theta = sample_wnaught(u1u2);
                auto pdf = pdf_wnaught(phi_and_theta);
                Vector3 wnaught = wnaught_from_angles(phi_and_theta, n);
                Ray scattered {p, wnaught, Real(1e-5), infinity<Real>()};
                L += refl * max(dot(n, wnaught), 0.0) / pdf / c_PI * radiance_roulette(scene, scattered, rng, depth-1);
            }
            refl = Vector3{0, 0, 0};
        }
        return L / Real(0.9);
    }
    return scene.background_color; //scene.background_color;
}

Image3 hw_4_1(const std::vector<std::string> &params) {
    // Homework 4.1: diffuse interreflection
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
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
                    img(x, y) += radiance(scene, ray, rng, max_depth) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_4_2(const std::vector<std::string> &params) {
    // Homework 4.2: adding more materials
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
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
                    img(x, y) += radiance(scene, ray, rng, max_depth) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_4_3(const std::vector<std::string> &params) {
    // Homework 4.3: multiple importance sampling

    // Added Russian Roulette Pathtracing
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
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
                    img(x, y) += radiance_roulette(scene, ray, rng, max_depth) / Real(spp);
                }
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}
