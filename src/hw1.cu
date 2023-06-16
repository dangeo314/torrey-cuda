#include "hw1.cuh"
#include "parallel.cuh"
#include "pcg.cuh"
#include "sphere.cuh"
#include "ray.cuh"
#include "vector.cuh"
#include "hw1_scenes.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

using namespace hw1;

__global__ void generate_parallel_image(Vector3* img, int w, int h, Vector3 origin, Vector3 horizontal, Vector3 vertical, Vector3 top_left_corner) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < w && y < h) {
        Real u = (x + 0.5f) / w;
        Real v = (y + 0.5f) / h;
        Vector3 pixel_color = normalize(top_left_corner + u * horizontal - v * vertical - origin);

        img[y*w+x] = pixel_color;
    }
}

Image3 hw_1_1(const std::vector<std::string> &/*params*/) {
    // Homework 1.1: generate camera rays and output the ray directions
    // The camera is positioned at (0, 0, 0), facing towards (0, 0, -1),
    // with an up vector (0, 1, 0) and a vertical field of view of 90 degree.

    int w = 640;
    int h = 480;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0;
    Real viewport_width = aspect_ratio * viewport_height;
    Real focal_length = 1.0;

    Vector3 origin = Vector3{0, 0, 0};
    Vector3 horizontal = Vector3{viewport_width, Real(0), Real(0)};
    Vector3 vertical = Vector3{Real(0), viewport_height, Real(0)};
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) -
        Vector3{Real(0), Real(0), focal_length};

    Vector3* img;
    size_t size = w * h * sizeof(Vector3);

    // Assuming Vector3 is a struct with three float members (x, y, z)
    checkCudaErrors(cudaMallocManaged(&img, size));

    // Assuming that 'blockSize' is a dim3 type specifying the number of threads per block
    // and 'numBlocks' is a dim3 type specifying the number of blocks.
    // You need to choose these values according to your GPU capabilities and image size.
    dim3 blockSize(16, 16);
    dim3 numBlocks((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    generate_parallel_image<<<numBlocks, blockSize >>>(img, w, h, origin, horizontal, vertical, top_left_corner);

    // cudaMemCpy from DeviceImage back to HostImage
    cudaDeviceSynchronize();

    return Image3(img, w,h);
}


/* Homework 1.2
Image3 hw_1_2(const std::vector<std::string> & params) {
    // Homework 1.2: intersect the rays generated from hw_1_1
    // with a unit sphere located at (0, 0, -2)

    // Image3 img(640, 480);
    Sphere sph{Vector3{0, 0, -2}, Real(1), -1};

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0;
    Real viewport_width = aspect_ratio * viewport_height;
    Real focal_length = 1.0;

    Vector3 origin = Vector3{0, 0, 0};
    Vector3 horizontal = Vector3{viewport_width, Real(0), Real(0)};
    Vector3 vertical = Vector3{Real(0), viewport_height, Real(0)};
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) -
        Vector3{Real(0), Real(0), focal_length};

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            Real u = (x + Real(0.5)) / w;
            Real v = (y + Real(0.5)) / h;

            Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
            Ray ray{origin, dir, Real(0), infinity<Real>()};

            if (auto isect = intersect(sph, ray)) {
                Vector3 n = isect->geometric_normal;
                img(x, y) = (n + Real(1)) / Real(2);
            } else {
                img(x, y) = Vector3{0.5, 0.5, 0.5};
            }
        }
    }

    return img;
}
*/


/* Homework 1.3
Image3 hw_1_3(const std::vector<std::string> &params) {
    // Homework 1.3: add camera control to hw_1_2. 
    // We will use a look at transform:
    // The inputs are "lookfrom" (camera position),
    //                "lookat" (target),
    //                and the up vector
    // and the vertical field of view (in degrees).
    // If the user did not specify, fall back to the default
    // values below.
    // If you use the default values, it should render
    // the same image as hw_1_2.

    Vector3 lookfrom = Vector3{0, 0,  0};
    Vector3 lookat   = Vector3{0, 0, -2};
    Vector3 up       = Vector3{0, 1,  0};
    Real    vfov     = 90;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-lookfrom") {
            Real x = std::stof(params[++i]);
            Real y = std::stof(params[++i]);
            Real z = std::stof(params[++i]);
            lookfrom = Vector3{x, y, z};
        } else if (params[i] == "-lookat") {
            Real x = std::stof(params[++i]);
            Real y = std::stof(params[++i]);
            Real z = std::stof(params[++i]);
            lookat = Vector3{x, y, z};
        } else if (params[i] == "-up") {
            Real x = std::stof(params[++i]);
            Real y = std::stof(params[++i]);
            Real z = std::stof(params[++i]);
            up = Vector3{x, y, z};
        } else if (params[i] == "-vfov") {
            vfov = std::stof(params[++i]);
        }
    }

    Image3 img(640, 480);
    Sphere sph{Vector3{0, 0, -2}, Real(1), -1};

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0 * tan(radians(vfov / 2));
    Real viewport_width = aspect_ratio * viewport_height;

    Vector3 cam_dir = normalize(lookat - lookfrom);
    Vector3 right = normalize(cross(cam_dir, up));
    Vector3 new_up = cross(right, cam_dir);

    Vector3 origin = lookfrom;
    Vector3 horizontal = viewport_width * right;
    Vector3 vertical = viewport_height * new_up;
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) + cam_dir;

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            Real u = (x + Real(0.5)) / w;
            Real v = (y + Real(0.5)) / h;

            Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
            Ray ray{origin, dir, Real(0), infinity<Real>()};

            if (auto isect = intersect(sph, ray)) {
                Vector3 n = isect->geometric_normal;
                img(x, y) = (n + Real(1)) / Real(2);
            } else {
                img(x, y) = Vector3{0.5, 0.5, 0.5};
            }
        }
    }

    return img;
}
*/

/* Homework 1.4
Image3 hw_1_4(const std::vector<std::string> &params) {
    // Homework 1.4: render the scenes defined in hw1_scenes.h
    // output their diffuse color directly.
    if (params.size() == 0) {
        return Image3(0, 0);
    }

    int scene_id = std::stoi(params[0]);

    Image3 img(640, 480);
    const Scene &scene = hw1_scenes[scene_id];
    const Camera &cam = scene.camera;

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0 * tan(radians(cam.vfov / 2));
    Real viewport_width = aspect_ratio * viewport_height;

    Vector3 cam_dir = normalize(cam.lookat - cam.lookfrom);
    Vector3 right = normalize(cross(cam_dir, cam.up));
    Vector3 new_up = cross(right, cam_dir);

    Vector3 origin = cam.lookfrom;
    Vector3 horizontal = viewport_width * right;
    Vector3 vertical = viewport_height * new_up;
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) + cam_dir;

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            Real u = (x + Real(0.5)) / w;
            Real v = (y + Real(0.5)) / h;

            Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
            Ray ray{origin, dir, Real(0), infinity<Real>()};

            Vector3 color = Vector3{0.5, 0.5, 0.5};
            for (const Sphere &sph : scene.shapes) {
                if (auto isect = intersect(sph, ray)) {
                    color = scene.materials[sph.material_id].color;
                    ray.tfar = isect->distance;
                }
            }
            img(x, y) = color;
        }
    }
    return img;
}
*/

/* Homework 1.5 
Image3 hw_1_5(const std::vector<std::string> &params) {
    // Homework 1.5: render the scenes defined in hw1_scenes.h,
    // light them using the point lights in the scene.
    if (params.size() == 0) {
        return Image3(0, 0);
    }

    int scene_id = std::stoi(params[0]);
    Image3 img(640, 480);
    const Scene &scene = hw1_scenes[scene_id];
    const Camera &cam = scene.camera;

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0 * tan(radians(cam.vfov / 2));
    Real viewport_width = aspect_ratio * viewport_height;

    Vector3 cam_dir = normalize(cam.lookat - cam.lookfrom);
    Vector3 right = normalize(cross(cam_dir, cam.up));
    Vector3 new_up = cross(right, cam_dir);

    Vector3 origin = cam.lookfrom;
    Vector3 horizontal = viewport_width * right;
    Vector3 vertical = viewport_height * new_up;
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) + cam_dir;

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            Real u = (x + Real(0.5)) / w;
            Real v = (y + Real(0.5)) / h;

            Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
            Ray ray{origin, dir, Real(0), infinity<Real>()};

            Vector3 color = Vector3{0.5, 0.5, 0.5};
            bool hit = false;
            Intersection hit_isect;
            for (const auto &shape : scene.shapes) {
                if (auto isect = intersect(shape, ray)) {
                    ray.tfar = isect->distance;
                    hit_isect = *isect;
                    hit = true;
                }
            }
            if (hit) {
                Vector3 L = Vector3{0, 0, 0};
                Vector3 p = hit_isect.position;
                Vector3 n = hit_isect.geometric_normal;
                if (dot(-dir, n) < 0) {
                    n = -n;
                }
                // loop over the lights
                for (const PointLight &light : scene.lights) {
                    Vector3 l = light.position - p;
                    Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                    if (!occluded(scene, shadow_ray)) {
                        L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                             (light.intensity / length_squared(l)) *
                             scene.materials[hit_isect.material_id].color;
                    }
                }
                color = L;
            }
            img(x, y) = color;
        }
    }
    return img;
}
*/

/* Homework 1.6: add antialiasing to homework 1.5 
Image3 hw_1_6(const std::vector<std::string> &params) {
    // Homework 1.6: add antialiasing to homework 1.5
    if (params.size() == 0) {
        return Image3(0, 0);
    }

    int scene_id = 0;
    int spp = 64;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-spp") {
            spp = std::stoi(params[++i]);
        } else {
            scene_id = std::stoi(params[i]);
        }
    }

    Image3 img(160, 120);

    const Scene &scene = hw1_scenes[scene_id];
    const Camera &cam = scene.camera;

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0 * tan(radians(cam.vfov / 2));
    Real viewport_width = aspect_ratio * viewport_height;

    Vector3 cam_dir = normalize(cam.lookat - cam.lookfrom);
    Vector3 right = normalize(cross(cam_dir, cam.up));
    Vector3 new_up = cross(right, cam_dir);

    Vector3 origin = cam.lookfrom;
    Vector3 horizontal = viewport_width * right;
    Vector3 vertical = viewport_height * new_up;
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) + cam_dir;

    pcg32_state rng = init_pcg32();

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int s = 0; s < spp; s++) {
                Real u, v;
                if (spp == 1) {
                    u = (x + Real(0.5)) / w;
                    v = (y + Real(0.5)) / h;
                } else {
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;
                }

                Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
                Ray ray{origin, dir, Real(0), infinity<Real>()};

                Vector3 color = Vector3{0.5, 0.5, 0.5};
                bool hit = false;
                Intersection hit_isect;
                for (const auto &shape : scene.shapes) {
                    if (auto isect = intersect(shape, ray)) {
                        ray.tfar = isect->distance;
                        hit_isect = *isect;
                        hit = true;
                    }
                }
                if (hit) {
                    Vector3 L = Vector3{0, 0, 0};
                    Vector3 p = hit_isect.position;
                    Vector3 n = hit_isect.geometric_normal;
                    if (dot(-dir, n) < 0) {
                        n = -n;
                    }
                    // loop over the lights
                    for (const PointLight &light : scene.lights) {
                        Vector3 l = light.position - p;
                        Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                        if (!occluded(scene, shadow_ray)) {
                            L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                                 (light.intensity / length_squared(l)) *
                                 scene.materials[hit_isect.material_id].color;
                        }
                    }
                    color = L;
                }
                img(x, y) += color / Real(spp);
            }
        }
    }
    return img;
}
*/

/* Radiance function for Homework 1.7
Vector3 radiance(const Scene &scene, Ray ray) {
    Vector3 color = Vector3{0.5, 0.5, 0.5};
    bool hit = false;
    Intersection hit_isect;
    for (const auto &shape : scene.shapes) {
        if (auto isect = intersect(shape, ray)) {
            ray.tfar = isect->distance;
            hit_isect = *isect;
            hit = true;
        }
    }
    if (hit) {
        Vector3 p = hit_isect.position;
        Vector3 n = hit_isect.geometric_normal;
        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }
        const Material &mat = scene.materials[hit_isect.material_id];
        if (mat.type == MaterialType::Diffuse) {
            Vector3 L = Vector3{0, 0, 0};
            // loop over the lights
            for (const PointLight &light : scene.lights) {
                Vector3 l = light.position - p;
                Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                if (!occluded(scene, shadow_ray)) {
                    L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                         (light.intensity / length_squared(l)) *
                         mat.color;
                }
            }
            color = L;
        } else {
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            return mat.color * radiance(scene, refl_ray);
        }
    }
    return color;
}
*/

/* Homework 1.7: add mirror materials to homework 1.6
Image3 hw_1_7(const std::vector<std::string> &params) {
    // Homework 1.7: add mirror materials to homework 1.6
    if (params.size() == 0) {
        return Image3(0, 0);
    }

    int scene_id = 0;
    int spp = 64;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-spp") {
            spp = std::stoi(params[++i]);
        } else {
            scene_id = std::stoi(params[i]);
        }
    }

    Image3 img(640, 480);

    const Scene &scene = hw1_scenes[scene_id];
    const Camera &cam = scene.camera;

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0 * tan(radians(cam.vfov / 2));
    Real viewport_width = aspect_ratio * viewport_height;

    Vector3 cam_dir = normalize(cam.lookat - cam.lookfrom);
    Vector3 right = normalize(cross(cam_dir, cam.up));
    Vector3 new_up = cross(right, cam_dir);

    Vector3 origin = cam.lookfrom;
    Vector3 horizontal = viewport_width * right;
    Vector3 vertical = viewport_height * new_up;
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) + cam_dir;

    pcg32_state rng = init_pcg32();

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int s = 0; s < spp; s++) {
                Real u, v;
                if (spp == 1) {
                    u = (x + Real(0.5)) / w;
                    v = (y + Real(0.5)) / h;
                } else {
                    u = (x + next_pcg32_real<Real>(rng)) / w;
                    v = (y + next_pcg32_real<Real>(rng)) / h;
                }

                Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
                Ray ray{origin, dir, Real(0), infinity<Real>()};
                Vector3 color = radiance(scene, ray);
                img(x, y) += color / Real(spp);
            }
        }
    }
    return img;
}
*/

/* Homework 1.8: parallelize HW 1.7
Image3 hw_1_8(const std::vector<std::string> &params) {
    // Homework 1.8: parallelize HW 1.7
    if (params.size() == 0) {
        return Image3(0, 0);
    }

    int scene_id = 0;
    int spp = 64;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-spp") {
            spp = std::stoi(params[++i]);
        } else {
            scene_id = std::stoi(params[i]);
        }
    }

    Image3 img(1280, 960);

    const Scene &scene = hw1_scenes[scene_id];
    const Camera &cam = scene.camera;

    int w = img.width;
    int h = img.height;

    Real aspect_ratio = Real(w) / Real(h);
    Real viewport_height = 2.0 * tan(radians(cam.vfov / 2));
    Real viewport_width = aspect_ratio * viewport_height;

    Vector3 cam_dir = normalize(cam.lookat - cam.lookfrom);
    Vector3 right = normalize(cross(cam_dir, cam.up));
    Vector3 new_up = cross(right, cam_dir);

    Vector3 origin = cam.lookfrom;
    Vector3 horizontal = viewport_width * right;
    Vector3 vertical = viewport_height * new_up;
    Vector3 top_left_corner =
        origin - horizontal / Real(2) + vertical / Real(2) + cam_dir;

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

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

                    Vector3 dir = normalize(top_left_corner + u * horizontal - v * vertical - origin);
                    Ray ray{origin, dir, Real(0), infinity<Real>()};
                    Vector3 color = radiance(scene, ray);
                    img(x, y) += color / Real(spp);
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}
*/