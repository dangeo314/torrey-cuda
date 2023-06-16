#include "hw2.cuh"
#include "bbox.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "parallel.cuh"
#include "parse_scene.cuh"
// #include "pcg.cuh"
#include "print_scene.cuh"
#include "progressreporter.cuh"
#include "ray.cuh"
#include "scene.cuh"
#include "timer.cuh"
#include <optional>

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

/* Homework 2.1
Image3 hw_2_1(const std::vector<std::string> &params) {
    // Homework 2.1: render a single triangle and outputs
    // its barycentric coordinates.
    // We will use the following camera parameter
    // lookfrom = (0, 0,  0)
    // lookat   = (0, 0, -1)
    // up       = (0, 1,  0)
    // vfov     = 45
    // and we will parse the triangle vertices from params
    // The three vertices are stored in v0, v1, and v2 below.

    std::vector<float> tri_params;
    int spp = 16;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-spp") {
            spp = std::stoi(params[++i]);
        } else {
            tri_params.push_back(std::stof(params[i]));
        }
    }

    if (tri_params.size() < 9) {
        // Not enough parameters to parse the triangle vertices.
        return Image3(0, 0);
    }

    Vector3 p0{tri_params[0], tri_params[1], tri_params[2]};
    Vector3 p1{tri_params[3], tri_params[4], tri_params[5]};
    Vector3 p2{tri_params[6], tri_params[7], tri_params[8]};

    Image3 img(640, 480);
    Camera cam{
        Vector3{0, 0,  0}, // lookfrom
        Vector3{0, 0, -1}, // lookat
        Vector3{0, 1,  0}, // up
        Real(45)
    };

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(cam, w, h);

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

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    if (auto uvt_ = intersect_triangle(ray, p0, p1, p2)) {
                        Vector3 uvt = *uvt_;
                        Vector3 wuv{1 - uvt[0] - uvt[1], uvt[0], uvt[1]};
                        img(x, y) += wuv / Real(spp);
                    } else {
                        img(x, y) += Vector3{0.5, 0.5, 0.5} / Real(spp);
                    }
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}
*/

/* Homework 2.2
Image3 hw_2_2(const std::vector<std::string> &params) {
    // Homework 2.2: render a triangle mesh.
    // We will use the same camera parameter:
    // lookfrom = (0, 0,  0)
    // lookat   = (0, 0, -1)
    // up       = (0, 1,  0)
    // vfov     = 45
    // and we will use a fixed triangle mesh: a tetrahedron!
    int spp = 16;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-spp") {
            spp = std::stoi(params[++i]);
        }
    }

    std::vector<Vector3> positions = {
        Vector3{ 0.0,  0.5, -2.0},
        Vector3{ 0.0, -0.3, -1.0},
        Vector3{ 1.0, -0.5, -3.0},
        Vector3{-1.0, -0.5, -3.0}
    };
    std::vector<Vector3i> indices = {
        Vector3i{0, 1, 2},
        Vector3i{0, 3, 1},
        Vector3i{0, 2, 3},
        Vector3i{1, 2, 3}
    };

    Image3 img(640, 480);
    Camera cam{
        Vector3{0, 0,  0}, // lookfrom
        Vector3{0, 0, -1}, // lookat
        Vector3{0, 1,  0}, // up
        Real(45)
    };

    int w = img.width;
    int h = img.height;

    CameraRayData cam_ray_data = compute_camera_ray_data(cam, w, h);

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

                    Ray ray = generate_primary_ray(cam_ray_data, u, v);
                    Vector3 color = Vector3{0.5, 0.5, 0.5};
                    Real dist = infinity<Real>();
                    Vector3 wuv = Vector3{0.5, 0.5, 0.5};
                    for (int i = 0; i < (int)indices.size(); i++) {
                        Vector3 p0 = positions[indices[i][0]];
                        Vector3 p1 = positions[indices[i][1]];
                        Vector3 p2 = positions[indices[i][2]];
                        if (auto uvt_ = intersect_triangle(ray, p0, p1, p2)) {
                            Vector3 uvt = *uvt_;
                            if (uvt[2] < dist) {
                                wuv = Vector3{1 - uvt[0] - uvt[1], uvt[0], uvt[1]};
                                dist = uvt[2];
                            }
                        }
                    }

                    img(x, y) += wuv / Real(spp);
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}
*/

__device__ static Vector3 radiance(const Scene &scene, Ray ray) {
    Vector3 color = Vector3{0.5, 0.5, 0.5};
    auto hit_isect = intersect(scene, ray);
    if (hit_isect.distance < infinity<Real>()) {
        Vector3 p = hit_isect.position;
        Vector3 n = hit_isect.geometric_normal;
        if (dot(-ray.dir, n) < 0) {
            n = -n;
        }
        
        const Material &mat = scene.materials_ptr[hit_isect.material_id];
        if (mat.type == DIFFUSE) {
            const Diffuse* diffuse = &mat.data.diffuse;
            Vector3 L = Vector3{0, 0, 0};
            // Loop over the lights
            for (int i = 0; i < scene.num_lights; i++) {
                // Assume point lights for now.
                const Light &light = scene.lights_ptr[i];
                const PointLight &point_light = light.data.point;
                Vector3 l = point_light.position - p;
                Ray shadow_ray{p, normalize(l), Real(1e-4), (1 - Real(1e-4)) * length(l)};
                if (!occluded(scene, shadow_ray)) {
                    ConstantTexture c = diffuse->reflectance.data.constant;
                    L += (max(dot(n, normalize(l)), Real(0)) / c_PI) *
                         (point_light.intensity / length_squared(l)) *
                         c.color;
                }
            }
            color = L;
        } else if (mat.type == MIRROR) {
            const Mirror* mirror = &mat.data.mirror;
            ConstantTexture c = mirror->reflectance.data.constant;
            Ray refl_ray{p, ray.dir - 2 * dot(ray.dir, n) * n, Real(1e-4), infinity<Real>()};
            return c.color * radiance(scene, refl_ray);
        }
    }
    return color;
}

__global__ void generate_parallel_image(Vector3* img, Scene* scene, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if (x < w && y < h) {
        Real u = (x + 0.5f) / w;
        Real v = (y + 0.5f) / h;

        CameraRayData cam_ray_data = compute_camera_ray_data(scene->camera, w, h);
        Ray ray = generate_primary_ray(cam_ray_data, u, v);

        Vector3 pixel_color = radiance(*scene, ray);    // insert radiance logic here;

        img[y*w+x] = pixel_color;
    }
}

/* Homework 2.3 
Image3 hw_2_3(const std::vector<std::string> &params) {
    // Homework 2.3: render a scene file provided by our parser.
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    Timer timer;
    tick(timer);
    ParsedScene parsed_scene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;
    std::cout << parsed_scene << std::endl;

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
                    img(x, y) += radiance(scene, ray) / Real(spp);
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}
*/

/* Homework 2.4
Image3 hw_2_4(const std::vector<std::string> &params) {
    // Homework 2.4: render AABBs of the shapes
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    Timer timer;
    tick(timer);
    ParsedScene parsed_scene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;
    std::cout << parsed_scene << std::endl;
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

    std::vector<BBox> bboxes(scene.shapes.size());
    for (int i = 0; i < (int)bboxes.size(); i++) {
        if (auto *sph = std::get_if<Sphere>(&scene.shapes[i])) {
            Vector3 p_min = sph->center - sph->radius;
            Vector3 p_max = sph->center + sph->radius;
            bboxes[i] = BBox{p_min, p_max};
        } else if (auto *tri = std::get_if<Triangle>(&scene.shapes[i])) {
            const TriangleMesh *mesh = tri->mesh;
            Vector3i index = mesh->indices[tri->face_index];
            Vector3 p0 = mesh->positions[index[0]];
            Vector3 p1 = mesh->positions[index[1]];
            Vector3 p2 = mesh->positions[index[2]];
            Vector3 p_min = min(min(p0, p1), p2);
            Vector3 p_max = max(max(p0, p1), p2);
            bboxes[i] = BBox{p_min, p_max};
        }
    }

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
                    bool hit = false;
                    for (const BBox &bbox : bboxes) {
                        if (intersect(bbox, ray)) {
                            hit = true;
                            break;
                        }
                    }
                    if (hit) {
                        img(x, y) += Vector3{1, 1, 1} / Real(spp);
                    } else {
                        img(x, y) += Vector3{0.5, 0.5, 0.5} / Real(spp);
                    }
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}
*/

Image3 hw_2_5(const std::vector<std::string> &params) {
    // Homework 2.5: rendering with BVHs
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

    int w = scene.width;
    int h = scene.height;

    Vector3* img;
    size_t size = w * h * sizeof(Vector3);

    Scene* sceneUnified;

    // Assuming Vector3 is a struct with three float members (x, y, z)
    checkCudaErrors(cudaMallocManaged(&img, size));
    checkCudaErrors(cudaMallocManaged(&sceneUnified, sizeof(Scene)));
    
    cudaMemcpy(sceneUnified, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

    // Assuming that 'blockSize' is a dim3 type specifying the number of threads per block
    // and 'numBlocks' is a dim3 type specifying the number of blocks.
    // You need to choose these values according to your GPU capabilities and image size.
    dim3 blockSize(16, 16);
    dim3 numBlocks((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    generate_parallel_image<<<numBlocks, blockSize>>>(img, sceneUnified, w, h);

    // cudaMemCpy from DeviceImage back to HostImage
    cudaDeviceSynchronize();

    // Got rid of PCG and Antialiasing until later

    tick(timer);
    std::cout << "Rendering done. Took " << tick(timer) << " seconds." << std::endl;

    return Image3(img, w,h);
}
