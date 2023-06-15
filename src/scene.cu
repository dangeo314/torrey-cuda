#include "scene.cuh"
#include "flexception.cuh"
#include "image.cuh"
#include "ray.cuh"
#include "texture.cuh"
#include <algorithm>

Texture parsed_color_to_texture(const ParsedColor &color) {
    if (auto *constant = std::get_if<Vector3>(&color)) {
        return ConstantTexture{*constant};
    } else if (auto *image = std::get_if<ParsedImageTexture>(&color)) {
        Image3 img = imread3(image->filename);
        return ImageTexture{img,
            image->uscale, image->vscale,
            image->uoffset, image->voffset};
    } else {
        Error("Unhandled ParsedColor");
    }
}

Scene::Scene(const ParsedScene &scene) :
        camera(from_parsed_camera(scene.camera)),
        width(scene.camera.width),
        height(scene.camera.height),
        background_color(scene.background_color),
        samples_per_pixel(scene.samples_per_pixel) {
    // Extract triangle meshes from the parsed scene.
    int tri_mesh_count = 0;
    for (const ParsedShape &parsed_shape : scene.shapes) {
        if (std::get_if<ParsedTriangleMesh>(&parsed_shape)) {
            tri_mesh_count++;
        }
    }
    meshes.resize(tri_mesh_count);

    // Extract the shapes
    tri_mesh_count = 0;
    for (int i = 0; i < (int)scene.shapes.size(); i++) {
        const ParsedShape &parsed_shape = scene.shapes[i];
        if (auto *sph = std::get_if<ParsedSphere>(&parsed_shape)) {
            if (sph->area_light_id >= 0) {
                const ParsedDiffuseAreaLight &light =
                    std::get<ParsedDiffuseAreaLight>(
                        scene.lights[sph->area_light_id]);
                lights.push_back(DiffuseAreaLight{
                    (int)shapes.size(), light.radiance});
            }
            shapes.push_back(
                Sphere{{sph->material_id, sph->area_light_id},
                    sph->position, sph->radius});
        } else if (auto *parsed_mesh = std::get_if<ParsedTriangleMesh>(&parsed_shape)) {
            meshes[tri_mesh_count] = TriangleMesh{
                {parsed_mesh->material_id, parsed_mesh->area_light_id},
                parsed_mesh->positions, parsed_mesh->indices,
                parsed_mesh->normals, parsed_mesh->uvs};
            // Extract all the individual triangles
            for (int face_index = 0; face_index < (int)parsed_mesh->indices.size(); face_index++) {
                if (parsed_mesh->area_light_id >= 0) {
                    const ParsedDiffuseAreaLight &light =
                        std::get<ParsedDiffuseAreaLight>(
                            scene.lights[parsed_mesh->area_light_id]);
                    lights.push_back(DiffuseAreaLight{
                        (int)shapes.size(), light.radiance});
                }
                shapes.push_back(Triangle{face_index, &meshes[tri_mesh_count]});
            }
            tri_mesh_count++;
        } else {
            // Unhandled case
            assert(false);
        }
    }

    // Copy the materials
    for (const ParsedMaterial &parsed_mat : scene.materials) {
        if (auto *diffuse = std::get_if<ParsedDiffuse>(&parsed_mat)) {
            materials.push_back(Diffuse{
                parsed_color_to_texture(diffuse->reflectance)});
        } else if (auto *mirror = std::get_if<ParsedMirror>(&parsed_mat)) {
            materials.push_back(Mirror{
                parsed_color_to_texture(mirror->reflectance)});
        } else if (auto *plastic = std::get_if<ParsedPlastic>(&parsed_mat)) {
            materials.push_back(Plastic{
                plastic->eta,
                parsed_color_to_texture(plastic->reflectance)});
        } else {
            // Unhandled case
            assert(false);
        }
    }

    // Copy the lights
    for (const ParsedLight &parsed_light : scene.lights) {
        if (auto *point_light = std::get_if<ParsedPointLight>(&parsed_light)) {
            lights.push_back(PointLight{point_light->intensity, point_light->position});
        }
    }

    // Build BVH
    std::vector<BBoxWithID> bboxes(shapes.size());
    for (int i = 0; i < (int)bboxes.size(); i++) {
        if (auto *sph = std::get_if<Sphere>(&shapes[i])) {
            Vector3 p_min = sph->center - sph->radius;
            Vector3 p_max = sph->center + sph->radius;
            bboxes[i] = {BBox{p_min, p_max}, i};
        } else if (auto *tri = std::get_if<Triangle>(&shapes[i])) {
            const TriangleMesh *mesh = tri->mesh;
            Vector3i index = mesh->indices[tri->face_index];
            Vector3 p0 = mesh->positions[index[0]];
            Vector3 p1 = mesh->positions[index[1]];
            Vector3 p2 = mesh->positions[index[2]];
            Vector3 p_min = min(min(p0, p1), p2);
            Vector3 p_max = max(max(p0, p1), p2);
            bboxes[i] = {BBox{p_min, p_max}, i};
        }
    }
    bvh_root_id = construct_bvh(bboxes, bvh_nodes);
}

std::optional<Intersection> intersect(const Scene &scene, const BVHNode &node, Ray ray) {
    if (node.primitive_id != -1) {
        return intersect(scene.shapes[node.primitive_id], ray);
    }
    const BVHNode &left = scene.bvh_nodes[node.left_node_id];
    const BVHNode &right = scene.bvh_nodes[node.right_node_id];
    std::optional<Intersection> isect_left;
    if (intersect(left.box, ray)) {
        isect_left = intersect(scene, left, ray);
        if (isect_left) {
            ray.tfar = isect_left->distance;
        }
    }
    if (intersect(right.box, ray)) {
        // Since we've already set ray.tfar to the left node
        // if we still hit something on the right, it's closer
        // and we should return that.
        if (auto isect_right = intersect(scene, right, ray)) {
            return isect_right;
        }
    }
    return isect_left;
}

bool occluded(const Scene &scene, const BVHNode &node, Ray ray) {
    if (node.primitive_id != -1) {
        return occluded(scene.shapes[node.primitive_id], ray);
    }
    const BVHNode &left = scene.bvh_nodes[node.left_node_id];
    const BVHNode &right = scene.bvh_nodes[node.right_node_id];
    if (intersect(left.box, ray)) {
        if (occluded(scene, left, ray)) {
            return true;
        }
    }
    if (intersect(right.box, ray)) {
        if (occluded(scene, right, ray)) {
            return true;
        }
    }
    return false;
}

std::optional<Intersection> intersect(const Scene &scene, Ray ray) {
    // std::optional<Intersection> hit_isect;
    // for (const auto &shape : scene.shapes) {
    //     if (auto isect = intersect(shape, ray)) {
    //         ray.tfar = isect->distance;
    //         hit_isect = *isect;
    //     }
    // }
    // return hit_isect;
    return intersect(scene, scene.bvh_nodes[scene.bvh_root_id], ray);
}

bool occluded(const Scene &scene, const Ray &ray) {
    // for (const Shape &shape : scene.shapes) {
    //     if (occluded(shape, ray)) {
    //         return true;
    //     }
    // }
    // return false;
    return occluded(scene, scene.bvh_nodes[scene.bvh_root_id], ray);
}
