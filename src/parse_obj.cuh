#pragma once

#include "torrey.cuh"
#include "matrix.cuh"
#include "parse_scene.cuh"
#include <filesystem>

/// Parse Wavefront obj files. Currently only supports triangles and quads.
/// Throw errors if encountered general polygons.
ParsedTriangleMesh parse_obj(const fs::path &filename, const Matrix4x4 &to_world);
