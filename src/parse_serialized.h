#pragma once

#include "torrey.h"
#include "matrix.cuh"
#include "parse_scene.cuh"
#include <filesystem>

/// Parse Mitsuba's serialized file format.
ParsedTriangleMesh parse_serialized(const fs::path &filename,
                                    int shape_index,
                                    const Matrix4x4 &to_world);
