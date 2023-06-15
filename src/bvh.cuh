#pragma once

#include "bbox.cuh"
#include <vector>

struct BVHNode {
    BBox box;
    int left_node_id;
    int right_node_id;
    int primitive_id;
};

struct BBoxWithID {
    BBox box;
    int id;
};

int construct_bvh(const std::vector<BBoxWithID> &boxes,
                  std::vector<BVHNode> &node_pool);
