#pragma once

#include "bbox.cuh"
#include <vector>
#include <thrust/universal_vector.h>

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

int construct_bvh(const thrust::universal_vector<BBoxWithID> &boxes,
                  thrust::universal_vector<BVHNode> &node_pool);
