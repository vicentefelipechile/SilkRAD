/**
 * For routines that are shared between BSP and CUDABSP.
 */

#include "cuda_runtime.h"

#include "bsp_shared.h"
#include "cudautils.h"


__device__ __host__ int16_t BSPShared::cluster_for_pos(
        const BSP::DPlane* planes,
        const BSP::DNode* nodes,
        const BSP::DLeaf* leaves,
        float3 pos
        ) {

    const float EPSILON = 0.1f;

    int32_t nodeIndexStack[1024];   // empty ascending stack
    size_t stackSize = 0;

    nodeIndexStack[stackSize++] = 0;

    while (stackSize > 0) {
        if (stackSize >= 1024) {
            printf("ALERT: PVS stack size too big!!!\n");
            return -1;
        }

        int32_t nodeIndex = nodeIndexStack[--stackSize];

        if (nodeIndex < 0) {
            // We found a leaf!
            int32_t leafIndex = -1 - nodeIndex;
            const BSP::DLeaf& leaf = leaves[leafIndex];

            int16_t cluster = leaf.cluster;

            if (cluster != -1) {
                // The leaf is valid. We're done!
                return cluster;
            }
            else {
                // The leaf was invalid. Keep trying to find a valid leaf...
                continue;
            }
        }

        const BSP::DNode& node = nodes[nodeIndex];
        const BSP::DPlane& plane = planes[node.planeNum];

        float planeDist = plane.dist;
        float3 planeNormal = make_float3(
            plane.normal.x, plane.normal.y, plane.normal.z
        );

        float diff = dot(pos, planeNormal) - planeDist;

        if (diff > EPSILON) {
            // The point is in front of the node's partition.
            nodeIndexStack[stackSize++] = node.children[0];
        }
        else if (diff < -EPSILON) {
            // The point is behind the node's partition.
            nodeIndexStack[stackSize++] = node.children[1];
        }
        else {
            // The point is close enough to the node's partition that we can't
            // really be sure which side it's on... recurse on both sides to
            // try and find the leaf.
            nodeIndexStack[stackSize++] = node.children[0];
            nodeIndexStack[stackSize++] = node.children[1];
        }
    }

    // Didn't find any leaf... welp.
    return -1;
}


__host__ int16_t BSPShared::cluster_for_pos(
        const BSP::BSP& bsp, const BSP::Vec3<float>& pos
        ) {

    const BSP::DPlane* planes = bsp.get_planes().data();
    const BSP::DNode* nodes = bsp.get_nodes().data();
    const BSP::DLeaf* leaves = bsp.get_leaves().data();

    return cluster_for_pos(planes, nodes, leaves, make_float3(pos));
}

