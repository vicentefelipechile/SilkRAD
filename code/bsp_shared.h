#ifndef __BSP_SHARED_H_
#define __BSP_SHARED_H_

#include <cstdint>
#include "cuda_runtime.h"

#include "bsp.h"


namespace BSPShared {
    __device__ __host__ int16_t cluster_for_pos(
        const BSP::DPlane* planes,
        const BSP::DNode* nodes,
        const BSP::DLeaf* leaves,
        float3 pos
    );

    __host__ int16_t cluster_for_pos(
        const BSP::BSP& bsp,
        const BSP::Vec3<float>& pos
    );
}

#endif
